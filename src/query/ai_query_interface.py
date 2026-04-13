"""
AI Query Interface: catalog-backed schema → intent router → LangChain pandas agent
(self-correction on errors) → optional LLM auditor. Deterministic planner/churn unchanged.
"""

import logging
import json
import os
import re
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from src.query.catalog import validate_router_targets
from src.query.intent_codes import EXECUTOR_INTENT_CODES, UNKNOWN_INTENT, canonical_intent
from src.query.intent_payload import IntentPayload
from src.query.intent_router import IntentRouter
from src.query.query_planner import QueryPlanner
from src.query.query_normalization import apply_default_year_if_vague

logger = logging.getLogger(__name__)

ENGLISH_ONLY_NOTICE = "Please use English for your query. Non-English questions are not supported."

_RANKED_DATA_KEYS: Tuple[Tuple[str, str], ...] = (
    ("top_customers", "Customers (revenue)"),
    ("top_products", "Products (revenue)"),
    ("top_brands", "Brands (revenue)"),
    ("top_categories", "Categories (revenue)"),
)


def _numeric_sort_key(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _sorted_ranked_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        if k is None:
            continue
        items.append((str(k), v))
    return sorted(items, key=lambda kv: _numeric_sort_key(kv[1]), reverse=True)


def _format_ranked_dict_sections(data: Dict[str, Any], max_rows: int = 50) -> List[str]:
    lines: List[str] = []
    for key, heading in _RANKED_DATA_KEYS:
        raw = data.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        lines.append("")
        lines.append(f"{heading}:")
        for i, (name, amt) in enumerate(_sorted_ranked_items(raw)[:max_rows], 1):
            if isinstance(amt, (int, float)):
                lines.append(f"{i}. {name}: {amt:,.2f}")
            else:
                lines.append(f"{i}. {name}: {amt}")
    return lines


def _markdown_rows_display(rows: List[Dict[str, Any]], *, max_rows: int = 50) -> List[str]:
    """Readable list/table-style rows for web UI (bullets; de-duplicate party_desc when repeated)."""
    if not rows:
        return []
    head = rows[:max_rows]
    if all(isinstance(r, dict) and r.get("party_desc") is not None for r in head):
        names = [str(r.get("party_desc", "")).strip() for r in head]
        names = [n for n in names if n]
        if names:
            counts = Counter(names)
            if len(counts) < len(head) or len(head) <= 12:
                lines: List[str] = ["", "Customers (unique):"]
                for name in sorted(counts.keys()):
                    c = counts[name]
                    if c > 1:
                        lines.append(f"- {name} ({c} line items in this sample)")
                    else:
                        lines.append(f"- {name}")
                return lines
    lines = ["", "Details:"]
    for row in head:
        if not isinstance(row, dict):
            continue
        parts = [f"{k}={v}" for k, v in row.items()]
        lines.append("- " + ", ".join(parts))
    return lines


def compact_ranked_preview(step_data: Any, *, max_items: int = 10, max_chars: int = 900) -> str:
    """Short grounded string for multi-step synthesis (entity + amount)."""
    if not isinstance(step_data, dict):
        return ""
    chunks: List[str] = []
    for key, heading in _RANKED_DATA_KEYS:
        raw = step_data.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        pairs = _sorted_ranked_items(raw)[:max_items]
        part = heading + ": " + "; ".join(f"{n} ({amt})" for n, amt in pairs)
        chunks.append(part)
    out = "\n".join(chunks)
    return out[:max_chars]


class AIQueryInterface:
    """AI query interface with planner toggle and agent fallback."""

    def __init__(self, data_path: str = "data/processed", companyfn: Optional[str] = None):
        self.data_path = data_path
        self.companyfn = companyfn or "p11011004464072155"
        self.query_history: List[Dict[str, Any]] = []
        self._pandas_agent = None
        self._data_context = None
        self._schema_hint = ""
        self._query_planner = None
        self._query_executor = None
        self._query_validator = None
        self._llm = None
        self._config = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._query_log_dir = os.path.join(project_root, "logs")
        self._query_log_base = "query_history"
        self._query_log_max_bytes = 10 * 1024 * 1024
        self._session_memory: Dict[str, Dict[str, Any]] = {}

    def _init_agent(self) -> None:
        """Lazy initialize query pipeline components."""
        try:
            from src.query.data_context import DataContext
            from src.query.query_executor import QueryExecutor
            from src.query.query_config import load_query_config
            from src.query.query_planner import QueryPlanner
            from src.query.query_validator import QueryValidator

            self._data_context = DataContext(data_path=self.data_path, companyfn=self.companyfn)
            dataframes = self._data_context.get_dataframes()
            self._schema_hint = self._data_context.get_schema_description()
            if not dataframes:
                raise ValueError("No DataFrames loaded from data/processed/")

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self._config = load_query_config(project_root)
            self._query_executor = QueryExecutor(dataframes=dataframes)
            self._query_validator = QueryValidator(config=self._config)

            # LLM-dependent components (planner + agent) are optional at runtime.
            try:
                from src.query.llm_engine import get_llm
                from src.query.pandas_agent import PandasQueryAgent

                llm = get_llm()
                self._llm = llm
                self._query_planner = QueryPlanner(llm=llm)
                self._pandas_agent = PandasQueryAgent(
                    llm=llm,
                    dataframes=dataframes,
                    schema_description=self._schema_hint,
                )
                logger.info("[AIQueryInterface] Planner and Agent ready")
            except Exception as llm_exc:
                self._llm = None
                self._query_planner = None
                self._pandas_agent = None
                logger.warning(f"[AIQueryInterface] LLM components unavailable: {llm_exc}")

            logger.info("[AIQueryInterface] Core deterministic pipeline ready")
        except Exception as exc:
            logger.error(f"[AIQueryInterface] Could not initialize agent: {exc}")
            self._pandas_agent = None

    def _loaded_non_empty_datasets(self) -> Set[str]:
        if self._data_context is None:
            self._init_agent()
        if self._data_context is None:
            return set()
        dfs = self._data_context.get_dataframes()
        return {k for k, v in dfs.items() if v is not None and len(v) > 0}

    def _prepare_router_and_schema(self, query: str) -> Tuple[Dict[str, Any], List[str], str]:
        loaded = self._loaded_non_empty_datasets()
        if self._llm is not None:
            payloads = IntentRouter(self._llm).route(query, loaded)
        else:
            payloads = IntentRouter.fallback_for_loaded(loaded)
        targets = IntentRouter.merge_target_files(payloads, loaded)
        if not targets:
            targets = IntentRouter.merge_target_files(IntentRouter.fallback_for_loaded(loaded), loaded)
        route = IntentRouter.payloads_to_agent_route_dict(payloads)
        schema = ""
        if self._data_context is not None:
            schema = self._data_context.get_schema_description(targets)
        return route, targets, schema

    def process_query_with_agent(
        self,
        query: str,
        precomputed_route: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            if self._pandas_agent is None and self._query_executor is None:
                self._init_agent()
            if self._pandas_agent is None:
                return {"success": False, "error": "Agent initialization failed"}
            if precomputed_route is not None:
                route = precomputed_route
                loaded = self._loaded_non_empty_datasets()
                targets = validate_router_targets(route.get("target_files") or [], loaded)
                if not targets:
                    ips = route.get("intent_payloads") or []
                    plist = [IntentPayload.from_dict(x) for x in ips if isinstance(x, dict)]
                    if plist:
                        targets = IntentRouter.merge_target_files(plist, loaded)
                    if not targets:
                        targets = IntentRouter.merge_target_files(IntentRouter.fallback_for_loaded(loaded), loaded)
                schema = ""
                if self._data_context is not None:
                    schema = self._data_context.get_schema_description(targets)
            else:
                route, targets, schema = self._prepare_router_and_schema(query)
            self._pandas_agent.schema_description = schema
            retries = int(self._config.get("executor_max_runtime_retries", 3))
            retries = max(1, min(retries, 8))
            router_meta = {
                "intent_code": route.get("intent_code"),
                "target_files": targets,
                "task_type": route.get("task_type"),
                "rationale": route.get("rationale"),
                "intent_payloads": route.get("intent_payloads"),
                "execution_plan": route.get("execution_plan"),
            }
            timeout_seconds = int(self._config.get("agent_timeout_seconds", 25))
            timeout_seconds = max(5, min(timeout_seconds, 120))

            def _call() -> Dict[str, Any]:
                assert self._pandas_agent is not None
                return self._pandas_agent.query(
                    query,
                    ordered_dataset_names=targets,
                    router_payload=router_meta,
                    max_runtime_retries=retries,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                try:
                    return future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    return {
                        "success": False,
                        "error": f"Agent timeout after {timeout_seconds}s",
                        "source": "agent_timeout",
                        "reason_code": "agent_timeout",
                    }
        except Exception as exc:
            logger.warning(f"[AIQueryInterface] Agent query error: {exc}")
            return {"success": False, "error": str(exc), "reason_code": "agent_exception"}

    def terminal_parquet_and_df_map_line(self) -> str:
        """
        One line for the server console: df1/df2 in the pandas agent map to which dataset + parquet path.
        Same files back the intent planner / QueryExecutor in-memory tables.
        """
        try:
            if self._data_context is None:
                self._init_agent()
            if self._data_context is None:
                return "[AIQuery] parquet: (no data context)"
            rows = self._data_context.list_priority_sources_in_agent_order()
            if not rows:
                return "[AIQuery] parquet: (no non-empty sales_data/sales_main for agent)"
            parts = [
                f"{slot}={name} rows={n} file={path or '?'}"
                for slot, name, path, n in rows
            ]
            return f"[AIQuery] companyfn={self.companyfn}  parquet: " + " | ".join(parts)
        except Exception as exc:
            return f"[AIQuery] parquet: (error: {exc})"

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query through toggleable runtime flow."""
        context = context or {}
        resolved_query = self._resolve_contextual_query(query, context)
        if self._config.get("default_year_when_unspecified", True):
            resolved_query = apply_default_year_if_vague(resolved_query, datetime.now().year)
        started = time.perf_counter()
        try:
            if self._query_executor is None:
                self._init_agent()

            planner_enabled = bool(self._config.get("enable_intent_planner", True))

            if not self._is_english_query(resolved_query):
                result = {
                    "summary": ENGLISH_ONLY_NOTICE,
                    "data": {},
                    "insights": [],
                    "source": "language_policy",
                    "intent_planner_config_enabled": planner_enabled,
                }
                self._append_history(resolved_query, "language_policy")
                return result

            route_snapshot: Optional[Dict[str, Any]] = None
            payload_snapshots: List[IntentPayload] = []
            if planner_enabled:
                loaded = self._loaded_non_empty_datasets()
                if self._llm is not None:
                    payload_snapshots = IntentRouter(self._llm).route(resolved_query, loaded)
                else:
                    payload_snapshots = IntentRouter.fallback_for_loaded(loaded)
                if self._payloads_allow_deterministic_template(payload_snapshots):
                    route_snapshot = IntentRouter.payloads_to_agent_route_dict(payload_snapshots)
                else:
                    exec_plan_fb: Optional[Dict[str, Any]] = None
                    if self._query_planner:
                        exec_plan_fb = self._query_planner.plan(
                            resolved_query,
                            companyfn=self.companyfn,
                            router_payloads=payload_snapshots,
                        )
                    if exec_plan_fb is None:
                        exec_plan_fb = QueryPlanner.build_router_only_execution_plan(payload_snapshots)
                    route_snapshot = IntentRouter.payloads_to_agent_route_dict(
                        payload_snapshots, exec_plan_fb
                    )

            # Planner-first: all routed intents executable as templates → chained executor (no ReAct).
            if planner_enabled and route_snapshot is not None and self._payloads_allow_deterministic_template(
                payload_snapshots
            ):
                planned_result = self._run_planner_and_executor(
                    resolved_query, route_snapshot, payload_snapshots
                )
                if planned_result is not None:
                    self._maybe_synthesize_multi_step_result(planned_result, resolved_query)
                    self._append_history(
                        resolved_query,
                        planned_result.get("query_type", "planned"),
                        planned_result.get("source", "planner_executor"),
                        int((time.perf_counter() - started) * 1000),
                    )
                    self._update_session_memory(context, planned_result)
                    return planned_result
                self._append_history(
                    resolved_query,
                    "template_executor_unsatisfied",
                    "template_executor_unsatisfied",
                    int((time.perf_counter() - started) * 1000),
                )
                first_ic = (
                    canonical_intent(payload_snapshots[0].intent_code) if payload_snapshots else UNKNOWN_INTENT
                )
                return {
                    "summary": (
                        "This question maps to fixed analytics template(s) but could not be computed "
                        "(missing period such as month/year, or no matching rows). "
                        "Rephrase with an explicit date or simpler wording."
                    ),
                    "data": {"intent_code": first_ic, "step_count": len(payload_snapshots)},
                    "insights": [
                        "Template path exhausted; ReAct agent is not used when all router intents are template-safe.",
                    ],
                    "source": "template_executor_unsatisfied",
                    "intent_planner_config_enabled": planner_enabled,
                    "intent_router": route_snapshot,
                }

            churn_result = self._try_deterministic_churn(resolved_query)
            if churn_result is not None:
                self._append_history(
                    resolved_query,
                    "deterministic_churn",
                    "deterministic_churn",
                    int((time.perf_counter() - started) * 1000),
                )
                self._update_session_memory(context, churn_result)
                return churn_result

            agent_result = self.process_query_with_agent(
                resolved_query,
                precomputed_route=route_snapshot if planner_enabled else None,
            )
            answer = agent_result.get("answer", "")
            if agent_result.get("success"):
                audit_notes: List[str] = []
                if self._llm is not None:
                    try:
                        from src.query.response_auditor import ResponseAuditor

                        aud = ResponseAuditor(self._llm).audit_and_polish(resolved_query, answer)
                        answer = aud.get("final_answer", answer)
                        if aud.get("issues"):
                            audit_notes.append(f"Auditor notes: {aud.get('issues')}")
                        risk = str(aud.get("sensitive_risk") or "none")
                        if risk in ("medium", "high"):
                            audit_notes.append(f"Sensitivity flag: {risk}")
                    except Exception as aud_exc:
                        logger.warning(f"[AIQueryInterface] audit_and_polish failed: {aud_exc}")
                else:
                    audit_notes.append("ResponseAuditor skipped: no LLM (agent answer not polished).")

                insights = [
                    "Answered by LangChain Pandas Agent over catalog allowlisted columns.",
                ]
                if agent_result.get("router"):
                    insights.append(f"Intent router: {agent_result.get('router')}")
                if agent_result.get("active_datasets"):
                    insights.append(f"Active datasets: {agent_result.get('active_datasets')}")
                insights.extend(audit_notes)

                result = {
                    "summary": answer,
                    "data": {},
                    "insights": insights,
                    "source": "langchain_agent",
                    "intent_planner_config_enabled": planner_enabled,
                    "intent_router": agent_result.get("router"),
                    "active_datasets": agent_result.get("active_datasets"),
                }
                self._append_history(
                    resolved_query,
                    "langchain_agent",
                    "langchain_agent",
                    int((time.perf_counter() - started) * 1000),
                )
                self._update_session_memory(context, result)
                return result

            if self._is_rate_limit_error(agent_result.get("error", "")):
                wait_seconds = self._extract_retry_after_seconds(agent_result.get("error", ""))
                result = {
                    "summary": (
                        f"Groq is currently rate-limited. Please wait about {wait_seconds} seconds "
                        "and ask again."
                    ),
                    "data": {"retry_after_seconds": wait_seconds},
                    "insights": ["Temporary token-per-minute limit from provider."],
                    "source": "rate_limited",
                    "reason_code": "rate_limited",
                    "intent_planner_config_enabled": planner_enabled,
                }
                self._append_history(
                    resolved_query,
                    "rate_limited",
                    "rate_limited",
                    int((time.perf_counter() - started) * 1000),
                )
                return result

            reason_code = str(agent_result.get("reason_code") or "agent_failed")
            result = {
                "summary": f"No data found for the requested query. (reason: {reason_code})",
                "data": {"reason_code": reason_code},
                "insights": [f"Agent could not produce a grounded result in this turn ({reason_code})."],
                "source": "no_data",
                "reason_code": reason_code,
                "intent_planner_config_enabled": planner_enabled,
            }
            self._append_history(
                resolved_query,
                "no_data",
                "no_data",
                int((time.perf_counter() - started) * 1000),
            )
            return result
        except Exception as exc:
            logger.error(f"[AIQueryInterface] Error processing query: {exc}")
            return {"error": str(exc), "query": resolved_query, "source": "system_error"}

    def _try_deterministic_churn(self, query: str) -> Optional[Dict[str, Any]]:
        """Fixed pandas churn for explicit MM/YYYY or year; skips LLM."""
        try:
            from src.query.churn_deterministic import compute_churn_result, parse_churn_query

            spec = parse_churn_query(query)
            if not spec:
                return None
            if self._query_executor is None:
                self._init_agent()
            if not self._query_executor:
                return None
            sm = self._query_executor.dataframes.get("sales_main")
            if sm is None or len(sm) == 0:
                return None
            out = compute_churn_result(sm, spec)
            if out is None:
                return None
            planner_enabled = bool(self._config.get("enable_intent_planner", True))
            return {
                "summary": out["summary"],
                "data": out.get("data") or {},
                "insights": ["Deterministic churn from sales_main."],
                "source": "deterministic_churn",
                "intent_planner_config_enabled": planner_enabled,
            }
        except Exception as exc:
            logger.warning(f"[AIQueryInterface] deterministic churn skipped: {exc}")
            return None

    def _maybe_synthesize_multi_step_result(self, result: Dict[str, Any], original_query: str) -> None:
        """Replace raw step-stitched summary with one ERP narrative (summaries only → LLM)."""
        if not bool(self._config.get("enable_multi_step_synthesis", True)) or self._llm is None:
            return
        if result.get("query_type") != "multi_step":
            return
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            return
        trace: List[Dict[str, Any]] = []
        for s in steps:
            if isinstance(s, dict):
                sd = s.get("data")
                trace.append(
                    {
                        "step_id": s.get("step_id"),
                        "intent_code": s.get("intent_code"),
                        "summary": s.get("summary"),
                        "ranked_preview": compact_ranked_preview(sd) if isinstance(sd, dict) else "",
                    }
                )
        chain_complete = bool(result.get("chain_complete", True))
        failure_reason = data.get("chain_failure_reason")
        failed_step_id = data.get("chain_failed_at_step")
        try:
            from src.query.response_auditor import ResponseAuditor

            syn = ResponseAuditor(self._llm).synthesize_chain(
                original_query,
                step_summaries=trace,
                chain_complete=chain_complete,
                failure_reason=str(failure_reason) if failure_reason is not None else None,
                failed_step_id=int(failed_step_id) if isinstance(failed_step_id, int) else None,
            )
            fa = syn.get("final_answer")
            if isinstance(fa, str) and fa.strip():
                result["summary"] = fa.strip()
            result.setdefault("insights", []).append(
                "ResponseAuditor: final answer synthesized from step summaries and ranked_preview (capped)."
                if syn.get("synthesized")
                else "ResponseAuditor: synthesis fallback (concatenated summaries and ranked_preview)."
            )
        except Exception as exc:
            logger.warning(f"[AIQueryInterface] multi-step synthesis skipped: {exc}")

    def _payloads_allow_deterministic_template(self, payloads: List[IntentPayload]) -> bool:
        if not payloads or self._query_validator is None:
            return False
        for p in payloads:
            ic = canonical_intent(p.intent_code)
            if ic == UNKNOWN_INTENT or ic not in EXECUTOR_INTENT_CODES:
                return False
            if ic not in self._query_validator.allowed_intents:
                return False
        return True

    def _minimal_multi_plan_from_payloads(self, payloads: List[IntentPayload]) -> Dict[str, Any]:
        steps: List[Dict[str, Any]] = []
        for i, p in enumerate(payloads):
            t = dict(p.time) if isinstance(p.time, dict) else {}
            month = t.get("month")
            year = t.get("year")
            flt: Dict[str, Any] = dict({"companyfn": self.companyfn} if self.companyfn else {})
            if isinstance(p.params, dict):
                for pk, pv in p.params.items():
                    if isinstance(pk, str) and isinstance(pv, (str, int, float, bool)):
                        flt.setdefault(pk, pv)
            steps.append(
                {
                    "step_id": i,
                    "intent": canonical_intent(p.intent_code),
                    "dependency_source": None,
                    "top_n": p.top_n,
                    "time": {
                        "month": int(month) if isinstance(month, int) and 1 <= month <= 12 else None,
                        "year": int(year) if isinstance(year, int) and 1900 <= year <= 2100 else None,
                    },
                    "dataset": None,
                    "dimensions": [],
                    "metrics": [],
                    "filters": flt,
                    "sort": {},
                    "limit": 20,
                    "view": None,
                    "target_files": list(p.target_files),
                    "confidence": 0.72,
                    "sub_query": p.sub_query,
                }
            )
        return {"steps": steps, "confidence": 0.72}

    def _run_planner_and_executor(
        self,
        query: str,
        route_snapshot: Dict[str, Any],
        payloads: List[IntentPayload],
    ) -> Optional[Dict[str, Any]]:
        if not self._query_planner or not self._query_executor or not self._query_validator:
            return None
        if any(canonical_intent(p.intent_code) == UNKNOWN_INTENT for p in payloads):
            return None
        plan = self._query_planner.plan(
            query=query,
            companyfn=self.companyfn,
            router_payloads=payloads,
            router_context=route_snapshot,
        )
        if not plan:
            plan = self._minimal_multi_plan_from_payloads(payloads)
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        for step in steps:
            if isinstance(step, dict) and not str(step.get("sub_query") or "").strip():
                step["sub_query"] = query
        if not self._query_validator.validate_template_plan(plan):
            return None
        for step in steps:
            if not isinstance(step, dict):
                return None
            sub_q = step.get("sub_query") or query
            if not self._is_plan_semantically_compatible(sub_q, {"intent": step.get("intent")}):
                return None
        planner_min_confidence = float(self._config.get("planner_min_confidence", 0.55))
        if float((plan or {}).get("confidence", 0.0)) < planner_min_confidence:
            return None
        if any(canonical_intent(s.get("intent")) == UNKNOWN_INTENT for s in steps):
            return None
        result = self._query_executor.execute(plan)
        if result:
            result.setdefault("insights", [])
            result["source"] = "planner_executor"
            last_intent = ""
            if steps:
                last_intent = str(steps[-1].get("intent") or "")
            result["plan_intent"] = last_intent
            result["intent_planner_config_enabled"] = True
            result["intent_router"] = dict(route_snapshot)
        return result

    def _is_plan_semantically_compatible(self, query: str, plan: Dict[str, Any]) -> bool:
        q = (query or "").lower()
        intent = canonical_intent((plan or {}).get("intent"))
        has_customer = "customer" in q or "khách" in q
        has_product = "product" in q or "sản phẩm" in q
        has_buy_signal = any(k in q for k in ["buy", "bought", "purchase", "mua", "most", "highest"])
        asks_latest = any(k in q for k in ["latest records", "latest record", "recent records", "recent rows"])
        asks_each_customer = any(k in q for k in ["each customer", "from each customer", "per customer", "mỗi khách", "từng khách"])

        # Hard intent guards to prevent planner hallucinating unrelated intent.
        if intent == "TOP_CUSTOMERS_TOP_PRODUCT":
            return has_customer and has_product and has_buy_signal and not asks_each_customer
        if intent == "LATEST_RECORDS_TOP_PRODUCT":
            return asks_latest and has_product
        if intent == "TOP_PRODUCTS":
            return not (has_customer and asks_each_customer)

        # Customer + latest-per-customer queries should stay generic DATA_QUERY.
        if has_customer and asks_latest and asks_each_customer:
            return intent == "DATA_QUERY"

        if has_customer and has_product and has_buy_signal and not asks_latest:
            return intent in {"TOP_CUSTOMERS_TOP_PRODUCT", "DATA_QUERY"}
        if asks_latest and "top product" in q:
            return intent in {"LATEST_RECORDS_TOP_PRODUCT", "DATA_QUERY"}
        return True

    def _resolve_contextual_query(self, query: str, context: Dict[str, Any]) -> str:
        session_id = str(context.get("session_id") or "").strip()
        if not session_id:
            return query
        memory = self._session_memory.get(session_id, {})
        last_top_product = memory.get("last_top_product")
        q_lower = (query or "").lower()
        refers_previous_product = any(
            token in q_lower for token in ["that product", "the product above", "sản phẩm đó", "top product đó"]
        )
        if refers_previous_product:
            period = memory.get("last_period")
            if "latest" in q_lower or "recent" in q_lower:
                if isinstance(period, str) and period:
                    return f"{query}. Interpret as: show latest records of top product {period}. Product reference: {last_top_product or ''}".strip()
                return f"{query}. Interpret as: show latest records of top product. Product reference: {last_top_product or ''}".strip()
            if last_top_product:
                return f"{query}. Product reference: {last_top_product}"
        return query

    def _update_session_memory(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        session_id = str(context.get("session_id") or "").strip()
        if not session_id:
            return
        memory = self._session_memory.setdefault(session_id, {})
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        final = data.get("final")
        if isinstance(final, dict):
            data = final
        top_product = data.get("top_product")
        if isinstance(top_product, str) and top_product.strip():
            memory["last_top_product"] = top_product.strip()
        period = data.get("period")
        if isinstance(period, str) and period.strip():
            memory["last_period"] = period.strip()

    def _append_history(self, query: str, query_type: str, source: str = "unknown", latency_ms: int = 0) -> None:
        record = {
            "query": query,
            "query_type": [query_type],
            "source": source,
            "latency_ms": latency_ms,
            "companyfn": self.companyfn,
            "timestamp": datetime.now().isoformat(),
        }
        self.query_history.append(record)
        self._persist_query_record(record)

    def _persist_query_record(self, record: Dict[str, Any]) -> None:
        try:
            os.makedirs(self._query_log_dir, exist_ok=True)
            log_path = self._resolve_query_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as exc:
            logger.warning(f"[AIQueryInterface] Could not persist query record: {exc}")

    def _resolve_query_log_path(self) -> str:
        date_tag = datetime.now().strftime("%Y%m%d")
        base_name = f"{self._query_log_base}_{date_tag}"
        candidate = os.path.join(self._query_log_dir, f"{base_name}.jsonl")
        if not os.path.exists(candidate):
            return candidate

        # Size-based rollover within the same day.
        if os.path.getsize(candidate) < self._query_log_max_bytes:
            return candidate

        index = 1
        while True:
            rotated = os.path.join(self._query_log_dir, f"{base_name}_{index}.jsonl")
            if not os.path.exists(rotated):
                return rotated
            if os.path.getsize(rotated) < self._query_log_max_bytes:
                return rotated
            index += 1

    def _is_english_query(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        has_latin = bool(re.search(r"[A-Za-z]", query))
        has_non_ascii = any(ord(ch) > 127 for ch in query)
        return has_latin and not has_non_ascii

    def _is_rate_limit_error(self, message: str) -> bool:
        lowered = (message or "").lower()
        return ("rate limit" in lowered) or ("rate_limit_exceeded" in lowered) or ("429" in lowered)

    def _extract_retry_after_seconds(self, message: str, default_seconds: int = 20) -> int:
        match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", message or "", flags=re.IGNORECASE)
        if not match:
            return default_seconds
        try:
            return max(1, int(round(float(match.group(1)))))
        except ValueError:
            return default_seconds

    def get_query_history(self) -> List[Dict[str, Any]]:
        return self.query_history

    def get_runtime_metrics(self) -> Dict[str, Any]:
        history = self.query_history
        total = len(history)
        by_source: Dict[str, int] = {}
        latency_values = []
        for item in history:
            src = str(item.get("source", "unknown"))
            by_source[src] = by_source.get(src, 0) + 1
            lat = item.get("latency_ms")
            if isinstance(lat, int):
                latency_values.append(lat)
        avg_latency = round(sum(latency_values) / len(latency_values), 2) if latency_values else 0.0
        p95_latency = 0
        if latency_values:
            sorted_lat = sorted(latency_values)
            idx = max(0, int(0.95 * (len(sorted_lat) - 1)))
            p95_latency = sorted_lat[idx]
        return {
            "companyfn": self.companyfn,
            "total_queries": total,
            "by_source": by_source,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "agent_initialized": self._pandas_agent is not None,
            "query_log_path": self._resolve_query_log_path(),
            "query_log_dir": self._query_log_dir,
            "query_log_max_mb": round(self._query_log_max_bytes / (1024 * 1024), 2),
        }

    def format_response(self, result: Dict[str, Any]) -> str:
        if "error" in result:
            return f"Error: {result['error']}"
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        records = data.get("latest_records")
        if isinstance(records, list) and records:
            lines = [result.get("summary", "No answer"), "", "Latest records:"]
            for idx, row in enumerate(records, 1):
                date_val = row.get("date_trans", "N/A")
                amount_val = row.get("amount_local", 0)
                party_val = row.get("party_desc", "N/A")
                qty_val = row.get("qnty_total", "N/A")
                lines.append(
                    f"{idx}. date={date_val}, amount={amount_val}, qty={qty_val}, customer={party_val}"
                )
            return "\n".join(lines)
        top_customers_for_product = data.get("top_customers_for_top_product")
        if isinstance(top_customers_for_product, list) and top_customers_for_product:
            lines = [result.get("summary", "No answer"), "", "Top customers for the top product:"]
            for idx, row in enumerate(top_customers_for_product, 1):
                lines.append(f"{idx}. customer={row.get('customer', 'N/A')}, amount={row.get('amount_local', 0)}")
            return "\n".join(lines)
        rows = data.get("rows")
        if isinstance(rows, list) and rows:
            dict_rows = [r for r in rows if isinstance(r, dict)]
            extra = _markdown_rows_display(dict_rows)
            if extra:
                return "\n".join([result.get("summary", "No answer")] + extra)
        ranked_extra = _format_ranked_dict_sections(data)
        if ranked_extra:
            return "\n".join([result.get("summary", "No answer")] + ranked_extra)
        msteps = data.get("steps")
        if isinstance(msteps, list) and msteps and result.get("query_type") == "multi_step":
            lines = [result.get("summary", "No answer")]
            for i, block in enumerate(msteps, 1):
                if isinstance(block, dict):
                    lines.append(f"\n--- Step {i} ---\n{block.get('summary', '')}")
                    bd = block.get("data")
                    if isinstance(bd, dict):
                        lines.extend(_format_ranked_dict_sections(bd))
            return "\n".join(lines)
        if result.get("source") in {"langchain_agent", "langchain_agent_retry"}:
            return result.get("summary", "No answer")
        if result.get("source") in {"deterministic_executor", "planner_executor", "deterministic_churn"}:
            return result.get("summary", "No answer")
        return result.get("summary", "No answer")
