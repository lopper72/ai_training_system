"""
LLM planner: natural language → plan with `steps[]` (each step may depend on a prior step).
Backward compatible: single-step plans are wrapped as one element in `steps`.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.query.intent_codes import (
    DEFAULT_RANK_TOP_N,
    PLANNER_ALLOWED_INTENTS,
    RANKING_INTENTS_DEFAULT_TOP_N,
    UNKNOWN_INTENT,
    canonical_intent,
)

if TYPE_CHECKING:
    from src.query.intent_payload import IntentPayload

logger = logging.getLogger(__name__)

_INTENT_LINE = "DATA_QUERY|TOP_PRODUCTS|LATEST_RECORDS_TOP_PRODUCT|TOP_CUSTOMERS_TOP_PRODUCT|TOP_CUSTOMERS|GET_REVENUE|REVENUE_GROWTH_MOM|TOP_BRANDS|TOP_CATEGORIES|COMPARE_CHURN|" + UNKNOWN_INTENT

PLANNER_SYSTEM_PROMPT = f"""
You are a query planner for business analytics.
Return ONLY valid JSON (no markdown).

Schema:
{{
  "steps": [
    {{
      "step_id": 0,
      "intent": "{_INTENT_LINE}",
      "dependency_source": null or {{
        "from_step_id": 0,
        "output_key": "top_product|top_customer|party_desc|...",
        "filter_column": "stkcode_desc|party_desc|brand_desc|stkcate_desc|companyfn"
      }},
      "top_n": 1..20 or null,
      "time": {{"month": 1..12 or null, "year": 4-digit or null}},
      "dataset": "sales_data|sales_main|null",
      "dimensions": ["column_name", "..."],
      "metrics": [{{"name":"sum|count|avg|max|min", "column":"amount_local|..."}}],
      "filters": {{"column": "value"}},
      "sort": {{"by":"column_or_metric", "order":"asc|desc"}},
      "limit": 1..100 or null,
      "view": "table|latest_records|distinct_customers|null",
      "target_files": ["sales_data", "sales_main"],
      "params": {{}},
      "confidence": 0..1
    }}
  ],
  "confidence": 0..1
}}

Optional "params" per step: flat key-value hints merged into filters (e.g. {{"party_desc":"Customer A"}}).
Rules:
- Use one step for a simple question; multiple ordered steps for compound questions.
- dependency_source: null for the first step or independent steps; later steps may filter using a scalar from data[output_key] of from_step_id.
- Example chain: step0 TOP_CUSTOMERS (who bought most); step1 TOP_PRODUCTS with dependency_source {{"from_step_id":0,"output_key":"top_customer","filter_column":"party_desc"}}.
- If unsure for a step, set intent="{UNKNOWN_INTENT}".
- Never output executable code.
- target_files must only use allowed dataset names from the user message context.
- For "all customers", "every customer", "list all customers", "get all customers", "list customers in system" (distinct party list, not transaction lines): set "view": "distinct_customers", set "dataset": "sales_main" (invoice header / scm_sal_main), leave dimensions/metrics empty unless aggregating.
""".strip()

PLANNER_PARAM_SINGLE_PROMPT = """
The intent for step 0 is FIXED as: {fixed_intent}
Return ONLY valid JSON (no markdown). Include a single step in "steps".

Schema:
{{
  "steps": [
    {{
      "step_id": 0,
      "dependency_source": null,
      "top_n": 1..20 or null,
      "time": {{"month": 1..12 or null, "year": 4-digit or null}},
      "dataset": "sales_data|sales_main|null",
      "dimensions": ["column_name", "..."],
      "metrics": [{{"name":"sum|count|avg|max|min", "column":"amount_local|..."}}],
      "filters": {{"column": "value"}},
      "sort": {{"by":"column_or_metric", "order":"asc|desc"}},
      "limit": 1..100 or null,
      "view": "table|latest_records|distinct_customers|null",
      "target_files": ["sales_data"],
      "confidence": 0..1
    }}
  ],
  "confidence": 0..1
}}

Rules:
- Do NOT include "intent" inside the step; it is fixed externally.
- Never output executable code.
""".strip()

PLANNER_MULTI_PROMPT = """
You plan ordered execution steps for a compound analytics question.
Router decomposition (trusted order and intents):
{router_json}

Return ONLY valid JSON (no markdown). Fill dependency_source when step B must use output from step A
(e.g. filter stkcode_desc by top_product from step 0).

Schema:
{{
  "steps": [ ... same shape as multi-step schema with intent per step matching the router ... ],
  "confidence": 0..1
}}

Rules:
- step_id must be 0,1,2,... in order.
- Align each step's intent with the router entry at the same index when possible.
- For "customer X then their top product": step0 should constrain party_desc (params or filters); step1 uses dependency from step0 (output_key top_customer or party_desc, filter_column party_desc) then TOP_PRODUCTS or DATA_QUERY grouped by stkcode_desc.
- Never output executable code.
""".strip()


class QueryPlanner:
    """Plan JSON generator using the configured LLM."""

    def __init__(self, llm):
        self.llm = llm

    def plan(
        self,
        query: str,
        companyfn: Optional[str] = None,
        router_context: Optional[Dict[str, Any]] = None,
        router_payloads: Optional[List["IntentPayload"]] = None,
    ) -> Optional[Dict[str, Any]]:
        from src.query.intent_payload import IntentPayload

        payloads: List[IntentPayload] = []
        if router_payloads:
            payloads = list(router_payloads)
        elif router_context and isinstance(router_context, dict):
            if router_context.get("intent_payloads") and isinstance(router_context["intent_payloads"], list):
                payloads = [IntentPayload.from_dict(x) for x in router_context["intent_payloads"] if isinstance(x, dict)]
            else:
                payloads = [IntentPayload.from_legacy_router_dict(router_context)]

        if not payloads:
            return None

        if len(payloads) == 1 and canonical_intent(payloads[0].intent_code) not in (UNKNOWN_INTENT,):
            fixed = canonical_intent(payloads[0].intent_code)
            if fixed in PLANNER_ALLOWED_INTENTS:
                system = PLANNER_PARAM_SINGLE_PROMPT.format(fixed_intent=fixed)
                user_prompt = (
                    f"Query: {query}\n"
                    f"Company code: {companyfn or 'null'}\n"
                    f"Suggested top_n: {payloads[0].top_n}\n"
                    f"Suggested time: {json.dumps(payloads[0].time)}\n"
                    f"Sub-query: {payloads[0].sub_query or query}\n"
                    "Emit JSON now."
                )
            else:
                system = PLANNER_SYSTEM_PROMPT
                user_prompt = (
                    f"Query: {query}\n"
                    f"Company code: {companyfn or 'null'}\n"
                    "Generate plan JSON now."
                )
        else:
            system = PLANNER_MULTI_PROMPT.format(router_json=json.dumps([p.to_dict() for p in payloads], indent=0))
            user_prompt = (
                f"Full query: {query}\n"
                f"Company code: {companyfn or 'null'}\n"
                "Emit JSON now."
            )

        try:
            msg = self.llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]
            )
            raw = getattr(msg, "content", "") or ""
            parsed = self._parse_json(raw)
            if not parsed:
                return None
            return self._normalize_plan_document(parsed, companyfn, payloads)
        except Exception as exc:
            logger.warning(f"[QueryPlanner] Planner failed: {exc}")
            return None

    @staticmethod
    def build_router_only_execution_plan(payloads: List["IntentPayload"]) -> Dict[str, Any]:
        """Deterministic execution outline when LLM planner is unavailable (agent fallback)."""
        from src.query.intent_payload import IntentPayload

        steps: List[Dict[str, Any]] = []
        for i, p in enumerate(payloads):
            if not isinstance(p, IntentPayload):
                continue
            flt: Dict[str, Any] = {}
            if isinstance(p.params, dict):
                for pk, pv in p.params.items():
                    if isinstance(pk, str) and isinstance(pv, (str, int, float, bool)):
                        flt[pk] = pv
            steps.append(
                {
                    "step_id": i,
                    "intent": canonical_intent(p.intent_code),
                    "dependency_source": None,
                    "top_n": p.top_n,
                    "time": dict(p.time),
                    "dataset": None,
                    "dimensions": [],
                    "metrics": [],
                    "filters": flt,
                    "sort": {},
                    "limit": 20,
                    "view": None,
                    "target_files": list(p.target_files),
                    "confidence": 0.5,
                    "sub_query": p.sub_query,
                }
            )
        return {"steps": steps, "confidence": 0.5, "source": "router_only"}

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        text = (raw or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            out = json.loads(text)
            return out if isinstance(out, dict) else None
        except Exception:
            return None

    def _normalize_plan_document(
        self,
        doc: Dict[str, Any],
        companyfn: Optional[str],
        payloads: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Normalize to {{steps, confidence}}; accept legacy flat plan."""
        if isinstance(doc.get("steps"), list) and doc["steps"]:
            steps_out: List[Dict[str, Any]] = []
            for i, s in enumerate(doc["steps"]):
                ns = self._normalize_step(s, companyfn, payloads, i)
                if ns is None:
                    return None
                steps_out.append(ns)
            for i, p in enumerate(payloads):
                if i < len(steps_out):
                    self._merge_payload_into_step(steps_out[i], p, i)
            conf = doc.get("confidence")
            if not isinstance(conf, (int, float)):
                conf = min(float(x.get("confidence") or 0.0) for x in steps_out) if steps_out else 0.0
            conf = float(max(0.0, min(1.0, float(conf))))
            return {"steps": steps_out, "confidence": conf}

        # Legacy: flat single plan
        legacy = self._normalize_flat_as_step(doc, companyfn)
        if not legacy:
            return None
        if payloads:
            self._merge_payload_into_step(legacy, payloads[0], 0)
            if len(payloads) == 1 and canonical_intent(payloads[0].intent_code) != UNKNOWN_INTENT:
                legacy["intent"] = canonical_intent(payloads[0].intent_code)
        conf = float(legacy.get("confidence", 0.0))
        conf = float(max(0.0, min(1.0, conf)))
        legacy["step_id"] = 0
        if "dependency_source" not in legacy:
            legacy["dependency_source"] = None
        return {"steps": [legacy], "confidence": conf}

    def _merge_payload_into_step(self, step: Dict[str, Any], payload: Any, index: int) -> None:
        if payload.top_n is not None and step.get("top_n") is None:
            step["top_n"] = payload.top_n
        pt = payload.time if isinstance(payload.time, dict) else {}
        st = step.get("time") if isinstance(step.get("time"), dict) else {}
        if st.get("month") is None and isinstance(pt.get("month"), int):
            st["month"] = pt["month"]
        if st.get("year") is None and isinstance(pt.get("year"), int):
            st["year"] = pt["year"]
        step["time"] = st
        if not step.get("target_files") and payload.target_files:
            step["target_files"] = list(payload.target_files)
        if payload.sub_query and not step.get("sub_query"):
            step["sub_query"] = payload.sub_query
        if getattr(payload, "params", None) and isinstance(payload.params, dict):
            filt = dict(step.get("filters") or {})
            for k, v in payload.params.items():
                if isinstance(k, str) and isinstance(v, (str, int, float, bool)) and k not in filt:
                    filt[k] = v
            step["filters"] = filt
        step.setdefault("step_id", index)

    def _normalize_step(
        self,
        step: Any,
        companyfn: Optional[str],
        payloads: List[Any],
        index: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(step, dict):
            return None
        intent = canonical_intent(step.get("intent"))
        if index < len(payloads):
            pi = canonical_intent(payloads[index].intent_code)
            if pi != UNKNOWN_INTENT:
                intent = pi
        top_n = step.get("top_n")
        if isinstance(top_n, int):
            top_n = max(1, min(top_n, 20))
        else:
            top_n = None
        if top_n is None and intent in RANKING_INTENTS_DEFAULT_TOP_N:
            top_n = DEFAULT_RANK_TOP_N
        time = step.get("time") if isinstance(step.get("time"), dict) else {}
        month = time.get("month")
        year = time.get("year")
        month = int(month) if isinstance(month, int) and 1 <= month <= 12 else None
        year = int(year) if isinstance(year, int) and 1900 <= year <= 2100 else None
        conf = step.get("confidence")
        if isinstance(conf, (int, float)):
            conf = float(max(0.0, min(1.0, conf)))
        else:
            conf = 0.0
        dep = step.get("dependency_source")
        if dep is not None and not isinstance(dep, dict):
            dep = None
        if isinstance(dep, dict) and dep:
            dep = {
                "from_step_id": int(dep["from_step_id"]) if isinstance(dep.get("from_step_id"), int) else None,
                "output_key": str(dep["output_key"]) if isinstance(dep.get("output_key"), str) else None,
                "filter_column": str(dep["filter_column"]) if isinstance(dep.get("filter_column"), str) else None,
            }
            if dep["from_step_id"] is None or not dep["output_key"] or not dep["filter_column"]:
                dep = None
        else:
            dep = None
        tf = step.get("target_files")
        if isinstance(tf, list):
            target_files = [str(x).strip() for x in tf if isinstance(x, str) and str(x).strip()]
        else:
            target_files = []
        raw_params = step.get("params")
        extra_filters: Dict[str, Any] = {}
        if isinstance(raw_params, dict):
            for pk, pv in raw_params.items():
                if isinstance(pk, str) and isinstance(pv, (str, int, float, bool)):
                    extra_filters[pk] = pv
        base_filters = self._normalize_filters(step.get("filters"), companyfn)
        for k, v in extra_filters.items():
            base_filters.setdefault(k, v)
        return {
            "step_id": int(step.get("step_id")) if isinstance(step.get("step_id"), int) else index,
            "intent": intent,
            "dependency_source": dep,
            "top_n": top_n,
            "time": {"month": month, "year": year},
            "dataset": step.get("dataset") if isinstance(step.get("dataset"), str) else None,
            "dimensions": step.get("dimensions") if isinstance(step.get("dimensions"), list) else [],
            "metrics": step.get("metrics") if isinstance(step.get("metrics"), list) else [],
            "filters": base_filters,
            "sort": self._normalize_sort(step.get("sort")),
            "limit": self._normalize_limit(step.get("limit")),
            "view": step.get("view") if isinstance(step.get("view"), str) else None,
            "target_files": target_files,
            "confidence": conf,
            "sub_query": step.get("sub_query") if isinstance(step.get("sub_query"), str) else None,
        }

    def _normalize_flat_as_step(self, plan: Dict[str, Any], companyfn: Optional[str]) -> Optional[Dict[str, Any]]:
        intent = canonical_intent(plan.get("intent"))
        top_n = plan.get("top_n")
        if isinstance(top_n, int):
            top_n = max(1, min(top_n, 20))
        else:
            top_n = None
        if top_n is None and intent in RANKING_INTENTS_DEFAULT_TOP_N:
            top_n = DEFAULT_RANK_TOP_N
        time = plan.get("time") if isinstance(plan.get("time"), dict) else {}
        month = time.get("month")
        year = time.get("year")
        month = int(month) if isinstance(month, int) and 1 <= month <= 12 else None
        year = int(year) if isinstance(year, int) and 1900 <= year <= 2100 else None
        confidence = plan.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence = float(max(0.0, min(1.0, confidence)))
        else:
            confidence = 0.0
        base_f = self._normalize_filters(plan.get("filters"), companyfn)
        pr = plan.get("params")
        if isinstance(pr, dict):
            for pk, pv in pr.items():
                if isinstance(pk, str) and isinstance(pv, (str, int, float, bool)):
                    base_f.setdefault(pk, pv)
        return {
            "step_id": 0,
            "intent": intent,
            "dependency_source": None,
            "top_n": top_n,
            "time": {"month": month, "year": year},
            "dataset": plan.get("dataset") if isinstance(plan.get("dataset"), str) else None,
            "dimensions": plan.get("dimensions") if isinstance(plan.get("dimensions"), list) else [],
            "metrics": plan.get("metrics") if isinstance(plan.get("metrics"), list) else [],
            "filters": base_f,
            "sort": self._normalize_sort(plan.get("sort")),
            "limit": self._normalize_limit(plan.get("limit")),
            "view": plan.get("view") if isinstance(plan.get("view"), str) else None,
            "target_files": [],
            "confidence": confidence,
        }

    def _normalize_filters(self, filters: Any, companyfn: Optional[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(filters, dict):
            for k, v in filters.items():
                if isinstance(k, str) and isinstance(v, (str, int, float, bool)):
                    out[k] = v
        if companyfn:
            out["companyfn"] = companyfn
        return out

    def _normalize_sort(self, sort: Any) -> Dict[str, str]:
        if not isinstance(sort, dict):
            return {}
        by = sort.get("by")
        order = sort.get("order")
        if not isinstance(by, str):
            return {}
        if order not in ("asc", "desc"):
            order = "desc"
        return {"by": by, "order": order}

    def _normalize_limit(self, limit_val: Any) -> Optional[int]:
        if isinstance(limit_val, int):
            return max(1, min(limit_val, 100))
        return None
