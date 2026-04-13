"""
Intent router: decomposes user questions into List[IntentPayload].
UNKNOWN steps or parse failure → caller uses PandasQueryAgent with execution_plan.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from src.query.catalog import PRIORITY_DATASETS, catalog_dataset_names, validate_router_targets
from src.query.intent_codes import (
    DEFAULT_RANK_TOP_N,
    RANKING_INTENTS_DEFAULT_TOP_N,
    ROUTER_INTENT_CODES,
    UNKNOWN_INTENT,
    canonical_intent,
)
from src.query.intent_payload import IntentPayload

logger = logging.getLogger(__name__)

TASK_TYPES = frozenset({"data_query", "comparison", "chart"})

_INTENT_ENUM = "|".join(sorted(c for c in ROUTER_INTENT_CODES if c != UNKNOWN_INTENT)) + f"|{UNKNOWN_INTENT}"

ROUTER_SYSTEM_TEMPLATE = """You decompose ERP analytics questions into one or more intent steps (multi-step if the user chains questions with then/and/also/first/then).
Return ONLY valid JSON (no markdown fences).

Allowed dataset keys (exact strings in each intent's target_files):
{allowed_list}

intent_code per step (choose exactly one per step):
- GET_REVENUE — total revenue for a specific month-year.
- REVENUE_GROWTH_MOM — month-over-month revenue change.
- TOP_PRODUCTS — top products by revenue.
- TOP_CUSTOMERS — top customers by revenue.
- TOP_BRANDS — top brands by revenue.
- TOP_CATEGORIES — top categories by revenue.
- LATEST_RECORDS_TOP_PRODUCT — latest line records for the #1 product by revenue.
- TOP_CUSTOMERS_TOP_PRODUCT — top customers for the #1 product.
- DATA_QUERY — custom aggregates/filters not covered above.
- COMPARE_CHURN — churn / attrition (MoM or YoY).
- UNKNOWN — step is ambiguous or not expressible as a template; use for that step only.

JSON schema:
{{
  "intents": [
    {{
      "step_id": 0,
      "intent_code": "{intent_enum}",
      "target_files": ["<one or more allowed keys>"],
      "task_type": "data_query|comparison|chart",
      "rationale": "short English phrase",
      "top_n": null or integer 1-20,
      "time": {{"month": null or 1-12, "year": null or 4-digit}},
      "sub_query": "optional English slice for this step only",
      "params": {{}}
    }}
  ]
}}

The "params" object is optional: use for explicit filters the next planner step needs, e.g.
{{"party_desc": "Exact customer name from the question"}} or {{"stkcode_desc": "SKU name"}}.
For compound questions like "Find customer X and the product they bought most":
  step 0: identify/locate the customer (often DATA_QUERY or TOP_CUSTOMERS with params.party_desc if named).
  step 1: depend on step 0 output (top_customer or party_desc) to filter lines and rank products (TOP_PRODUCTS or DATA_QUERY on sales_data).

Rules:
- Use a single intent in the array for a simple question.
- Use multiple intents in order for compound questions (e.g. top product then revenue for that product).
- sales_main: invoices, churn, top customers by invoice revenue.
- sales_data: line/SKU detail.
- Include BOTH datasets when the step needs totals and line detail.
- task_type = "chart" only if the user explicitly wants a chart.
- If any part of a compound question cannot be templated, output UNKNOWN for that step (do not guess).
""".replace("{intent_enum}", _INTENT_ENUM)


class IntentRouter:
    """Routes natural language to ordered IntentPayload list (LLM or deterministic fallback)."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def route(self, query: str, loaded_datasets: Set[str]) -> List[IntentPayload]:
        allowed = [k for k in catalog_dataset_names() if k in loaded_datasets]
        fallback = IntentRouter.fallback_for_loaded(loaded_datasets)
        if not allowed:
            return fallback
        system = ROUTER_SYSTEM_TEMPLATE.format(allowed_list=", ".join(allowed))
        user = f"User question:\n{query}\n\nEmit JSON now."
        try:
            msg = self.llm.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            raw = getattr(msg, "content", "") or ""
            parsed = self._parse_json(raw)
            if not parsed:
                return fallback
            return self._normalize_document(parsed, allowed, fallback)
        except Exception as exc:
            logger.warning(f"[IntentRouter] route failed: {exc}")
            return fallback

    def _parse_json(self, raw: str) -> Optional[Any]:
        text = (raw or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _normalize_time(self, raw: Any) -> Dict[str, Optional[int]]:
        if not isinstance(raw, dict):
            return {"month": None, "year": None}
        month = raw.get("month")
        year = raw.get("year")
        mi = int(month) if isinstance(month, int) and 1 <= month <= 12 else None
        yi = int(year) if isinstance(year, int) and 1900 <= year <= 2100 else None
        return {"month": mi, "year": yi}

    def _normalize_document(
        self,
        data: Any,
        allowed: List[str],
        fallback: List[IntentPayload],
    ) -> List[IntentPayload]:
        """Accept new multi-intent shape or legacy single-object shape."""
        if isinstance(data, dict) and isinstance(data.get("intents"), list) and data["intents"]:
            return self._normalize_intent_list(data["intents"], allowed, fallback)
        if isinstance(data, dict) and (data.get("intent_code") is not None or data.get("intent") is not None):
            one = self._normalize_single_dict(data, allowed, fallback)
            return [one] if one else fallback
        return fallback

    def _normalize_intent_list(
        self,
        items: List[Any],
        allowed: List[str],
        fallback: List[IntentPayload],
    ) -> List[IntentPayload]:
        allowed_set = set(allowed)
        out: List[IntentPayload] = []
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                return fallback
            p = self._dict_to_payload(raw, allowed_set, i)
            out.append(p)
        return out if out else fallback

    def _normalize_single_dict(
        self,
        data: Dict[str, Any],
        allowed: List[str],
        fallback: List[IntentPayload],
    ) -> Optional[IntentPayload]:
        allowed_set = set(allowed)
        return self._dict_to_payload(data, allowed_set, 0)

    def _dict_to_payload(self, data: Dict[str, Any], allowed_set: Set[str], default_step: int) -> IntentPayload:
        raw_files = data.get("target_files")
        if not isinstance(raw_files, list):
            raw_files = []
        targets = [str(x).strip() for x in raw_files if str(x).strip() in allowed_set]
        if not targets:
            for name in PRIORITY_DATASETS:
                if name in allowed_set:
                    targets.append(name)
                    break

        ordered: List[str] = []
        for name in PRIORITY_DATASETS:
            if name in targets:
                ordered.append(name)
        for name in targets:
            if name not in ordered:
                ordered.append(name)

        tt = str(data.get("task_type") or "data_query").strip().lower()
        if tt not in TASK_TYPES:
            tt = "data_query"
        rationale = data.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        raw_code = data.get("intent_code")
        if raw_code is None:
            raw_code = data.get("intent")
        intent_code = canonical_intent(str(raw_code) if raw_code is not None else UNKNOWN_INTENT)
        if intent_code not in ROUTER_INTENT_CODES:
            intent_code = UNKNOWN_INTENT

        top_n = data.get("top_n")
        if isinstance(top_n, int):
            top_n = max(1, min(top_n, 20))
        else:
            top_n = None
        if top_n is None and intent_code in RANKING_INTENTS_DEFAULT_TOP_N:
            top_n = DEFAULT_RANK_TOP_N

        time_block = self._normalize_time(data.get("time"))
        step_id = data.get("step_id")
        sid = int(step_id) if isinstance(step_id, int) else default_step
        sub = data.get("sub_query")
        sub_q = sub.strip() if isinstance(sub, str) and sub.strip() else None
        praw = data.get("params")
        params: Dict[str, Any] = {}
        if isinstance(praw, dict):
            for pk, pv in praw.items():
                if isinstance(pk, str) and isinstance(pv, (str, int, float, bool)):
                    params[pk] = pv

        return IntentPayload(
            intent_code=intent_code,
            target_files=ordered,
            task_type=tt,
            rationale=rationale.strip(),
            top_n=top_n,
            time=time_block,
            step_id=sid,
            sub_query=sub_q,
            params=params,
        )

    @staticmethod
    def fallback_for_loaded(loaded: Set[str]) -> List[IntentPayload]:
        """UNKNOWN + priority datasets when LLM is missing or JSON fails."""
        allowed = [k for k in PRIORITY_DATASETS if k in loaded]
        if not allowed:
            allowed = [k for k in catalog_dataset_names() if k in loaded]
        ordered = [n for n in PRIORITY_DATASETS if n in allowed]
        if not ordered:
            ordered = list(allowed)
        return [
            IntentPayload(
                intent_code=UNKNOWN_INTENT,
                target_files=ordered,
                task_type="data_query",
                rationale="deterministic: LLM unavailable or parse failure; defer to agent",
                top_n=None,
                time={"month": None, "year": None},
                step_id=0,
                sub_query=None,
            )
        ]

    @staticmethod
    def merge_target_files(payloads: List[IntentPayload], loaded: Set[str]) -> List[str]:
        merged: List[str] = []
        for p in payloads:
            for name in p.target_files:
                if name not in merged:
                    merged.append(name)
        return validate_router_targets(merged, loaded)

    @staticmethod
    def payloads_to_legacy_route_dict(payloads: List[IntentPayload]) -> Dict[str, Any]:
        """First payload as the historical single-intent dict (backward compatible consumers)."""
        if not payloads:
            return {
                "intent_code": UNKNOWN_INTENT,
                "target_files": [],
                "task_type": "data_query",
                "rationale": "",
                "top_n": None,
                "time": {"month": None, "year": None},
            }
        p = payloads[0]
        return {
            "intent_code": p.intent_code,
            "target_files": list(p.target_files),
            "task_type": p.task_type,
            "rationale": p.rationale,
            "top_n": p.top_n,
            "time": dict(p.time),
        }

    @staticmethod
    def payloads_to_agent_route_dict(
        payloads: List[IntentPayload],
        execution_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Bundle for PandasQueryAgent (multi-intent + optional planner steps)."""
        if not payloads:
            return IntentRouter.payloads_to_legacy_route_dict([])
        merged_targets: List[str] = []
        for p in payloads:
            for t in p.target_files:
                if t not in merged_targets:
                    merged_targets.append(t)
        return {
            "intent_code": payloads[0].intent_code,
            "multi_step": len(payloads) > 1,
            "target_files": merged_targets,
            "task_type": payloads[0].task_type,
            "rationale": " | ".join(x.rationale for x in payloads if x.rationale),
            "intent_payloads": [x.to_dict() for x in payloads],
            "execution_plan": execution_plan,
        }
