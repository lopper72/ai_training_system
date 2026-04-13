"""
Post-validation for planned execution and agent responses.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from src.query.intent_codes import (
    EXECUTOR_INTENT_CODES,
    PLANNER_ALLOWED_INTENTS,
    UNKNOWN_INTENT,
    canonical_intent,
)


class QueryValidator:
    """Validate planner output and final response safety/quality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        intents = cfg.get("allowed_intents") if isinstance(cfg.get("allowed_intents"), list) else []
        columns = cfg.get("known_columns") if isinstance(cfg.get("known_columns"), list) else []
        if intents:
            self.allowed_intents = {canonical_intent(x) for x in intents}
        else:
            self.allowed_intents = set(PLANNER_ALLOWED_INTENTS) | {UNKNOWN_INTENT}
        self.known_columns = set(columns) if columns else {
            "stkcode_desc",
            "amount_local",
            "party_desc",
            "brand_desc",
            "stkcate_desc",
        }

    def validate_plan(self, plan: Optional[Dict[str, Any]]) -> bool:
        if not plan or not isinstance(plan, dict):
            return False
        steps = plan.get("steps")
        if isinstance(steps, list) and steps:
            for step in steps:
                if not isinstance(step, dict) or not self._validate_single_step_plan(step):
                    return False
            return True
        return self._validate_single_step_plan(plan)

    def validate_template_plan(self, plan: Optional[Dict[str, Any]]) -> bool:
        """Planner plan that will run in QueryExecutor: no UNKNOWN steps."""
        if not self.validate_plan(plan):
            return False
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else None
        if steps:
            for step in steps:
                if canonical_intent(step.get("intent")) not in EXECUTOR_INTENT_CODES:
                    return False
            return True
        return canonical_intent(plan.get("intent")) in EXECUTOR_INTENT_CODES

    def _validate_single_step_plan(self, step: Dict[str, Any]) -> bool:
        intent = canonical_intent(step.get("intent"))
        if intent not in self.allowed_intents:
            return False
        time = step.get("time", {})
        if not isinstance(time, dict):
            return False
        month = time.get("month")
        year = time.get("year")
        if month is not None and (not isinstance(month, int) or month < 1 or month > 12):
            return False
        if year is not None and (not isinstance(year, int) or year < 1900 or year > 2100):
            return False
        top_n = step.get("top_n")
        if top_n is not None and (not isinstance(top_n, int) or top_n < 1 or top_n > 20):
            return False
        dep = step.get("dependency_source")
        if dep is not None and not isinstance(dep, dict):
            return False
        return True

    def validate_agent_answer(self, answer: str) -> bool:
        if not answer or not answer.strip():
            return False
        lower = answer.lower()
        if "agent stopped due to iteration limit" in lower:
            return False
        if self._looks_like_column_name_answer(answer):
            return False
        return True

    def _looks_like_column_name_answer(self, answer: str) -> bool:
        lower = (answer or "").lower()
        if any(col == lower.strip() for col in self.known_columns):
            return True
        if re.search(r"\b(column|field)\b", lower) and re.search(r"\b(top product|best product)\b", lower):
            return True
        return False
