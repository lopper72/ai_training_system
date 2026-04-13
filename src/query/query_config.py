"""
Query engine config loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "planner_min_confidence": 0.55,
    "agent_timeout_seconds": 25,
    # Intent router (LLM) is always used when LLM is available; on failure see IntentRouter fallback.
    "executor_max_runtime_retries": 3,
    "enable_llm_response_auditor": False,
    # Multi-step template path: merge step summaries into one ERP-style narrative (summaries only → LLM).
    "enable_multi_step_synthesis": True,
    "default_year_when_unspecified": True,
    # Planner-first template path (Router → Planner params → QueryExecutor).
    # - False: skip template path; churn then PandasQueryAgent (router inside agent call).
    # - True: Router intent_code; only UNKNOWN uses the ReAct agent.
    "enable_intent_planner": True,
    "allowed_intents": [
        "DATA_QUERY",
        "TOP_PRODUCTS",
        "LATEST_RECORDS_TOP_PRODUCT",
        "TOP_CUSTOMERS_TOP_PRODUCT",
        "TOP_CUSTOMERS",
        "GET_REVENUE",
        "REVENUE_GROWTH_MOM",
        "TOP_BRANDS",
        "TOP_CATEGORIES",
        "COMPARE_CHURN",
        "UNKNOWN",
    ],
    "known_columns": [
        "stkcode_desc",
        "amount_local",
        "party_desc",
        "brand_desc",
        "stkcate_desc",
        "month",
        "year",
    ],
}


def load_query_config(project_root: str) -> Dict[str, Any]:
    path = Path(project_root) / "config" / "query_engine.json"
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return DEFAULT_CONFIG.copy()
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()
