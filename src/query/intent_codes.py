"""
Canonical intent codes for Router → Planner → Template Executor.
Legacy lowercase planner intents are normalized to these codes.
"""

from __future__ import annotations

from typing import Optional

UNKNOWN_INTENT = "UNKNOWN"

# All codes the router may emit (including UNKNOWN).
ROUTER_INTENT_CODES = frozenset(
    {
        UNKNOWN_INTENT,
        "DATA_QUERY",
        "TOP_PRODUCTS",
        "TOP_CUSTOMERS",
        "TOP_BRANDS",
        "TOP_CATEGORIES",
        "LATEST_RECORDS_TOP_PRODUCT",
        "TOP_CUSTOMERS_TOP_PRODUCT",
        "GET_REVENUE",
        "REVENUE_GROWTH_MOM",
        "COMPARE_CHURN",
    }
)

# Intents the template executor implements (not UNKNOWN).
EXECUTOR_INTENT_CODES = ROUTER_INTENT_CODES - {UNKNOWN_INTENT}

# Ranking-style intents: when the user does not say "top N", default to this many rows.
RANKING_INTENTS_DEFAULT_TOP_N = frozenset(
    {"TOP_CUSTOMERS", "TOP_PRODUCTS", "TOP_BRANDS", "TOP_CATEGORIES"}
)
DEFAULT_RANK_TOP_N = 10

# Planner / config allowlist = executor intents + UNKNOWN (planner may output unknown).
PLANNER_ALLOWED_INTENTS = frozenset(EXECUTOR_INTENT_CODES | {UNKNOWN_INTENT})

LEGACY_TO_CANONICAL = {
    "unknown": UNKNOWN_INTENT,
    "data_query": "DATA_QUERY",
    "top_products": "TOP_PRODUCTS",
    "top_customers": "TOP_CUSTOMERS",
    "top_brands": "TOP_BRANDS",
    "top_categories": "TOP_CATEGORIES",
    "latest_records_of_top_product": "LATEST_RECORDS_TOP_PRODUCT",
    "top_customers_of_top_product": "TOP_CUSTOMERS_TOP_PRODUCT",
    "revenue_by_date": "GET_REVENUE",
    "revenue_growth_mom": "REVENUE_GROWTH_MOM",
    "compare_churn": "COMPARE_CHURN",
}

# Stable API-style query_type strings in executor responses
INTENT_TO_QUERY_TYPE = {
    "DATA_QUERY": "data_query",
    "TOP_PRODUCTS": "top_products",
    "TOP_CUSTOMERS": "top_customers",
    "TOP_BRANDS": "top_brands",
    "TOP_CATEGORIES": "top_categories",
    "LATEST_RECORDS_TOP_PRODUCT": "latest_records_of_top_product",
    "TOP_CUSTOMERS_TOP_PRODUCT": "top_customers_of_top_product",
    "GET_REVENUE": "revenue_by_date",
    "REVENUE_GROWTH_MOM": "revenue_growth_mom",
    "COMPARE_CHURN": "compare_churn",
}


def canonical_intent(intent: Optional[str]) -> str:
    if intent is None:
        return UNKNOWN_INTENT
    s = str(intent).strip()
    if not s:
        return UNKNOWN_INTENT
    u = s.upper().replace("-", "_")
    if u in ROUTER_INTENT_CODES:
        return u
    low = s.lower()
    return LEGACY_TO_CANONICAL.get(low, UNKNOWN_INTENT)


def query_type_for_intent(intent_code: str) -> str:
    c = canonical_intent(intent_code)
    return INTENT_TO_QUERY_TYPE.get(c, c.lower())
