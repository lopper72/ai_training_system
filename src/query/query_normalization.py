"""
Pre-process user queries: English-only is enforced upstream; optional default calendar year
when the question is time-sensitive but no period is given.
"""

from __future__ import annotations

import re
from typing import Optional

_ALL_TIME = re.compile(r"(?i)\b(all[\s-]?time|all\s+periods|entire\s+history|ever)\b")
_EXPLICIT_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_MMYYYY = re.compile(r"\b(\d{1,2})\s*/\s*(\d{4})\b")
_YYYYMM = re.compile(r"\b(\d{4})\s*[-/](\d{1,2})\b")
_MONTH_YEAR = re.compile(
    r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b"
)
_RELATIVE = re.compile(
    r"(?i)\b(last|this|next)\s+(month|year|quarter|week)\b|"
    r"\bytd\b|\bq[1-4]\s+\d{4}\b|\b\d{4}\s+q[1-4]\b"
)
# Broad triggers: analytics questions that usually need a period
_TIME_SENSITIVE = re.compile(
    r"(?i)\b("
    r"top|rank|ranking|revenue|sales|customer|customers|product|products|"
    r"churn|growth|compare|comparison|trend|monthly|quarter|year|"
    r"best|highest|most|largest|smallest|latest|recent|last\s+order"
    r")\b"
)
# Do not append default-year system block: it confuses planner/router sub_query (e.g. "all-time customers")
# and breaks distinct-customer template logic — user asked for the full customer universe, not a year slice.
_SKIP_DEFAULT_YEAR_FOR_CUSTOMER_DIRECTORY = re.compile(
    r"(?i)(\b(get|list|show|find)\s+all\s+customers?\b|"
    r"\ball\s+customers?\s+in\s+(the\s+)?system\b|"
    r"\bevery\s+customer\b|"
    r"\b(entire|full|complete)\s+customer\s+(list|directory|base|set)\b)"
)


def has_explicit_time_period(query: str) -> bool:
    if not query or not query.strip():
        return False
    if _ALL_TIME.search(query):
        return True
    if _EXPLICIT_YEAR.search(query):
        return True
    if _MMYYYY.search(query) or _YYYYMM.search(query):
        return True
    if _MONTH_YEAR.search(query):
        return True
    if _RELATIVE.search(query):
        return True
    return False


def apply_default_year_if_vague(query: str, default_year: int) -> str:
    """
    If the query looks time-sensitive but has no explicit period, append a system line
    so the agent filters on `default_year` instead of asking for clarification.
    """
    q = (query or "").strip()
    if not q:
        return q
    if _ALL_TIME.search(q):
        return q
    first_user_block = q.split("\n\n[")[0].strip()
    if _SKIP_DEFAULT_YEAR_FOR_CUSTOMER_DIRECTORY.search(first_user_block):
        return q
    if has_explicit_time_period(q):
        return q
    if not _TIME_SENSITIVE.search(q):
        return q
    suffix = (
        f"\n\n[System context: No calendar period was specified. "
        f"Use calendar year {default_year} when filtering `year` or `date_trans`, "
        f"unless the user clearly asked for all-time. State this assumption briefly in the Final Answer.]"
    )
    return q + suffix
