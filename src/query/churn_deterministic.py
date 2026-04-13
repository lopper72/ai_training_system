"""
Deterministic customer churn from sales_main (valid-invoice rules aligned with QueryExecutor).
MoM: baseline = prior calendar month with ≥1 invoice; churned = no invoice in report month.
YoY: baseline = year Y-1; churned = had invoice in Y-1, none in Y.
"""

from __future__ import annotations

import calendar
import re
from typing import Any, Dict, Optional

_CHURN_KW = re.compile(
    r"(?i)\b(churn|attrition|churned|churn\s*rate)\b"
)
_MMYYYY = re.compile(r"\b(\d{1,2})\s*/\s*(\d{4})\b")
_YYYYMM = re.compile(r"\b(\d{4})\s*[-/](\d{1,2})\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def valid_sales_main(df):
    v = df[
        (df["tag_table_usage"] == "sal_soc")
        | ((df["tag_table_usage"] == "sal_soe") & (df["tag_closed02_yn"] == "n"))
    ]
    v = v[(v["tag_deleted_yn"] == "n") & (v["tag_void_yn"] == "n")]
    return v


def _party_set_month(v, y: int, m: int) -> set:
    sub = v[(v["year"] == y) & (v["month"] == m)]
    s = sub["party_desc"].dropna().astype(str).str.strip()
    return {x for x in s.unique() if x}


def _party_set_year(v, y: int) -> set:
    sub = v[v["year"] == y]
    s = sub["party_desc"].dropna().astype(str).str.strip()
    return {x for x in s.unique() if x}


def parse_churn_query(query: str) -> Optional[Dict[str, Any]]:
    if not query or not _CHURN_KW.search(query):
        return None
    m = _MMYYYY.search(query)
    if m:
        mm, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12 and 1900 <= yy <= 2100:
            return {"mode": "mom", "curr_y": yy, "curr_m": mm}
    m2 = _YYYYMM.search(query)
    if m2:
        yy, mm = int(m2.group(1)), int(m2.group(2))
        if 1 <= mm <= 12 and 1900 <= yy <= 2100:
            return {"mode": "mom", "curr_y": yy, "curr_m": mm}
    ym = _YEAR.search(query)
    if ym:
        y = int(ym.group(0))
        if 1900 <= y <= 2100:
            return {"mode": "yoy", "year": y}
    return None


def compute_churn_result(df_main, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if df_main is None or len(df_main) == 0:
        return None
    required = (
        "party_desc",
        "year",
        "month",
        "tag_table_usage",
        "tag_deleted_yn",
        "tag_void_yn",
        "tag_closed02_yn",
    )
    if not all(c in df_main.columns for c in required):
        return None

    v = valid_sales_main(df_main)

    if spec["mode"] == "mom":
        cy, cm = int(spec["curr_y"]), int(spec["curr_m"])
        if cm > 1:
            py, pm = cy, cm - 1
        else:
            py, pm = cy - 1, 12
        P = _party_set_month(v, py, pm)
        C = _party_set_month(v, cy, cm)
        churned = P - C
        base_n, churn_n = len(P), len(churned)
        pct = (100.0 * churn_n / base_n) if base_n else None
        summary = _format_mom_summary(py, pm, cy, cm, base_n, churn_n, pct)
        return {
            "summary": summary,
            "data": {
                "churn_kind": "mom",
                "prior_year": py,
                "prior_month": pm,
                "report_year": cy,
                "report_month": cm,
                "baseline_active": base_n,
                "churned_count": churn_n,
                "churn_rate_pct": pct,
            },
        }

    if spec["mode"] == "yoy":
        y = int(spec["year"])
        yp = y - 1
        P = _party_set_year(v, yp)
        Cy = _party_set_year(v, y)
        churned = P - Cy
        base_n, churn_n = len(P), len(churned)
        pct = (100.0 * churn_n / base_n) if base_n else None
        summary = _format_yoy_summary(yp, y, base_n, churn_n, pct)
        return {
            "summary": summary,
            "data": {
                "churn_kind": "yoy",
                "base_year": yp,
                "report_year": y,
                "baseline_active": base_n,
                "churned_count": churn_n,
                "churn_rate_pct": pct,
            },
        }
    return None


def _format_mom_summary(py, pm, cy, cm, base_n, churn_n, pct) -> str:
    pnm = calendar.month_name[pm]
    cnm = calendar.month_name[cm]
    lines = [
        "**Customer churn (month-over-month, transactional)**",
        "",
        f"- **Definition:** Customers with ≥1 valid invoice in **{pnm} {py}** who had **no** valid invoice in **{cnm} {cy}**.",
        f"- **Baseline month:** {pnm} {py}",
        f"- **Reporting month:** {cnm} {cy}",
        f"- **Active customers (baseline):** {base_n}",
        f"- **Churned:** {churn_n}",
    ]
    if base_n == 0:
        lines.append("- **Churn rate:** N/A (no active customers in baseline month).")
    else:
        lines.append(f"- **Churn rate:** {pct:.2f}%")
    return "\n".join(lines)


def _format_yoy_summary(yp, y, base_n, churn_n, pct) -> str:
    lines = [
        "**Customer churn (year-over-year, transactional)**",
        "",
        f"- **Definition:** Customers with ≥1 valid invoice in **{yp}** who had **no** valid invoice in **{y}**.",
        f"- **Baseline year:** {yp}",
        f"- **Reporting year:** {y}",
        f"- **Active customers (baseline):** {base_n}",
        f"- **Churned:** {churn_n}",
    ]
    if base_n == 0:
        lines.append("- **Churn rate:** N/A (no active customers in baseline year).")
    else:
        lines.append(f"- **Churn rate:** {pct:.2f}%")
    return "\n".join(lines)
