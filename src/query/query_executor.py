"""
Template query executor: each canonical intent_code maps to fixed pandas logic.
Supports chained steps with dependency_source and per-step catalog checks.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.query.catalog import CATALOG, validate_router_targets
from src.query.intent_codes import (
    EXECUTOR_INTENT_CODES,
    canonical_intent,
    query_type_for_intent,
)


def _distinct_customers_view(view: str) -> bool:
    v = (view or "").lower().strip()
    return v in ("distinct_customers", "all_customers", "unique_customers")


def _sub_query_implies_all_customers(plan: Dict[str, Any]) -> bool:
    """Heuristic: user wants every distinct customer, not a preview of N transaction rows."""
    sq = str(plan.get("sub_query") or "").lower()
    if not sq.strip():
        return False
    customer = any(
        w in sq for w in ("customer", "customers", "party", "client", "buyer", "khách")
    )
    if not customer:
        return False
    phrases = (
        "all customer",
        "all customers",
        "every customer",
        "each customer",
        "entire ",
        "full list",
        "complete list",
        "list all",
        "get all",
        "show all",
        "names of all",
        "get every",
        "tất cả",
        "toàn bộ",
    )
    if any(p in sq for p in phrases):
        return True
    # "list customers in system" / directory without the word "all"
    if ("list" in sq or "get" in sq or "show" in sq) and "system" in sq:
        return True
    if "in system" in sq and ("customer" in sq or "customers" in sq):
        return True
    # After default-year system lines, router/planner may emit "all-time customers" (not substring "all customer")
    if re.search(r"(?i)\ball[\s-]?time\b.*\bcustomers?\b", sq) or (
        "all-time" in sq and ("customer" in sq or "customers" in sq)
    ):
        return True
    return False


def _plan_metrics_effectively_empty(plan: Dict[str, Any]) -> bool:
    mets = plan.get("metrics") if isinstance(plan.get("metrics"), list) else []
    for m in mets:
        if not isinstance(m, dict):
            continue
        col = m.get("column")
        if isinstance(col, str) and col.strip():
            return False
    return True


def _plan_dimensions_allow_distinct_party_list(plan: Dict[str, Any]) -> bool:
    dims = plan.get("dimensions") if isinstance(plan.get("dimensions"), list) else []
    cleaned = [d.strip().lower() for d in dims if isinstance(d, str) and d.strip()]
    if not cleaned:
        return True
    # Planner sometimes sends a logical name, not a column id
    if cleaned in (["party_desc"], ["customer"]):
        return True
    return False


class QueryExecutor:
    """Execute plan JSON using in-memory DataFrames."""

    def __init__(self, dataframes: Dict[str, Any]):
        self.dataframes = dataframes
        self._loaded_names: Set[str] = {
            k for k, v in (dataframes or {}).items() if v is not None and not getattr(v, "empty", True)
        }

    def execute(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else None
        if steps and len(steps) > 1:
            return self.execute_chained(plan)
        if steps and len(steps) == 1:
            return self._execute_single(self._step_to_single_plan(steps[0]))
        return self._execute_single(plan)

    def execute_chained(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        steps = plan.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            return None
        ordered = sorted(steps, key=lambda s: int(s.get("step_id", 0)))
        # Full outputs for in-process dependency resolution
        state: Dict[int, Dict[str, Any]] = {}
        # Token-light snapshots for downstream synthesis / APIs
        execution_state: Dict[int, Dict[str, Any]] = {}
        step_results: List[Dict[str, Any]] = []
        for raw_step in ordered:
            sid = int(raw_step.get("step_id", 0))
            merged, dep_ok = self._apply_dependency_to_step(copy.deepcopy(raw_step), state)
            if not self._step_passes_catalog_security(merged):
                return self._chain_incomplete_result(
                    step_results,
                    execution_state,
                    failed_step_id=sid,
                    reason_code="catalog_security",
                    user_message=(
                        "Execution stopped: a step referenced datasets outside the approved catalog "
                        "or data that is not loaded."
                    ),
                )
            if not dep_ok:
                return self._chain_incomplete_result(
                    step_results,
                    execution_state,
                    failed_step_id=sid,
                    reason_code="dependency_unresolved",
                    user_message=(
                        "Execution stopped: the next step needed a value from the previous step "
                        "(e.g. customer or product id), but that value was missing or empty."
                    ),
                )
            single = self._step_to_single_plan(merged)
            out = self._execute_single(single)
            if not out:
                return self._chain_incomplete_result(
                    step_results,
                    execution_state,
                    failed_step_id=sid,
                    reason_code="execution_failed",
                    user_message=(
                        "Execution stopped: this analytics step could not produce a result "
                        "(for example missing month/year, or no rows after filters)."
                    ),
                )
            state[sid] = out
            execution_state[sid] = self._compact_execution_entry(out)
            step_results.append({"step_id": sid, **out})
        last = step_results[-1]
        summaries = [f"[Step {i + 1}] {r.get('summary', '')}" for i, r in enumerate(step_results)]
        return {
            "summary": " ".join(summaries),
            "data": {
                "steps": step_results,
                "final": last.get("data"),
                "execution_state": execution_state,
            },
            "source": "planner_executor",
            "query_type": "multi_step",
            "intent_code": str(last.get("intent_code") or ""),
            "step_count": len(step_results),
            "chain_complete": True,
        }

    def _compact_execution_entry(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Token-light snapshot: summary + tiny data sketch (no full tables)."""
        summ = str(out.get("summary") or "")[:900]
        sketch: Dict[str, Any] = {}
        data = out.get("data")
        if isinstance(data, dict):
            for k, v in data.items():
                if k in ("latest_records", "rows", "top_customers_for_top_product"):
                    if isinstance(v, list):
                        sketch[k] = {"row_count": len(v)}
                    else:
                        sketch[k] = "present" if v else "empty"
                elif isinstance(v, dict) and k in ("top_customers", "top_products", "top_brands", "top_categories"):
                    sketch[k] = {"keys": len(v)}
                elif isinstance(v, (str, int, float, bool)) or v is None:
                    sketch[k] = v
                else:
                    sketch[k] = "<summary_only>"
        return {
            "intent_code": out.get("intent_code"),
            "query_type": out.get("query_type"),
            "summary": summ,
            "data_sketch": sketch,
        }

    def _chain_incomplete_result(
        self,
        step_results: List[Dict[str, Any]],
        execution_state: Dict[int, Dict[str, Any]],
        *,
        failed_step_id: int,
        reason_code: str,
        user_message: str,
    ) -> Dict[str, Any]:
        partial_summaries = [f"[Step {i + 1}] {r.get('summary', '')}" for i, r in enumerate(step_results)]
        glue = " ".join(partial_summaries) if partial_summaries else ""
        summary = (glue + " " + user_message).strip()
        last_data = step_results[-1].get("data") if step_results else {}
        return {
            "summary": summary,
            "data": {
                "steps": list(step_results),
                "final": last_data if isinstance(last_data, dict) else {},
                "execution_state": dict(execution_state),
                "chain_failed_at_step": failed_step_id,
                "chain_failure_reason": reason_code,
            },
            "source": "planner_executor",
            "query_type": "multi_step",
            "intent_code": str(step_results[-1].get("intent_code") or "") if step_results else "",
            "step_count": len(step_results),
            "chain_complete": False,
        }

    def _step_to_single_plan(self, step: Dict[str, Any]) -> Dict[str, Any]:
        # Keep sub_query for DATA_QUERY heuristics (e.g. distinct customers from sales_main).
        skip = {"step_id", "dependency_source"}
        return {k: v for k, v in step.items() if k not in skip}

    def _step_passes_catalog_security(self, step: Dict[str, Any]) -> bool:
        names = step.get("target_files") if isinstance(step.get("target_files"), list) else []
        if not names:
            return True
        validated = validate_router_targets(names, self._loaded_names)
        if len(validated) != len([n for n in names if str(n).strip()]):
            return False
        for n in names:
            s = str(n).strip()
            if s and s not in CATALOG:
                return False
        return True

    def _apply_dependency_to_step(
        self,
        step: Dict[str, Any],
        state: Dict[int, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool]:
        dep = step.get("dependency_source")
        if not isinstance(dep, dict) or not dep:
            return step, True
        from_id = dep.get("from_step_id")
        output_key = dep.get("output_key")
        filter_col = dep.get("filter_column")
        if from_id is None or not output_key or not filter_col:
            return step, True
        prev = state.get(int(from_id))
        if not prev:
            return step, False
        val = self._resolve_binding_value(prev, str(output_key))
        if val is None:
            return step, False
        filters = dict(step.get("filters") or {})
        filters[str(filter_col)] = val
        step["filters"] = filters
        return step, True

    def _resolve_binding_value(self, prev_result: Dict[str, Any], output_key: str) -> Any:
        data = prev_result.get("data") if isinstance(prev_result.get("data"), dict) else {}
        if output_key in data and data[output_key] is not None:
            return data[output_key]
        if output_key == "top_customer":
            tc = data.get("top_customer")
            if tc is not None:
                return tc
            td = data.get("top_customers")
            if isinstance(td, dict) and td:
                return max(td, key=lambda k: float(td[k]))
        if output_key in ("top_product", "stkcode_desc"):
            tp = data.get("top_product")
            if tp is not None:
                return tp
            tpd = data.get("top_products")
            if isinstance(tpd, dict) and tpd:
                return max(tpd, key=lambda k: float(tpd[k]))
        if output_key in ("party_desc", "top_party"):
            pd = data.get("party_desc")
            if pd is not None:
                return pd
            tc = data.get("top_customer")
            if tc is not None:
                return tc
        return None

    def _execute_single(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intent = canonical_intent(plan.get("intent"))
        if intent not in EXECUTOR_INTENT_CODES:
            return None
        if intent == "DATA_QUERY":
            return self._data_query(plan)
        if intent == "TOP_PRODUCTS":
            return self._top_products(plan)
        if intent == "LATEST_RECORDS_TOP_PRODUCT":
            return self._latest_records_of_top_product(plan)
        if intent == "TOP_CUSTOMERS_TOP_PRODUCT":
            return self._top_customers_of_top_product(plan)
        if intent == "TOP_CUSTOMERS":
            return self._top_customers(plan)
        if intent == "GET_REVENUE":
            return self._revenue_by_date(plan)
        if intent == "REVENUE_GROWTH_MOM":
            return self._revenue_growth_mom(plan)
        if intent == "TOP_BRANDS":
            return self._top_brands(plan)
        if intent == "TOP_CATEGORIES":
            return self._top_categories(plan)
        if intent == "COMPARE_CHURN":
            return self._compare_churn(plan)
        return None

    def _apply_plan_filters(self, df, plan: Dict[str, Any]):
        filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
        for col, val in filters.items():
            if col in df.columns and isinstance(val, (str, int, float, bool)):
                df = df[df[col] == val]
        return df

    def _valid_sales(self, df):
        valid_df = df[
            (df["tag_table_usage"] == "sal_soc")
            | ((df["tag_table_usage"] == "sal_soe") & (df["tag_closed02_yn"] == "n"))
        ]
        valid_df = valid_df[(valid_df["tag_deleted_yn"] == "n") & (valid_df["tag_void_yn"] == "n")]
        return valid_df

    def _apply_time(self, df, plan: Dict[str, Any]):
        month = (plan.get("time") or {}).get("month")
        year = (plan.get("time") or {}).get("year")
        if month and year and "month" in df.columns and "year" in df.columns:
            return df[(df["year"] == year) & (df["month"] == month)]
        return df

    def _compare_churn(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from src.query.churn_deterministic import compute_churn_result

        sm = self.dataframes.get("sales_main")
        if sm is None or len(sm) == 0:
            return None
        t = plan.get("time") or {}
        y, m = t.get("year"), t.get("month")
        spec = None
        if isinstance(y, int) and isinstance(m, int) and 1 <= m <= 12:
            spec = {"mode": "mom", "curr_y": y, "curr_m": m}
        elif isinstance(y, int) and m is None:
            spec = {"mode": "yoy", "year": y}
        if not spec:
            return None
        out = compute_churn_result(sm, spec)
        if not out:
            return None
        ic = "COMPARE_CHURN"
        return {
            "summary": out["summary"],
            "data": out.get("data") or {},
            "source": "deterministic_executor",
            "query_type": query_type_for_intent(ic),
            "intent_code": ic,
        }

    def _top_products(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = self._apply_plan_filters(df, plan)
        df = df[df["stkcode_desc"].notna() & (df["stkcode_desc"].astype(str).str.strip() != "")]
        top_n = plan.get("top_n") or 5
        grouped = df.groupby("stkcode_desc")["amount_local"].sum().sort_values(ascending=False).head(top_n)
        period = self._period_text(plan)
        if top_n == 1 and not grouped.empty:
            name = str(grouped.index[0])
            rev = float(grouped.iloc[0])
            period_value = None
            month = (plan.get("time") or {}).get("month")
            year = (plan.get("time") or {}).get("year")
            if month and year:
                period_value = f"{month:02d}/{year}"
            return {
                "summary": f"Top product{period}: {name} ({rev:,.2f} VND)",
                "data": {"top_product": name, "top_revenue": rev, "period": period_value},
                "source": "deterministic_executor",
                "query_type": query_type_for_intent("TOP_PRODUCTS"),
                "intent_code": "TOP_PRODUCTS",
            }
        return {
            "summary": f"Top {top_n} products by revenue{period}",
            "data": {"top_products": grouped.to_dict()},
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("TOP_PRODUCTS"),
            "intent_code": "TOP_PRODUCTS",
        }

    def _top_customers(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = self._apply_plan_filters(df, plan)
        df = df[df["party_desc"].notna() & (df["party_desc"].astype(str).str.strip() != "")]
        top_n = plan.get("top_n") or 10
        grouped = df.groupby("party_desc")["amount_local"].sum().sort_values(ascending=False).head(top_n)
        data: Dict[str, Any] = {"top_customers": grouped.to_dict()}
        if top_n == 1 and not grouped.empty:
            name = str(grouped.index[0])
            data["top_customer"] = name
            data["party_desc"] = name
        return {
            "summary": f"Top {top_n} customers by revenue{self._period_text(plan)}",
            "data": data,
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("TOP_CUSTOMERS"),
            "intent_code": "TOP_CUSTOMERS",
        }

    def _latest_records_of_top_product(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = df[df["stkcode_desc"].notna() & (df["stkcode_desc"].astype(str).str.strip() != "")]
        df = self._apply_plan_filters(df, plan)
        filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
        forced = filters.get("stkcode_desc")

        if df.empty:
            return {
                "summary": f"No records found{self._period_text(plan)}.",
                "data": {"latest_records": []},
                "source": "deterministic_executor",
                "query_type": query_type_for_intent("LATEST_RECORDS_TOP_PRODUCT"),
                "intent_code": "LATEST_RECORDS_TOP_PRODUCT",
            }

        if forced:
            top_product = str(forced)
            subset_pre = df[df["stkcode_desc"] == top_product]
            top_revenue = float(subset_pre["amount_local"].sum()) if not subset_pre.empty else 0.0
        else:
            top_series = df.groupby("stkcode_desc")["amount_local"].sum().sort_values(ascending=False).head(1)
            top_product = str(top_series.index[0])
            top_revenue = float(top_series.iloc[0])

        subset = df[df["stkcode_desc"] == top_product].copy()
        if "date_trans" in subset.columns:
            subset["date_trans"] = subset["date_trans"].astype(str)
            subset = subset.sort_values("date_trans", ascending=False)

        top_n = int(plan.get("top_n") or 5)
        keep_cols = [
            "date_trans",
            "stkcode_desc",
            "amount_local",
            "qnty_total",
            "party_desc",
        ]
        use_cols = [c for c in keep_cols if c in subset.columns]
        latest = subset[use_cols].head(top_n)
        latest_records = latest.to_dict(orient="records")

        return {
            "summary": (
                f"Top product{self._period_text(plan)}: {top_product} ({top_revenue:,.2f} VND). "
                f"Showing {len(latest_records)} latest records."
            ),
            "data": {
                "top_product": top_product,
                "top_revenue": top_revenue,
                "latest_records": latest_records,
            },
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("LATEST_RECORDS_TOP_PRODUCT"),
            "intent_code": "LATEST_RECORDS_TOP_PRODUCT",
        }

    def _top_customers_of_top_product(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = df[df["stkcode_desc"].notna() & (df["stkcode_desc"].astype(str).str.strip() != "")]
        df = df[df["party_desc"].notna() & (df["party_desc"].astype(str).str.strip() != "")]
        df = self._apply_plan_filters(df, plan)
        filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
        if df.empty:
            return {
                "summary": f"No records found{self._period_text(plan)}.",
                "data": {},
                "source": "deterministic_executor",
                "query_type": query_type_for_intent("TOP_CUSTOMERS_TOP_PRODUCT"),
                "intent_code": "TOP_CUSTOMERS_TOP_PRODUCT",
            }

        if filters.get("stkcode_desc"):
            top_product = str(filters["stkcode_desc"])
            product_df = df[df["stkcode_desc"] == top_product]
            top_product_revenue = float(product_df["amount_local"].sum()) if not product_df.empty else 0.0
        else:
            top_product_series = df.groupby("stkcode_desc")["amount_local"].sum().sort_values(ascending=False).head(1)
            top_product = str(top_product_series.index[0])
            top_product_revenue = float(top_product_series.iloc[0])
            product_df = df[df["stkcode_desc"] == top_product]

        top_n = int(plan.get("top_n") or 10)
        by_customer = product_df.groupby("party_desc")["amount_local"].sum().sort_values(ascending=False).head(top_n)
        result_rows = [{"customer": str(name), "amount_local": float(val)} for name, val in by_customer.items()]
        top_customer = result_rows[0]["customer"] if result_rows else "N/A"
        top_customer_amount = result_rows[0]["amount_local"] if result_rows else 0.0

        return {
            "summary": (
                f"Top product{self._period_text(plan)} is {top_product} ({top_product_revenue:,.2f} VND). "
                f"Top customer buying this product: {top_customer} ({top_customer_amount:,.2f} VND)."
            ),
            "data": {
                "top_product": top_product,
                "top_product_revenue": top_product_revenue,
                "top_customers_for_top_product": result_rows,
            },
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("TOP_CUSTOMERS_TOP_PRODUCT"),
            "intent_code": "TOP_CUSTOMERS_TOP_PRODUCT",
        }

    def _revenue_by_date(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        month = (plan.get("time") or {}).get("month")
        year = (plan.get("time") or {}).get("year")
        if not month or not year:
            return None
        df = self._valid_sales(self.dataframes["sales_main"])
        df = df[(df["year"] == year) & (df["month"] == month)]
        df = self._apply_plan_filters(df, plan)
        total = float(df["amount_local"].sum()) if not df.empty else 0.0
        txn = int(len(df))
        return {
            "summary": f"Revenue for {month:02d}/{year}: {total:,.2f} VND ({txn} transactions)",
            "data": {"period": f"{month:02d}/{year}", "total_revenue": total, "transaction_count": txn},
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("GET_REVENUE"),
            "intent_code": "GET_REVENUE",
        }

    def _revenue_growth_mom(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        month = (plan.get("time") or {}).get("month")
        year = (plan.get("time") or {}).get("year")
        if not month or not year:
            return None
        df = self._valid_sales(self.dataframes["sales_main"])
        df = self._apply_plan_filters(df, plan)
        current_df = df[(df["year"] == year) & (df["month"] == month)]
        current = float(current_df["amount_local"].sum()) if not current_df.empty else 0.0
        prev_month = 12 if month == 1 else month - 1
        prev_year = year - 1 if month == 1 else year
        prev_df = df[(df["year"] == prev_year) & (df["month"] == prev_month)]
        previous = float(prev_df["amount_local"].sum()) if not prev_df.empty else 0.0
        growth = ((current - previous) / previous * 100.0) if previous > 0 else 0.0
        return {
            "summary": (
                f"Revenue growth for {month:02d}/{year} vs {prev_month:02d}/{prev_year}: "
                f"{growth:.2f}% (current: {current:,.2f} VND, previous: {previous:,.2f} VND)"
            ),
            "data": {
                "period": f"{month:02d}/{year}",
                "previous_period": f"{prev_month:02d}/{prev_year}",
                "current_revenue": current,
                "previous_revenue": previous,
                "growth_rate_pct": growth,
            },
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("REVENUE_GROWTH_MOM"),
            "intent_code": "REVENUE_GROWTH_MOM",
        }

    def _top_brands(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = self._apply_plan_filters(df, plan)
        df = df[df["brand_desc"].notna() & (df["brand_desc"].astype(str).str.strip() != "")]
        top_n = plan.get("top_n") or 5
        grouped = df.groupby("brand_desc")["amount_local"].sum().sort_values(ascending=False).head(top_n)
        return {
            "summary": f"Top {top_n} brands by revenue{self._period_text(plan)}",
            "data": {"top_brands": grouped.to_dict()},
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("TOP_BRANDS"),
            "intent_code": "TOP_BRANDS",
        }

    def _top_categories(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        df = self._valid_sales(self.dataframes["sales_data"])
        df = self._apply_time(df, plan)
        df = self._apply_plan_filters(df, plan)
        df = df[df["stkcate_desc"].notna() & (df["stkcate_desc"].astype(str).str.strip() != "")]
        top_n = plan.get("top_n") or 5
        grouped = df.groupby("stkcate_desc")["amount_local"].sum().sort_values(ascending=False).head(top_n)
        return {
            "summary": f"Top {top_n} categories by revenue{self._period_text(plan)}",
            "data": {"top_categories": grouped.to_dict()},
            "source": "deterministic_executor",
            "query_type": query_type_for_intent("TOP_CATEGORIES"),
            "intent_code": "TOP_CATEGORIES",
        }

    def _period_text(self, plan: Dict[str, Any]) -> str:
        month = (plan.get("time") or {}).get("month")
        year = (plan.get("time") or {}).get("year")
        if month and year:
            return f" for {month:02d}/{year}"
        return ""

    def _data_query(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dataset = plan.get("dataset")
        if dataset not in self.dataframes:
            dataset = "sales_data" if "sales_data" in self.dataframes else None
        if not dataset:
            return None

        df = self.dataframes[dataset].copy()
        if dataset in {"sales_data", "sales_main"}:
            df = self._valid_sales(df)
        df = self._apply_time(df, plan)

        filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]

        dimensions = [c for c in (plan.get("dimensions") or []) if isinstance(c, str) and c in df.columns]
        metrics = plan.get("metrics") if isinstance(plan.get("metrics"), list) else []
        limit = int(plan.get("limit") or 20)
        limit = max(1, min(limit, 100))
        view = (plan.get("view") or "").lower()
        sort = plan.get("sort") if isinstance(plan.get("sort"), dict) else {}

        # Full distinct customer list from invoice headers (sales_main / scm_sal_main), not line-item sample
        if (
            _sub_query_implies_all_customers(plan)
            and _plan_dimensions_allow_distinct_party_list(plan)
            and _plan_metrics_effectively_empty(plan)
        ):
            view = "distinct_customers"
        if _distinct_customers_view(view):
            eff_ds: Optional[str] = None
            sm = self.dataframes.get("sales_main")
            if sm is not None and not getattr(sm, "empty", True):
                eff_ds = "sales_main"
            elif dataset in self.dataframes and not getattr(self.dataframes[dataset], "empty", True):
                eff_ds = dataset
            elif "sales_data" in self.dataframes:
                eff_ds = "sales_data"
            if not eff_ds:
                return None
            df = self.dataframes[eff_ds].copy()
            dataset = eff_ds
            if dataset in {"sales_data", "sales_main"}:
                df = self._valid_sales(df)
            df = self._apply_time(df, plan)
            filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
            for col, val in filters.items():
                if col in df.columns:
                    df = df[df[col] == val]
            col = "party_desc"
            if col not in df.columns:
                return None
            ser = df[col].dropna().astype(str).str.strip()
            ser = ser[ser != ""]
            full_n = int(ser.nunique())
            unique_vals = sorted(ser.unique().tolist())
            max_distinct = 20000
            truncated = len(unique_vals) > max_distinct
            unique_vals = unique_vals[:max_distinct]
            records = [{col: v} for v in unique_vals]
            src_label = "sales_main (invoice headers; scm_sal_main)"
            if dataset == "sales_data":
                src_label = "sales_data (line items; scm_sal_data)"
            summary = (
                f"Found {full_n} distinct customers by party_desc from {src_label}{self._period_text(plan)}."
            )
            if truncated:
                summary += f" Listing first {len(records)} names (display cap {max_distinct})."
            return {
                "summary": summary,
                "data": {"rows": records},
                "source": "planner_executor",
                "query_type": query_type_for_intent("DATA_QUERY"),
                "intent_code": "DATA_QUERY",
            }

        if view == "latest_records":
            sort_col = "date_trans" if "date_trans" in df.columns else None
            if sort_col:
                df = df.sort_values(sort_col, ascending=False)
            rows = df.head(limit)
            keep_cols = dimensions if dimensions else [c for c in ["date_trans", "stkcode_desc", "party_desc", "amount_local"] if c in rows.columns]
            records = rows[keep_cols].to_dict(orient="records") if keep_cols else []
            return {
                "summary": f"Showing {len(records)} latest records from {dataset}{self._period_text(plan)}.",
                "data": {"latest_records": records},
                "source": "planner_executor",
                "query_type": query_type_for_intent("DATA_QUERY"),
                "intent_code": "DATA_QUERY",
            }

        if dimensions and metrics:
            agg_map = {}
            rename_map = {}
            for i, m in enumerate(metrics):
                if not isinstance(m, dict):
                    continue
                func = str(m.get("name", "sum")).lower()
                col = m.get("column")
                if col not in df.columns:
                    continue
                if func not in {"sum", "count", "avg", "max", "min"}:
                    func = "sum"
                pandas_func = "mean" if func == "avg" else func
                out_col = f"m{i}_{func}_{col}"
                agg_map[out_col] = (col, pandas_func)
                rename_map[out_col] = f"{func}_{col}"
            if not agg_map:
                return None
            grouped = df.groupby(dimensions).agg(**agg_map).reset_index()
            grouped = grouped.rename(columns=rename_map)
            sort_by = sort.get("by")
            if sort_by in grouped.columns:
                grouped = grouped.sort_values(sort_by, ascending=(sort.get("order") == "asc"))
            rows = grouped.head(limit).to_dict(orient="records")
            return {
                "summary": f"Computed grouped result from {dataset}{self._period_text(plan)} with {len(rows)} rows.",
                "data": {"rows": rows},
                "source": "planner_executor",
                "query_type": query_type_for_intent("DATA_QUERY"),
                "intent_code": "DATA_QUERY",
            }

        # Fallback table preview
        sort_col = sort.get("by") if sort.get("by") in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=(sort.get("order") == "asc"))
        rows = df.head(limit)
        keep_cols = dimensions if dimensions else [c for c in ["date_trans", "stkcode_desc", "party_desc", "amount_local"] if c in rows.columns]
        records = rows[keep_cols].to_dict(orient="records") if keep_cols else rows.head(limit).to_dict(orient="records")
        return {
            "summary": f"Showing {len(records)} records from {dataset}{self._period_text(plan)}.",
            "data": {"rows": records},
            "source": "planner_executor",
            "query_type": query_type_for_intent("DATA_QUERY"),
            "intent_code": "DATA_QUERY",
        }
