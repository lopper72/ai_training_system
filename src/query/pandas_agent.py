"""
Pandas Query Agent — LangChain create_pandas_dataframe_agent (in-memory Parquet-backed frames).
Pipeline: catalog-aligned schema, intent router context, runtime self-correction on exceptions.
"""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_RETRIES = int(os.getenv("AGENT_RATE_LIMIT_RETRIES", "3"))
DEFAULT_SCHEMA_MAX_CHARS = int(os.getenv("AGENT_SCHEMA_MAX_CHARS", "1200"))
DEFAULT_AGENT_MAX_ITERATIONS = int(os.getenv("PANDAS_AGENT_MAX_ITERATIONS", "10"))

AGENT_CORE = """
You are an expert ERP data analyst. DataFrames are READ-ONLY and already loaded in memory as df1, df2, ... by dataset order.
Never use read_csv, read_parquet, open() to load data, never import os/subprocess, never eval(input), never write files.
Always start with Thought: to plan steps before Action.
For nulls: dropna() or fillna with column mean/median when appropriate; say what you did briefly in Final Answer.

ROUTING: Invoice revenue / top customers by revenue / churn / invoice counts → prefer df* that is sales_main: groupby party_desc sum amount_local.
Line/SKU quantity or "which customer bought most of product X" → sales_data lines: filter product, groupby party_desc (qnty_total vs amount_local per question).
Never use top-N revenue share as churn; churn uses valid invoice rules on sales_main-style party sets only.

LIST / CUSTOMERS IN SYSTEM: If the user asks to list customers, all customers, "customers in the system", or any request for a customer directory (not line-level transactions), do NOT print head(N) of raw line-item rows — those repeat party_desc for every invoice line. Instead:
- Prefer sales_main (invoice headers; DB table scm_sal_main): df = valid_sales(sales_main); then party_desc.drop_duplicates() or groupby('party_desc') — one row per invoice, correct customer universe.
- Only if sales_main is unavailable, fall back to sales_data (scm_sal_data line items) with unique() on party_desc — never dump duplicate transaction lines for a "list" question.
- Final Answer: one Markdown bullet per distinct customer ("- Customer Name") or per aggregated row; optional amount on the same line if you grouped revenue.

TIME: If the question includes an explicit period (4-digit year, MM/YYYY, month name + year, YYYY-MM, or [System context: ... year ...]), apply it — never ask the user to specify a period via tools.
Filter examples: month → (v['year']==Y)&(v['month']==M); year only → (v['year']==Y); parenthesize each & term.

CHURN (plain, sales_main as df2 or the frame that holds invoices): MoM = customers with invoice in prior month, none in report month; YoY = invoiced in Y-1, not in Y.
After valid_df2: MoM: compute P and C sets on party_desc; print CHURN_MOM ... pct. YoY: print CHURN_YOY.

FILTERS (exact — use the df index matching the dataset name in ACTIVE FRAMES):
On sales_main-like frame (df2 if present, else the invoice frame): valid_df2=df2[(df2['tag_table_usage']=='sal_soc')|((df2['tag_table_usage']=='sal_soe')&(df2['tag_closed02_yn']=='n'))]; valid_df2=valid_df2[(valid_df2['tag_deleted_yn']=='n')&(valid_df2['tag_void_yn']=='n')]
On sales_data-like frame (df1 when it is line items): valid_df1=df1[(df1['tag_table_usage']=='sal_soc')|((df1['tag_table_usage']=='sal_soe')&(df1['tag_closed02_yn']=='n'))]; valid_df1=valid_df1[(valid_df1['tag_deleted_yn']=='n')&(valid_df1['tag_void_yn']=='n')]

EXEC: Print scalars or .head(5) only for exploratory checks; for user-facing lists use unique/groupby as above. Minimize tool calls. Need Observation before Final Answer.
Final Answer: clear English only — Markdown bullets (lines starting with "- ") or one small markdown table; easy to read on web/mobile. No Python/JSON in Final Answer.
If task_type is chart from the router, describe the chart type and key series briefly (you cannot render pixels here).
"""

AGENT_FORMAT_INSTRUCTIONS = """
After Observation: Thought: I now have the answer. then Final Answer: ... Never another Action after that Thought.
Thought/Action in English. Top-N lists → compact markdown table or "- " bullets from Observation values only; do not invent numbers.
For "list" questions about customers/suppliers, never paste repetitive raw rows — use unique() or groupby(party_desc) per LIST / CUSTOMERS IN SYSTEM above.
Action Input = pure Python + print(...); tools only python_repl_ast, market_web_search.
"""


def _active_frames_preamble(ordered_names: List[str]) -> str:
    lines = ["ACTIVE FRAMES (read-only):"]
    for i, name in enumerate(ordered_names, start=1):
        lines.append(f"  df{i} = dataset `{name}`")
    return "\n".join(lines)


class PandasQueryAgent:
    """LangChain Pandas agent with per-query dataframe subset and runtime retry on exceptions."""

    def __init__(self, llm: Any, dataframes: Dict[str, Any], schema_description: str = "") -> None:
        self.llm = llm
        self.dataframes = dataframes
        self.schema_description = schema_description
        self._agent = None
        self._active_df_names: Optional[Tuple[str, ...]] = None

    def _build_agent(self, ordered_names: Tuple[str, ...]) -> None:
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
        except ImportError as e:
            logger.error(f"[PandasAgent] Missing dependency: {e}")
            raise

        df_list: List[Any] = []
        for name in ordered_names:
            if name not in self.dataframes:
                continue
            df = self.dataframes[name]
            if df is not None and not getattr(df, "empty", True):
                df_list.append(df)
                logger.info(f"[PandasAgent] Bound df slot: {name} ({len(df)} rows)")

        if not df_list:
            raise ValueError("No DataFrames available to build agent")

        # Always pass a list so locals are df1, df2, ... (even for a single frame).
        df_input = df_list

        try:
            from langchain_core.tools import Tool
            from src.query.web_search_tool import search_market_trends

            web_tool = Tool(
                name="market_web_search",
                func=search_market_trends,
                description="Market trends: comma-separated product names. Strategic: TOP_GLOBAL: <product> | <short goal>.",
            )
            extra_tools = [web_tool]
        except Exception as e:
            logger.warning(f"[PandasAgent] Web search tool unavailable: {e}")
            extra_tools = []

        preamble = _active_frames_preamble(list(ordered_names))
        prefix = f"{preamble}\n{AGENT_CORE}"

        self._agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df_input,
            agent_type="zero-shot-react-description",
            verbose=True,
            allow_dangerous_code=True,
            prefix=prefix,
            max_iterations=max(4, min(DEFAULT_AGENT_MAX_ITERATIONS, 15)),
            early_stopping_method="force",
            number_of_head_rows=2,
            include_df_in_prompt=False,
            extra_tools=extra_tools,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        self._active_df_names = tuple(ordered_names)
        logger.info(f"[PandasAgent] Agent built for {ordered_names}")

    def _ensure_agent(self, ordered_names: Tuple[str, ...]) -> None:
        if self._agent is not None and self._active_df_names == ordered_names:
            return
        self._agent = None
        self._active_df_names = None
        self._build_agent(ordered_names)

    def query(
        self,
        question: str,
        *,
        ordered_dataset_names: Optional[Sequence[str]] = None,
        router_payload: Optional[Dict[str, Any]] = None,
        max_runtime_retries: int = 3,
    ) -> Dict[str, Any]:
        names = self._resolve_order(ordered_dataset_names)
        correction_suffix = ""
        attempts = max(1, int(max_runtime_retries))

        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                self._ensure_agent(tuple(names))
                schema_hint = self._compact_schema_description(self.schema_description)
                router_line = ""
                if router_payload:
                    router_line = f"\nRouter JSON: {router_payload}\n"
                    ep = router_payload.get("execution_plan")
                    if ep:
                        router_line += f"\nExecution plan (multi-step — run steps in order; use prior step outputs as filters when indicated): {ep}\n"
                full_prompt = (
                    f"{schema_hint}{router_line}\n"
                    f"Q: {question}{correction_suffix}\n"
                    f"{AGENT_FORMAT_INSTRUCTIONS}"
                )
                result = self._invoke_with_rate_limit_retry({"input": full_prompt})
                answer = result.get("output", "")

                if not answer or not str(answer).strip():
                    last_error = "Agent returned empty response"
                    correction_suffix = self._correction_block(last_error, attempt)
                    self.reset()
                    continue

                if "Agent stopped due to iteration limit or time limit" in answer:
                    last_error = "Agent stopped due to iteration/time limit"
                    correction_suffix = self._correction_block(last_error, attempt)
                    self.reset()
                    continue

                if self._looks_like_template_answer(answer):
                    last_error = "Agent returned template output instead of grounded result"
                    correction_suffix = self._correction_block(last_error, attempt)
                    self.reset()
                    continue

                return {
                    "success": True,
                    "answer": answer,
                    "source": "langchain_agent",
                    "router": router_payload,
                    "active_datasets": list(names),
                    "runtime_attempts": attempt + 1,
                }

            except Exception as e:
                last_error = str(e)
                tb = traceback.format_exc()
                logger.warning(f"[PandasAgent] attempt {attempt + 1}/{attempts} failed: {e}")
                correction_suffix = self._correction_block(f"{last_error}\n{tb}", attempt)
                self.reset()

        reason_code = "agent_runtime_error"
        lower = (last_error or "").lower()
        if "not a valid tool" in lower or "could not parse" in lower or "parsing" in lower:
            reason_code = "agent_parse_error"
        return {
            "success": False,
            "error": last_error or "Agent failed after retries",
            "source": "agent",
            "reason_code": reason_code,
            "active_datasets": list(names),
            "runtime_attempts": attempts,
        }

    def _resolve_order(self, ordered_dataset_names: Optional[Sequence[str]]) -> List[str]:
        from src.query.catalog import PRIORITY_DATASETS

        if ordered_dataset_names:
            out = [n for n in ordered_dataset_names if n in self.dataframes]
            if out:
                return out
        out = []
        for name in PRIORITY_DATASETS:
            if name in self.dataframes:
                df = self.dataframes[name]
                if df is not None and not getattr(df, "empty", True):
                    out.append(name)
        return out

    @staticmethod
    def _correction_block(message: str, attempt: int) -> str:
        return (
            f"\n\n[Previous run failed (attempt {attempt + 1}). Fix the Python; follow catalog columns only.]\n"
            f"{message}\n"
        )

    def _invoke_with_rate_limit_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        attempts = max(1, DEFAULT_RATE_LIMIT_RETRIES)
        for attempt in range(1, attempts + 1):
            try:
                assert self._agent is not None
                return self._agent.invoke(payload)
            except Exception as exc:
                message = str(exc)
                if not self._is_rate_limit_error(message) or attempt >= attempts:
                    raise
                wait_seconds = self._extract_retry_after_seconds(message, default_seconds=18.0)
                wait_seconds = min(60.0, wait_seconds + (attempt - 1) * 2.0)
                logger.warning(
                    f"[PandasAgent] Rate limit, retry in {wait_seconds:.1f}s ({attempt}/{attempts})"
                )
                time.sleep(wait_seconds)
        assert self._agent is not None
        return self._agent.invoke(payload)

    def _is_rate_limit_error(self, message: str) -> bool:
        lowered = (message or "").lower()
        return (
            ("rate limit" in lowered)
            or ("rate_limit_exceeded" in lowered)
            or ("429" in lowered)
            or ("413" in lowered)
            or ("tokens per minute" in lowered)
        )

    def _extract_retry_after_seconds(self, message: str, default_seconds: float = 18.0) -> float:
        match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
        if not match:
            return default_seconds
        try:
            return max(1.0, float(match.group(1)))
        except ValueError:
            return default_seconds

    def _compact_schema_description(self, schema_text: str) -> str:
        text = (schema_text or "").strip()
        if not text:
            return "Schema:\n(catalog unavailable)"
        header = "Schema:\n"
        body = text
        if len(body) > DEFAULT_SCHEMA_MAX_CHARS:
            body = body[:DEFAULT_SCHEMA_MAX_CHARS].rstrip() + "\n...[truncated]"
        return header + body

    def _looks_like_template_answer(self, answer: str) -> bool:
        lower = (answer or "").lower()
        if "customer name -" in lower or "product name -" in lower:
            return True
        if "123,456.78" in answer and "98,765.43" in answer:
            return True
        return False

    def reset(self) -> None:
        self._agent = None
        self._active_df_names = None
        logger.info("[PandasAgent] Agent reset")
