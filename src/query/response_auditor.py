"""
LLM auditor: groundedness check + final synthesis for multi-step ERP answers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Polite empty-state when there is nothing grounded to synthesize (no LLM call needed).
EMPTY_EXECUTION_MESSAGE = (
    "We could not find any matching data for your question in the available extracts "
    "for the selected period or filters. Please try a different date range, scope, or wording."
)

AUDITOR_SYSTEM = """You audit analytics answers over internal ERP extracts.
Return ONLY valid JSON (no markdown):
{
  "answers_question": true|false,
  "sensitive_risk": "none|low|medium|high",
  "issues": ["short English notes if any"],
  "polished_answer": "optional: rewrite the answer in clear English with short bullets or one markdown table; empty string to keep original"
}

Rules:
- answers_question=false if numbers or conclusions are missing for what was asked.
- sensitive_risk>none if the text appears to dump raw personal identifiers beyond business need (e.g. full unrelated ID lists).
- polished_answer must not invent numbers; only reorganize wording. If unsure, set polished_answer to "".
- Hallucination guard: use only numbers and real entity names that already appear in the draft answer or in evidence the user supplied in the conversation. If a figure or name is not present, do not infer, interpolate, or round values unless you explicitly state that you are rounding and what the exact source values were; otherwise omit the claim.
- Lists (customers, products, SKUs, categories, any finite set): ALWAYS use Markdown bullet lines (each line exactly starting with "- " then the item). Do not leave a list as one dense comma-separated line or as "key=value" blobs. If the draft looks like raw ERP rows (repeated fields like date_trans=, party_desc=), rewrite into one "- " line per logical item, preserving every name and number from the draft.
- Ranked or Top-N lists: same rule — one "- " per row of the ranking; keep amounts on the same bullet as the entity when present.
- Currency and amounts: keep currency units and labels exactly as in the draft (VND, USD, etc.). Keep original numeric formatting from the source (e.g. thousand separators); do not rewrite amounts into shorthand like turning 1.000.000 into "1M" unless the draft already used that form.
"""

SYNTHESIS_SYSTEM = """You are a senior ERP business analyst. You receive:
- The user's original question
- execution_history: a compact list of steps. Each entry has "summary" and may include "ranked_preview": short lines with entity names and amounts copied from rolled-up extracts (not full raw tables). Summaries and ranked_preview together are the ONLY source of grounded numbers and entity names.

Your job: produce ONE professional answer in English suitable for web/mobile readers.

Hallucination guard (mandatory):
- Use ONLY numbers and real entity names that explicitly appear in execution_history (summary and ranked_preview). If information is not there, do not guess, infer, or silently round; never invent metrics. If you must round or aggregate, state clearly that you are doing so and cite the exact figures from execution_history you used — otherwise omit the claim.

Formatting:
- Do NOT label steps as "Step 1", "Step 2", or similar runbook headers.
- When execution_history implies a finite ranked list (e.g. Top 5 products, top customers, multiple line items), present those items as Markdown bullet points (each line: "- item with amounts copied verbatim from summary or ranked_preview"). If ranked_preview lists concrete names and amounts, you MUST include them in the bullets — do not answer with only a title line.
- For a single scalar outcome (one revenue total, one churn rate), concise prose without bullets is fine.
- Currency and units: preserve currency codes and wording exactly as they appear in execution_history (VND, USD, etc.). Keep original currency and number formatting from the ERP summaries (including thousand separators and decimal style); never compress numbers into informal shorthand (e.g. do not change 1.000.000 into "1M") unless that exact shorthand already appears in execution_history.

Narrative:
- Weave non-list facts into clear prose (e.g. overall totals, period, partial-chain explanation).
- If the chain did not complete, explain in business language what was determined and what could not be finished — no stack traces or internal error codes.

Length: keep reasonable; prefer clarity over length.

Empty or failed runs:
- If execution_history is missing, empty, or every step has an empty summary with no usable figures or names, you MUST NOT invent content. Set final_answer to a single polite, clear English sentence stating that no matching data was found or the request could not be completed from the extracts, and set chain_addressed to false. Do not output an empty string for final_answer.

Return ONLY valid JSON (no markdown fences):
{
  "final_answer": "the synthesized answer (may include Markdown bullets for lists)",
  "chain_addressed": true|false
}
"""


class ResponseAuditor:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def audit_and_polish(self, original_query: str, draft_answer: str) -> Dict[str, Any]:
        """Grounded audit + optional rewrite for user-facing text (e.g. LangChain agent output)."""
        return self.audit(original_query, draft_answer)

    def audit(self, original_query: str, draft_answer: str) -> Dict[str, Any]:
        user = (
            f"Original question:\n{original_query}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Return audit JSON."
        )
        try:
            msg = self.llm.invoke(
                [
                    {"role": "system", "content": AUDITOR_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
            raw = getattr(msg, "content", "") or ""
            parsed = self._parse_json(raw)
            if not parsed:
                return self._passthrough(draft_answer)
            polished = parsed.get("polished_answer")
            if isinstance(polished, str) and polished.strip():
                final_text = polished.strip()
            else:
                final_text = draft_answer
            return {
                "final_answer": final_text,
                "answers_question": bool(parsed.get("answers_question", True)),
                "sensitive_risk": str(parsed.get("sensitive_risk") or "none"),
                "issues": parsed.get("issues") if isinstance(parsed.get("issues"), list) else [],
                "audited": True,
            }
        except Exception as exc:
            logger.warning(f"[ResponseAuditor] audit failed: {exc}")
            return self._passthrough(draft_answer)

    def synthesize_chain(
        self,
        original_query: str,
        *,
        step_summaries: List[Dict[str, Any]],
        chain_complete: bool,
        failure_reason: Optional[str] = None,
        failed_step_id: Optional[int] = None,
        max_summary_chars: int = 600,
        max_ranked_preview_chars: int = 900,
    ) -> Dict[str, Any]:
        """
        Build one ERP-style narrative from multi-step summaries (token-efficient).
        step_summaries items: {step_id?, intent_code?, summary?, ranked_preview?}
        """
        compact: List[Dict[str, Any]] = []
        for i, row in enumerate(step_summaries):
            if not isinstance(row, dict):
                continue
            summ = str(row.get("summary") or "").strip()[:max_summary_chars]
            rprev = str(row.get("ranked_preview") or "").strip()[:max_ranked_preview_chars]
            if not summ and not rprev:
                continue
            entry: Dict[str, Any] = {
                "step": row.get("step_id", i),
                "intent": str(row.get("intent_code") or row.get("intent") or ""),
                "summary": summ,
            }
            if rprev:
                entry["ranked_preview"] = rprev
            compact.append(entry)
        if not compact:
            return {
                "final_answer": EMPTY_EXECUTION_MESSAGE,
                "chain_addressed": False,
                "synthesized": False,
            }
        payload = {
            "original_query": original_query,
            "execution_history": compact,
            "chain_complete": chain_complete,
            "failure_reason": failure_reason,
            "failed_step_id": failed_step_id,
        }
        user = (
            "Synthesize the final answer from this JSON. "
            "The field execution_history is the only grounded evidence (per-step summaries and ranked_preview when present):\n"
            f"{json.dumps(payload, ensure_ascii=True)}\n\n"
            "Return synthesis JSON only."
        )
        try:
            msg = self.llm.invoke(
                [
                    {"role": "system", "content": SYNTHESIS_SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
            raw = getattr(msg, "content", "") or ""
            parsed = self._parse_json(raw)
            if not parsed:
                return self._synthesis_fallback(original_query, compact, chain_complete)
            fa = parsed.get("final_answer")
            if isinstance(fa, str) and fa.strip():
                return {
                    "final_answer": fa.strip(),
                    "chain_addressed": bool(parsed.get("chain_addressed", True)),
                    "synthesized": True,
                }
            return {
                "final_answer": EMPTY_EXECUTION_MESSAGE,
                "chain_addressed": False,
                "synthesized": False,
            }
        except Exception as exc:
            logger.warning(f"[ResponseAuditor] synthesize_chain failed: {exc}")
        return self._synthesis_fallback(original_query, compact, chain_complete)

    def _synthesis_fallback(
        self,
        query: str,
        compact: List[Dict[str, Any]],
        chain_complete: bool,
    ) -> Dict[str, Any]:
        parts: List[str] = []
        for x in compact:
            s = str(x.get("summary") or "").strip()
            r = str(x.get("ranked_preview") or "").strip()
            if s and r:
                parts.append(f"{s}\n{r}")
            elif s:
                parts.append(s)
            elif r:
                parts.append(r)
        if not parts:
            return {
                "final_answer": EMPTY_EXECUTION_MESSAGE,
                "chain_addressed": False,
                "synthesized": False,
            }
        text = " ".join(parts)
        if not chain_complete:
            text = (
                "Part of your request could be answered from the data we processed, "
                "but the full chain did not complete. " + text
            )
        return {
            "final_answer": text,
            "chain_addressed": chain_complete,
            "synthesized": False,
        }

    def _passthrough(self, draft_answer: str) -> Dict[str, Any]:
        return {
            "final_answer": draft_answer,
            "answers_question": True,
            "sensitive_risk": "none",
            "issues": [],
            "audited": False,
        }

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
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            try:
                out = json.loads(m.group(0))
                return out if isinstance(out, dict) else None
            except Exception:
                return None
