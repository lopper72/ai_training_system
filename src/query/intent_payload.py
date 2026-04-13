"""
Structured intent units from the router (single- or multi-step).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.query.intent_codes import UNKNOWN_INTENT, canonical_intent


@dataclass
class IntentPayload:
    """One routed sub-question with catalog-scoped datasets."""

    intent_code: str
    target_files: List[str]
    task_type: str = "data_query"
    rationale: str = ""
    top_n: Optional[int] = None
    time: Dict[str, Any] = field(default_factory=lambda: {"month": None, "year": None})
    step_id: int = 0
    sub_query: Optional[str] = None
    # Named filters / hints for planner (e.g. party_desc, stkcode_desc) — token-light.
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intent_code = canonical_intent(self.intent_code)
        self.target_files = [str(x).strip() for x in (self.target_files or []) if str(x).strip()]
        self.task_type = (self.task_type or "data_query").strip().lower()
        if not isinstance(self.params, dict):
            self.params = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> IntentPayload:
        t = raw.get("time")
        if not isinstance(t, dict):
            t = {"month": None, "year": None}
        pr = raw.get("params")
        if not isinstance(pr, dict):
            pr = {}
        return cls(
            intent_code=str(raw.get("intent_code") or raw.get("intent") or UNKNOWN_INTENT),
            target_files=list(raw.get("target_files") or []),
            task_type=str(raw.get("task_type") or "data_query"),
            rationale=str(raw.get("rationale") or ""),
            top_n=raw.get("top_n") if isinstance(raw.get("top_n"), int) else None,
            time=dict(t),
            step_id=int(raw.get("step_id") or 0),
            sub_query=raw.get("sub_query") if isinstance(raw.get("sub_query"), str) else None,
            params=dict(pr),
        )

    @classmethod
    def from_legacy_router_dict(cls, d: Dict[str, Any]) -> IntentPayload:
        """Backward compatibility: old single-intent router dict."""
        t = d.get("time")
        if not isinstance(t, dict):
            t = {"month": None, "year": None}
        return cls(
            intent_code=str(d.get("intent_code") or UNKNOWN_INTENT),
            target_files=list(d.get("target_files") or []),
            task_type=str(d.get("task_type") or "data_query"),
            rationale=str(d.get("rationale") or ""),
            top_n=d.get("top_n") if isinstance(d.get("top_n"), int) else None,
            time=dict(t),
            step_id=0,
            sub_query=None,
            params={},
        )
