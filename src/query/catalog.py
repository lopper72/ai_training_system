"""
Data catalog & allowlist: only datasets and columns registered here are exposed to the LLM
(soft security via prompts; executor remains a Python REPL — see project docs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """One Parquet-backed dataset the agent may use."""

    name: str
    meaning: str
    allowed_columns: Sequence[str]
    column_types: Mapping[str, str] = field(default_factory=dict)


# Canonical names must match Parquet stems under data/processed (e.g. sales_data.parquet).
CATALOG: Dict[str, DatasetCatalogEntry] = {
    "sales_data": DatasetCatalogEntry(
        name="sales_data",
        meaning="Order line items (SKU-level): quantities and revenue per product per customer.",
        allowed_columns=(
            "date_trans",
            "year",
            "month",
            "amount_local",
            "party_desc",
            "stkcode_desc",
            "brand_desc",
            "stkcate_desc",
            "qnty_total",
            "companyfn",
            "tag_void_yn",
            "tag_deleted_yn",
            "tag_table_usage",
            "tag_closed02_yn",
            "transaction_type_name",
        ),
        column_types={
            "date_trans": "datetime64",
            "year": "int",
            "month": "int",
            "amount_local": "float",
            "party_desc": "string",
            "stkcode_desc": "string",
            "qnty_total": "float",
        },
    ),
    "sales_main": DatasetCatalogEntry(
        name="sales_main",
        meaning="Invoice headers (source table scm_sal_main): one row per invoice; use for distinct customers (group by party_desc), revenue totals, top customers, churn.",
        allowed_columns=(
            "date_trans",
            "year",
            "month",
            "amount_local",
            "party_desc",
            "transaction_type_name",
            "companyfn",
            "tag_void_yn",
            "tag_deleted_yn",
            "tag_table_usage",
            "tag_closed02_yn",
        ),
        column_types={
            "date_trans": "datetime64",
            "year": "int",
            "month": "int",
            "amount_local": "float",
            "party_desc": "string",
        },
    ),
}

# Load order when multiple frames are active (matches historical agent behavior).
PRIORITY_DATASETS: List[str] = ["sales_data", "sales_main"]


def catalog_dataset_names() -> List[str]:
    return list(CATALOG.keys())


def format_catalog_excerpt(dataset_names: Sequence[str], max_columns: int = 24) -> str:
    """Compact allowlisted schema for prompts (only listed datasets)."""
    lines: List[str] = ["=== CATALOG (allowed sources & columns) ===", ""]
    for name in dataset_names:
        entry = CATALOG.get(name)
        if not entry:
            continue
        cols = list(entry.allowed_columns)[:max_columns]
        typed = [f"{c} ({entry.column_types.get(c, 'unknown')})" for c in cols]
        lines.append(f"• {name}: {entry.meaning}")
        lines.append(f"  Allowed columns: {', '.join(typed)}")
        lines.append("")
    lines.append("Do not reference columns outside the allowlist above.")
    return "\n".join(lines).strip()


def is_catalog_dataset(name: str) -> bool:
    """True if name is a registered dataset key (Parquet security boundary)."""
    n = (name or "").strip()
    return bool(n) and n in CATALOG


def filter_to_catalog_datasets(names: Sequence[str]) -> List[str]:
    """Keep only names present in CATALOG (does not check loaded)."""
    out: List[str] = []
    for raw in names:
        n = str(raw).strip()
        if n and n in CATALOG and n not in out:
            out.append(n)
    return out


def validate_router_targets(targets: Sequence[str], loaded: Set[str]) -> List[str]:
    """Keep only targets that exist in catalog and are loaded; preserve PRIORITY_DATASETS order."""
    want = {t.strip() for t in targets if t and t.strip() in CATALOG}
    ordered: List[str] = []
    for name in PRIORITY_DATASETS:
        if name in want and name in loaded:
            ordered.append(name)
    for name in targets:
        n = name.strip()
        if n in want and n in loaded and n not in ordered:
            ordered.append(n)
    return ordered
