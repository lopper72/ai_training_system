"""
Data Context - Load DataFrames from Parquet for the query pipeline.
Schema text for the LLM is derived from the catalog allowlist (see catalog.py).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.query.catalog import PRIORITY_DATASETS, format_catalog_excerpt

logger = logging.getLogger(__name__)


class DataContext:
    """
    Load và quản lý DataFrames từ data/processed/.
    Cung cấp schema mô tả (catalog-driven) cho Agent.
    """

    def __init__(self, data_path: str = "data/processed", companyfn: Optional[str] = None):
        self.data_path = Path(data_path)
        self.companyfn = companyfn
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._source_paths: Dict[str, str] = {}
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return

        ordered_files: List[Tuple[str, Path, str]] = []
        for name in PRIORITY_DATASETS:
            for ext in ["parquet", "csv"]:
                path = self.data_path / f"{name}.{ext}"
                if path.exists():
                    ordered_files.append((name, path, ext))
                    break

        for name, path, ext in ordered_files:
            try:
                if ext == "parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)

                if self.companyfn and "companyfn" in df.columns:
                    original_len = len(df)
                    df = df[df["companyfn"] == self.companyfn].copy()
                    logger.info(
                        f"[DataContext] {name}: filtered by companyfn={self.companyfn} "
                        f"({original_len} → {len(df)} rows)"
                    )

                if "date_trans" in df.columns:
                    df["date_trans"] = pd.to_datetime(df["date_trans"], errors="coerce")

                self._dataframes[name] = df
                self._source_paths[name] = str(path.resolve())
                logger.info(f"[DataContext] Loaded '{name}': {df.shape[0]} rows × {df.shape[1]} cols")

            except Exception as e:
                logger.warning(f"[DataContext] Could not load '{name}': {e}")

        self._loaded = True
        logger.info(f"[DataContext] Total datasets loaded: {list(self._dataframes.keys())}")

    def get_resolved_source_paths(self) -> Dict[str, str]:
        self._load_all()
        return dict(self._source_paths)

    def list_priority_sources_in_agent_order(self) -> List[Tuple[str, str, str, int]]:
        self._load_all()
        out: List[Tuple[str, str, str, int]] = []
        slot = 1
        for name in PRIORITY_DATASETS:
            if name not in self._dataframes:
                continue
            df = self._dataframes[name]
            if df is None or len(df) == 0:
                continue
            path = self._source_paths.get(name, "")
            out.append((f"df{slot}", name, path, len(df)))
            slot += 1
        return out

    def get_dataframes(self) -> Dict[str, pd.DataFrame]:
        self._load_all()
        ordered: Dict[str, pd.DataFrame] = {}
        for name in PRIORITY_DATASETS:
            if name in self._dataframes:
                ordered[name] = self._dataframes[name]
        for name, df in self._dataframes.items():
            if name not in ordered:
                ordered[name] = df
        return ordered

    def get_priority_dataframes(self) -> Dict[str, pd.DataFrame]:
        self._load_all()
        return {name: self._dataframes[name] for name in PRIORITY_DATASETS if name in self._dataframes}

    def get_schema_description(self, dataset_names: Optional[Sequence[str]] = None) -> str:
        """Catalog-based schema plus row/year hints for listed datasets (default: priority loaded)."""
        self._load_all()
        names: List[str] = list(dataset_names) if dataset_names else []
        if not names:
            names = [n for n in PRIORITY_DATASETS if n in self._dataframes]
        lines = [
            "=== AVAILABLE DATA (catalog) ===",
            format_catalog_excerpt(names),
            "",
        ]
        for name in names:
            if name not in self._dataframes:
                continue
            df = self._dataframes[name]
            lines.append(f"📦 {name}: {len(df):,} rows | {len(df.columns)} columns in file")
            if "year" in df.columns:
                years = sorted(df["year"].dropna().unique().tolist())
                lines.append(f"   Sample years in data: {years[:12]}{'...' if len(years) > 12 else ''}")
            if "month" in df.columns:
                months = sorted(df["month"].dropna().unique().tolist())
                lines.append(f"   Months present: {months}")
            lines.append("")
        return "\n".join(lines).strip()
