"""General utility functions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def normalise_name(name: str) -> str:
    """Normalize a column name for keyword and fuzzy matching."""
    lowered = str(name).strip().lower()
    lowered = re.sub(r"[\s\-\/]+", "_", lowered)
    lowered = re.sub(r"[^a-z0-9_]+", "", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered


def contains_keyword(normalised_name: str, keywords: Iterable[str]) -> str | None:
    """Return the first keyword that matches a normalized column name."""
    for keyword in keywords:
        keyword_norm = normalise_name(keyword)
        if normalised_name == keyword_norm or keyword_norm in normalised_name:
            return keyword
    return None


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_from_mixed(series: pd.Series) -> pd.Series:
    """Coerce numeric-looking values, including percent strings, into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    as_string = series.astype("string").str.strip()
    percent_mask = as_string.str.endswith("%", na=False)
    cleaned = (
        as_string.str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    numeric.loc[percent_mask] = numeric.loc[percent_mask] / 100.0
    return numeric


def dataframe_to_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    frame = df.head(max_rows) if max_rows else df
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def to_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def score_column_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return round(float(value), 2)
