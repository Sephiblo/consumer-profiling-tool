"""Robust normalisation helpers."""

from __future__ import annotations

import pandas as pd


def robust_minmax(series: pd.Series) -> pd.Series:
    """Robust min-max scaling with 1st and 99th percentile winsorisation."""
    numeric = pd.to_numeric(series, errors="coerce")
    lower = numeric.quantile(0.01)
    upper = numeric.quantile(0.99)
    if pd.isna(lower) or pd.isna(upper):
        return pd.Series(0.5, index=series.index)
    clipped = numeric.clip(lower, upper)
    if upper == lower:
        return pd.Series(0.5, index=series.index)
    return (clipped - lower) / (upper - lower)


def apply_polarity(normalised: pd.Series, polarity: str) -> pd.Series:
    """Apply metric polarity to a normalized score."""
    if polarity == "negative":
        return 1 - normalised
    return normalised

