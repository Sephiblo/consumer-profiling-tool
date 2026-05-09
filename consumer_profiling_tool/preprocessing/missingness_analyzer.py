"""Practical missingness diagnostics."""

from __future__ import annotations

import pandas as pd


def analyze_missingness(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Classify field-level missingness without overclaiming MCAR/MAR/MNAR."""
    result: dict[str, dict[str, object]] = {}
    for column in df.columns:
        rate = float(df[column].isna().mean())
        if rate < 0.05:
            pattern = "low_missingness"
            warning = "Low missingness; standard imputation or exclusion is usually acceptable."
        elif rate < 0.25:
            pattern = "moderate_missingness"
            warning = "Moderate missingness; interpret summaries with some caution."
        elif rate < 0.6:
            pattern = "high_missingness"
            warning = "High missingness; analysis using this field may be biased."
        else:
            pattern = "possibly_structural_missingness"
            warning = "Very high missingness may reflect structural data capture differences."
        result[column] = {"missing_rate": round(rate, 4), "pattern": pattern, "warning": warning}
    return result

