"""Data cleaning utilities for customer profiling."""

from __future__ import annotations

import pandas as pd

from core.models import FieldTypeProfile
from core.utils import normalise_name, numeric_from_mixed

PERCENT_HINTS = ("rate", "ratio", "percent", "percentage")


def _looks_like_percentage_scale(series: pd.Series, column_name: str) -> bool:
    normalised = normalise_name(column_name)
    if not any(hint in normalised for hint in PERCENT_HINTS):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    return bool(non_null.quantile(0.75) > 1 and non_null.max() <= 100)


def clean_dataframe(
    df: pd.DataFrame,
    type_profiles: list[FieldTypeProfile] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a cleaned copy of the source data plus cleaning metadata."""
    original_shape = df.shape
    cleaned = df.copy()
    cleaned = cleaned.dropna(how="all").dropna(axis=1, how="all")
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    profile_by_name = {profile.name: profile for profile in type_profiles or []}
    coerced_numeric_columns: list[str] = []

    for column in cleaned.columns:
        profile = profile_by_name.get(column)
        numeric_candidate = numeric_from_mixed(cleaned[column])
        numeric_rate = numeric_candidate.notna().mean() if len(cleaned) else 0.0
        should_coerce = bool((profile and profile.is_numeric) or numeric_rate >= 0.8)
        if should_coerce:
            if _looks_like_percentage_scale(numeric_candidate, column):
                numeric_candidate = numeric_candidate / 100.0
            cleaned[column] = numeric_candidate
            median = cleaned[column].median()
            if pd.notna(median):
                cleaned[column] = cleaned[column].fillna(median)
            coerced_numeric_columns.append(column)
        else:
            cleaned[column] = cleaned[column].fillna("Unknown")

    metadata = {
        "original_shape": original_shape,
        "cleaned_shape": cleaned.shape,
        "duplicate_row_count": duplicate_count,
        "coerced_numeric_columns": coerced_numeric_columns,
    }
    return cleaned, metadata

