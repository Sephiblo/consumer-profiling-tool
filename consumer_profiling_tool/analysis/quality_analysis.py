"""Data quality and privacy-aware diagnostics."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping, FieldTypeProfile
from core.utils import normalise_name, numeric_from_mixed


def analyze_data_quality(
    df: pd.DataFrame,
    type_profiles: list[FieldTypeProfile],
    mappings: list[ConfirmedFieldMapping] | None = None,
) -> dict[str, object]:
    """Summarize quality issues expected by the v2 specification."""
    profile_by_name = {profile.name: profile for profile in type_profiles}
    duplicate_rows = int(df.duplicated().sum())
    missing = {column: round(float(df[column].isna().mean()), 4) for column in df.columns if df[column].isna().any()}
    constant_columns = [column for column in df.columns if df[column].nunique(dropna=True) <= 1]
    near_constant_columns = [
        column
        for column in df.columns
        if df[column].value_counts(normalize=True, dropna=True).head(1).sum() >= 0.98 and column not in constant_columns
    ]
    high_cardinality = [
        profile.name
        for profile in type_profiles
        if profile.is_categorical and profile.unique_count > 50
    ]
    id_like_columns = [profile.name for profile in type_profiles if profile.is_id_like]
    duplicate_ids = {column: int(df[column].duplicated().sum()) for column in id_like_columns if column in df.columns}

    numeric_summary: dict[str, dict[str, float | None]] = {}
    categorical_summary: dict[str, dict[str, int | float | list[str]]] = {}
    issues: list[str] = []
    impossible_values: list[str] = []
    suspicious_negative_values: list[str] = []
    percentage_scale_issues: list[str] = []
    datetime_issues: list[str] = []

    for column in df.columns:
        profile = profile_by_name[column]
        normalised = normalise_name(column)
        numeric = numeric_from_mixed(df[column])
        if profile.is_numeric:
            numeric_summary[column] = {
                "min": profile.min_value,
                "max": profile.max_value,
                "mean": profile.mean_value,
                "median": profile.median_value,
            }
            if any(word in normalised for word in ["spend", "revenue", "amount", "orders", "count", "sessions"]) and (numeric.dropna() < 0).any():
                suspicious_negative_values.append(column)
            if any(word in normalised for word in ["rate", "ratio", "percent"]) and numeric.dropna().between(0, 1).any() and (numeric.dropna() > 1).any():
                percentage_scale_issues.append(column)
            if "age" in normalised and ((numeric.dropna() < 0).any() or (numeric.dropna() > 120).any()):
                impossible_values.append(column)
        if profile.is_categorical:
            categorical_summary[column] = {
                "unique_count": profile.unique_count,
                "unique_ratio": profile.unique_ratio,
                "mode_values": profile.mode_values,
            }
        if profile.is_datetime_like and profile.inferred_type != "datetime":
            datetime_issues.append(column)

    for column, rate in missing.items():
        if rate > 0.5:
            issues.append(f"{column} has more than 50% missing values.")
    for column in constant_columns:
        issues.append(f"{column} is constant.")
    for column in near_constant_columns:
        issues.append(f"{column} is near-constant.")
    for column in suspicious_negative_values:
        issues.append(f"{column} has suspicious negative values for a spend/count/activity field.")
    for column in percentage_scale_issues:
        issues.append(f"{column} may mix 0-1 and 0-100 percentage scales.")
    for column in impossible_values:
        issues.append(f"{column} contains impossible or implausible values.")
    for column in datetime_issues:
        issues.append(f"{column} may have datetime parsing inconsistencies.")

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_row_count": duplicate_rows,
        "duplicate_id_counts": duplicate_ids,
        "missing_value_summary": missing,
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
        "high_cardinality_categorical_columns": high_cardinality,
        "id_like_columns": id_like_columns,
        "numeric_column_summary": numeric_summary,
        "categorical_column_summary": categorical_summary,
        "impossible_values": impossible_values,
        "suspicious_negative_values": suspicious_negative_values,
        "percentage_scale_inconsistencies": percentage_scale_issues,
        "datetime_parsing_issues": datetime_issues,
        "potential_data_issues": issues,
    }

