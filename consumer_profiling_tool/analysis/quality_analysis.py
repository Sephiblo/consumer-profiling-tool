"""Data quality and privacy-aware diagnostics."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping, FieldTypeProfile
from core.utils import normalise_name, numeric_from_mixed


def _columns_by_role(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping] | None,
    roles: set[str],
) -> list[str]:
    if not mappings:
        return []
    return [mapping.name for mapping in mappings if mapping.role in roles and mapping.name in df.columns]


def _columns_by_keywords(df: pd.DataFrame, keywords: set[str]) -> list[str]:
    matches = []
    for column in df.columns:
        normalised = normalise_name(column)
        tokens = set(normalised.split("_"))
        if any(_keyword_matches(normalised, tokens, keyword) for keyword in keywords):
            matches.append(column)
    return matches


def _keyword_matches(normalised_name: str, tokens: set[str], keyword: str) -> bool:
    keyword_norm = normalise_name(keyword)
    if normalised_name == keyword_norm or keyword_norm in tokens:
        return True
    if len(keyword_norm) <= 3:
        return False
    return keyword_norm in normalised_name


def _candidate_columns(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping] | None,
    roles: set[str],
    keywords: set[str],
) -> list[str]:
    ordered: list[str] = []
    for column in _columns_by_role(df, mappings, roles) + _columns_by_keywords(df, keywords):
        if column not in ordered:
            ordered.append(column)
    return ordered


def _numeric_without_percent_scaling(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _count_pair_violations(
    df: pd.DataFrame,
    left_columns: list[str],
    right_columns: list[str],
    left_condition,
    right_condition,
) -> list[tuple[str, str, int]]:
    violations: list[tuple[str, str, int]] = []
    for left in left_columns:
        left_numeric = numeric_from_mixed(df[left])
        for right in right_columns:
            if left == right:
                continue
            right_numeric = numeric_from_mixed(df[right])
            mask = left_condition(left_numeric) & right_condition(right_numeric)
            count = int(mask.fillna(False).sum())
            if count:
                violations.append((left, right, count))
    return violations


def _business_rule_quality_checks(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping] | None,
) -> list[str]:
    """Find cross-field business inconsistencies that type checks cannot catch."""
    issues: list[str] = []
    frequency_columns = _candidate_columns(
        df,
        mappings,
        {"purchase_frequency"},
        {"frequency", "order_count", "orders", "purchase_count", "transactions"},
    )
    monetary_columns = _candidate_columns(
        df,
        mappings,
        {"monetary_value", "avg_order_value", "clv_or_ltv", "contract_value"},
        {"monetary", "spend", "revenue", "sales", "amount", "order_value", "basket_value"},
    )
    for frequency, monetary, count in _count_pair_violations(
        df,
        frequency_columns,
        monetary_columns,
        lambda series: series == 0,
        lambda series: series > 0,
    ):
        issues.append(f"Business rule: {frequency} is 0 but {monetary} is positive in {count} rows.")

    session_columns = _candidate_columns(
        df,
        mappings,
        {"session_activity", "app_engagement"},
        {"session", "sessions", "login", "app_open"},
    )
    click_columns = _candidate_columns(df, mappings, {"click_activity"}, {"click", "clicks"})
    for session, clicks, count in _count_pair_violations(
        df,
        session_columns,
        click_columns,
        lambda series: series == 0,
        lambda series: series > 0,
    ):
        issues.append(f"Business rule: {session} is 0 but {clicks} is positive in {count} rows.")

    cart_rate_columns = [
        column
        for column in _candidate_columns(df, mappings, {"cart_abandonment"}, {"cart_abandon", "abandonment"})
        if any(token in normalise_name(column) for token in {"rate", "ratio", "percent", "pct"})
    ]
    for column in cart_rate_columns:
        numeric = _numeric_without_percent_scaling(df[column])
        count = int(((numeric < 0) | (numeric > 100)).fillna(False).sum())
        if count:
            issues.append(f"Business rule: {column} is outside the 0-100 percentage range in {count} rows.")

    age_columns = _candidate_columns(df, mappings, {"age"}, {"age", "age_group", "age_band"})
    for column in age_columns:
        numeric = numeric_from_mixed(df[column])
        count = int(((numeric < 0) | (numeric > 120)).fillna(False).sum())
        if count:
            issues.append(f"Business rule: {column} is outside the reasonable 0-120 age range in {count} rows.")

    spend_columns = _candidate_columns(
        df,
        mappings,
        {"monetary_value", "avg_order_value", "clv_or_ltv", "contract_value"},
        {"spend", "revenue", "sales", "amount", "order_value", "basket_value"},
    )
    for column in spend_columns:
        numeric = numeric_from_mixed(df[column])
        count = int((numeric < 0).fillna(False).sum())
        if count:
            issues.append(f"Business rule: {column} contains negative spend/value in {count} rows.")

    return_columns = _candidate_columns(df, mappings, {"return_refund"}, {"return", "returns", "refund", "refunds"})
    order_columns = _candidate_columns(
        df,
        mappings,
        {"purchase_frequency"},
        {"order_count", "orders", "purchase_count", "transactions"},
    )
    for returns, orders, count in _count_pair_violations(
        df,
        return_columns,
        order_columns,
        lambda series: series > 0,
        lambda series: series >= 0,
    ):
        return_numeric = numeric_from_mixed(df[returns])
        order_numeric = numeric_from_mixed(df[orders])
        mask = return_numeric > order_numeric
        count = int(mask.fillna(False).sum())
        if count:
            issues.append(f"Business rule: {returns} is greater than {orders} in {count} rows.")

    return issues


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
    business_rule_issues = _business_rule_quality_checks(df, mappings)

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
            tokens = set(normalised.split("_"))
            if _keyword_matches(normalised, tokens, "age") and ((numeric.dropna() < 0).any() or (numeric.dropna() > 120).any()):
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
    issues.extend(business_rule_issues)

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
        "business_rule_issues": business_rule_issues,
        "potential_data_issues": issues,
    }
