"""Column type detection for arbitrary customer, account, and contact schemas."""

from __future__ import annotations

import re

import pandas as pd
from dateutil.parser import parse

from core.models import FieldTypeProfile
from core.privacy import detect_sensitive_field
from core.utils import normalise_name, numeric_from_mixed

ID_NAME_HINTS = ("id", "customer_id", "user_id", "client_id", "account_id", "member_id", "contact_id", "company_id", "household_id")
BINARY_VALUE_SETS = [
    {"0", "1"},
    {"0.0", "1.0"},
    {"yes", "no"},
    {"true", "false"},
    {"y", "n"},
    {"responded", "not responded"},
    {"converted", "not converted"},
]
ORDINAL_HINTS = ("tier", "level", "grade", "rank", "stage", "band", "size")
RATE_HINTS = ("rate", "ratio", "percent", "percentage", "pct")
MONEY_HINTS = ("spend", "revenue", "sales", "amount", "value", "profit", "margin", "price", "cost", "clv", "ltv")
LOCATION_HINTS = ("postcode", "postal", "zip", "latitude", "longitude", "city", "country", "region")


def _sample_values(series: pd.Series, n: int = 5) -> list[str]:
    return series.dropna().astype(str).drop_duplicates().head(n).tolist()


def _mode_values(series: pd.Series, n: int = 5) -> list[str]:
    modes = series.dropna().astype(str).mode().head(n).tolist()
    return modes


def _binary_equivalent(series: pd.Series) -> bool:
    values = set(series.dropna().astype(str).str.strip().str.lower().unique().tolist())
    return len(values) == 2 and any(values == candidate for candidate in BINARY_VALUE_SETS)


def _generated_identifier_like(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(200)
    if values.empty:
        return False
    patterns = [
        re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$", re.I),
        re.compile(r"^[a-z]{0,8}[-_]?\d{3,}$", re.I),
        re.compile(r"^[a-z0-9]{10,}$", re.I),
    ]
    matched = values.map(lambda item: any(pattern.match(item.strip()) for pattern in patterns))
    return bool(matched.mean() >= 0.75)


def _datetime_parse_rate(series: pd.Series) -> float:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return 0.0
    parsed = 0
    sample = non_null.head(100)
    for value in sample:
        if len(value) < 5 or value.replace(".", "", 1).isdigit():
            continue
        try:
            parse(value, fuzzy=False)
            parsed += 1
        except (TypeError, ValueError, OverflowError):
            continue
    return parsed / max(len(sample), 1)


def detect_column_type(series: pd.Series) -> FieldTypeProfile:
    """Create an enriched type profile for one column."""
    name = str(series.name)
    normalised = normalise_name(name)
    row_count = len(series)
    non_null = series.dropna()
    missing_rate = float(series.isna().mean()) if row_count else 0.0
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = float(unique_count / max(len(non_null), 1))
    raw_dtype = str(series.dtype)

    numeric_series = numeric_from_mixed(series)
    numeric_rate = float(numeric_series.notna().mean()) if row_count else 0.0
    is_numeric = pd.api.types.is_numeric_dtype(series) or numeric_rate >= 0.8
    is_binary = _binary_equivalent(series)

    name_has_id_hint = normalised == "id" or any(hint in normalised for hint in ID_NAME_HINTS)
    is_id_like = (unique_ratio > 0.9 and name_has_id_hint) or (
        unique_ratio > 0.9 and _generated_identifier_like(series)
    )

    date_name_hint = any(hint in normalised for hint in ["date", "time", "created", "joined", "signup", "renewal"])
    date_rate = _datetime_parse_rate(series)
    is_datetime_like = date_rate >= 0.75 or date_name_hint

    string_lengths = non_null.astype(str).str.len()
    avg_string_length = float(string_lengths.mean()) if not string_lengths.empty else 0.0
    is_text_like = bool(
        not is_numeric and not is_id_like and avg_string_length > 30 and unique_ratio > 0.5
    )

    is_categorical = bool(
        not is_id_like
        and not is_text_like
        and (
            (series.dtype == "object" and unique_ratio < 0.3)
            or (unique_count <= 30 and not is_numeric)
            or (unique_count <= 15 and is_numeric)
        )
    )

    numeric_stats = numeric_series.dropna() if is_numeric else pd.Series(dtype=float)
    min_value = float(numeric_stats.min()) if not numeric_stats.empty else None
    max_value = float(numeric_stats.max()) if not numeric_stats.empty else None
    mean_value = float(numeric_stats.mean()) if not numeric_stats.empty else None
    median_value = float(numeric_stats.median()) if not numeric_stats.empty else None

    is_rate = any(hint in normalised for hint in RATE_HINTS) or (
        is_numeric and min_value is not None and max_value is not None and min_value >= 0 and max_value <= 1
    )
    is_money = any(hint in normalised for hint in MONEY_HINTS)
    is_count = is_numeric and any(hint in normalised for hint in ["count", "orders", "purchases", "sessions", "clicks", "views", "employees"])
    is_ordinal = is_categorical and any(hint in normalised for hint in ORDINAL_HINTS)
    is_location = any(hint in normalised for hint in LOCATION_HINTS)

    sensitive, _, _ = detect_sensitive_field(name, series, row_count)

    if is_id_like:
        inferred_type = "id"
    elif is_binary:
        inferred_type = "binary"
    elif is_datetime_like:
        inferred_type = "datetime"
    elif is_money:
        inferred_type = "currency_money"
    elif is_rate:
        inferred_type = "percentage_rate"
    elif is_count:
        inferred_type = "numeric_count"
    elif is_numeric:
        inferred_type = "numeric_continuous"
    elif is_ordinal:
        inferred_type = "ordinal_categorical"
    elif is_location:
        inferred_type = "postcode_location"
    elif is_text_like:
        inferred_type = "free_text"
    elif is_categorical:
        inferred_type = "categorical"
    else:
        inferred_type = "text"

    return FieldTypeProfile(
        name=name,
        raw_dtype=raw_dtype,
        inferred_type=inferred_type,
        missing_rate=round(missing_rate, 4),
        unique_count=unique_count,
        unique_ratio=round(unique_ratio, 4),
        sample_values=_sample_values(series),
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        median_value=median_value,
        mode_values=_mode_values(series),
        avg_string_length=round(avg_string_length, 2),
        is_numeric=bool(is_numeric),
        is_categorical=bool(is_categorical),
        is_binary=bool(is_binary),
        is_datetime_like=bool(is_datetime_like),
        is_text_like=bool(is_text_like),
        is_id_like=bool(is_id_like),
        is_sensitive_candidate=bool(sensitive),
    )


def detect_field_types(df: pd.DataFrame) -> list[FieldTypeProfile]:
    """Detect enriched type profiles for every column in a dataframe."""
    return [detect_column_type(df[column]) for column in df.columns]


def type_profiles_to_frame(profiles: list[FieldTypeProfile]) -> pd.DataFrame:
    return pd.DataFrame([profile.model_dump() for profile in profiles])
