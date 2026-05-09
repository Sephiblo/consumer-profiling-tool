"""Privacy and sensitive-field guardrails."""

from __future__ import annotations

import re

import pandas as pd

from core.constants import SENSITIVE_NAME_HINTS
from core.models import FieldTypeProfile, PrivacyFieldFlag, PrivacyScanResult
from core.utils import normalise_name

PRIVACY_NOTICE = (
    "Privacy notice: Some uploaded fields may contain personal or sensitive information. "
    "The app will avoid exposing raw identifiers in the report by default. "
    "This tool does not provide legal advice; review applicable data-protection rules before operational use."
)


def detect_sensitive_field(column_name: str, series: pd.Series | None = None, row_count: int | None = None) -> tuple[bool, str, list[str]]:
    """Flag likely personal or sensitive fields by name and simple value patterns."""
    normalised = normalise_name(column_name)
    reasons: list[str] = []
    risk_level = "low"

    for hint in SENSITIVE_NAME_HINTS:
        if hint in normalised:
            reasons.append(f"Column name contains privacy/sensitive hint '{hint}'.")
            risk_level = "medium"
            break

    if series is not None:
        text_sample = series.dropna().astype(str).head(100)
        if not text_sample.empty:
            email_rate = text_sample.str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", regex=True).mean()
            phone_rate = text_sample.str.contains(r"\+?\d[\d\-\s()]{7,}", regex=True).mean()
            if email_rate > 0.3:
                reasons.append("Values look like email addresses.")
                risk_level = "high"
            if phone_rate > 0.3:
                reasons.append("Values look like phone numbers.")
                risk_level = "high"

        if "age" in normalised and row_count is not None and row_count < 100:
            reasons.append("Exact age in a small sample can be identifying.")
            risk_level = "medium"

    return bool(reasons), risk_level, reasons


def scan_privacy(
    df: pd.DataFrame,
    type_profiles: list[FieldTypeProfile],
) -> PrivacyScanResult:
    """Return privacy flags and fields excluded from narrative reports by default."""
    flags: list[PrivacyFieldFlag] = []
    excluded: list[str] = []
    profile_by_name = {profile.name: profile for profile in type_profiles}
    for column in df.columns:
        profile = profile_by_name.get(column)
        flagged, risk, reasons = detect_sensitive_field(column, df[column], len(df))
        if profile and profile.is_id_like:
            flagged = True
            risk = "medium"
            reasons.append("Column looks like an identifier.")
        if flagged:
            flags.append(PrivacyFieldFlag(name=column, risk_level=risk, reasons=reasons))
            if risk in {"medium", "high"}:
                excluded.append(column)
    return PrivacyScanResult(
        privacy_notice=PRIVACY_NOTICE,
        flagged_fields=flags,
        report_excluded_fields=excluded,
    )

