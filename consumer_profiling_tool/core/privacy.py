"""Privacy and sensitive-field guardrails."""

from __future__ import annotations

import pandas as pd

from core.constants import SENSITIVE_NAME_HINTS
from core.models import FieldTypeProfile, PrivacyFieldFlag, PrivacyScanResult
from core.utils import normalise_name

PRIVACY_NOTICE = (
    "Privacy notice: Some uploaded fields may contain personal or sensitive information. "
    "The app will avoid exposing raw identifiers in the report by default. "
    "This tool does not provide legal advice; review applicable data-protection rules before operational use."
)

DIRECT_PII_HINTS = {
    "email",
    "e_mail",
    "phone",
    "mobile",
    "telephone",
    "tel",
    "address",
    "home_address",
    "street_address",
    "postal_address",
    "postcode",
    "postal_code",
    "zip",
    "latitude",
    "longitude",
}

PERSON_NAME_HINTS = {"name", "full_name", "first_name", "last_name", "surname"}
PERSON_CONTEXT_HINTS = {"customer", "client", "contact", "user", "member", "person", "individual"}
SPECIAL_CATEGORY_HINTS = {
    "gender",
    "sex",
    "ethnicity",
    "religion",
    "political",
    "biometric",
    "medical",
    "diagnosis",
    "disability",
}
BUSINESS_HEALTH_HINTS = {"account_health", "customer_health_score", "health_score", "account_health_score"}
BEHAVIOURAL_BUSINESS_HINTS = {
    "spend",
    "revenue",
    "sales",
    "amount",
    "monetary",
    "frequency",
    "orders",
    "order_count",
    "purchase",
    "recency",
    "session",
    "click",
    "page_view",
    "engagement",
    "open_rate",
    "email_open",
    "cart",
    "abandon",
    "churn",
    "return",
    "refund",
    "complaint",
    "conversion",
    "response",
    "score",
}


def detect_sensitive_field(column_name: str, series: pd.Series | None = None, row_count: int | None = None) -> tuple[bool, str, list[str]]:
    """Flag likely PII/sensitive fields without treating ordinary behaviour metrics as sensitive."""
    normalised = normalise_name(column_name)
    tokens = set(normalised.split("_"))
    reasons: list[str] = []
    risk_level = "low"
    is_business_metric = any(business_hint in normalised for business_hint in BEHAVIOURAL_BUSINESS_HINTS)

    if normalised in BUSINESS_HEALTH_HINTS:
        return False, risk_level, []

    direct_hint = next((hint for hint in DIRECT_PII_HINTS if normalised == hint or hint in normalised), None)
    if direct_hint and not (direct_hint in {"email", "phone", "mobile", "telephone", "tel"} and is_business_metric):
        reasons.append(f"Column name contains direct PII/location hint '{direct_hint}'.")
        risk_level = "medium"

    name_hint = normalised in PERSON_NAME_HINTS or any(hint in normalised for hint in {"full_name", "first_name", "last_name"})
    contextual_name = "name" in tokens and bool(tokens & PERSON_CONTEXT_HINTS)
    if name_hint or contextual_name:
        reasons.append("Column name appears to identify a person.")
        risk_level = "medium"

    special_hint = next(
        (
            hint
            for hint in SPECIAL_CATEGORY_HINTS
            if normalised == hint or hint in tokens or hint in normalised
        ),
        None,
    )
    if special_hint:
        reasons.append(f"Column name contains sensitive demographic/category hint '{special_hint}'.")
        risk_level = "medium"

    if "health" in tokens and normalised not in BUSINESS_HEALTH_HINTS:
        reasons.append("Column name may refer to personal health rather than account health.")
        risk_level = "medium"

    if not reasons:
        for hint in SENSITIVE_NAME_HINTS:
            if hint in normalised and not is_business_metric:
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
