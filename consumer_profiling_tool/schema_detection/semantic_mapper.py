"""Semantic business-role mapping for v2 theory-enhanced profiling."""

from __future__ import annotations

import pandas as pd

from core.constants import (
    B2B_ROLES,
    BEHAVIOURAL_ROLES,
    FIELD_ROLES,
    ROLE_KEYWORDS,
    TARGET_ROLES,
)
from core.models import ConfirmedFieldMapping, FieldSemanticProfile, FieldTypeProfile
from core.utils import contains_keyword, normalise_name
from schema_detection.polarity_detector import detect_polarity

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


ROLE_PRIORITY = [
    "customer_id",
    "household_id",
    "account_id",
    "company_id",
    "existing_segment",
    "avg_order_value",
    "clv_or_ltv",
    "profitability",
    "contract_value",
    "monetary_value",
    "purchase_frequency",
    "recency",
    "cart_abandonment",
    "churn_indicator",
    "return_refund",
    "complaint_support",
    "cancellation",
    "unsubscribe",
    "failed_payment",
    "negative_feedback",
    "conversion_or_response",
    "promotion_usage",
    "campaign_exposure",
    "email_engagement",
    "session_activity",
    "page_view",
    "click_activity",
    "social_engagement",
    "app_engagement",
    "engagement",
    "product_interest",
    "category_preference",
    "channel",
    "device_platform",
    "age",
    "gender",
    "income",
    "education",
    "occupation",
    "marital_status",
    "household_structure",
    "life_stage",
    "socioeconomic_status",
    "country",
    "region",
    "city",
    "postcode",
    "geographic_area",
    "urban_rural",
    "language",
    "climate_or_environment",
    "price_sensitivity",
    "brand_affinity",
    "innovation_adoption",
    "satisfaction",
    "nps",
    "interest",
    "lifestyle",
    "values_attitudes",
    "motivation",
    "pain_point",
    "benefit_sought",
    "industry",
    "company_size",
    "annual_revenue",
    "employee_count",
    "department",
    "job_title",
    "seniority",
    "decision_role",
    "technology_stack",
    "renewal_date",
    "account_health",
    "date_or_time",
]


def _keyword_role_score(column_name: str) -> tuple[str | None, float, str | None]:
    normalised = normalise_name(column_name)
    best_role: str | None = None
    best_score = 0.0
    best_keyword: str | None = None

    for role in ROLE_PRIORITY:
        for keyword in ROLE_KEYWORDS.get(role, []):
            keyword_norm = normalise_name(keyword)
            if normalised == keyword_norm:
                return role, 0.96, keyword
            if keyword_norm in normalised:
                score = 0.88
                if score > best_score:
                    best_role, best_score, best_keyword = role, score, keyword
            elif fuzz:
                ratio = fuzz.partial_ratio(normalised, keyword_norm) / 100
                if ratio >= 0.9 and ratio * 0.78 > best_score:
                    best_role, best_score, best_keyword = role, ratio * 0.78, keyword
    return best_role, best_score, best_keyword


def _adjust_role_for_type(profile: FieldTypeProfile, role: str | None, score: float) -> tuple[str, float, list[str]]:
    reasons: list[str] = []

    if profile.is_id_like and (role is None or role in {"customer_id", "household_id", "account_id", "company_id"}):
        name = normalise_name(profile.name)
        if "company" in name:
            return "company_id", max(score, 0.9), ["Values look like company/account identifiers."]
        if "account" in name:
            return "account_id", max(score, 0.9), ["Values look like account identifiers."]
        if "household" in name:
            return "household_id", max(score, 0.9), ["Values look like household identifiers."]
        return "customer_id", max(score, 0.9), ["Values look like customer/contact identifiers."]

    if role is None:
        if profile.is_datetime_like:
            return "date_or_time", 0.82, ["Values or column name look date/time-like."]
        if profile.is_text_like:
            return "free_text", 0.8, ["Long high-cardinality text detected."]
        if profile.is_binary:
            return "binary_target", 0.45, ["Binary field with no stronger business keyword."]
        return "unknown", 0.3, ["No confident semantic keyword or statistical pattern was found."]

    if role == "conversion_or_response" and profile.is_binary:
        return "binary_target", max(score, 0.88), ["Binary conversion/response field can be used as a response target."]

    if role in {"annual_revenue", "contract_value"} and profile.is_numeric:
        return role, max(score, 0.9), [f"Numeric B2B value-like field mapped to {role}."]

    if role in BEHAVIOURAL_ROLES and not (profile.is_numeric or profile.is_binary or profile.is_categorical):
        reasons.append(f"Role '{role}' usually expects numeric, binary, or categorical business data; confidence reduced.")
        score = min(score, 0.65)

    if role in B2B_ROLES:
        reasons.append("Column is a B2B firmographic or buying-committee signal.")

    if role == "date_or_time" and not profile.is_datetime_like:
        score = min(score, 0.7)
        reasons.append("Name suggests date/time but values were not strongly parseable.")

    return role, score, reasons


def map_field_semantics(
    type_profiles: list[FieldTypeProfile],
    df: pd.DataFrame | None = None,
) -> list[FieldSemanticProfile]:
    """Map type profiles to business roles, confidence, polarity, and proxy flags."""
    profiles: list[FieldSemanticProfile] = []
    for profile in type_profiles:
        role, score, keyword = _keyword_role_score(profile.name)
        adjusted_role, adjusted_score, type_reasons = _adjust_role_for_type(profile, role, score)
        reasons = list(type_reasons)
        if keyword:
            reasons.insert(0, f"Column name matched keyword '{keyword}' for role '{role}'.")
        if profile.inferred_type in {"percentage_rate", "currency_money", "numeric_count"}:
            reasons.append(f"Type detector classified field as {profile.inferred_type}.")

        polarity, polarity_confidence, polarity_reasons = detect_polarity(profile.name, adjusted_role)
        reasons.extend(polarity_reasons)

        is_proxy = bool(adjusted_role in BEHAVIOURAL_ROLES and adjusted_role not in TARGET_ROLES)
        profiles.append(
            FieldSemanticProfile(
                name=profile.name,
                inferred_type=profile.inferred_type,
                suggested_role=adjusted_role if adjusted_role in FIELD_ROLES else "unknown",
                role_confidence=round(float(min(max(adjusted_score, 0.0), 0.99)), 2),
                suggested_polarity=polarity,
                polarity_confidence=round(float(polarity_confidence), 2),
                is_sensitive_candidate=profile.is_sensitive_candidate,
                is_proxy_inference=is_proxy,
                reasons=reasons,
            )
        )
    return profiles


def semantic_profiles_to_confirmed(profiles: list[FieldSemanticProfile]) -> list[ConfirmedFieldMapping]:
    return [
        ConfirmedFieldMapping(
            name=profile.name,
            inferred_type=profile.inferred_type,
            role=profile.suggested_role,
            role_confidence=profile.role_confidence,
            polarity=profile.suggested_polarity,
            polarity_confidence=profile.polarity_confidence,
            is_sensitive_candidate=profile.is_sensitive_candidate,
            is_proxy_inference=profile.is_proxy_inference,
            reasons=profile.reasons,
        )
        for profile in profiles
    ]


def mappings_by_name(mappings: list[ConfirmedFieldMapping]) -> dict[str, ConfirmedFieldMapping]:
    return {mapping.name: mapping for mapping in mappings}
