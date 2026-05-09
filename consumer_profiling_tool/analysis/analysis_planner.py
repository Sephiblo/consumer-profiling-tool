"""Dynamic analysis planner for the theory-enhanced platform."""

from __future__ import annotations

from collections import Counter

from core.constants import B2B_ROLES, BEHAVIOURAL_ROLES, GEOGRAPHIC_ROLES, PSYCHOGRAPHIC_ROLES
from core.models import AnalysisPlan, ConfirmedFieldMapping, CoverageAssessment


def build_analysis_plan(
    mappings: list[ConfirmedFieldMapping],
    coverage: CoverageAssessment | None = None,
) -> AnalysisPlan:
    """Decide supported, skipped, and proxy analyses from confirmed mappings."""
    roles = Counter(mapping.role for mapping in mappings if mapping.role != "ignore")
    supported: list[str] = []
    skipped: dict[str, str] = {}
    proxy: dict[str, str] = {}
    warnings: list[str] = []

    has_value = roles["monetary_value"] or roles["avg_order_value"] or roles["clv_or_ltv"] or roles["contract_value"]
    has_frequency = roles["purchase_frequency"] or roles["loyalty"] or roles["tenure"]
    has_recency = roles["recency"]

    if has_value and has_frequency and has_recency:
        supported.append("RFM value-lifecycle analysis")
    else:
        skipped["RFM value-lifecycle analysis"] = "requires value, frequency/loyalty, and recency fields"

    if roles["existing_segment"]:
        supported.append("Existing segment profiling and data-based ranking")
    else:
        supported.append("Automatic behavioural clustering")

    if any(roles[role] for role in BEHAVIOURAL_ROLES):
        supported.append("Behavioural, digital, and transactional profiling")
        supported.append("EDA and association diagnostics")
    else:
        skipped["Behavioural profiling"] = "no behavioural, digital, transactional, or risk fields detected"

    if roles["engagement"] or roles["session_activity"] or roles["page_view"] or roles["click_activity"] or roles["email_engagement"]:
        supported.append("Engagement and funnel analysis")
    else:
        skipped["Engagement and funnel analysis"] = "no engagement or funnel fields detected"

    if roles["risk_or_friction"] or roles["recency"] or roles["cart_abandonment"] or roles["churn_indicator"] or roles["return_refund"]:
        supported.append("Risk, friction, and negative persona analysis")
    else:
        skipped["Risk and negative persona analysis"] = "no risk, recency, return, complaint, or friction fields detected"

    if roles["conversion_or_response"] or roles["binary_target"]:
        supported.append("Response/conversion analysis")
    else:
        skipped["Response/conversion analysis"] = "no campaign response, conversion, or binary target detected"

    if roles["binary_target"]:
        supported.append("Response prediction model")
    else:
        skipped["Response prediction model"] = "no confirmed binary target detected"

    if any(roles[role] for role in GEOGRAPHIC_ROLES):
        supported.append("Geographic and environmental profiling")
    else:
        skipped["Geographic profiling"] = "no geographic or environmental fields detected"

    if roles["age"] or roles["gender"] or roles["income"] or roles["education"] or roles["life_stage"]:
        supported.append("Demographic and socioeconomic profiling")
    else:
        skipped["Demographic profiling"] = "no demographic or socioeconomic fields detected"

    if any(roles[role] for role in PSYCHOGRAPHIC_ROLES):
        supported.append("Direct psychographic and motivation analysis")
    elif any(roles[role] for role in BEHAVIOURAL_ROLES):
        proxy["Psychographic/motivation interpretation"] = (
            "No direct psychographic variables were detected; behavioural intent can be discussed only as a proxy."
        )
    else:
        skipped["Psychographic and motivation analysis"] = "no direct psychographic fields or behavioural proxies detected"

    if any(roles[role] for role in B2B_ROLES):
        supported.append("B2B ICP and buying-committee analysis")
    else:
        skipped["B2B ICP analysis"] = "no firmographic or decision-role fields detected"

    if roles["campaign_exposure"] or roles["promotion_usage"] or roles["channel"] or has_value:
        supported.append("ROI and marketing-effectiveness diagnostics")
    else:
        skipped["ROI diagnostics"] = "no cost, value, campaign, promotion, or channel fields detected"

    if coverage and coverage.data_completeness_score < 35:
        warnings.append("Profile coverage is limited; recommendations will rely on sparse evidence.")
    if any(mapping.is_sensitive_candidate for mapping in mappings):
        warnings.append("Sensitive or PII-like fields were detected; avoid discriminatory or privacy-invasive recommendations.")

    return AnalysisPlan(
        supported_analyses=supported,
        skipped_analyses=skipped,
        proxy_analyses=proxy,
        warnings=warnings,
    )

