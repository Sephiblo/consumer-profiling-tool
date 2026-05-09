"""Persona and business recommendation generation with evidence."""

from __future__ import annotations

import pandas as pd


def _bucket(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    if value >= 70:
        return "high"
    if value >= 40:
        return "medium"
    return "low"


def determine_profile_type(row: pd.Series) -> tuple[str, str, str, str, str, str]:
    """Return persona, action, evidence, confidence, assumptions, limitation."""
    value = _bucket(row.get("value_score"))
    frequency = _bucket(row.get("frequency_loyalty_score"))
    engagement = _bucket(row.get("engagement_score"))
    conversion = _bucket(row.get("conversion_score"))
    risk = _bucket(row.get("risk_score_raw"))
    b2b_fit = _bucket(row.get("b2b_account_fit_score"))
    negative = bool(row.get("negative_persona_candidate", False))

    if negative:
        return (
            "Negative Persona Candidate",
            "Diagnose service/product mismatch, use low-cost automation, and avoid high-cost acquisition until the issue is understood.",
            "High risk/friction with low value or weak value evidence.",
            "medium",
            "Risk fields are valid proxies for friction or poor ROI.",
            "Do not use unfair or discriminatory exclusion rules.",
        )
    if b2b_fit == "high" and value in {"high", "medium"}:
        return (
            "B2B Strategic Account",
            "Use account-based marketing, executive outreach, and renewal/expansion planning.",
            "Strong B2B account-fit score and value signal.",
            "medium",
            "Firmographic fields are good proxies for account fit.",
            "Needs qualitative sales context before operational prioritisation.",
        )
    if value == "high" and frequency in {"high", "medium"} and engagement in {"high", "medium"} and risk != "high":
        return (
            "High-Value Loyalist",
            "VIP retention, loyalty benefits, early access, personalised cross-sell, and avoid over-discounting.",
            "High value, frequency/loyalty, engagement, and low or medium risk.",
            "high",
            "Composite scores reflect recent customer behaviour.",
            "Does not explain motivations unless psychographic fields exist.",
        )
    if engagement == "high" and conversion in {"low", "unknown"} and risk == "high":
        return (
            "High-Intent Non-Converter",
            "Simplify checkout, use basket reminders, trust signals, limited-time nudges, and delivery/return reassurance.",
            "High engagement or product interest with low conversion and high friction.",
            "medium",
            "Engagement is interpreted as intent proxy.",
            "Intent is behavioural proxy, not direct motivation.",
        )
    if value == "high" and risk == "high":
        return (
            "At-Risk Valuable Customer",
            "Run win-back campaign, personalised recovery offer, service follow-up, and reason-for-lapse survey.",
            "High value but elevated recency/churn/friction risk.",
            "high",
            "Risk score captures lapse or friction signals.",
            "Cannot confirm reason for lapse without survey/support notes.",
        )
    if conversion == "high" and value in {"medium", "unknown"}:
        return (
            "Promotion-Sensitive Responder",
            "Use controlled discounting, loyalty points, threshold incentives, and margin-aware targeting.",
            "Strong campaign/promotion response with medium or unclear value.",
            "medium",
            "Promotion and conversion fields are meaningful response signals.",
            "Margin or campaign cost data is needed to validate ROI.",
        )
    if value == "low" and engagement == "low" and conversion in {"low", "unknown"}:
        return (
            "Low-Value Low-Engagement Customer",
            "Use low-cost lifecycle automation and avoid expensive paid retargeting.",
            "Low value, low engagement, and weak response evidence.",
            "medium",
            "Score thresholds are suitable for a first-pass triage.",
            "Future behaviour may change; avoid permanent exclusion.",
        )
    return (
        "General Nurture Customer",
        "Use lifecycle messaging, monitor future behaviour, and improve data coverage for sharper profiling.",
        "No dominant high-confidence persona pattern emerged.",
        "low",
        "Available fields provide partial customer behaviour evidence.",
        "Additional value, response, and motivation fields would improve confidence.",
    )


def add_customer_recommendations(scored_df: pd.DataFrame) -> pd.DataFrame:
    result = scored_df.copy()
    recommendations = result.apply(determine_profile_type, axis=1)
    result["recommended_profile_type"] = [item[0] for item in recommendations]
    result["recommended_action"] = [item[1] for item in recommendations]
    result["recommendation_evidence"] = [item[2] for item in recommendations]
    result["recommendation_confidence"] = [item[3] for item in recommendations]
    result["recommendation_assumptions"] = [item[4] for item in recommendations]
    result["recommendation_limitation"] = [item[5] for item in recommendations]
    return result


def generate_segment_recommendations(scored_df: pd.DataFrame, segment_column: str) -> list[dict[str, str]]:
    """Generate segment/cluster-level strategy with evidence fields."""
    if segment_column not in scored_df.columns:
        return []
    output: list[dict[str, str]] = []
    for segment, group in scored_df.groupby(segment_column, dropna=False):
        persona = group["recommended_profile_type"].mode().iloc[0] if "recommended_profile_type" in group and not group.empty else "Mixed Profile"
        action = group["recommended_action"].mode().iloc[0] if "recommended_action" in group and not group.empty else "Review behaviour."
        evidence = group["recommendation_evidence"].mode().iloc[0] if "recommendation_evidence" in group and not group.empty else "Mixed evidence."
        confidence = group["recommendation_confidence"].mode().iloc[0] if "recommendation_confidence" in group and not group.empty else "low"
        output.append(
            {
                "segment": str(segment),
                "dominant_persona": str(persona),
                "recommendation": str(action),
                "evidence_from_data": str(evidence),
                "confidence_level": str(confidence),
                "assumptions": "Segment-level averages represent meaningful group behaviour.",
                "limitation": "Recommendations are suggestive and should be checked against campaign constraints and fairness rules.",
            }
        )
    return output


def data_collection_recommendations(missing_pillars: list[str]) -> list[str]:
    suggestions = []
    if "Demographic and socioeconomic" in missing_pillars:
        suggestions.append("Collect age bands, life stage, and income bands where lawful and relevant.")
    if "Geographic and environmental" in missing_pillars:
        suggestions.append("Collect region, city, delivery zone, or store catchment area.")
    if "Psychographic and motivational" in missing_pillars:
        suggestions.append("Collect satisfaction/NPS, benefit-sought, price sensitivity, and reason-for-abandonment survey fields.")
    if "B2B firmographic and decision-role" in missing_pillars:
        suggestions.append("Collect industry, company size, contract value, account health, and buying-committee role.")
    if not suggestions:
        suggestions.append("Improve longitudinal tracking, campaign costs, margin, and qualitative feedback for deeper ROI diagnostics.")
    return suggestions

