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


def _mean_score(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    return None if pd.isna(value) else float(value)


def _message_strategy_for_persona(persona: str) -> str:
    if "Strategic Account" in persona:
        return "Executive value narrative, ROI proof, renewal/expansion messaging, and account-specific next-best action."
    if "High-Value Loyalist" in persona:
        return "VIP recognition, early access, personalised cross-sell, service assurance, and low-discount loyalty messaging."
    if "High-Intent Non-Converter" in persona:
        return "Friction-removal messaging: trust proof, delivery/return reassurance, basket reminders, and concise conversion nudges."
    if "At-Risk" in persona:
        return "Win-back and recovery messaging focused on service fix, reason-for-lapse learning, and controlled retention offer."
    if "Promotion-Sensitive" in persona:
        return "Margin-aware promotion, threshold incentives, loyalty points, and urgency only where response evidence supports it."
    if "Negative Persona" in persona:
        return "Low-cost diagnostic messaging, product-fit learning, service triage, and no high-cost acquisition pressure."
    if "Low-Value" in persona:
        return "Low-cost education or lifecycle automation, preference capture, and selective reactivation."
    return "General nurture, behaviour-based education, data enrichment prompts, and gradual lifecycle progression."


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
    result["message_strategy"] = result["recommended_profile_type"].map(_message_strategy_for_persona)
    return result


def _segment_key_traits(group: pd.DataFrame) -> str:
    traits = []
    for column, label in [
        ("value_score", "value"),
        ("frequency_loyalty_score", "frequency/loyalty"),
        ("engagement_score", "engagement"),
        ("conversion_score", "conversion"),
        ("risk_score_raw", "risk"),
        ("b2b_account_fit_score", "B2B fit"),
    ]:
        bucket = _bucket(_mean_score(group, column))
        if bucket != "unknown":
            traits.append(f"{bucket} {label}")
    return "; ".join(traits) if traits else "Limited score evidence"


def _layered_segment_strategy(group: pd.DataFrame, dominant_persona: str) -> tuple[str, str]:
    value = _bucket(_mean_score(group, "value_score"))
    engagement = _bucket(_mean_score(group, "engagement_score"))
    conversion = _bucket(_mean_score(group, "conversion_score"))
    risk = _bucket(_mean_score(group, "risk_score_raw"))

    if value == "high" and risk == "high":
        return (
            "Prioritise service recovery, personalised win-back, and retention economics before broad upsell.",
            "Use apology/fix, reason-for-lapse, and controlled high-value recovery messaging.",
        )
    if value == "high":
        return (
            "Protect margin with VIP retention, cross-sell, loyalty recognition, and early-access propositions.",
            "Use premium loyalty, recognition, and relevance-led recommendations rather than blanket discounts.",
        )
    if value == "medium" and engagement == "high" and conversion in {"low", "unknown"}:
        return (
            "Treat as a conversion-development layer: reduce friction, clarify value, and test targeted incentives.",
            "Use reassurance, product education, comparison proof, and light urgency.",
        )
    if value == "medium" and risk == "high":
        return (
            "Use selective reactivation with modest incentives and collect churn/friction reasons.",
            "Use needs-check, preference refresh, and controlled win-back messaging.",
        )
    if value == "medium":
        return (
            "Use nurture-to-grow journeys, bundle education, and next-best-category recommendations.",
            "Use benefit-led education and relevance, with incentives reserved for response-tested customers.",
        )
    if risk == "high":
        return (
            "Keep outreach low-cost, diagnose mismatch, and avoid expensive paid retargeting until fit improves.",
            "Use concise diagnostic or preference-capture messaging rather than repeated sales pressure.",
        )
    if engagement == "high":
        return (
            "Develop intent with product education, social proof, and conversion-friction experiments.",
            "Use browse-based education, trust proof, and gentle next-step prompts.",
        )
    if dominant_persona == "General Nurture Customer":
        return (
            "Use data-enrichment and low-cost lifecycle nurture before assigning high-budget campaigns.",
            "Use preference capture, onboarding, and broad benefit education.",
        )
    return (
        "Use low-cost lifecycle automation and monitor for stronger future value, response, or risk signals.",
        _message_strategy_for_persona(dominant_persona),
    )


def generate_segment_recommendations(scored_df: pd.DataFrame, segment_column: str) -> list[dict[str, str]]:
    """Generate segment/cluster-level strategy with evidence fields."""
    if segment_column not in scored_df.columns:
        return []
    output: list[dict[str, str]] = []
    for segment, group in scored_df.groupby(segment_column, dropna=False):
        persona = group["recommended_profile_type"].mode().iloc[0] if "recommended_profile_type" in group and not group.empty else "Mixed Profile"
        action, message_strategy = _layered_segment_strategy(group, str(persona))
        evidence = group["recommendation_evidence"].mode().iloc[0] if "recommendation_evidence" in group and not group.empty else "Mixed evidence."
        confidence = group["recommendation_confidence"].mode().iloc[0] if "recommendation_confidence" in group and not group.empty else "low"
        output.append(
            {
                "segment": str(segment),
                "size": str(len(group)),
                "dominant_segment": str(segment),
                "dominant_persona": str(persona),
                "key_traits": _segment_key_traits(group),
                "business_value": _bucket(_mean_score(group, "value_score")),
                "risk": _bucket(_mean_score(group, "risk_score_raw")),
                "recommendation": str(action),
                "recommended_action": str(action),
                "message_strategy": message_strategy,
                "evidence_from_data": str(evidence),
                "confidence_level": str(confidence),
                "assumptions": "Segment-level averages represent meaningful group behaviour.",
                "limitation": "Recommendations are suggestive and should be checked against campaign constraints and fairness rules.",
            }
        )
    return output


def build_persona_summary(scored_df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Return the persona table required by the strategy section."""
    if scored_df.empty:
        return pd.DataFrame()
    if not segment_column or segment_column not in scored_df.columns:
        segment_column = "recommended_profile_type"
    rows = []
    for segment, group in scored_df.groupby(segment_column, dropna=False):
        persona = group["recommended_profile_type"].mode().iloc[0] if "recommended_profile_type" in group else str(segment)
        action, message_strategy = _layered_segment_strategy(group, str(persona))
        rows.append(
            {
                "size": int(len(group)),
                "dominant segment": str(segment),
                "key traits": _segment_key_traits(group),
                "business value": _bucket(_mean_score(group, "value_score")),
                "risk": _bucket(_mean_score(group, "risk_score_raw")),
                "recommended action": action,
                "message strategy": message_strategy,
            }
        )
    return pd.DataFrame(rows).sort_values(["business value", "risk", "size"], ascending=[True, False, False])


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
