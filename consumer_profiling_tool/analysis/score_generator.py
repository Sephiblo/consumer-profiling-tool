"""Dynamic scoring engine for v2 profile dimensions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.recommendation_engine import add_customer_recommendations
from core.constants import IDENTITY_ROLES, SCORE_GROUPS
from core.models import ConfirmedFieldMapping
from preprocessing.normalizer import apply_polarity, robust_minmax


RICHNESS_GROUPS = {
    "identity_completeness_score",
    "demographic_richness_score",
    "geographic_richness_score",
    "psychographic_richness_score",
}

SCORE_LABELS = {
    "identity_completeness_score": "Identity completeness",
    "demographic_richness_score": "Demographic richness",
    "geographic_richness_score": "Geographic richness",
    "psychographic_richness_score": "Psychographic richness",
    "value_score": "Customer/account value",
    "frequency_loyalty_score": "Frequency and loyalty",
    "engagement_score": "Engagement",
    "conversion_score": "Conversion/response",
    "risk_score": "Risk/friction",
    "b2b_account_fit_score": "B2B account fit",
    "profile_quality_score": "Overall profile quality",
}


def _fields_for_group(mappings: list[ConfirmedFieldMapping], group_name: str, df: pd.DataFrame) -> list[ConfirmedFieldMapping]:
    roles = set(SCORE_GROUPS[group_name])
    return [mapping for mapping in mappings if mapping.role in roles and mapping.name in df.columns and mapping.role != "ignore"]


def _score_numeric_field(series: pd.Series, polarity: str) -> pd.Series | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    filled = numeric.fillna(numeric.median())
    return apply_polarity(robust_minmax(filled), polarity)


def calculate_score_group(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], group_name: str) -> pd.Series | None:
    """Calculate a 0-100 score group, or None if the group is unavailable."""
    if group_name not in SCORE_GROUPS:
        raise ValueError(f"Unknown score group: {group_name}")
    group_mappings = _fields_for_group(mappings, group_name, df)
    if not group_mappings:
        return None

    if group_name in RICHNESS_GROUPS:
        completeness = pd.concat([df[mapping.name].notna().astype(float) for mapping in group_mappings], axis=1).mean(axis=1)
        return (completeness * 100).round(2)

    scores: list[pd.Series] = []
    for mapping in group_mappings:
        score = _score_numeric_field(df[mapping.name], mapping.polarity)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return (pd.concat(scores, axis=1).mean(axis=1) * 100).round(2)


def calculate_risk_scores(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> tuple[pd.Series | None, pd.Series | None]:
    risk_roles = set(SCORE_GROUPS["risk_score"])
    raw_scores: list[pd.Series] = []
    health_scores: list[pd.Series] = []
    for mapping in mappings:
        if mapping.role not in risk_roles or mapping.name not in df.columns:
            continue
        numeric = pd.to_numeric(df[mapping.name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        normalised = robust_minmax(numeric.fillna(numeric.median()))
        raw = normalised if mapping.polarity == "negative" else 1 - normalised
        raw_scores.append(raw)
        health_scores.append(1 - raw)
    if not raw_scores:
        return None, None
    return (
        (pd.concat(raw_scores, axis=1).mean(axis=1) * 100).round(2),
        (pd.concat(health_scores, axis=1).mean(axis=1) * 100).round(2),
    )


def build_scoring_methodology(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    """Explain formulas, field weights, and polarity treatment for generated scores."""
    methodology: list[dict[str, object]] = []
    for group_name in SCORE_GROUPS:
        group_mappings = _fields_for_group(mappings, group_name, df)
        if not group_mappings:
            continue
        weight = round(1 / len(group_mappings), 4)
        fields = [
            {
                "field": mapping.name,
                "role": mapping.role,
                "polarity": mapping.polarity,
                "weight": weight,
                "treatment": _field_treatment(group_name, mapping),
            }
            for mapping in group_mappings
        ]
        if group_name in RICHNESS_GROUPS:
            formula = "mean(non_missing(field_i)) * 100"
        elif group_name == "risk_score":
            formula = "mean(risk_normalized(field_i)) * 100; risk_score_health = 100 - risk_score_raw"
        else:
            formula = "mean(polarity_adjusted_robust_minmax(field_i)) * 100"
        methodology.append(
            {
                "score": group_name,
                "label": SCORE_LABELS.get(group_name, group_name),
                "formula": formula,
                "fields": fields,
            }
        )
    return {
        "normalisation": (
            "Numeric fields are coerced to numbers, missing values are median-imputed, "
            "and values are robust-minmax scaled using the 1st and 99th percentiles to reduce outlier impact."
        ),
        "polarity": (
            "Positive polarity means higher raw values raise the score. Negative polarity means higher raw values are inverted "
            "because they represent risk, friction, recency/lapse, refunds, complaints, churn, or abandonment."
        ),
        "weights": "Within each score group, available fields currently use equal weights.",
        "score_groups": methodology,
    }


def _field_treatment(group_name: str, mapping: ConfirmedFieldMapping) -> str:
    if group_name in RICHNESS_GROUPS:
        return "Completeness indicator: present = 1, missing = 0."
    if group_name == "risk_score":
        if mapping.polarity == "negative":
            return "Higher values increase raw risk and reduce risk health."
        return "Higher values are treated as protective and inverted for raw risk."
    if mapping.polarity == "negative":
        return "Robust-minmax scaled, then inverted so lower-risk values score higher."
    if mapping.polarity == "positive":
        return "Robust-minmax scaled so higher values score higher."
    return "Robust-minmax scaled with neutral/unknown polarity; review mapping if this should be directional."


def generate_customer_scores(
    df: pd.DataFrame,
    mappings: list[ConfirmedFieldMapping],
    cluster_labels: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Append dynamic profile scores and recommended actions to customers."""
    scored = df.copy()
    id_field = next((mapping.name for mapping in mappings if mapping.role in IDENTITY_ROLES and mapping.name in scored.columns), None)
    segment_field = next((mapping.name for mapping in mappings if mapping.role == "existing_segment" and mapping.name in scored.columns), None)
    scored["_customer_profile_id"] = scored[id_field].astype(str) if id_field else [f"record_{idx + 1}" for idx in range(len(scored))]
    if segment_field:
        scored["_original_segment"] = scored[segment_field].astype(str)
    if cluster_labels is not None:
        scored["_generated_cluster"] = pd.Series(cluster_labels, index=scored.index).astype(str)

    for group_name in SCORE_GROUPS:
        if group_name == "risk_score":
            continue
        score = calculate_score_group(scored, mappings, group_name)
        scored[group_name] = score if score is not None else np.nan

    raw_risk, health_risk = calculate_risk_scores(scored, mappings)
    scored["risk_score_raw"] = raw_risk if raw_risk is not None else np.nan
    scored["risk_score_health"] = health_risk if health_risk is not None else np.nan
    scored["negative_persona_candidate"] = (
        (pd.to_numeric(scored["risk_score_raw"], errors="coerce") >= 70)
        & (pd.to_numeric(scored.get("value_score", pd.Series(np.nan, index=scored.index)), errors="coerce") < 40)
    )
    return add_customer_recommendations(scored)
