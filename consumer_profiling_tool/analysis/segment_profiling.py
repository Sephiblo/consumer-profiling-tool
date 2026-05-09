"""Existing segment ranking and interpretation."""

from __future__ import annotations

import pandas as pd

SEGMENT_WEIGHTS = {
    "value_score": 0.30,
    "frequency_loyalty_score": 0.20,
    "engagement_score": 0.20,
    "conversion_score": 0.15,
    "risk_score_health": 0.15,
}


def _safe_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    return None if pd.isna(value) else round(float(value), 2)


def _available_weights(df: pd.DataFrame) -> dict[str, float]:
    available = {column: weight for column, weight in SEGMENT_WEIGHTS.items() if column in df.columns and df[column].notna().any()}
    total = sum(available.values())
    return {column: weight / total for column, weight in available.items()} if total else {}


def rank_segments(scored_df: pd.DataFrame, segment_column: str = "_original_segment") -> tuple[pd.DataFrame, list[str]]:
    """Rank segments from data-derived patterns, never label names."""
    if segment_column not in scored_df.columns:
        return pd.DataFrame(), ["No existing segment column was available."]
    weights = _available_weights(scored_df)
    rows: list[dict[str, object]] = []
    for segment, group in scored_df.groupby(segment_column, dropna=False):
        composite = None
        if weights:
            composite = sum(group[column].mean() * weight for column, weight in weights.items())
        confidence = "high" if len(weights) >= 4 else "medium" if len(weights) >= 2 else "low"
        rows.append(
            {
                "Segment": str(segment),
                "Count": int(len(group)),
                "Share": round(len(group) / max(len(scored_df), 1), 4),
                "Value": _safe_mean(group, "value_score"),
                "Frequency/Loyalty": _safe_mean(group, "frequency_loyalty_score"),
                "Engagement": _safe_mean(group, "engagement_score"),
                "Conversion": _safe_mean(group, "conversion_score"),
                "Risk": _safe_mean(group, "risk_score_raw"),
                "Composite Score": round(float(composite), 2) if composite is not None and pd.notna(composite) else None,
                "Confidence": confidence,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking["Composite Score"].notna().any():
        ranking = ranking.sort_values("Composite Score", ascending=False).reset_index(drop=True)
        ranking["Inferred Rank"] = [f"Rank {idx + 1}" for idx in range(len(ranking))]
    else:
        ranking = ranking.sort_values("Count", ascending=False).reset_index(drop=True)
        ranking["Inferred Rank"] = "Unranked"
    return ranking, build_segment_interpretations(ranking, weights)


def build_segment_interpretations(ranking: pd.DataFrame, weights: dict[str, float]) -> list[str]:
    if ranking.empty:
        return ["No segment ranking could be generated."]
    if not weights:
        return ["Segment ranking confidence is limited because no monetary, frequency, or response variables were detected."]
    top = ranking.iloc[0]
    messages = [
        (
            f"The system infers that {top['Segment']} is the highest-value segment because it shows the strongest available composite pattern. "
            "This is inferred from data rather than from the label name."
        )
    ]
    if len(weights) < 3:
        messages.append("Segment ranking confidence is limited because only a small number of score groups were available.")
    return messages


def metric_comparison_by_segment(scored_df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    score_columns = [
        column
        for column in [
            "value_score",
            "frequency_loyalty_score",
            "engagement_score",
            "conversion_score",
            "risk_score_raw",
            "risk_score_health",
            "profile_quality_score",
            "b2b_account_fit_score",
        ]
        if column in scored_df.columns and scored_df[column].notna().any()
    ]
    if not score_columns or segment_column not in scored_df.columns:
        return pd.DataFrame()
    return scored_df.groupby(segment_column)[score_columns].mean().round(2).reset_index()

