"""Negative persona candidate analysis."""

from __future__ import annotations

import pandas as pd


def run_negative_persona_analysis(scored_df: pd.DataFrame, segment_column: str | None = None) -> pd.DataFrame:
    if "negative_persona_candidate" not in scored_df:
        return pd.DataFrame()
    candidates = scored_df[scored_df["negative_persona_candidate"]]
    if candidates.empty:
        return pd.DataFrame(
            [
                {
                    "Negative Persona Candidate": "None detected",
                    "Evidence": "No group combines high risk/friction with low value under current thresholds.",
                    "Business Risk": "Monitor as more risk and margin data becomes available.",
                    "Recommended Handling": "Do not create exclusion rules without stronger evidence.",
                }
            ]
        )
    if segment_column and segment_column in candidates.columns:
        rows = []
        for segment, group in candidates.groupby(segment_column):
            rows.append(
                {
                    "Negative Persona Candidate": str(segment),
                    "Evidence": f"{len(group)} records show high risk/friction and low value.",
                    "Business Risk": "May consume campaign or service resources with poor ROI.",
                    "Recommended Handling": "Use low-cost automation, diagnose mismatch, and avoid discriminatory exclusion rules.",
                }
            )
        return pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "Negative Persona Candidate": "High-risk low-value records",
                "Evidence": f"{len(candidates)} records meet the negative-persona threshold.",
                "Business Risk": "May consume marketing or service resources with poor ROI.",
                "Recommended Handling": "Use low-cost automation and diagnose product/service mismatch.",
            }
        ]
    )

