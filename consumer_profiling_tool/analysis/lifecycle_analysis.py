"""Lifecycle analysis from recency, tenure, and subscription signals."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping


def run_lifecycle_analysis(scored_df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    if "risk_score_raw" not in scored_df and "frequency_loyalty_score" not in scored_df:
        return {"available": False, "summary": "No lifecycle, recency, tenure, or subscription fields were detected.", "table": pd.DataFrame()}
    labels = []
    for _, row in scored_df.iterrows():
        risk = row.get("risk_score_raw")
        freq = row.get("frequency_loyalty_score")
        value = row.get("value_score")
        if pd.notna(risk) and risk >= 75:
            label = "at-risk customer"
        elif pd.notna(freq) and freq >= 70 and pd.notna(value) and value >= 60:
            label = "loyal customer"
        elif pd.notna(freq) and freq < 30 and pd.notna(risk) and risk < 40:
            label = "new/promising customer"
        elif pd.notna(risk) and risk >= 50:
            label = "declining customer"
        else:
            label = "active customer"
        labels.append(label)
    table = pd.Series(labels, name="Lifecycle Stage").value_counts().reset_index()
    table.columns = ["Lifecycle Stage", "Count"]
    return {"available": True, "summary": "Lifecycle stages are inferred from available recency/risk/frequency/value scores.", "table": table, "labels": labels}

