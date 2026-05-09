"""RFM and partial value-lifecycle analysis."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping


def run_rfm_analysis(scored_df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    recency = next((m.name for m in mappings if m.role == "recency" and m.name in scored_df.columns), None)
    frequency = next((m.name for m in mappings if m.role == "purchase_frequency" and m.name in scored_df.columns), None)
    monetary = next((m.name for m in mappings if m.role in {"monetary_value", "avg_order_value", "clv_or_ltv"} and m.name in scored_df.columns), None)
    if not (recency and frequency and monetary):
        return {
            "available": False,
            "summary": "Exact RFM is unavailable; missing recency, frequency, or monetary fields.",
            "table": pd.DataFrame(),
        }
    frame = scored_df[["_customer_profile_id", recency, frequency, monetary]].copy()
    frame["R"] = pd.qcut(-pd.to_numeric(frame[recency], errors="coerce").rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    frame["F"] = pd.qcut(pd.to_numeric(frame[frequency], errors="coerce").rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    frame["M"] = pd.qcut(pd.to_numeric(frame[monetary], errors="coerce").rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    frame["RFM Code"] = frame["R"].astype(str) + frame["F"].astype(str) + frame["M"].astype(str)
    frame["RFM Persona"] = frame.apply(_rfm_persona, axis=1)
    summary = frame["RFM Persona"].value_counts().reset_index()
    summary.columns = ["RFM Persona", "Count"]
    return {"available": True, "summary": "Full RFM analysis was generated.", "table": summary, "customer_rfm": frame}


def _rfm_persona(row: pd.Series) -> str:
    r, f, m = int(row["R"]), int(row["F"]), int(row["M"])
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if f >= 4 and m >= 3:
        return "Loyal Customers"
    if r >= 4 and f >= 3:
        return "Potential Loyalists"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r <= 2 and f >= 4 and m >= 4:
        return "Cannot Lose Them"
    if r <= 2 and m >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Lost"
    if r == 3:
        return "Need Attention"
    return "Promising"

