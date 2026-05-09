"""ROI and marketing-effectiveness diagnostics."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping


def run_roi_analysis(df: pd.DataFrame, scored_df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], segment_column: str | None = None) -> dict[str, object]:
    value_fields = [m.name for m in mappings if m.role in {"monetary_value", "profitability", "contract_value", "clv_or_ltv"} and m.name in df.columns]
    response_fields = [m.name for m in mappings if m.role in {"conversion_or_response", "binary_target"} and m.name in df.columns]
    channel_fields = [m.name for m in mappings if m.role == "channel" and m.name in df.columns]
    cost_fields = [m.name for m in mappings if "cost" in m.name.lower() and m.name in df.columns]

    rows = []
    revenue = pd.to_numeric(df[value_fields[0]], errors="coerce") if value_fields else pd.Series(dtype=float)
    response = pd.to_numeric(df[response_fields[0]], errors="coerce") if response_fields else pd.Series(dtype=float)
    if value_fields:
        rows.append({"Metric": "average_value_per_customer", "Value": round(float(revenue.mean()), 2)})
        rows.append({"Metric": "revenue_share_basis", "Value": value_fields[0]})
    if response_fields and response.notna().any():
        rows.append({"Metric": "response_rate", "Value": round(float(response.mean()), 4)})
    if value_fields and cost_fields:
        cost = pd.to_numeric(df[cost_fields[0]], errors="coerce")
        roi = ((revenue - cost) / cost.replace(0, pd.NA)).mean()
        rows.append({"Metric": "ROI proxy", "Value": round(float(roi), 4) if pd.notna(roi) else None})
        summary = "Cost-based ROI proxy was calculated from detected value and cost fields."
    else:
        summary = (
            "Cost-based ROI cannot be calculated because campaign cost or margin fields were not detected. "
            "The tool reports response lift and revenue contribution as proxy effectiveness indicators."
        )
    if segment_column and segment_column in scored_df.columns and response_fields and response.notna().any():
        tmp = scored_df[[segment_column]].copy()
        tmp["_response"] = response
        lift = tmp.groupby(segment_column)["_response"].mean() / response.mean() if response.mean() else pd.Series(dtype=float)
        for segment, value in lift.items():
            rows.append({"Metric": f"segment_lift_{segment}", "Value": round(float(value), 3)})
    return {"available": bool(rows), "summary": summary, "table": pd.DataFrame(rows), "channel_fields": channel_fields}

