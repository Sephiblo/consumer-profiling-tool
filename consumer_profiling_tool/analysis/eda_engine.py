"""Business-readable EDA and association diagnostics."""

from __future__ import annotations

import pandas as pd

from core.constants import TARGET_ROLES
from core.models import ConfirmedFieldMapping


def run_eda(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], group_column: str | None = None) -> dict[str, object]:
    numeric = df.select_dtypes(include="number")
    categorical = [column for column in df.columns if column not in numeric.columns and df[column].nunique(dropna=True) <= 30]
    summaries = {
        "numeric_summary": numeric.describe().round(2).to_dict() if not numeric.empty else {},
        "categorical_summary": {
            column: df[column].value_counts(dropna=False).head(10).to_dict() for column in categorical[:20]
        },
        "outlier_warnings": [],
        "correlations": {},
        "group_comparisons": {},
        "response_lift": {},
        "notes": ["EDA relationships are suggestive, not causal."],
    }
    for column in numeric.columns:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr and ((numeric[column] < q1 - 3 * iqr) | (numeric[column] > q3 + 3 * iqr)).mean() > 0.02:
            summaries["outlier_warnings"].append(f"{column} has visible outliers.")
    if len(numeric.columns) >= 2:
        summaries["correlations"] = numeric.corr(method="spearman").round(2).to_dict()
    if group_column and group_column in df.columns and not numeric.empty:
        summaries["group_comparisons"] = df.groupby(group_column)[numeric.columns.tolist()].mean().round(2).to_dict()

    target = next((m.name for m in mappings if m.role in TARGET_ROLES | {"conversion_or_response"} and m.name in df.columns), None)
    if target and group_column and group_column in df.columns:
        y = pd.to_numeric(df[target], errors="coerce")
        if y.notna().any():
            overall = y.mean()
            lift = df.assign(_target=y).groupby(group_column)["_target"].mean() / overall if overall else pd.Series(dtype=float)
            summaries["response_lift"] = lift.round(2).to_dict()
    return summaries

