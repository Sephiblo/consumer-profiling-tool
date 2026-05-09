"""B2B ICP and buying-committee analysis."""

from __future__ import annotations

import pandas as pd

from core.constants import B2B_ROLES
from core.models import ConfirmedFieldMapping, ModeDetectionResult


def run_b2b_icp_analysis(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], mode_result: ModeDetectionResult) -> dict[str, object]:
    fields = [m.name for m in mappings if m.role in B2B_ROLES and m.name in df.columns]
    if not fields:
        return {"available": False, "summary": "No B2B firmographic or decision-role fields were detected.", "tables": {}}
    tables = {}
    for field in fields:
        if pd.api.types.is_numeric_dtype(df[field]):
            tables[field] = pd.DataFrame({"Metric": ["mean", "median"], "Value": [df[field].mean(), df[field].median()]})
        else:
            tables[field] = df[field].value_counts(dropna=False).head(15).reset_index()
    return {
        "available": True,
        "summary": f"{mode_result.mode} mode supports ICP and buying-committee analysis where fields are available.",
        "tables": tables,
        "messaging_implications": [
            "Decision makers need ROI and risk-reduction messaging.",
            "Technical influencers need proof, documentation, and integration support.",
            "Procurement/gatekeeper roles need compliance, cost, and vendor-risk evidence.",
        ],
    }

