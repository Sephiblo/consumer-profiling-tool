"""Geographic and environmental profiling."""

from __future__ import annotations

import pandas as pd

from core.constants import GEOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping


def run_geographic_analysis(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    fields = [m.name for m in mappings if m.role in GEOGRAPHIC_ROLES and m.name in df.columns]
    if not fields:
        return {"available": False, "summary": "No geographic or environmental fields were detected.", "tables": {}}
    tables = {field: df[field].value_counts(dropna=False).head(20).reset_index() for field in fields}
    return {
        "available": True,
        "summary": "Geographic fields support regional concentration, coverage, and market white-space analysis.",
        "tables": tables,
    }

