"""Demographic and socioeconomic profiling."""

from __future__ import annotations

import pandas as pd

from core.constants import DEMOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping


def run_demographic_analysis(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping], segment_column: str | None = None) -> dict[str, object]:
    fields = [m.name for m in mappings if m.role in DEMOGRAPHIC_ROLES and m.name in df.columns]
    if not fields:
        return {"available": False, "summary": "No demographic or socioeconomic fields were detected.", "tables": {}}
    tables = {}
    for field in fields:
        tables[field] = df[field].value_counts(dropna=False).head(15).reset_index().rename(columns={"index": field, field: "Count"})
    summary = "Demographic fields are available for descriptive analysis only; they are not treated as better/worse scores."
    return {"available": True, "summary": summary, "tables": tables}

