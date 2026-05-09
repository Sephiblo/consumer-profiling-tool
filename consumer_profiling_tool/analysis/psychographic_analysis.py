"""Psychographic and motivational analysis."""

from __future__ import annotations

import pandas as pd

from core.constants import PSYCHOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping


def run_psychographic_analysis(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    fields = [m.name for m in mappings if m.role in PSYCHOGRAPHIC_ROLES and m.name in df.columns]
    if not fields:
        return {
            "available": False,
            "summary": (
                "No direct psychographic variables were detected. Behavioural intent from clicks, sessions, wishlist, or cart abandonment "
                "can be used only as a proxy and should not be interpreted as direct evidence of values, beliefs, or motivations."
            ),
            "tables": {},
        }
    tables = {field: df[field].value_counts(dropna=False).head(15).reset_index() for field in fields if not pd.api.types.is_numeric_dtype(df[field])}
    numeric = {field: round(float(pd.to_numeric(df[field], errors="coerce").mean()), 2) for field in fields if pd.api.types.is_numeric_dtype(df[field])}
    return {
        "available": True,
        "summary": "Direct psychographic fields support benefit, motivation, sensitivity, satisfaction, and message-framing analysis.",
        "dominant_motivations": list(tables.keys())[:5],
        "message_framing_recommendations": [
            "Align creative messaging to detected benefits, pain points, satisfaction, and price-sensitivity signals."
        ],
        "numeric_means": numeric,
        "tables": tables,
    }

