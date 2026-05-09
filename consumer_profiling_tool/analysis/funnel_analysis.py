"""Dynamic funnel analysis."""

from __future__ import annotations

import pandas as pd

from core.models import ConfirmedFieldMapping

FUNNEL_ROLE_MAP = {
    "campaign_exposure": "Awareness",
    "engagement": "Engagement",
    "session_activity": "Engagement",
    "page_view": "Engagement",
    "click_activity": "Engagement",
    "product_interest": "Consideration/Intent",
    "category_preference": "Consideration/Intent",
    "cart_abandonment": "Intent/Friction",
    "conversion_or_response": "Conversion",
    "binary_target": "Conversion",
    "purchase_frequency": "Retention",
    "loyalty": "Retention",
    "subscription": "Retention",
}


def run_funnel_analysis(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> dict[str, object]:
    rows = []
    for mapping in mappings:
        stage = FUNNEL_ROLE_MAP.get(mapping.role)
        if not stage or mapping.name not in df.columns:
            continue
        numeric = pd.to_numeric(df[mapping.name], errors="coerce")
        signal = numeric.mean() if numeric.notna().any() else df[mapping.name].notna().mean()
        rows.append({"Stage": stage, "Field": mapping.name, "Average Signal": round(float(signal), 3)})
    table = pd.DataFrame(rows)
    if table.empty:
        return {"available": False, "summary": "No funnel-stage fields were detected.", "table": table}
    return {"available": True, "summary": "Funnel signals are mapped dynamically from awareness through retention.", "table": table}

