"""Existing segment field detection."""

from __future__ import annotations

import pandas as pd

from core.constants import ROLE_KEYWORDS
from core.models import ConfirmedFieldMapping, FieldTypeProfile, SegmentDetectionResult
from core.utils import contains_keyword, normalise_name


def detect_existing_segment(
    df: pd.DataFrame,
    type_profiles: list[FieldTypeProfile],
    mappings: list[ConfirmedFieldMapping] | None = None,
) -> SegmentDetectionResult:
    """Find a reliable segment-like column without ranking by label name."""
    profile_by_name = {profile.name: profile for profile in type_profiles}
    mapping_by_name = {mapping.name: mapping for mapping in mappings or []}
    best_field: str | None = None
    best_score = 0.0
    best_reasons: list[str] = []

    for column in df.columns:
        profile = profile_by_name.get(column)
        mapping = mapping_by_name.get(column)
        unique_count = int(df[column].dropna().nunique())
        if unique_count < 2:
            continue
        counts = df[column].value_counts(dropna=True)
        min_class_size = int(counts.min()) if not counts.empty else 0
        enough_rows = min_class_size >= max(10, int(len(df) * 0.01)) or (len(df) < 100 and min_class_size >= 1)
        keyword = contains_keyword(normalise_name(column), ROLE_KEYWORDS["existing_segment"])

        score = 0.0
        reasons: list[str] = []
        if mapping and mapping.role == "existing_segment":
            score += 0.55
            reasons.append("Mapping assigns this field to existing_segment.")
        if keyword:
            score += 0.25
            reasons.append(f"Column name contains segment hint '{keyword}'.")
        if profile and profile.is_categorical and 2 <= unique_count <= 15:
            score += 0.2
            reasons.append("Categorical field has 2-15 classes.")
        if enough_rows:
            score += 0.1
            reasons.append("Segment classes have enough rows for profiling.")
        else:
            score -= 0.2
            reasons.append("Some segment classes are too small for reliable profiling.")
        if unique_count > 20:
            score -= 0.25
            reasons.append("Too many unique values for a segment-like field.")

        if score > best_score:
            best_field, best_score, best_reasons = column, score, reasons

    if best_score < 0.45:
        return SegmentDetectionResult(
            field_name=None,
            confidence=round(max(best_score, 0.0), 2),
            reasons=best_reasons or ["No reliable existing segment field detected."],
        )
    return SegmentDetectionResult(
        field_name=best_field,
        confidence=round(min(best_score, 0.99), 2),
        reasons=best_reasons,
    )

