"""Export helpers."""

from __future__ import annotations

import pandas as pd

from core.models import AnalysisPlan, ConfirmedFieldMapping, CoverageAssessment
from core.utils import to_json_bytes


def scored_customers_csv(scored_df: pd.DataFrame, include_sensitive: bool = False) -> bytes:
    export_df = scored_df.copy()
    if not include_sensitive:
        sensitive_like = [column for column in export_df.columns if column.lower() in {"name", "email", "phone", "address"}]
        export_df = export_df.drop(columns=sensitive_like, errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8-sig")


def mapping_json(mappings: list[ConfirmedFieldMapping]) -> bytes:
    return to_json_bytes([mapping.model_dump() for mapping in mappings])


def coverage_json(coverage: CoverageAssessment) -> bytes:
    return to_json_bytes(coverage.model_dump())


def analysis_summary_json(plan: AnalysisPlan, quality_report: dict[str, object], extra: dict[str, object] | None = None) -> bytes:
    payload = {"analysis_plan": plan.model_dump(), "quality_report": quality_report}
    if extra:
        payload.update(extra)
    return to_json_bytes(payload)

