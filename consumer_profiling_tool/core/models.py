"""Pydantic models used across the profiling pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FieldTypeProfile(BaseModel):
    name: str
    raw_dtype: str
    inferred_type: str
    missing_rate: float
    unique_count: int
    unique_ratio: float
    sample_values: list[str]
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    mode_values: list[str] = Field(default_factory=list)
    avg_string_length: float | None = None
    is_numeric: bool
    is_categorical: bool
    is_binary: bool
    is_datetime_like: bool
    is_text_like: bool
    is_id_like: bool
    is_sensitive_candidate: bool


class FieldSemanticProfile(BaseModel):
    name: str
    inferred_type: str
    suggested_role: str
    role_confidence: float
    suggested_polarity: str
    polarity_confidence: float
    is_sensitive_candidate: bool = False
    is_proxy_inference: bool = False
    reasons: list[str] = Field(default_factory=list)


class ConfirmedFieldMapping(BaseModel):
    name: str
    inferred_type: str
    role: str
    role_confidence: float = 0.0
    polarity: str
    polarity_confidence: float = 0.0
    is_sensitive_candidate: bool = False
    is_proxy_inference: bool = False
    reasons: list[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    supported_analyses: list[str]
    skipped_analyses: dict[str, str]
    proxy_analyses: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SegmentDetectionResult(BaseModel):
    field_name: str | None
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class ResponseModelResult(BaseModel):
    target_field: str
    metrics: dict[str, float | None]
    warnings: list[str] = Field(default_factory=list)
    top_positive_drivers: list[dict[str, Any]] = Field(default_factory=list)
    top_negative_drivers: list[dict[str, Any]] = Field(default_factory=list)


class PrivacyFieldFlag(BaseModel):
    name: str
    risk_level: str
    reasons: list[str] = Field(default_factory=list)


class PrivacyScanResult(BaseModel):
    privacy_notice: str
    flagged_fields: list[PrivacyFieldFlag] = Field(default_factory=list)
    report_excluded_fields: list[str] = Field(default_factory=list)


class CoverageDimension(BaseModel):
    dimension: str
    available: bool
    fields: list[str] = Field(default_factory=list)
    analysis_depth: str
    missing_fields: list[str] = Field(default_factory=list)
    suggested_data_to_collect: str
    proxy_only: bool = False


class CoverageAssessment(BaseModel):
    dimensions: list[CoverageDimension]
    available_pillars: list[str] = Field(default_factory=list)
    missing_pillars: list[str] = Field(default_factory=list)
    data_completeness_score: float = 0.0
    summary: str


class ModeDetectionResult(BaseModel):
    mode: str
    confidence: float
    b2c_signals: list[str] = Field(default_factory=list)
    b2b_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class IdentityResolutionResult(BaseModel):
    id_fields: list[str] = Field(default_factory=list)
    suggested_primary_id: str | None = None
    duplicate_id_counts: dict[str, int] = Field(default_factory=dict)
    pii_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

