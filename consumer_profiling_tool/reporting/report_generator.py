"""Markdown report generation for the v2 platform."""

from __future__ import annotations

import pandas as pd

from core.models import (
    AnalysisPlan,
    ConfirmedFieldMapping,
    CoverageAssessment,
    ModeDetectionResult,
    PrivacyScanResult,
    ResponseModelResult,
)


def generate_markdown_report(
    quality_report: dict[str, object],
    privacy_scan: PrivacyScanResult,
    mappings: list[ConfirmedFieldMapping],
    coverage: CoverageAssessment,
    analysis_plan: AnalysisPlan,
    mode_result: ModeDetectionResult,
    theory_narrative: str,
    scored_df: pd.DataFrame | None = None,
    segment_profile: pd.DataFrame | None = None,
    interpretations: list[str] | None = None,
    recommendations: list[dict[str, str]] | None = None,
    persona_summary: pd.DataFrame | None = None,
    negative_personas: pd.DataFrame | None = None,
    roi_result: dict[str, object] | None = None,
    response_result: ResponseModelResult | None = None,
    scoring_methodology: dict[str, object] | None = None,
) -> str:
    scored_df = scored_df if scored_df is not None else pd.DataFrame()
    segment_profile = segment_profile if segment_profile is not None else pd.DataFrame()
    interpretations = interpretations or []
    recommendations = recommendations or []
    persona_summary = persona_summary if persona_summary is not None else pd.DataFrame()
    negative_personas = negative_personas if negative_personas is not None else pd.DataFrame()
    roi_result = roi_result or {}
    scoring_methodology = scoring_methodology or {}

    lines = [
        "# Consumer Profiling Report",
        "",
        "## 1. Executive Summary",
        coverage.summary,
        f"Detected profile mode: {mode_result.mode} (confidence {mode_result.confidence:.2f}).",
        "",
        "## 2. Dataset Overview",
        f"- Rows: {quality_report.get('row_count', 'Unknown')}",
        f"- Columns: {quality_report.get('column_count', 'Unknown')}",
        f"- Duplicate rows: {quality_report.get('duplicate_row_count', 'Unknown')}",
        "",
        "## 3. Data Quality and Privacy Review",
        privacy_scan.privacy_notice,
    ]
    issues = quality_report.get("potential_data_issues", [])
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- No major automated quality warnings were detected.")
    if privacy_scan.flagged_fields:
        lines.append("- Privacy-flagged fields: " + ", ".join(flag.name for flag in privacy_scan.flagged_fields))

    lines.extend(["", "## 4. Detected Field Mapping", ""])
    lines.append("| Field | Role | Polarity | Confidence | Sensitive? | Proxy? |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for mapping in mappings:
        lines.append(
            f"| {mapping.name} | {mapping.role} | {mapping.polarity} | {mapping.role_confidence:.2f} | {mapping.is_sensitive_candidate} | {mapping.is_proxy_inference} |"
        )

    lines.extend(["", "## 5. Profile Coverage Matrix", ""])
    lines.append("| Dimension | Available? | Fields | Analysis Depth | Suggested Data to Collect |")
    lines.append("| --- | --- | --- | --- | --- |")
    for dimension in coverage.dimensions:
        lines.append(
            f"| {dimension.dimension} | {dimension.available} | {', '.join(dimension.fields) or 'None'} | {dimension.analysis_depth} | {dimension.suggested_data_to_collect} |"
        )

    lines.extend(["", "## 6. Supported, Skipped, and Proxy Analyses", ""])
    lines.extend(f"- Supported: {item}" for item in analysis_plan.supported_analyses)
    lines.extend(f"- Skipped: {name} because {reason}." for name, reason in analysis_plan.skipped_analyses.items())
    lines.extend(f"- Proxy: {name}: {reason}" for name, reason in analysis_plan.proxy_analyses.items())
    lines.extend(f"- Warning: {warning}" for warning in analysis_plan.warnings)

    lines.extend(["", "## 7. Theoretical Framework Applied", theory_narrative])
    lines.extend(["", "### 7.1 Demographic and Socioeconomic Profile", _dimension_text(coverage, "Demographic and socioeconomic")])
    lines.extend(["", "### 7.2 Geographic and Environmental Profile", _dimension_text(coverage, "Geographic and environmental")])
    lines.extend(["", "### 7.3 Psychographic and Motivational Profile", _dimension_text(coverage, "Psychographic and motivational")])
    lines.extend(["", "### 7.4 Behavioural and Transactional Profile", _dimension_text(coverage, "Behavioural, digital and transactional")])
    if mode_result.mode in {"b2b", "mixed_b2b_b2c"}:
        lines.extend(["", "### 7.5 B2B ICP and Decision-Role Profile", _dimension_text(coverage, "B2B firmographic and decision-role")])

    lines.extend(["", "## 8. Customer Value and Behavioural Scores"])
    if scored_df.empty:
        lines.append("- Scores were not generated.")
    else:
        for column in [
            "value_score",
            "frequency_loyalty_score",
            "engagement_score",
            "conversion_score",
            "risk_score_raw",
            "risk_score_health",
            "profile_quality_score",
            "b2b_account_fit_score",
        ]:
            if column in scored_df.columns and scored_df[column].notna().any():
                lines.append(f"- Mean {column}: {scored_df[column].mean():.2f}")
    lines.extend(["", "### 8.1 Scoring Methodology"])
    lines.extend(_scoring_methodology_lines(scoring_methodology))

    lines.extend(["", "## 8.2 Response Prediction Model"])
    lines.extend(_response_prediction_lines(response_result, analysis_plan))

    lines.extend(["", "## 9. Segment or Cluster Analysis"])
    if segment_profile.empty:
        lines.append("- No segment or cluster profile table was generated.")
    else:
        lines.append(segment_profile.to_markdown(index=False))
    lines.extend(f"- {text}" for text in interpretations)

    lines.extend(["", "## 10. Key Personas and Negative Personas"])
    if not persona_summary.empty:
        lines.append(persona_summary.to_markdown(index=False))
    if not scored_df.empty and "recommended_profile_type" in scored_df:
        for persona, count in scored_df["recommended_profile_type"].value_counts().items():
            lines.append(f"- {persona}: {count} records")
    if not negative_personas.empty:
        lines.append(negative_personas.to_markdown(index=False))

    lines.extend(["", "## 11. Marketing and Business Recommendations"])
    if recommendations:
        for item in recommendations:
            lines.append(
                f"- {item['segment']}: {item.get('recommended_action', item['recommendation'])} "
                f"Message strategy: {item.get('message_strategy', 'N/A')} "
                f"Evidence: {item['evidence_from_data']} Confidence: {item['confidence_level']}."
            )
    else:
        lines.append("- Use customer-level recommended actions as a first-pass lifecycle plan.")

    lines.extend(["", "## 12. ROI and Marketing Effectiveness Diagnostics"])
    lines.append(str(roi_result.get("summary", "ROI diagnostics were not generated.")))
    roi_table = roi_result.get("table")
    if isinstance(roi_table, pd.DataFrame) and not roi_table.empty:
        lines.append(roi_table.to_markdown(index=False))

    lines.extend(["", "## 13. Limitations"])
    lines.extend(_limitations(coverage, analysis_plan))

    lines.extend(["", "## 14. Recommended Additional Data to Collect"])
    for dimension in coverage.dimensions:
        if not dimension.available or dimension.proxy_only:
            lines.append(f"- {dimension.suggested_data_to_collect}")

    return "\n".join(lines).strip() + "\n"


def _dimension_text(coverage: CoverageAssessment, name: str) -> str:
    dimension = next((item for item in coverage.dimensions if item.dimension == name), None)
    if not dimension:
        return "Not assessed."
    if dimension.available:
        return f"Available at {dimension.analysis_depth} depth using fields: {', '.join(dimension.fields)}."
    if dimension.proxy_only:
        return "No direct fields detected; any interpretation must be treated as behavioural proxy analysis only."
    return f"Not available. Suggested data: {dimension.suggested_data_to_collect}"


def _scoring_methodology_lines(scoring_methodology: dict[str, object]) -> list[str]:
    if not scoring_methodology:
        return ["- Scoring methodology was not generated."]
    lines = [
        f"- Normalisation: {scoring_methodology.get('normalisation', 'N/A')}",
        f"- Polarity handling: {scoring_methodology.get('polarity', 'N/A')}",
        f"- Field weights: {scoring_methodology.get('weights', 'N/A')}",
    ]
    for group in scoring_methodology.get("score_groups", []):
        fields = group.get("fields", [])
        field_text = "; ".join(
            f"{field['field']} ({field['role']}, {field['polarity']}, weight {field['weight']})"
            for field in fields
        )
        lines.append(f"- {group.get('score')}: {group.get('formula')} Fields: {field_text}")
    return lines


def _response_prediction_lines(
    response_result: ResponseModelResult | None,
    analysis_plan: AnalysisPlan,
) -> list[str]:
    if response_result is None:
        if "Response prediction model" in analysis_plan.supported_analyses:
            return ["- Response prediction is available but not included in this report."]
        return ["- Response prediction was not available because no confirmed binary response target was detected."]
    lines = [f"- Target field: {response_result.target_field}"]
    if response_result.metrics:
        metrics = ", ".join(f"{name}: {value}" for name, value in response_result.metrics.items())
        lines.append(f"- Model metrics: {metrics}")
    else:
        lines.append("- Model metrics were not produced.")
    lines.extend(f"- Model warning: {warning}" for warning in response_result.warnings)
    if response_result.top_positive_drivers:
        drivers = ", ".join(f"{item['feature']} ({item['coefficient']})" for item in response_result.top_positive_drivers[:5])
        lines.append(f"- Top positive response drivers: {drivers}")
    if response_result.top_negative_drivers:
        drivers = ", ".join(f"{item['feature']} ({item['coefficient']})" for item in response_result.top_negative_drivers[:5])
        lines.append(f"- Top negative response drivers: {drivers}")
    return lines


def _limitations(coverage: CoverageAssessment, analysis_plan: AnalysisPlan) -> list[str]:
    lines = []
    if coverage.missing_pillars:
        lines.append(
            "This dataset cannot support direct analysis for: " + ", ".join(coverage.missing_pillars) + "."
        )
    if analysis_plan.proxy_analyses:
        lines.append("Motivational interpretation should be treated as proxy analysis unless direct psychographic fields are collected.")
    if not lines:
        lines.append("All major profile dimensions have at least limited coverage, but recommendations remain suggestive rather than causal.")
    return [f"- {line}" for line in lines]
