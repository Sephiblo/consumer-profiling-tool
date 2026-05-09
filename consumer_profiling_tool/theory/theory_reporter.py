"""Generate theory-based interpretation without overclaiming."""

from __future__ import annotations

from core.models import CoverageAssessment, ModeDetectionResult
from theory.interpretation_templates import BEHAVIOURAL_PROXY_WARNING, RFM_INTERPRETATION, SENSITIVE_ATTRIBUTE_GUARDRAIL


def generate_theory_narrative(
    coverage: CoverageAssessment,
    mode: ModeDetectionResult,
    analysis_plan_proxy: dict[str, str] | None = None,
) -> str:
    """Return Markdown narrative linking available fields to theory frameworks."""
    available = coverage.available_pillars or ["No strong profile dimensions"]
    missing = coverage.missing_pillars or ["None"]
    lines = [
        "## Theoretical Interpretation",
        "",
        "### 1. Profile Dimensions Available",
        f"The dataset supports: {', '.join(available)}.",
        f"Missing or weak dimensions: {', '.join(missing)}.",
        "",
        "### 2. Behavioural Theory Interpretation",
    ]
    if any("Behavioural" in item for item in available):
        lines.append(RFM_INTERPRETATION)
        lines.append("Engagement without conversion is treated as a possible funnel-friction signal.")
    else:
        lines.append("Behavioural interpretation is limited because few behavioural or transactional variables were detected.")

    lines.extend(["", "### 3. Strategic Implication"])
    if "Psychographic and motivational" in missing:
        lines.append(
            "The dataset is weaker for creative persona development because no direct motivation, attitude, or lifestyle variables were detected."
        )
        lines.append(BEHAVIOURAL_PROXY_WARNING)
    else:
        lines.append("Direct psychographic fields can support message-framing, benefit-sought, and motivation analysis.")
    if mode.mode in {"b2b", "mixed_b2b_b2c"}:
        lines.append("B2B fields support ICP and buying-committee interpretation.")
    lines.append(SENSITIVE_ATTRIBUTE_GUARDRAIL)

    lines.extend(["", "### 4. Data Collection Improvement"])
    suggestions = [dimension.suggested_data_to_collect for dimension in coverage.dimensions if not dimension.available or dimension.proxy_only]
    if suggestions:
        lines.extend(f"- {suggestion}" for suggestion in suggestions)
    else:
        lines.append("- Add campaign cost, margin, satisfaction, and qualitative feedback to deepen ROI and motivation diagnostics.")

    if analysis_plan_proxy:
        lines.extend(["", "### 5. Proxy Analysis Warnings"])
        lines.extend(f"- {name}: {reason}" for name, reason in analysis_plan_proxy.items())
    return "\n".join(lines)

