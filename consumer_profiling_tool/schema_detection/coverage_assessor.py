"""Profile coverage and theory-readiness assessment."""

from __future__ import annotations

from core.constants import BEHAVIOURAL_ROLES, PROFILE_DIMENSIONS, PSYCHOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping, CoverageAssessment, CoverageDimension


def assess_profile_coverage(mappings: list[ConfirmedFieldMapping]) -> CoverageAssessment:
    """Build the profile coverage matrix required by v2."""
    dimensions: list[CoverageDimension] = []
    role_to_fields: dict[str, list[str]] = {}
    for mapping in mappings:
        if mapping.role == "ignore":
            continue
        role_to_fields.setdefault(mapping.role, []).append(mapping.name)

    available_pillars: list[str] = []
    missing_pillars: list[str] = []
    depth_scores: list[float] = []
    has_behavioural_proxy = any(role in role_to_fields for role in BEHAVIOURAL_ROLES)
    has_psychographic = any(role in role_to_fields for role in PSYCHOGRAPHIC_ROLES)

    for dimension, config in PROFILE_DIMENSIONS.items():
        fields = []
        for role in config["roles"]:
            fields.extend(role_to_fields.get(role, []))
        count = len(fields)
        available = count > 0
        if count >= 5:
            depth = "strong"
            score = 1.0
        elif count >= 2:
            depth = "moderate"
            score = 0.65
        elif count == 1:
            depth = "limited"
            score = 0.35
        else:
            depth = "not available"
            score = 0.0
        proxy_only = False
        if dimension == "Psychographic and motivational" and not has_psychographic and has_behavioural_proxy:
            available = False
            depth = "proxy only"
            proxy_only = True
            score = 0.2
        if available:
            available_pillars.append(dimension)
        else:
            missing_pillars.append(dimension)
        depth_scores.append(score)
        dimensions.append(
            CoverageDimension(
                dimension=dimension,
                available=available,
                fields=fields,
                analysis_depth=depth,
                missing_fields=list(config["missing"]),
                suggested_data_to_collect=str(config["suggest"]),
                proxy_only=proxy_only,
            )
        )

    completeness = round(sum(depth_scores) / max(len(depth_scores), 1) * 100, 1)
    summary = _coverage_summary(dimensions)
    return CoverageAssessment(
        dimensions=dimensions,
        available_pillars=available_pillars,
        missing_pillars=missing_pillars,
        data_completeness_score=completeness,
        summary=summary,
    )


def _coverage_summary(dimensions: list[CoverageDimension]) -> str:
    by_name = {dimension.dimension: dimension for dimension in dimensions}
    missing = [dimension.dimension for dimension in dimensions if not dimension.available]
    if by_name["Behavioural, digital and transactional"].available and {
        "Demographic and socioeconomic",
        "Geographic and environmental",
        "Psychographic and motivational",
        "B2B firmographic and decision-role",
    }.issubset(set(missing)):
        return (
            "This dataset supports strong behavioural, transactional, engagement, risk, and response analysis. "
            "It does not support demographic, geographic, psychographic, or B2B firmographic analysis because those fields were not detected."
        )
    available = [dimension.dimension for dimension in dimensions if dimension.available]
    if available:
        return "Available profile dimensions: " + ", ".join(available) + "."
    return "No strong profile dimension coverage was detected; collect behavioural, demographic, geographic, and response fields."
