"""Metric polarity detection with interpretation guardrails."""

from __future__ import annotations

from core.constants import (
    DEFAULT_POLARITY_BY_ROLE,
    NEGATIVE_POLARITY_WORDS,
    NEUTRAL_ROLES,
    POSITIVE_POLARITY_WORDS,
)
from core.utils import contains_keyword, normalise_name


def detect_polarity(column_name: str, role: str) -> tuple[str, float, list[str]]:
    """Infer metric polarity from role defaults and column-name hints."""
    normalised = normalise_name(column_name)

    if role in NEUTRAL_ROLES:
        return "neutral", 0.95, ["Role is descriptive or sensitive; it should not be treated as better/worse by default."]

    negative_match = contains_keyword(normalised, NEGATIVE_POLARITY_WORDS)
    if negative_match:
        return "negative", 0.95, [f"Column name contains negative/risk hint '{negative_match}'."]

    positive_match = contains_keyword(normalised, POSITIVE_POLARITY_WORDS)
    if positive_match:
        return "positive", 0.9, [f"Column name contains positive outcome hint '{positive_match}'."]

    default = DEFAULT_POLARITY_BY_ROLE.get(role, "unknown")
    confidence = 0.8 if default != "unknown" else 0.35
    return default, confidence, [f"Using default polarity for role '{role}'."]

