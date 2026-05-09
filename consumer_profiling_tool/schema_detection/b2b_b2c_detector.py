"""Detect whether a dataset is B2C, B2B, or mixed."""

from __future__ import annotations

from core.constants import B2B_ROLES, BEHAVIOURAL_ROLES, DEMOGRAPHIC_ROLES, GEOGRAPHIC_ROLES, PSYCHOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping, ModeDetectionResult


def detect_profile_mode(mappings: list[ConfirmedFieldMapping]) -> ModeDetectionResult:
    roles = {mapping.role for mapping in mappings if mapping.role != "ignore"}
    b2b_signals = sorted(role for role in roles if role in B2B_ROLES)
    b2c_roles = DEMOGRAPHIC_ROLES | GEOGRAPHIC_ROLES | PSYCHOGRAPHIC_ROLES | BEHAVIOURAL_ROLES | {"customer_id", "household_id"}
    b2c_signals = sorted(role for role in roles if role in b2c_roles)

    b2b_score = len(b2b_signals)
    b2c_score = len(b2c_signals)
    if b2b_score and b2c_score >= 3:
        mode = "mixed_b2b_b2c"
    elif b2b_score >= 2:
        mode = "b2b"
    else:
        mode = "b2c"
    total = max(b2b_score + b2c_score, 1)
    confidence = max(b2b_score, b2c_score) / total
    reasons = [
        f"Detected {b2c_score} B2C-like signals and {b2b_score} B2B-like signals.",
        "Mixed mode separates account-level and contact/customer-level interpretation." if mode == "mixed_b2b_b2c" else f"Dataset appears primarily {mode.upper()}.",
    ]
    return ModeDetectionResult(
        mode=mode,
        confidence=round(float(confidence), 2),
        b2c_signals=b2c_signals,
        b2b_signals=b2b_signals,
        reasons=reasons,
    )

