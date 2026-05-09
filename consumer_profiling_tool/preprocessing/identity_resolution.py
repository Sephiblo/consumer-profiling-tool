"""MVP identity-resolution diagnostics."""

from __future__ import annotations

import pandas as pd

from core.constants import IDENTITY_ROLES
from core.models import ConfirmedFieldMapping, IdentityResolutionResult


def analyze_identity_fields(df: pd.DataFrame, mappings: list[ConfirmedFieldMapping]) -> IdentityResolutionResult:
    """Detect possible ID fields, duplicate IDs, and PII fields without merging records."""
    id_fields = [mapping.name for mapping in mappings if mapping.role in IDENTITY_ROLES and mapping.name in df.columns]
    pii_fields = [mapping.name for mapping in mappings if mapping.is_sensitive_candidate]
    duplicate_counts: dict[str, int] = {}
    for field in id_fields:
        duplicate_counts[field] = int(df[field].duplicated().sum())

    suggested = None
    if id_fields:
        suggested = sorted(id_fields, key=lambda field: (duplicate_counts.get(field, 0), -df[field].nunique(dropna=True)))[0]

    warnings: list[str] = []
    if any(count > 0 for count in duplicate_counts.values()):
        warnings.append("Duplicate IDs were detected; do not assume one row equals one unique customer/account.")
    if pii_fields:
        warnings.append("PII-like fields were detected and should not be exposed in reports by default.")

    return IdentityResolutionResult(
        id_fields=id_fields,
        suggested_primary_id=suggested,
        duplicate_id_counts=duplicate_counts,
        pii_fields=pii_fields,
        warnings=warnings,
    )

