"""Feature selection helpers for scoring and modelling."""

from __future__ import annotations

from core.models import ConfirmedFieldMapping


def fields_for_roles(
    mappings: list[ConfirmedFieldMapping],
    roles: list[str] | set[str],
    include_unknown_polarity: bool = True,
) -> list[str]:
    """Return column names mapped to any of the requested roles."""
    role_set = set(roles)
    return [
        mapping.name
        for mapping in mappings
        if mapping.role in role_set
        and mapping.role != "ignore"
        and (include_unknown_polarity or mapping.polarity != "unknown")
    ]


def first_field_for_role(mappings: list[ConfirmedFieldMapping], role: str) -> str | None:
    for mapping in mappings:
        if mapping.role == role:
            return mapping.name
    return None

