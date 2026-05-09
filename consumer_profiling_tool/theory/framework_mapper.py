"""Map fields to consumer-profiling theoretical frameworks."""

from __future__ import annotations

from core.constants import BEHAVIOURAL_ROLES, B2B_ROLES, DEMOGRAPHIC_ROLES, GEOGRAPHIC_ROLES, PSYCHOGRAPHIC_ROLES
from core.models import ConfirmedFieldMapping


def map_frameworks(mappings: list[ConfirmedFieldMapping]) -> dict[str, list[str]]:
    role_fields: dict[str, list[str]] = {
        "Demographic/Socioeconomic": [],
        "Geographic/Environmental": [],
        "Psychographic/Motivational": [],
        "Behavioural/Digital/Transactional": [],
        "B2B ICP/Decision Role": [],
    }
    for mapping in mappings:
        if mapping.role in DEMOGRAPHIC_ROLES:
            role_fields["Demographic/Socioeconomic"].append(mapping.name)
        if mapping.role in GEOGRAPHIC_ROLES:
            role_fields["Geographic/Environmental"].append(mapping.name)
        if mapping.role in PSYCHOGRAPHIC_ROLES:
            role_fields["Psychographic/Motivational"].append(mapping.name)
        if mapping.role in BEHAVIOURAL_ROLES:
            role_fields["Behavioural/Digital/Transactional"].append(mapping.name)
        if mapping.role in B2B_ROLES:
            role_fields["B2B ICP/Decision Role"].append(mapping.name)
    return role_fields

