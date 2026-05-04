from .build import build_registry
from .registry import CredibilityRegistry, Tier, normalize_host
from .sources import (
    AUTHORITATIVE_DOMAINS,
    HEALTH_AUTHORITIES,
    MEDICAL_JOURNALS,
    SCIENCE_PUBLISHERS,
)

__all__ = [
    "AUTHORITATIVE_DOMAINS",
    "CredibilityRegistry",
    "HEALTH_AUTHORITIES",
    "MEDICAL_JOURNALS",
    "SCIENCE_PUBLISHERS",
    "Tier",
    "build_registry",
    "normalize_host",
]
