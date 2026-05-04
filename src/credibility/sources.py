"""Curated authoritative domains for COVID-era health/science journalism.

Composition: top-tier health authorities, peer-reviewed medical journals,
and major scientific publishers. Deliberately conservative — false positives
hurt the comparative analysis more than false negatives, since unmatched
domains fall back to "unknown" rather than getting mistagged.

This list is the v1 anchor; expand it via `CredibilityRegistry.extend_authoritative`
or by editing this file directly.
"""
from __future__ import annotations

# Health authorities (national + supranational)
HEALTH_AUTHORITIES = frozenset({
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "ecdc.europa.eu",
    "ema.europa.eu",
    "phe.gov.uk",
    "gov.uk",
    "canada.ca",
    "health.gov.au",
    "pasteur.fr",
    "rki.de",
    "iss.it",
})

# Peer-reviewed medical journals
MEDICAL_JOURNALS = frozenset({
    "nejm.org",
    "thelancet.com",
    "bmj.com",
    "jamanetwork.com",
    "annals.org",
    "cell.com",
    "plos.org",
    "ploscompbiol.org",
    "ploscollections.org",
})

# General scientific publishers
SCIENCE_PUBLISHERS = frozenset({
    "nature.com",
    "science.org",
    "sciencemag.org",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "biorxiv.org",
    "medrxiv.org",
    "europepmc.org",
})

AUTHORITATIVE_DOMAINS = HEALTH_AUTHORITIES | MEDICAL_JOURNALS | SCIENCE_PUBLISHERS
