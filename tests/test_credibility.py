"""Unit + smoke tests for the credibility module.

Pure unit tests exercise the lookup logic. The live test verifies that
collection 13559's seed list yields a non-trivial low-tier domain set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import httpx
import pytest

from credibility import (
    AUTHORITATIVE_DOMAINS,
    CredibilityRegistry,
    build_registry,
    normalize_host,
)
from ingest import ArchiveItClient


# ---- normalize_host ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("https://www.cdc.gov/covid-19/", "cdc.gov"),
    ("HTTP://Themindunleashed.com/", "themindunleashed.com"),
    ("http://www.WND.com:80/some/path", "wnd.com"),
    ("cdc.gov", "cdc.gov"),
    ("WWW.who.int.", "who.int"),
    ("", None),
    ("not a url", "not a url"),  # bare-ish input is still passed through, lowercased
    ("http://", None),
])
def test_normalize_host(raw: str, expected: str | None) -> None:
    assert normalize_host(raw) == expected


# ---- CredibilityRegistry.lookup ---------------------------------------------

@pytest.fixture
def registry() -> CredibilityRegistry:
    return CredibilityRegistry(
        low={"themindunleashed.com", "wnd.com", "zerohedge.com"},
        authoritative={"cdc.gov", "who.int", "nejm.org"},
        mixed={"cnn.com", "bbc.co.uk", "coronavirus.fr"},
    )


def test_lookup_exact_match(registry: CredibilityRegistry) -> None:
    assert registry.lookup("https://www.cdc.gov/covid/index.html") == "authoritative"
    assert registry.lookup("themindunleashed.com") == "low"
    assert registry.lookup("cnn.com") == "mixed"


def test_lookup_subdomain_walks_up(registry: CredibilityRegistry) -> None:
    assert registry.lookup("https://news.cdc.gov/feature/") == "authoritative"
    assert registry.lookup("blog.themindunleashed.com") == "low"
    assert registry.lookup("https://edition.cnn.com/health") == "mixed"


def test_lookup_unknown(registry: CredibilityRegistry) -> None:
    assert registry.lookup("https://nytimes.com/article") == "unknown"
    assert registry.lookup("") == "unknown"


def test_lookup_priority_authoritative_beats_low_and_mixed() -> None:
    """If a domain lands in multiple sets, authoritative wins, then low, then mixed."""
    reg = CredibilityRegistry(
        low={"cdc.gov"},
        authoritative={"cdc.gov"},
        mixed={"cdc.gov"},
    )
    assert reg.lookup("cdc.gov") == "authoritative"


def test_lookup_priority_low_beats_mixed() -> None:
    reg = CredibilityRegistry(
        low={"zerohedge.com"},
        mixed={"zerohedge.com"},
    )
    assert reg.lookup("zerohedge.com") == "low"


def test_lookup_normalizes_input(registry: CredibilityRegistry) -> None:
    assert registry.lookup("HTTPS://WWW.CDC.GOV/") == "authoritative"


# ---- save / load -------------------------------------------------------------

def test_save_load_roundtrip(registry: CredibilityRegistry, tmp_path: Path) -> None:
    out = tmp_path / "registry.json"
    registry.save(out)
    loaded = CredibilityRegistry.load(out)
    assert loaded.low == registry.low
    assert loaded.authoritative == registry.authoritative
    assert loaded.mixed == registry.mixed
    # behavior round-trips too
    assert loaded.lookup("news.cdc.gov") == "authoritative"
    assert loaded.lookup("edition.cnn.com") == "mixed"


# ---- live build smoke --------------------------------------------------------

@pytest.fixture(scope="module")
def live_client() -> Iterator[ArchiveItClient]:
    with ArchiveItClient(timeout=60.0) as c:
        yield c


def test_build_from_live_collections(live_client: ArchiveItClient) -> None:
    """Live build, capped to 50 seeds per mixed collection so the 16k-seed
    international collection doesn't dominate the test runtime."""
    try:
        reg = build_registry(live_client, max_seeds_per_collection=50)
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")

    # Collection 13559 has 99 low-credibility seeds — all <= 50 cap fits, so
    # we expect ~50 (capped) low domains, possibly fewer after dedup vs auth.
    assert 30 <= len(reg.low) <= 99, f"low={len(reg.low)}"
    # Mixed should pull from collections 4887 + 13529 (both tagged 'mixed').
    assert len(reg.mixed) >= 30, f"mixed={len(reg.mixed)}"
    # Curated authoritative is unchanged.
    assert reg.authoritative == AUTHORITATIVE_DOMAINS

    # Tier sets must be disjoint after the build's dedup pass.
    assert reg.low.isdisjoint(reg.authoritative)
    assert reg.mixed.isdisjoint(reg.authoritative)
    assert reg.low.isdisjoint(reg.mixed)

    # Known low-tier seeds resolve correctly.
    for url in ("themindunleashed.com", "wnd.com", "zerohedge.com"):
        assert reg.lookup(url) == "low", f"{url} should be tagged low"

    # Curated authoritative still resolves.
    assert reg.lookup("cdc.gov") == "authoritative"
    assert reg.lookup("nejm.org") == "authoritative"
