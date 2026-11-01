"""Unit + smoke tests for the credibility module.

Pure unit tests exercise the lookup logic. The live test verifies that
collection 13559's seed list yields a non-trivial low-tier domain set.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def test_tier_priority_applies_across_parent_and_child_domains() -> None:
    reg = CredibilityRegistry(
        authoritative={"nih.gov", "who.int"},
        low={"afro.who.int", "example.org"},
        mixed={"covid19treatmentguidelines.nih.gov", "news.example.org"},
    )
    assert reg.lookup("covid19treatmentguidelines.nih.gov") == "authoritative"
    assert reg.lookup("afro.who.int") == "authoritative"
    assert reg.lookup("news.example.org") == "low"


def test_extend_authoritative_preserves_overrides_and_prunes_shadowed_hosts() -> None:
    reg = CredibilityRegistry(
        authoritative={"custom.health.test"}, low={"other.test", "afro.who.int"},
        mixed={"news.cdc.gov", "nhs.uk", "news.other.test", "remaining.test"},
    ).with_authoritative(["https://www.cdc.gov/", "who.int", "nhs.uk"])
    assert reg.authoritative == {"custom.health.test", "cdc.gov", "who.int", "nhs.uk"}
    assert reg.low == {"other.test"}
    assert reg.mixed == {"remaining.test"}


@pytest.mark.parametrize("host", [
    "weekly.chinacdc.cn", "coronavirus.health.ny.gov", "cdph.ca.gov",
    "covid19.colorado.gov", "nhsinform.scot", "health.govt.nz", "mohfw.gov.in",
    "doh.gov.ph", "globalhealthnow.org", "cidrap.umn.edu", "health.harvard.edu",
])
def test_curated_expansion_recognizes_institutional_sources(host: str) -> None:
    reg = CredibilityRegistry(authoritative=AUTHORITATIVE_DOMAINS, mixed={host})
    assert reg.lookup(host) == "authoritative"


@pytest.mark.parametrize("host", [
    "cdc.news", "cdc.gov.example.org", "cnn.com", "outbreak.my",
    "github.com", "youtube.com", "student.harvard.edu", "mass.gov",
])
def test_curated_expansion_does_not_blanket_trust_similar_or_broad_hosts(host: str) -> None:
    reg = CredibilityRegistry(authoritative=AUTHORITATIVE_DOMAINS)
    assert reg.lookup(host) == "unknown"


def test_build_deduplicates_authoritative_parent_domains_offline() -> None:
    class Client:
        def iter_seeds(self, collection_id, page_size):
            for url in ["https://news.cdc.gov/", "https://example.org/", "https://extra.test/"]:
                yield SimpleNamespace(canonical_url=None, url=url)

    from config import ArchiveItCollection
    reg = build_registry(
        Client(), collections={1: ArchiveItCollection(1, "Test", "mixed")},
        extra_authoritative=["https://www.extra.test/path"],
    )
    assert reg.mixed == {"example.org"}
    assert reg.lookup("news.cdc.gov") == "authoritative"
    assert reg.lookup("extra.test") == "authoritative"


def test_offline_refresh_backs_up_existing_registry(tmp_path, monkeypatch) -> None:
    import runpy
    import sys

    path = tmp_path / "credibility.json"
    CredibilityRegistry(
        authoritative={"custom.health.test"}, mixed={"weekly.chinacdc.cn", "example.org"},
    ).save(path)
    original = path.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_credibility.py"
    main = runpy.run_path(str(script))["main"]

    def network_is_forbidden(*args, **kwargs):
        pytest.fail("Refresh must not initialize the Archive-It client")

    monkeypatch.setitem(main.__globals__, "ArchiveItClient", network_is_forbidden)
    monkeypatch.setattr(sys, "argv", [str(script), "--refresh-authoritative", "--output", str(path)])
    assert main() == 0
    refreshed = CredibilityRegistry.load(path)
    assert refreshed.lookup("weekly.chinacdc.cn") == "authoritative"
    assert refreshed.lookup("custom.health.test") == "authoritative"
    assert refreshed.mixed == {"example.org"}
    backups = list(tmp_path.glob("credibility.before-refresh-*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_notebook_refreshes_cached_document_and_mention_tiers(tmp_path) -> None:
    import ast
    import json
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "notebooks" / "02_georeferencing_eda.ipynb"
    tree = ast.parse("".join(json.loads(path.read_text())["cells"][3]["source"]))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    georef = tmp_path / "georef"
    georef.mkdir()
    mention = dict(toponym="Paris", name="Paris", lat=48.86, lon=2.35,
                   country_code="FR", feature_class="P", mention_count=2, confidence=.9)
    doc = dict(article_id="1:2:20200311000000", phase="P1_global_onset", seed_url="https://www.nhs.uk/",
               credibility_tier="mixed", event_date="2020-03-11", event_week="2020-W11",
               event_date_source="publish", mentions=[mention])
    (georef / "13559").mkdir()
    cached_path = georef / "13559/documents.jsonl"
    cached_path.write_text(json.dumps(doc) + "\n")
    original = cached_path.read_bytes()
    CredibilityRegistry(mixed={"nhs.uk"}).save(tmp_path / "credibility.json")
    namespace = dict(Path=Path, pd=pd, json=json, PROCESSED_DIR=tmp_path)
    exec(compile(tree, "notebook loader", "exec"), namespace)
    docs, mentions = namespace["load_georef"](georef)
    assert docs.credibility_tier.tolist() == ["authoritative"]
    assert mentions.tier.tolist() == ["authoritative"]
    assert docs.iloc[0]["mentions"] == [mention]
    assert cached_path.read_bytes() == original


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
