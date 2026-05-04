"""Live smoke test for the normalize layer.

Reuses the harvest fixture: harvests 2 seeds × 2 captures from collection 13559
in P1, then runs normalize end-to-end. Verifies the cache shortcut on a re-run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import httpx
import pytest

from config import PHASES_BY_NAME
from ingest import ArchiveItClient, harvest_phase
from normalize import load_normalized, normalize_phase


@pytest.fixture(scope="module")
def client() -> Iterator[ArchiveItClient]:
    with ArchiveItClient(timeout=60.0) as c:
        yield c


@pytest.fixture(scope="module")
def captures_jsonl(client: ArchiveItClient, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Harvest seed-URL captures (these collections are crawled at depth=0, so
    every capture is the seed homepage itself — perfect for exercising the
    fallback extraction path)."""
    out = tmp_path_factory.mktemp("captures") / "captures.jsonl"
    try:
        path, _ = harvest_phase(
            client,
            collection_id=13559,
            phase=PHASES_BY_NAME["P1_global_onset"],
            output_path=out,
            max_seeds=2,
            max_captures_per_seed=3,
        )
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")
    return path


def test_normalize_end_to_end(
    client: ArchiveItClient,
    captures_jsonl: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    out = tmp_path / "normalized.jsonl"

    try:
        path, stats = normalize_phase(
            client,
            captures_jsonl,
            output_path=out,
            cache_root=cache_root,
        )
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")

    assert path == out
    assert stats.total >= 1
    assert stats.fetched >= 1, "first run should hit the network"
    assert stats.from_cache == 0

    docs = load_normalized(out)
    assert len(docs) == stats.total

    # Every doc must have a valid status and shape consistent with that status.
    valid = {"ok", "fetch_failed", "extract_failed", "empty"}
    for d in docs:
        assert d.fetch_status in valid
        assert d.collection_id == 13559
        assert d.phase == "P1_global_onset"
        if d.fetch_status == "ok":
            assert d.text and len(d.text) > 0
            assert d.word_count == len(d.text.split())
            assert d.char_count == len(d.text)
        if d.fetch_status == "fetch_failed":
            assert d.error
            assert d.text is None

    # With host-match we expect at least one article-shaped page to extract.
    ok_docs = [d for d in docs if d.fetch_status == "ok"]
    assert ok_docs, (
        f"expected at least one extracted doc; got "
        f"{[d.fetch_status for d in docs]}"
    )


def test_normalize_uses_cache_on_rerun(
    client: ArchiveItClient,
    captures_jsonl: Path,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    out = tmp_path / "normalized.jsonl"

    try:
        normalize_phase(client, captures_jsonl, output_path=out, cache_root=cache_root)
        _, stats2 = normalize_phase(
            client, captures_jsonl, output_path=out, cache_root=cache_root
        )
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")

    assert stats2.from_cache >= 1, "second run should hit the HTML cache"
    assert stats2.fetched == 0 or stats2.fetched < stats2.total, (
        "second run should refetch fewer docs than the first"
    )
