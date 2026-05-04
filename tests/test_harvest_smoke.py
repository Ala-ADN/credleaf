"""Live smoke test for the harvest module.

Runs a tightly-bounded harvest (3 seeds × 3 captures each) for collection 13559
in phase P1, then re-reads the JSONL and validates shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import httpx
import pytest

from config import PHASES_BY_NAME
from ingest import ArchiveItClient, CaptureRecord, harvest_phase, load_jsonl


@pytest.fixture(scope="module")
def client() -> Iterator[ArchiveItClient]:
    with ArchiveItClient(timeout=60.0) as c:
        yield c


def test_harvest_collection_13559_p1_global_onset(
    client: ArchiveItClient, tmp_path: Path
) -> None:
    phase = PHASES_BY_NAME["P1_global_onset"]
    output = tmp_path / "captures.jsonl"

    try:
        path, stats = harvest_phase(
            client,
            collection_id=13559,
            phase=phase,
            output_path=output,
            max_seeds=3,
            max_captures_per_seed=3,
        )
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")

    assert path == output
    assert path.exists()
    assert stats.seeds_visited == 3
    assert stats.captures_kept >= 1, "expected at least one capture in P1 for collection 13559"

    records = list(load_jsonl(path))
    assert len(records) == stats.captures_kept
    for r in records:
        assert isinstance(r, CaptureRecord)
        assert r.collection_id == 13559
        assert r.phase == "P1_global_onset"
        assert r.capture.status and r.capture.status.startswith("2")
        assert r.capture.mimetype and "html" in r.capture.mimetype
        assert "2020" in r.capture.timestamp[:4] or r.capture.timestamp[:4] == "2020"

    digests = [r.capture.digest for r in records if r.capture.digest]
    assert len(digests) == len(set(digests)), "duplicate digests slipped through"
