"""Live smoke test against Archive-It collections.

Runs three checks per collection:
  1. fetch collection metadata
  2. paginate the first ~10 seeds
  3. fetch a small CDX sample for the first seed

Network-dependent. Skipped automatically if the API is unreachable.
"""
from __future__ import annotations
from typing import Iterator

import httpx
import pytest

from config import COVID_COLLECTIONS
from ingest import ArchiveItClient


@pytest.fixture(scope="module")
def client() -> Iterator[ArchiveItClient]:
    with ArchiveItClient(timeout=60.0) as c:
        yield c


@pytest.mark.parametrize("collection_id", list(COVID_COLLECTIONS))
def test_collection_metadata(client: ArchiveItClient, collection_id: int) -> None:
    try:
        coll = client.get_collection(collection_id)
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")
    assert coll.id == collection_id
    assert coll.publicly_visible is True
    assert coll.num_active_seeds > 0


@pytest.mark.parametrize("collection_id", list(COVID_COLLECTIONS))
def test_seed_pagination(client: ArchiveItClient, collection_id: int) -> None:
    try:
        seeds = list(client.iter_seeds(collection_id, page_size=5, max_seeds=10))
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")
    assert seeds, "expected at least one seed"
    assert all(s.collection == collection_id for s in seeds)
    assert all(s.url for s in seeds)


@pytest.mark.parametrize("collection_id", list(COVID_COLLECTIONS))
def test_cdx_sample(client: ArchiveItClient, collection_id: int) -> None:
    try:
        seeds = list(client.iter_seeds(collection_id, page_size=20, max_seeds=20))
        target = next((s for s in seeds if s.canonical_url or s.url), None)
        assert target is not None
        url = target.canonical_url or target.url
        captures = list(
            client.iter_captures(collection_id, url, match_type="exact", limit=3)
        )
    except httpx.HTTPError as e:
        pytest.skip(f"network/API unavailable: {e}")
    # CDX may legitimately return zero rows for some seeds; just assert shape.
    for cap in captures:
        assert len(cap.timestamp) == 14
        assert cap.url
