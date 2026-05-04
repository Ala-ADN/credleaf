"""Build a `CredibilityRegistry` by aggregating seeds from configured collections.

Each Archive-It collection in `config.COVID_COLLECTIONS` carries a
`credibility_tier`. This builder walks every collection whose tier is `low`
or `mixed`, collects the seed hostnames, and combines them with the curated
`AUTHORITATIVE_DOMAINS` list. After collection it dedupes across sets so each
domain appears in exactly one tier (priority: authoritative > low > mixed).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from config import COVID_COLLECTIONS, ArchiveItCollection
from ingest import ArchiveItClient

from .registry import CredibilityRegistry, normalize_host
from .sources import AUTHORITATIVE_DOMAINS

log = logging.getLogger(__name__)


def _seed_domains(
    client: ArchiveItClient,
    collection_id: int,
    max_seeds: int | None,
) -> set[str]:
    out: set[str] = set()
    yielded = 0
    for seed in client.iter_seeds(collection_id, page_size=100):
        host = normalize_host(seed.canonical_url or seed.url)
        if host:
            out.add(host)
        yielded += 1
        if max_seeds is not None and yielded >= max_seeds:
            break
    return out


def build_registry(
    client: ArchiveItClient,
    collections: Mapping[int, ArchiveItCollection] = COVID_COLLECTIONS,
    extra_authoritative: Iterable[str] | None = None,
    max_seeds_per_collection: int | None = None,
) -> CredibilityRegistry:
    """Aggregate seed hostnames by collection tier and combine with the curated
    authoritative list.

    `max_seeds_per_collection` is a cap to keep tests fast against the 16k-seed
    international collection; leave None for production builds.
    """
    low: set[str] = set()
    mixed: set[str] = set()
    counts: dict[int, int] = {}

    for cid, cfg in collections.items():
        if cfg.credibility_tier == "low":
            target = low
        elif cfg.credibility_tier == "mixed":
            target = mixed
        else:
            # 'authoritative' (or any other tier) is sourced from the curated
            # list, not from collection seeds — skip.
            continue
        n_before = len(target)
        target |= _seed_domains(client, cid, max_seeds_per_collection)
        counts[cid] = len(target) - n_before
        log.info(
            "collection %d (%s, tier=%s): +%d domains",
            cid, cfg.name, cfg.credibility_tier, counts[cid],
        )

    authoritative: set[str] = set(AUTHORITATIVE_DOMAINS)
    if extra_authoritative:
        authoritative |= {d for d in extra_authoritative if d}

    # Dedup across tiers (priority: authoritative > low > mixed). Each domain
    # ends up in exactly one set so set sizes are honest counts.
    low -= authoritative
    mixed -= authoritative
    mixed -= low

    log.info(
        "build_registry: low=%d authoritative=%d mixed=%d",
        len(low), len(authoritative), len(mixed),
    )
    return CredibilityRegistry(low=low, authoritative=authoritative, mixed=mixed)
