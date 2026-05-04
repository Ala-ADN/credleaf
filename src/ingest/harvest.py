"""Phase-windowed harvest: enumerate seeds, query CDX per phase, dedupe, write JSONL.

Output: one JSONL file per (collection, phase) at
    {RAW_DIR}/captures/{collection_id}/{phase_name}.jsonl

Each line is a serialized `CaptureRecord` (capture + seed + collection context).
Revisits and duplicate-digest captures are dropped so each line is one unique
archived document. Non-2xx and non-HTML rows are filtered.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from config import RAW_DIR, Phase

from .archive_it import ArchiveItClient
from .models import Capture, CaptureRecord, Seed

log = logging.getLogger(__name__)

HTML_MIMETYPES = frozenset({"text/html", "application/xhtml+xml"})


@dataclass
class HarvestStats:
    seeds_visited: int = 0
    seeds_with_captures: int = 0
    captures_seen: int = 0
    captures_kept: int = 0
    captures_skipped_status: int = 0
    captures_skipped_mimetype: int = 0
    captures_skipped_dup_digest: int = 0
    seed_errors: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "seeds_visited": self.seeds_visited,
            "seeds_with_captures": self.seeds_with_captures,
            "captures_seen": self.captures_seen,
            "captures_kept": self.captures_kept,
            "captures_skipped_status": self.captures_skipped_status,
            "captures_skipped_mimetype": self.captures_skipped_mimetype,
            "captures_skipped_dup_digest": self.captures_skipped_dup_digest,
            "seed_error_count": len(self.seed_errors),
        }


def _is_html(mimetype: str | None) -> bool:
    if not mimetype:
        return False
    return mimetype.split(";", 1)[0].strip().lower() in HTML_MIMETYPES


def _is_ok_status(status: str | None) -> bool:
    return bool(status) and status[0] == "2"


def _select_seeds(
    client: ArchiveItClient,
    collection_id: int,
    include_inactive: bool,
    max_seeds: int | None,
) -> Iterable[Seed]:
    yielded = 0
    for seed in client.iter_seeds(collection_id, page_size=100):
        if not include_inactive and not seed.active:
            continue
        yield seed
        yielded += 1
        if max_seeds is not None and yielded >= max_seeds:
            return


def _harvest_seed(
    client: ArchiveItClient,
    collection_id: int,
    seed: Seed,
    phase: Phase,
    seen_digests: set[str],
    max_captures_per_seed: int | None,
    stats: HarvestStats,
    match_type: str = "exact",
) -> Iterable[CaptureRecord]:
    from_ts, to_ts = phase.cdx_window()
    target = seed.canonical_url or seed.url
    yielded = 0
    for cap in client.iter_captures(
        collection_id, target,
        match_type=match_type, from_ts=from_ts, to_ts=to_ts,
        collapse="digest",
    ):
        stats.captures_seen += 1
        if not _is_ok_status(cap.status):
            stats.captures_skipped_status += 1
            continue
        if not _is_html(cap.mimetype):
            stats.captures_skipped_mimetype += 1
            continue
        if cap.digest and cap.digest in seen_digests:
            stats.captures_skipped_dup_digest += 1
            continue
        if cap.digest:
            seen_digests.add(cap.digest)
        yield CaptureRecord(
            collection_id=collection_id,
            seed_id=seed.id,
            seed_url=target,
            phase=phase.name,
            capture=cap,
        )
        yielded += 1
        if max_captures_per_seed is not None and yielded >= max_captures_per_seed:
            return


def harvest_phase(
    client: ArchiveItClient,
    collection_id: int,
    phase: Phase,
    output_path: Path | None = None,
    include_inactive: bool = True,
    max_seeds: int | None = None,
    max_captures_per_seed: int | None = None,
    match_type: str = "exact",
) -> tuple[Path, HarvestStats]:
    """Harvest one (collection, phase) into a JSONL file. Returns (path, stats).

    match_type: 'exact' yields only captures of the seed URL itself (typically
    homepages). Use 'host' or 'domain' to harvest articles under each seed's
    site — orders of magnitude more captures, and what you want for content
    analysis.
    """
    if output_path is None:
        output_path = RAW_DIR / "captures" / str(collection_id) / f"{phase.name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = HarvestStats()
    seen_digests: set[str] = set()

    log.info(
        "harvest start: collection=%d phase=%s match=%s window=%s..%s -> %s",
        collection_id, phase.name, match_type, *phase.cdx_window(), output_path,
    )

    with output_path.open("w", encoding="utf-8") as fh:
        for seed in _select_seeds(client, collection_id, include_inactive, max_seeds):
            stats.seeds_visited += 1
            kept_before = stats.captures_kept
            try:
                for record in _harvest_seed(
                    client, collection_id, seed, phase,
                    seen_digests, max_captures_per_seed, stats,
                    match_type=match_type,
                ):
                    fh.write(record.model_dump_json() + "\n")
                    stats.captures_kept += 1
            except httpx.HTTPError as e:
                log.warning("seed %s failed: %s", seed.id, e)
                stats.seed_errors.append((seed.id, str(e)))
                continue
            if stats.captures_kept > kept_before:
                stats.seeds_with_captures += 1

    log.info("harvest done: %s", stats.as_dict())
    return output_path, stats


def _phase_for_timestamp(timestamp: str, windows: list[tuple[Phase, str, str]]) -> Phase | None:
    for phase, from_ts, to_ts in windows:
        if from_ts <= timestamp <= to_ts:
            return phase
    return None


def harvest_all_phases(
    client: ArchiveItClient,
    collection_id: int,
    phases: list[Phase],
    output_dir: Path | None = None,
    include_inactive: bool = True,
    max_seeds: int | None = None,
    max_captures_per_seed: int | None = None,
    match_type: str = "exact",
) -> tuple[dict[str, Path], dict[str, HarvestStats]]:
    """Walk seeds once and bucket captures into per-phase JSONL files.

    Equivalent to running `harvest_phase` once per phase, but issues a single
    CDX query per seed spanning the union of all phase windows — saving the
    repeated seed pagination and per-phase CDX fetches.
    """
    if not phases:
        raise ValueError("phases must be non-empty")

    if output_dir is None:
        output_dir = RAW_DIR / "captures" / str(collection_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = [(p, *p.cdx_window()) for p in phases]
    union_from = min(w[1] for w in windows)
    union_to = max(w[2] for w in windows)

    paths: dict[str, Path] = {p.name: output_dir / f"{p.name}.jsonl" for p in phases}
    stats_by_phase: dict[str, HarvestStats] = {p.name: HarvestStats() for p in phases}
    seen_by_phase: dict[str, set[str]] = {p.name: set() for p in phases}

    log.info(
        "harvest start (multi-phase): collection=%d phases=%d match=%s window=%s..%s -> %s",
        collection_id, len(phases), match_type, union_from, union_to, output_dir,
    )

    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for seed in _select_seeds(client, collection_id, include_inactive, max_seeds):
            for s in stats_by_phase.values():
                s.seeds_visited += 1
            kept_before = {n: s.captures_kept for n, s in stats_by_phase.items()}
            yielded_for_seed = {p.name: 0 for p in phases}
            target = seed.canonical_url or seed.url
            try:
                for cap in client.iter_captures(
                    collection_id, target,
                    match_type=match_type, from_ts=union_from, to_ts=union_to,
                    collapse="digest",
                ):
                    phase = _phase_for_timestamp(cap.timestamp, windows)
                    if phase is None:
                        continue
                    if (
                        max_captures_per_seed is not None
                        and yielded_for_seed[phase.name] >= max_captures_per_seed
                    ):
                        if all(
                            yielded_for_seed[p.name] >= max_captures_per_seed
                            for p in phases
                        ):
                            break
                        continue
                    stats = stats_by_phase[phase.name]
                    stats.captures_seen += 1
                    if not _is_ok_status(cap.status):
                        stats.captures_skipped_status += 1
                        continue
                    if not _is_html(cap.mimetype):
                        stats.captures_skipped_mimetype += 1
                        continue
                    seen = seen_by_phase[phase.name]
                    if cap.digest and cap.digest in seen:
                        stats.captures_skipped_dup_digest += 1
                        continue
                    if cap.digest:
                        seen.add(cap.digest)
                    record = CaptureRecord(
                        collection_id=collection_id,
                        seed_id=seed.id,
                        seed_url=target,
                        phase=phase.name,
                        capture=cap,
                    )
                    handles[phase.name].write(record.model_dump_json() + "\n")
                    stats.captures_kept += 1
                    yielded_for_seed[phase.name] += 1
            except httpx.HTTPError as e:
                log.warning("seed %s failed: %s", seed.id, e)
                for s in stats_by_phase.values():
                    s.seed_errors.append((seed.id, str(e)))
                continue
            for name, stats in stats_by_phase.items():
                if stats.captures_kept > kept_before[name]:
                    stats.seeds_with_captures += 1
    finally:
        for fh in handles.values():
            fh.close()

    log.info(
        "harvest done (multi-phase): %s",
        {n: s.as_dict() for n, s in stats_by_phase.items()},
    )
    return paths, stats_by_phase


def load_jsonl(path: Path) -> Iterable[CaptureRecord]:
    """Read a harvest JSONL file back into CaptureRecord objects."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield CaptureRecord.model_validate(json.loads(line))
