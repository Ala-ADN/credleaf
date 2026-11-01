"""Phase-windowed harvest: enumerate seeds, query CDX per phase, dedupe, write JSONL.

Output: one JSONL file per collection at
    {RAW_DIR}/captures/{collection_id}/captures.jsonl

Each line is a serialized `CaptureRecord` (capture + seed + collection context).
Revisits and duplicate page/digest captures are dropped. By default each
canonical URL contributes at most one capture per capture phase; distinct
sources and later phases remain eligible. Non-2xx/non-HTML rows are filtered.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import httpx

from config import RAW_DIR, Phase
from url_utils import canonical_url

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
    captures_skipped_dup_url: int = 0
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
            "captures_skipped_dup_url": self.captures_skipped_dup_url,
            "seed_error_count": len(self.seed_errors),
        }


def _is_html(mimetype: str | None) -> bool:
    if not mimetype:
        return False
    return mimetype.split(";", 1)[0].strip().lower() in HTML_MIMETYPES


def _is_ok_status(status: str | None) -> bool:
    return bool(status) and status[0] == "2"


def _keep_capture(cap, seen_digests, url_counts, max_captures_per_url, stats):
    """Count each canonical page toward the budget only after validity checks."""
    stats.captures_seen += 1
    if not _is_ok_status(cap.status):
        stats.captures_skipped_status += 1
        return False
    if not _is_html(cap.mimetype):
        stats.captures_skipped_mimetype += 1
        return False
    page = canonical_url(cap.url) or cap.url
    key = (page, cap.digest)
    if cap.digest and key in seen_digests:
        stats.captures_skipped_dup_digest += 1
        return False
    if max_captures_per_url and url_counts[page] >= max_captures_per_url:
        stats.captures_skipped_dup_url += 1
        return False
    if cap.digest:
        seen_digests.add(key)
    url_counts[page] += 1
    return True


def _validate_limits(max_seeds, max_captures_per_seed, max_captures_per_url):
    if max_seeds is not None and max_seeds < 1:
        raise ValueError("max_seeds must be positive")
    if max_captures_per_seed is not None and max_captures_per_seed < 1:
        raise ValueError("max_captures_per_seed must be positive")
    if max_captures_per_url < 0:
        raise ValueError("max_captures_per_url must be nonnegative; zero means unlimited")


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
    seen_digests: set[tuple[str, str]],
    max_captures_per_seed: int | None,
    stats: HarvestStats,
    match_type: str = "exact",
    url_counts: Counter | None = None,
    max_captures_per_url: int = 1,
) -> Iterable[CaptureRecord]:
    from_ts, to_ts = phase.cdx_window()
    target = seed.canonical_url or seed.url
    yielded = 0
    if url_counts is None:
        url_counts = Counter()
    for cap in client.iter_captures(
        collection_id, target,
        match_type=match_type, from_ts=from_ts, to_ts=to_ts,
        collapse=None,
    ):
        if not _keep_capture(cap, seen_digests, url_counts, max_captures_per_url, stats):
            continue
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


def _harvest_phase(
    client: ArchiveItClient,
    collection_id: int,
    phase: Phase,
    output_path: Path | None = None,
    include_inactive: bool = True,
    max_seeds: int | None = None,
    max_captures_per_seed: int | None = None,
    match_type: str = "exact",
    max_captures_per_url: int = 1,
) -> tuple[Path, HarvestStats]:
    """Harvest one (collection, phase) into a JSONL file. Returns (path, stats).

    match_type: 'exact' yields only captures of the seed URL itself (typically
    homepages). Use 'host' or 'domain' to harvest articles under each seed's
    site — orders of magnitude more captures, and what you want for content
    analysis.
    max_captures_per_url: default 1 per capture phase; 0 retains changed revisions.
    """
    _validate_limits(max_seeds, max_captures_per_seed, max_captures_per_url)
    if output_path is None:
        output_path = RAW_DIR / "captures" / str(collection_id) / "captures.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = HarvestStats()
    seen_digests: set[tuple[str, str]] = set()
    url_counts = Counter()

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
                    url_counts=url_counts, max_captures_per_url=max_captures_per_url,
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


def _harvest_all_phases(
    client: ArchiveItClient,
    collection_id: int,
    phases: list[Phase],
    output_dir: Path | None = None,
    include_inactive: bool = True,
    max_seeds: int | None = None,
    max_captures_per_seed: int | None = None,
    match_type: str = "exact",
    max_captures_per_url: int = 1,
    progress=None,
) -> tuple[dict[str, Path], dict[str, HarvestStats]]:
    """Walk seeds once and write one collection JSONL file.

    Equivalent to running `harvest_phase` once per phase, but issues a single
    CDX query per seed spanning the union of all phase windows — saving the
    repeated seed pagination and per-phase CDX fetches.
    """
    if not phases:
        raise ValueError("phases must be non-empty")
    _validate_limits(max_seeds, max_captures_per_seed, max_captures_per_url)

    if output_dir is None:
        output_dir = RAW_DIR / "captures" / str(collection_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = [(p, *p.cdx_window()) for p in phases]
    union_from = min(w[1] for w in windows)
    union_to = max(w[2] for w in windows)

    paths: dict[str, Path] = {p.name: output_dir / "captures.jsonl" for p in phases}
    stats_by_phase: dict[str, HarvestStats] = {p.name: HarvestStats() for p in phases}
    seen_by_phase: dict[str, set[tuple[str, str]]] = {p.name: set() for p in phases}
    urls_by_phase = {p.name: Counter() for p in phases}

    log.info(
        "harvest start (multi-phase): collection=%d phases=%d match=%s window=%s..%s -> %s",
        collection_id, len(phases), match_type, union_from, union_to, output_dir,
    )

    handle = (output_dir / "captures.jsonl").open("w", encoding="utf-8")
    completed = 0
    if progress:
        progress(0, max_seeds)
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
                    collapse=None,
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
                    seen = seen_by_phase[phase.name]
                    if not _keep_capture(cap, seen, urls_by_phase[phase.name], max_captures_per_url, stats):
                        continue
                    record = CaptureRecord(
                        collection_id=collection_id,
                        seed_id=seed.id,
                        seed_url=target,
                        phase=phase.name,
                        capture=cap,
                    )
                    handle.write(record.model_dump_json() + "\n")
                    stats.captures_kept += 1
                    yielded_for_seed[phase.name] += 1
            except httpx.HTTPError as e:
                log.warning("seed %s failed: %s", seed.id, e)
                for s in stats_by_phase.values():
                    s.seed_errors.append((seed.id, str(e)))
                continue
            finally:
                completed += 1
                if progress:
                    progress(completed, max_seeds)
            for name, stats in stats_by_phase.items():
                if stats.captures_kept > kept_before[name]:
                    stats.seeds_with_captures += 1
    finally:
        handle.close()

    log.info(
        "harvest done (multi-phase): %s",
        {n: s.as_dict() for n, s in stats_by_phase.items()},
    )
    return paths, stats_by_phase


def _check_coverage(stats: HarvestStats, max_seed_error_fraction: float):
    if not 0 <= max_seed_error_fraction <= 1:
        raise ValueError("max_seed_error_fraction must be between 0 and 1")
    fraction = len(stats.seed_errors) / max(1, stats.seeds_visited)
    if fraction > max_seed_error_fraction:
        raise RuntimeError(f"Harvest rejected: {len(stats.seed_errors)}/{stats.seeds_visited} seeds failed "
                           f"(allowed fraction {max_seed_error_fraction})")


def harvest_phase(client, collection_id, phase, output_path=None, *, max_seed_error_fraction=0., **options):
    """Harvest a window, preserving other windows in canonical collection storage."""
    if output_path is None:
        paths, stats = harvest_all_phases(client, collection_id, [phase],
            max_seed_error_fraction=max_seed_error_fraction, **options)
        return paths[phase.name], stats[phase.name]
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harvest-", dir=output.parent) as temporary:
        staged, stats = _harvest_phase(client, collection_id, phase,
                                       output_path=Path(temporary) / output.name, **options)
        _check_coverage(stats, max_seed_error_fraction)
        staged.replace(output)
    return output, stats


def harvest_all_phases(client, collection_id, phases, output_dir=None, *, max_seed_error_fraction=0., **options):
    """Validate query windows, then atomically publish one capture file.

    Window labels control sampling and provenance only. Unselected windows in
    an existing collection are preserved; a full-study harvest replaces it.
    """
    from corpus_storage import CAPTURES, load_records, write_records, backup_directory, remove_legacy
    from config import PHASES_BY_NAME
    output = output_dir or RAW_DIR / "captures" / str(collection_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harvest-", dir=output.parent) as temporary:
        paths, stats = _harvest_all_phases(client, collection_id, phases, output_dir=Path(temporary), **options)
        for value in stats.values():
            _check_coverage(value, max_seed_error_fraction)
        selected = {p.name for p in phases}
        from document_time import parse_timestamp
        def in_selected_window(row):
            timestamp = parse_timestamp(row["capture"]["timestamp"], capture=True)
            day = timestamp.date().isoformat() if timestamp else None
            return (any(p.start <= day <= p.end for p in phases) if day else row.get("phase") in selected)
        preserved = ([] if selected == set(PHASES_BY_NAME) else
                     [row for row in load_records(output, CAPTURES) if not in_selected_window(row)])
        rows = preserved
        for path in set(paths.values()):
            with path.open(encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source if line.strip())
        backup_directory(output)
        write_records(output / CAPTURES, rows)
        remove_legacy(output)
    return {p.name: output / CAPTURES for p in phases}, stats


def load_jsonl(path: Path) -> Iterable[CaptureRecord]:
    """Read a harvest JSONL file back into CaptureRecord objects."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield CaptureRecord.model_validate(json.loads(line))
