"""Drive a harvest JSONL through fetch -> cache -> extract -> NormalizedDoc JSONL."""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

import httpx
from ingest.archive_it import ArchiveUnavailable

from config import CACHE_DIR, PROCESSED_DIR, RAW_DIR, PHASES_BY_NAME
from document_time import assign_document_time, phase_for_date, parse_timestamp
from ingest import ArchiveItClient, CaptureRecord, load_jsonl

from .extract import (
    Extracted,
    cache_path,
    extract,
    load_cached_html,
    save_cached_html,
)
from .models import FetchStatus, NormalizedDoc
from .corpus import write_collection
from corpus_storage import CAPTURES, DOCUMENTS, collection_files, load_records, write_records

log = logging.getLogger(__name__)


@dataclass
class NormalizeStats:
    total: int = 0
    fetched: int = 0
    from_cache: int = 0
    extracted: int = 0
    fetch_failed: int = 0
    extract_failed: int = 0
    empty: int = 0
    adopted: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "fetched": self.fetched,
            "from_cache": self.from_cache,
            "extracted": self.extracted,
            "fetch_failed": self.fetch_failed,
            "extract_failed": self.extract_failed,
            "empty": self.empty,
            "error_count": len(self.errors),
            "adopted": self.adopted,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_doc(
    rec: CaptureRecord,
    status: FetchStatus,
    *,
    extracted: Extracted | None = None,
    error: str | None = None,
) -> NormalizedDoc:
    content = extracted.model_dump() if extracted is not None else {}
    text = content.get("text") or ""
    return NormalizedDoc(**assign_document_time(dict(
        collection_id=rec.collection_id,
        seed_id=rec.seed_id,
        seed_url=rec.seed_url,
        phase=rec.phase,
        capture_phase=rec.phase,
        capture_timestamp=rec.capture.timestamp,
        capture_url=rec.capture.url,
        digest=rec.capture.digest,
        fetched_at=_now(),
        fetch_status=status,
        error=error,
        word_count=len(text.split()),
        char_count=len(text),
        **content,
    )))


def _process(
    client: ArchiveItClient,
    rec: CaptureRecord,
    cache_root: Path,
    use_cache: bool,
    stats: NormalizeStats,
) -> NormalizedDoc:
    digest = rec.capture.digest
    cpath = cache_path(cache_root, rec.collection_id, digest) if (use_cache and digest) else None

    html: bytes | None = None
    if cpath is not None:
        html = load_cached_html(cpath)
        if html is not None:
            stats.from_cache += 1

    if html is None:
        try:
            html = client.fetch_capture(
                rec.collection_id, rec.capture.timestamp, rec.capture.url
            )
            stats.fetched += 1
        except ArchiveUnavailable:
            # No row is committed for an outage: retry the same capture on resume.
            raise
        except httpx.HTTPError as e:
            stats.fetch_failed += 1
            stats.errors.append((rec.capture.timestamp, str(e)))
            return _make_doc(rec, "fetch_failed", error=str(e))
        if cpath is not None:
            save_cached_html(cpath, html)

    extracted = extract(html, url=rec.capture.url)
    if extracted is None:
        stats.extract_failed += 1
        return _make_doc(rec, "extract_failed")
    if extracted.is_empty:
        stats.empty += 1
        return _make_doc(rec, "empty", extracted=extracted)

    stats.extracted += 1
    return _make_doc(rec, "ok", extracted=extracted)


def normalize_phase(
    client: ArchiveItClient,
    captures_jsonl: Path,
    output_path: Path | None = None,
    cache_root: Path = CACHE_DIR,
    use_cache: bool = True,
    max_docs: int | None = None,
    progress=None,
    checkpoint_dir=None,
    checkpoint_identity=None,
    control=None,
) -> tuple[Path, NormalizeStats]:
    """Read a harvest JSONL, fetch+extract each capture, write normalized JSONL."""
    records = list(load_jsonl(captures_jsonl))
    total = min(len(records), max_docs) if max_docs is not None else len(records)
    if progress and (checkpoint_dir is None or not records):
        progress(0, total)
    if not records:
        log.warning(ValueError(f"no records in {captures_jsonl}"))
        return output_path or Path(), NormalizeStats()

    if output_path is None:
        cid = records[0].collection_id
        output_path = PROCESSED_DIR / "normalized_batches" / str(cid) / DOCUMENTS
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = NormalizeStats()
    log.info(
        "normalize start: %d records from %s -> %s",
        len(records), captures_jsonl, output_path,
    )

    journal = None
    started, prior_seconds = time.monotonic(), 0.
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            if checkpoint_dir is not None:
                from .checkpoint import DocumentJournal
                journal = DocumentJournal(checkpoint_dir, checkpoint_identity, records[:total])
                resumed, saved, errors, prior_seconds = journal.replay(fh)
                if saved:
                    for name in stats.__dataclass_fields__:
                        if name != "errors" and name in saved:
                            setattr(stats, name, saved[name])
                    stats.errors = errors
                started = time.monotonic()
                if progress and hasattr(progress, "set_context"):
                    progress.set_context(processing_seconds=prior_seconds, resumed_completed=resumed)
                if progress:
                    progress(stats.total, total)
            for i in range(stats.total, total):
                rec = records[i]
                if control:
                    control.wait(stats.total, total, checkpoint_dir)
                stats.total += 1
                doc = _process(client, rec, cache_root, use_cache, stats)
                active = prior_seconds + time.monotonic() - started - (control.paused_seconds if control else 0.)
                if journal:
                    journal.append(i, rec, doc.model_dump(mode="json"), stats.as_dict(), active)
                fh.write(doc.model_dump_json() + "\n")
                if progress:
                    if hasattr(progress, "set_context"):
                        progress.set_context(processing_seconds=active)
                    progress(stats.total, total)
    finally:
        if journal:
            journal.close()

    log.info("normalize done: %s", stats.as_dict())
    return output_path, stats


def load_normalized(path: Path) -> list[NormalizedDoc]:
    docs: list[NormalizedDoc] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            import json
            docs.append(NormalizedDoc.model_validate(json.loads(line)))
    return docs


def normalize_collection(
    client: ArchiveItClient, collection_id: int, capture_phases: list[str] | None = None, *,
    raw_root: Path = RAW_DIR / "captures", output_root: Path = PROCESSED_DIR / "normalized",
    cache_root: Path = CACHE_DIR, use_cache: bool = True, max_docs: int | None = None, progress=None,
    checkpoint_dir=None, checkpoint_identity=None, control=None, max_captures_per_url=0,
) -> tuple[Path, dict]:
    """Normalize a collection; optional capture-window filters are independent of storage.

    Partial updates preserve observations outside the selected capture dates.
    Document caps apply once across the selected collection, never per phase.
    """
    if capture_phases is not None and (not capture_phases or any(phase not in PHASES_BY_NAME for phase in capture_phases)):
        raise ValueError("Select at least one configured capture phase")
    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be positive")
    raw = raw_root / str(collection_id)
    if not collection_files(raw, CAPTURES):
        raise FileNotFoundError(f"Missing harvest data: {raw}")
    selected = set(capture_phases) if capture_phases is not None else None
    def capture_phase(record):
        timestamp = (record.get("capture") or {}).get("timestamp") or record.get("capture_timestamp")
        return (phase_for_date(parse_timestamp(timestamp, capture=True))
                or record.get("capture_phase") or record.get("phase"))
    captures = [r for r in load_records(raw, CAPTURES) if selected is None or capture_phase(r) in selected]
    input_captures = len(captures)
    if max_captures_per_url < 0:
        raise ValueError("max_captures_per_url must be nonnegative")
    if max_captures_per_url:
        from collections import Counter
        from url_utils import canonical_url
        counts = Counter()
        retained = []
        # Earliest captures first, independent of imported file ordering.
        for record in sorted(captures, key=lambda r: (r["capture"]["timestamp"], r["capture"]["url"], str(r["seed_id"]))):
            key = (canonical_url(record["capture"]["url"]) or record["capture"]["url"], capture_phase(record))
            if counts[key] < max_captures_per_url:
                retained.append(record)
                counts[key] += 1
        captures = retained
    output = output_root / str(collection_id)
    records = ([r for r in load_records(output) if capture_phase(r) not in selected]
               if selected is not None and selected != set(PHASES_BY_NAME) else [])
    with tempfile.TemporaryDirectory(prefix="credleaf-normalize-") as temporary:
        source = Path(temporary) / CAPTURES
        write_records(source, captures)
        batch, stats = normalize_phase(client, source, output_path=Path(temporary) / DOCUMENTS,
            cache_root=cache_root, use_cache=use_cache, max_docs=max_docs, progress=progress,
            checkpoint_dir=checkpoint_dir, checkpoint_identity=checkpoint_identity, control=control)
        if batch.is_file():
            with batch.open(encoding="utf-8") as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
    summary = write_collection(records, output)
    summary["normalization"] = {"collection": stats.as_dict()}
    summary["capture_selection"] = dict(input_captures=input_captures, selected_captures=len(captures),
        max_captures_per_url=max_captures_per_url, scope="canonical URL per capture phase", selection="earliest capture")
    return output, summary
