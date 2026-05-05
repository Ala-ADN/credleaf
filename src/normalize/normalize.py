"""Drive a harvest JSONL through fetch -> cache -> extract -> NormalizedDoc JSONL."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import CACHE_DIR, PROCESSED_DIR
from ingest import ArchiveItClient, CaptureRecord, load_jsonl

from .extract import (
    Extracted,
    cache_path,
    extract,
    load_cached_html,
    save_cached_html,
)
from .models import FetchStatus, NormalizedDoc

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
    return NormalizedDoc(
        collection_id=rec.collection_id,
        seed_id=rec.seed_id,
        seed_url=rec.seed_url,
        phase=rec.phase,
        capture_timestamp=rec.capture.timestamp,
        capture_url=rec.capture.url,
        digest=rec.capture.digest,
        fetched_at=_now(),
        fetch_status=status,
        error=error,
        word_count=len(text.split()),
        char_count=len(text),
        **content,
    )


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
) -> tuple[Path, NormalizeStats]:
    """Read a harvest JSONL, fetch+extract each capture, write normalized JSONL."""
    records = list(load_jsonl(captures_jsonl))
    if not records:
        log.warning(ValueError(f"no records in {captures_jsonl}"))
        return output_path or Path(), NormalizeStats()

    if output_path is None:
        cid = records[0].collection_id
        phase = records[0].phase
        output_path = PROCESSED_DIR / "normalized" / str(cid) / f"{phase}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = NormalizeStats()
    log.info(
        "normalize start: %d records from %s -> %s",
        len(records), captures_jsonl, output_path,
    )

    with output_path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            if max_docs is not None and i >= max_docs:
                break
            stats.total += 1
            doc = _process(client, rec, cache_root, use_cache, stats)
            fh.write(doc.model_dump_json() + "\n")

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
