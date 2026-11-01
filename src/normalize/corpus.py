"""Refresh document dates and publish one canonical file per collection."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from config import PHASES_BY_NAME
from document_time import OUTSIDE_STUDY, UNDATED, assign_document_time, document_time, parse_timestamp

from corpus_storage import DOCUMENTS, backup_directory, load_records, remove_legacy, write_records


def write_collection(records: list[dict], directory: Path) -> dict:
    """Preserve every record, sort by date, and back up previous outputs.

    Valid publication dates outside the configured study stay in the corpus;
    they never acquire a study phase from a later capture. Undated rows are also
    retained. The original capture-phase label is explicit in each record.
    """
    assigned = [assign_document_time(record) for record in records]
    collections = {row.get("collection_id") for row in assigned}
    if len(collections) > 1:
        raise ValueError("Collection file must contain only one collection")
    maximum = datetime.max.replace(tzinfo=timezone.utc)
    def sort_key(row):
        return (document_time(row).timestamp or maximum,
                parse_timestamp(row.get("capture_timestamp"), capture=True) or maximum,
                str(row.get("capture_url") or row.get("seed_url") or ""), str(row.get("digest") or ""))
    counts = Counter(row["phase"] for row in assigned)
    sources = Counter(row["event_date_source"] or "undated" for row in assigned)
    summary = dict(
        documents=len(assigned), moved=sum(a.get("phase") != b["phase"] for a, b in zip(records, assigned)),
        event_time_policy="valid publish_date, otherwise capture_timestamp; UTC",
        date_sources=dict(sources), counts={phase: counts[phase] for phase in [*PHASES_BY_NAME, OUTSIDE_STUDY, UNDATED]},
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_directory(directory)
    if backup:
        summary["backup"] = str(backup)
    write_records(directory / DOCUMENTS, sorted(assigned, key=sort_key))
    (directory / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    remove_legacy(directory)
    return summary


def refresh_collection_dates(directory: Path) -> dict:
    records = load_records(directory)
    if not records:
        raise ValueError(f"No normalized records in {directory}")
    return write_collection(records, directory)
