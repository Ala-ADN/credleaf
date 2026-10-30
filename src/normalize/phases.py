"""Partition existing normalized records by publication time, without refetching."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from config import PHASES_BY_NAME
from document_time import OUTSIDE_STUDY, UNDATED, assign_document_time, document_time, parse_timestamp

PHASE_FILES = {*PHASES_BY_NAME, OUTSIDE_STUDY, UNDATED, "P1_first_wave"}


def load_phase_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        if path.stem not in PHASE_FILES:
            continue
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    return records


def write_phase_partitions(records: list[dict], directory: Path) -> dict:
    """Preserve every record, sort within phases, and back up previous outputs.

    Valid publication dates outside the configured study are kept separately;
    they never acquire a study phase from a later capture. Undated rows are also
    retained. The original capture-phase label is explicit in each record.
    """
    assigned = [assign_document_time(record) for record in records]
    collections = {row.get("collection_id") for row in assigned}
    if len(collections) > 1:
        raise ValueError("Phase partitions must contain only one collection")
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
    with tempfile.TemporaryDirectory(prefix=".rephase-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        for phase in [*PHASES_BY_NAME, OUTSIDE_STUDY, UNDATED]:
            rows = sorted((row for row in assigned if row["phase"] == phase), key=sort_key)
            with (staging / f"{phase}.jsonl").open("w", encoding="utf-8") as target:
                for row in rows:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
        if directory.exists() and any(directory.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = directory.parent.parent / "phase_backups" / directory.parent.name / directory.name / stamp
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(directory, backup)
            summary["backup"] = str(backup)
        (staging / "phase_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(directory / path.name)
        legacy = directory / "P1_first_wave.jsonl"
        if legacy.exists():
            legacy.unlink()  # retained in the dated backup, now filed by event date
    return summary


def rephase_normalized(directory: Path) -> dict:
    records = load_phase_records(directory)
    if not records:
        raise ValueError(f"No normalized records in {directory}")
    return write_phase_partitions(records, directory)
