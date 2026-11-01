"""Collection storage independent of research phase boundaries.

Legacy partitions are readable for migration; a canonical file always takes
precedence so an interrupted cleanup cannot double the corpus.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

DOCUMENTS = "documents.jsonl"
CAPTURES = "captures.jsonl"


def legacy_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.jsonl")
                  if re.fullmatch(r"P\d+_.+|outside_study|undated", p.stem))


def collection_files(directory: Path, filename=DOCUMENTS) -> list[Path]:
    canonical = directory / filename
    return [canonical] if canonical.is_file() else legacy_files(directory)


def load_records(directory: Path, filename=DOCUMENTS) -> list[dict]:
    records = []
    for path in collection_files(directory, filename):
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    return records


def backup_directory(directory: Path) -> Path | None:
    if not directory.exists() or not any(directory.iterdir()):
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = directory.parent.parent / "collection_backups" / directory.parent.name / directory.name / stamp
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(directory, backup)
    return backup


def remove_legacy(directory: Path):
    for path in legacy_files(directory):
        path.unlink()
        for suffix in (".progress", ".progress.json"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    (directory / "phase_summary.json").unlink(missing_ok=True)


def write_records(path: Path, records):
    """Publish a complete JSONL file by atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".collection-", delete=False) as target:
        temporary = Path(target.name)
        try:
            for row in records:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(paths):
    """Order-sensitive, whitespace-independent record digest and count."""
    digest, count = hashlib.sha256(), 0
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a record object in {path}")
                digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                digest.update(b"\n")
                count += 1
    return dict(records=count, sha256=digest.hexdigest())


def migrate_collection(directory: Path, filename=DOCUMENTS) -> dict:
    """Losslessly consolidate legacy files, verify, back up, then retire them.

    No deduplication, date reinterpretation, model execution or ID changes.
    Published immutable snapshots and historical backups are never traversed.
    """
    sources = legacy_files(directory)
    destination = directory / filename
    if not sources:
        return dict(directory=str(directory), status="unchanged", **(_fingerprint([destination]) if destination.exists() else {}))
    expected = _fingerprint(sources)  # Reject malformed data before any mutation.
    if destination.exists() and _fingerprint([destination]) != expected:
        raise ValueError(f"Conflicting canonical and legacy data in {directory}; refusing to discard either")
    backup = backup_directory(directory)
    if not destination.exists():
        def rows():
            for path in sources:
                with path.open(encoding="utf-8") as source:
                    for line in source:
                        if line.strip():
                            yield json.loads(line)
        write_records(destination, rows())
    if _fingerprint([destination]) != expected:
        raise ValueError(f"Migration verification failed for {destination}; originals retained")
    summary = dict(directory=str(directory), status="migrated", file=filename,
                   sources=[p.name for p in sources], backup=str(backup), **expected)
    (directory / "collection_migration.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    remove_legacy(directory)
    return summary
