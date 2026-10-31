"""Collection-level deduplication shared by CLI and pipeline workers."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from config import PHASES_BY_NAME
from normalize.dedupe import THRESHOLD, deduplicate_documents
from normalize.phases import PHASE_FILES
from document_time import OUTSIDE_STUDY, UNDATED, assign_document_time


def dedupe_collection(normalized_root: Path, output_root: Path, *, max_docs=None, threshold=THRESHOLD):
    if normalized_root.resolve() == output_root.resolve():
        raise ValueError("Deduplication output must not overwrite normalized inputs")
    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be positive")
    files = sorted(p for p in normalized_root.glob("*.jsonl") if p.stem in PHASE_FILES)
    if not files:
        raise ValueError(f"No configured phase JSONL files in {normalized_root}")
    records = []
    for path in files:
        count = 0
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if max_docs is not None and count >= max_docs:
                    break
                records.append(dict(doc=assign_document_time(json.loads(line)), origin=f"{path.name}:{line_number}", file=path.name))
                count += 1
    result = deduplicate_documents(records, threshold=threshold)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dedupe-", dir=output_root.parent) as temporary:
        staging = Path(temporary)
        output_phases = [*PHASES_BY_NAME, OUTSIDE_STUDY, UNDATED]
        for phase_name in output_phases:
            rows = sorted((item for item in result.kept if item["doc"]["phase"] == phase_name),
                          key=lambda item: (item["doc"].get("event_date") or "9999",
                                            str(item["doc"].get("capture_timestamp") or ""),
                                            str(item["doc"].get("capture_url") or "")))
            with (staging / f"{phase_name}.jsonl").open("w", encoding="utf-8") as target:
                for item in rows:
                    target.write(json.dumps(item["doc"], ensure_ascii=False) + "\n")
        (staging / "audit").mkdir()
        with (staging / "audit/duplicates.jsonl").open("w", encoding="utf-8") as target:
            for record in result.audit:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = dict(result.stats, input_directory=str(normalized_root),
                       phases=output_phases,
                       scope="exact: collection + host + observation month + language; near: same page URL too",
                       selection="article extraction, available publication date, most complete text, earliest capture",
                       safeguards="direct representative match; number/negation changes retained; normalized inputs unchanged")
        (staging / "dedupe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if output_root.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = output_root.parent.parent / "dedupe_backups" / output_root.parent.name / output_root.name / stamp
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output_root, backup)
            summary["backup"] = str(backup)
            (staging / "dedupe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        output_root.mkdir(parents=True, exist_ok=True)
        for path in staging.rglob("*"):
            if path.is_file():
                destination = output_root / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
        legacy_alias = output_root / "P1_first_wave.jsonl"
        if legacy_alias.exists():
            legacy_alias.unlink()  # preserved in the dated backup
    return summary

