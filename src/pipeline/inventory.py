"""Small artifact inventories; never imports the ML stack."""
from collections import Counter
import csv
import json
from pathlib import Path


def describe(path: Path, root: Path) -> dict:
    result = dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path), exists=path.is_file())
    if not result["exists"]:
        return result
    result["bytes"] = path.stat().st_size
    if path.suffix == ".jsonl":
        counts = {key: Counter() for key in ("fetch_status", "ner_status", "language", "event_date_source")}
        records, mentions = 0, 0
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                records += 1
                for key, counter in counts.items():
                    if row.get(key) is not None:
                        counter[str(row[key])] += 1
                mentions += len(row.get("mentions") or [])
        result.update(records=records, counts={key: dict(value.most_common(20)) for key, value in counts.items() if value})
        if mentions:
            result["mentions"] = mentions
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            result["summary"] = {key: item for key, item in value.items() if item is None or isinstance(item, (str, int, float, bool))}
            result["domain_counts"] = {tier: len(value[tier]) for tier in ("low", "authoritative", "mixed") if tier in value}
        elif isinstance(value, list):
            result["records"] = len(value)
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            rows = csv.reader(source)
            next(rows, None)  # Header is not a data record.
            result["records"] = sum(1 for row in rows if row)
    return result

