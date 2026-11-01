"""Losslessly consolidate live phase partitions into collection files, offline.

Each directory is backed up and its ordered record hash/count verified before
legacy partitions are retired. Expensive embeddings/geography are preserved;
immutable snapshots, model caches and historical backups are never rewritten.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_DIR
from corpus_storage import CAPTURES, DOCUMENTS, migrate_collection


def migrate(data_dir):
    results = []
    folders = [(data_dir / "raw/captures", CAPTURES)]
    folders += [(data_dir / "processed" / name, DOCUMENTS)
                for name in ("normalized", "deduped", "deduped_v2", "georef")]
    for root, filename in folders:
        for directory in sorted(root.iterdir()) if root.exists() else []:
            if not directory.is_dir() or not directory.name.isdigit():
                continue
            result = migrate_collection(directory, filename)
            if root.name == "normalized" and (directory / DOCUMENTS).is_file():
                # Derive an inventory without modifying any preserved row.
                from collections import Counter
                from corpus_storage import load_records
                from document_time import document_time
                rows = load_records(directory)
                summary = dict(documents=len(rows),
                    counts=dict(Counter(document_time(row).phase for row in rows)),
                    date_sources=dict(Counter(document_time(row).source or "undated" for row in rows)),
                    event_time_policy="valid publish_date, otherwise capture_timestamp; UTC")
                (directory / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    for result in migrate(args.data_dir):
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
