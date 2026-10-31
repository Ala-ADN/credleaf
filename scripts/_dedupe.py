"""Deduplicate a collection across phase files, preserving sources and months.

Usage: uv run python scripts/_dedupe.py <collection_id> [--max-docs N]
Outputs phase JSONL files plus duplicate provenance and a summary. Existing
outputs are backed up before replacement; normalized inputs are never modified.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import COVID_COLLECTIONS, PROCESSED_DIR
from normalize.dedupe import THRESHOLD


from normalize.collection import dedupe_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "deduped",
                        help="Output root; collection ID is appended")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        summary = dedupe_collection(PROCESSED_DIR / "normalized" / str(args.collection_id),
                                    args.output_dir / str(args.collection_id),
                                    max_docs=args.max_docs, threshold=args.threshold)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
