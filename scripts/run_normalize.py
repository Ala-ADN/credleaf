"""Normalize capture windows and partition by publication date, then capture time.

Use --rephase-only to regroup existing JSONL locally without fetching or extracting.
Raw harvest files remain partitioned by capture windows.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import COVID_COLLECTIONS, PHASES_BY_NAME, PROCESSED_DIR
from ingest import ArchiveItClient
from normalize import normalize_collection
from normalize.phases import rephase_normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    parser.add_argument("phase_name", choices=list(PHASES_BY_NAME) + ["all"],
                        help="Capture window(s) to fetch; outputs are regrouped by event date")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rephase-only", action="store_true",
                        help="Repartition all existing normalized records, without network or extraction")
    parser.add_argument("--timeout", type=float, default=60.)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.rephase_only:
            path = PROCESSED_DIR / "normalized" / str(args.collection_id)
            summary = rephase_normalized(path)
        else:
            phases = list(PHASES_BY_NAME) if args.phase_name == "all" else [args.phase_name]
            with ArchiveItClient(timeout=args.timeout) as client:
                path, summary = normalize_collection(client, args.collection_id, phases,
                                                       use_cache=not args.no_cache, max_docs=args.max_docs)
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"Wrote sorted event-phase partitions to {path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
