"""Normalize a collection into documents.jsonl.

Use --rephase-only to refresh derived date metadata without fetching or extracting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import COVID_COLLECTIONS, PHASES_BY_NAME, PROCESSED_DIR
from ingest import ArchiveItClient
from normalize import normalize_collection
from normalize.corpus import refresh_collection_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    parser.add_argument("phase_name", nargs="?", default="all", choices=list(PHASES_BY_NAME) + ["all"],
                        help="Capture window(s) to fetch; outputs stay in one collection file")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rephase-only", action="store_true",
                        help="Refresh derived dates in the existing collection file, without network or extraction")
    parser.add_argument("--timeout", type=float, default=60.)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.rephase_only:
            path = PROCESSED_DIR / "normalized" / str(args.collection_id)
            summary = refresh_collection_dates(path)
        else:
            phases = None if args.phase_name == "all" else [args.phase_name]
            with ArchiveItClient(timeout=args.timeout) as client:
                path, summary = normalize_collection(client, args.collection_id, phases,
                                                       use_cache=not args.no_cache, max_docs=args.max_docs)
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"Wrote collection corpus to {path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
