"""Fetch + extract main text for every capture in a harvest JSONL.

Usage:
    uv run python scripts/run_normalize.py <collection_id> <[phase_name|all]> [--max-docs N] [--no-cache]

"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import COVID_COLLECTIONS, PHASES_BY_NAME, RAW_DIR
from ingest import ArchiveItClient
from normalize import normalize_phase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument("phase_name", choices=list(PHASES_BY_NAME) + ["all"])
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.phase_name == "all":
        for p in PHASES_BY_NAME:
            print(f"\n=== Normalizing phase {p} ===")
            sys.argv[2] = p
            if main() != 0:
                return 1
    
    captures_jsonl = (
        RAW_DIR / "captures" / str(args.collection_id) / f"{args.phase_name}.jsonl"
    )
    if not captures_jsonl.exists():
        print(
            f"missing harvest file: {captures_jsonl}\n"
            f"run `uv run python scripts/harvest.py {args.collection_id} "
            f"{args.phase_name}` first.",
            file=sys.stderr,
        )
        return 2
    with ArchiveItClient(timeout=args.timeout) as client:
        path, stats = normalize_phase(
            client,
            captures_jsonl,
            use_cache=not args.no_cache,
            max_docs=args.max_docs,
        )

    print(f"\nwrote {path}")
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
