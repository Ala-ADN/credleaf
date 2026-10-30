"""Build the domain → credibility-tier registry and save it to disk.

Walks every collection in `config.COVID_COLLECTIONS` whose tier is `low` or
`mixed`, then dedupes against the curated `AUTHORITATIVE_DOMAINS` list.

Usage:
    uv run python scripts/build_credibility.py [--output PATH] [--max-seeds N]
    uv run python scripts/build_credibility.py --refresh-authoritative

The refresh mode updates an existing registry locally, preserves custom
authoritative entries and backs up the previous JSON. It makes no API calls.

Default output: data/processed/credibility.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from config import PROCESSED_DIR
from credibility import AUTHORITATIVE_DOMAINS, CredibilityRegistry, build_registry
from ingest import ArchiveItClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "credibility.json",
        help="JSON file to write the registry to.",
    )
    p.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Cap seeds-per-collection (useful for the 16k-seed international "
             "collection during testing). Leave unset for a full build.",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--refresh-authoritative", action="store_true",
        help="Merge the current curated domains into an existing registry without network access.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.refresh_authoritative:
        if not args.output.is_file():
            raise SystemExit(f"No registry at {args.output}; run a full build first.")
        registry = CredibilityRegistry.load(args.output).with_authoritative(AUTHORITATIVE_DOMAINS)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = args.output.with_name(f"{args.output.stem}.before-refresh-{stamp}.json")
        copy2(args.output, backup)
        print(f"backup: {backup}")
    else:
        with ArchiveItClient(timeout=args.timeout) as client:
            registry = build_registry(client, max_seeds_per_collection=args.max_seeds)
    registry.save(args.output)

    print(f"\nwrote {args.output}")
    print(f"  low           : {len(registry.low):>5} domains")
    print(f"  authoritative : {len(registry.authoritative):>5} domains")
    print(f"  mixed         : {len(registry.mixed):>5} domains")
    print(f"  sample low    : {sorted(registry.low)[:5]}")
    print(f"  sample auth   : {sorted(registry.authoritative)[:5]}")
    print(f"  sample mixed  : {sorted(registry.mixed)[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
