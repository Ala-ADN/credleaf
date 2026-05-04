"""Run a phase-windowed harvest for one collection.

Usage:
    uv run python scripts/harvest.py <collection_id> [phase_name|all] [--max-seeds N] [--max-captures-per-seed N] [--active-only]

`phase_name` defaults to "all", which walks the seed list once and issues
a single CDX query per seed covering every phase window — much cheaper
than running the script once per phase.

Examples:
    uv run python scripts/harvest.py 13559  # all phases, one pass
    uv run python scripts/harvest.py 13559 P1_global_onset --max-seeds 5 --max-captures-per-seed 10
    uv run python scripts/harvest.py 4887 P3_vaccine_rollout --active-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import COVID_COLLECTIONS, PANDEMIC_PHASES, PHASES_BY_NAME
from ingest import ArchiveItClient, harvest_all_phases, harvest_phase

ALL_PHASES = "all"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument(
        "phase_name",
        nargs="?",
        choices=[*PHASES_BY_NAME, ALL_PHASES],
        default=ALL_PHASES,
        help="Phase to harvest. Default 'all' harvests every phase in a single seed pass.",
    )
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--max-captures-per-seed", type=int, default=None)
    p.add_argument("--active-only", action="store_true",
                   help="Skip seeds where active=False")
    p.add_argument(
        "--match-type",
        choices=["exact", "prefix", "host", "domain"],
        default="exact",
        help="CDX match scope. Use 'host' or 'domain' to harvest articles "
             "under each seed; 'exact' only returns the seed URL itself.",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with ArchiveItClient(timeout=args.timeout) as client:
        if args.phase_name == ALL_PHASES:
            paths, stats_by_phase = harvest_all_phases(
                client,
                args.collection_id,
                PANDEMIC_PHASES,
                include_inactive=not args.active_only,
                max_seeds=args.max_seeds,
                max_captures_per_seed=args.max_captures_per_seed,
                match_type=args.match_type,
            )
            for name, path in paths.items():
                print(f"\nwrote {path}")
                print(json.dumps(stats_by_phase[name].as_dict(), indent=2))
        else:
            phase = PHASES_BY_NAME[args.phase_name]
            path, stats = harvest_phase(
                client,
                args.collection_id,
                phase,
                include_inactive=not args.active_only,
                max_seeds=args.max_seeds,
                max_captures_per_seed=args.max_captures_per_seed,
                match_type=args.match_type,
            )
            print(f"\nwrote {path}")
            print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
