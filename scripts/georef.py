"""Run toponym extraction + gazetteer lookup over normalized articles.

Writes one GeoDoc JSONL per collection at
`data/processed/georef/{collection_id}/documents.jsonl`. Append-only with a
`*.progress` sidecar so reruns resume where they left off.

Usage:
    uv run python scripts/georef.py <collection_id> [all]
        [--max-docs N] [--no-resume] [--device cpu|cuda]

Examples:
    uv run python scripts/georef.py 13559 --max-docs 50
    uv run python scripts/georef.py 13559 all
    uv run python scripts/georef.py 4887 all --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys

from config import COVID_COLLECTIONS, PROCESSED_DIR
from credibility import CredibilityRegistry
from georeferencing import ToponymExtractor, georef_phase
from georeferencing.gazetteer import Gazetteer
from georeferencing.setup_gazetteer import gazetteer_pickle_path

CREDIBILITY_PATH = PROCESSED_DIR / "credibility.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument("phase_name", nargs="?", default="all", choices=["all"], help="Optional legacy all argument; process the entire collection")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="overwrite outputs instead of resuming")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="force device for spaCy (default: auto-detect)")
    p.add_argument("--model", default="en_core_web_trf",
                   help="spaCy model name (default: en_core_web_trf)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return _run_collection(args)


def _run_collection(args: argparse.Namespace, *, shared_extractor: ToponymExtractor | None = None,
                   shared_gazetteer: Gazetteer | None = None,
                   shared_registry: CredibilityRegistry | None = None) -> int:
    deduped_path = (
        PROCESSED_DIR / "deduped" / str(args.collection_id) / "documents.jsonl"
    )
    if not deduped_path.exists():
        print(
            f"missing deduped file: {deduped_path}\n"
            f"run `uv run python scripts/_dedupe.py {args.collection_id}` first.",
            file=sys.stderr,
        )
        return 2

    if not CREDIBILITY_PATH.exists():
        print(
            f"missing credibility registry: {CREDIBILITY_PATH}\n"
            f"run `uv run python scripts/build_credibility.py` first.",
            file=sys.stderr,
        )
        return 3

    registry = shared_registry or CredibilityRegistry.load(CREDIBILITY_PATH)

    pkl = gazetteer_pickle_path()
    if not pkl.exists():
        print(
            f"missing gazetteer pickle: {pkl}\n"
            f"run `uv run python -m georeferencing.setup_gazetteer` first.",
            file=sys.stderr,
        )
        return 4
    gazetteer = shared_gazetteer or Gazetteer.load_pickle(pkl)

    extractor = shared_extractor or ToponymExtractor(model_name=args.model, device=args.device)

    path, stats = georef_phase(
        deduped_path,
        extractor=extractor,
        gazetteer=gazetteer,
        registry=registry,
        max_docs=args.max_docs,
        resume=not args.no_resume,
        resume_key=hashlib.sha256(pkl.read_bytes()).hexdigest(),
    )
    print(f"\nwrote {path}")
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
