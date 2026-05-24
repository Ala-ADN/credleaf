"""Embed normalized documents into Qdrant using chonkie for semantic chunking.

Requires a running Qdrant instance (local or remote).

Usage:
    uv run python scripts/embed.py <collection_id> <[phase_name|all]> [--qdrant-url URL] [--model MODEL] [--batch-size N]

Qdrant Setup (quick start):
    docker run -p 6333:6333 qdrant/qdrant

Examples:
    uv run python scripts/embed.py 13559 P0_outbreak
    uv run python scripts/embed.py 4887 all --qdrant-url http://localhost:6333
    uv run python scripts/embed.py 13559 all --model bge-m3
    uv run python scripts/embed.py 13559 all --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import PROCESSED_DIR, PHASES_BY_NAME, COVID_COLLECTIONS
from embed import embed_to_qdrant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed normalized documents into Qdrant using semantic chunking."
    )
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument("phase_name", choices=list(PHASES_BY_NAME) + ["all"])
    p.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Embedding model (default: BAAI/bge-m3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of chunks to buffer before writing to Qdrant (default: 256)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Handle "all" phases recursively
    if args.phase_name == "all":
        for phase_name in PHASES_BY_NAME:
            print(f"\n=== Embedding phase {phase_name} ===")
            sys.argv[2] = phase_name
            if main() != 0:
                return 1
        return 0

    # Check Qdrant connection
    try:
        import httpx
        resp = httpx.get(f"{args.qdrant_url}/healthz", timeout=5.0)
        if resp.status_code != 200:
            print(f"Qdrant not responding: {resp.status_code}", file=sys.stderr)
            print(f"Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant", file=sys.stderr)
            return 3
    except Exception as e:
        print(f"Cannot reach Qdrant at {args.qdrant_url}: {e}", file=sys.stderr)
        print(f"Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant", file=sys.stderr)
        return 3

    # Load normalized JSONL
    normalized_path = (
        PROCESSED_DIR / "normalized" / str(args.collection_id) / f"{args.phase_name}.jsonl"
    )
    if not normalized_path.exists():
        print(
            f"missing normalized file: {normalized_path}\n"
            f"run `uv run python scripts/run_normalize.py {args.collection_id} "
            f"{args.phase_name}` first.",
            file=sys.stderr,
        )
        return 2

    # Qdrant collection name: "{collection_id}_{phase_name}"
    collection_name = f"collection_{args.collection_id}_{args.phase_name}"

    # Embed and write to Qdrant
    chunk_count = embed_to_qdrant(
        normalized_path,
        qdrant_url=args.qdrant_url,
        collection_name=collection_name,
        embedding_model=args.model,
        batch_size=args.batch_size,
        show_progress=True,
    )

    print(f"\nSuccess!")
    print(json.dumps({
        "chunks_written": chunk_count,
        "collection_id": args.collection_id,
        "phase": args.phase_name,
        "qdrant_collection": collection_name,
        "model": args.model,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

