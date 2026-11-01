"""Embed normalized documents into Qdrant using chonkie for semantic chunking.

Requires a running Qdrant instance (local or remote).

Usage:
    uv run python scripts/run_embed.py <collection_id> [all] [--qdrant-url URL] [--model MODEL] [--batch-size N]

Qdrant Setup (quick start):
    docker run -p 6333:6333 qdrant/qdrant

Examples:
    uv run python scripts/run_embed.py 13559
    uv run python scripts/run_embed.py 4887 all --qdrant-url http://localhost:6333
    uv run python scripts/run_embed.py 13559 all --model bge-m3
    uv run python scripts/run_embed.py 13559 all --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from config import PROCESSED_DIR, COVID_COLLECTIONS
from embed import embed_to_qdrant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed normalized documents into Qdrant using semantic chunking."
    )
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument("phase_name", nargs="?", default="all", choices=["all"], help="Optional legacy all argument; the entire collection is embedded")
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
        "--collection-name", default="credleaf",
        help="Target collection; use credleaf_dedup_v2 when rebuilding legacy data",
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

    # Load deduped JSONL
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

    # Qdrant collection name: one collection for hnsw index optimization, not per phase"
    collection_name = args.collection_name

    # Embed and write to Qdrant
    chunk_count = embed_to_qdrant(
        deduped_path,
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

