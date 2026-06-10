"""Update existing Qdrant payloads to add publish_date metadata.

Reads original normalized documents, extracts publish_date, and updates
the corresponding chunks in Qdrant without re-embedding.

Uses set_payload to update only metadata, preserving all vectors.

Usage:
    uv run python scripts/_amend_embedding_metadata.py <collection_id> <[phase_name|all]> [--qdrant-url URL]

Examples:
    uv run python scripts/_amend_embedding_metadata.py 13559 P0_outbreak
    uv run python scripts/_amend_embedding_metadata.py 13559 all
    uv run python scripts/_amend_embedding_metadata.py 4887 all --qdrant-url http://localhost:6333
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import PointId
from qdrant_client.http.models import Record

from config import PROCESSED_DIR, PHASES_BY_NAME, COVID_COLLECTIONS

log = logging.getLogger(__name__)

COLLECTION_NAME = "credleaf"
SCROLL_BATCH_SIZE = 500
UPDATE_BATCH_SIZE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update existing Qdrant chunk metadata with publish_date."
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
        "--dry-run",
        action="store_true",
        help="Preview changes without actually updating Qdrant",
    )
    return p.parse_args()


def load_document_metadata(
    deduped_path: Path,
) -> Dict[str, Optional[str]]:
    """Load publish_date from normalized documents, keyed by digest.
    
    Returns dict mapping digest -> publish_date (or None if not available).
    """
    metadata_map: Dict[str, Optional[str]] = {}
    
    if not deduped_path.exists():
        log.warning("Deduped file not found: %s", deduped_path)
        return metadata_map
    
    with deduped_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping invalid JSON at line %d: %s", line_num, e)
                continue
            
            digest = doc.get("digest")
            if not digest:
                log.debug("Skipping document at line %d: no digest", line_num)
                continue
            
            publish_date = doc.get("publish_date")
            metadata_map[digest] = publish_date
    
    log.info("Loaded metadata for %d unique documents", len(metadata_map))
    return metadata_map


def update_qdrant_metadata(
    client: QdrantClient,
    collection_name: str,
    metadata_map: Dict[str, Optional[str]],
    scroll_batch_size: int = SCROLL_BATCH_SIZE,
    dry_run: bool = False,
) -> int:
    """Update existing points in Qdrant with publish_date using set_payload.
    
    Scrolls through all points, groups them by digest, and updates
    all chunks from the same document with the same publish_date.
    
    Uses set_payload which only updates payload fields, preserving vectors.
    
    Returns count of updated points.
    """
    updated_count = 0
    offset: Optional[PointId] = None
    total_scrolled = 0
    skipped_digests: set = set()
    missing_digests: set = set()
    
    while True:
        # Scroll through points in batches
        points: list[Record]
        next_offset: Optional[PointId]
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=scroll_batch_size,
            offset=offset,
            with_payload=["digest", "seed_id", "chunk_idx"],  # Only fetch needed fields
            with_vectors=False,
        )
        
        if not points:
            break
        
        total_scrolled += len(points)
        
        # Group points by digest for batch update
        digest_to_points: Dict[str, list] = {}
        for point in points:
            if point.payload is None:
                continue
            
            digest = point.payload.get("digest")
            if not digest:
                continue
            
            # Check if we have metadata for this digest
            if digest not in metadata_map:
                if digest not in missing_digests:
                    missing_digests.add(digest)
                    log.debug("No metadata found for digest: %s", digest[:20])
                continue
            
            publish_date = metadata_map[digest]
            if publish_date is None:
                if digest not in skipped_digests:
                    skipped_digests.add(digest)
                    log.debug("Skipping digest with no publish_date: %s", digest[:20])
                continue
            
            if digest not in digest_to_points:
                digest_to_points[digest] = []
            digest_to_points[digest].append(point)
        
        # Update payloads in batches
        if not dry_run and digest_to_points:
            digest_items = list(digest_to_points.items())
            for i in range(0, len(digest_items), UPDATE_BATCH_SIZE):
                batch = digest_items[i:i + UPDATE_BATCH_SIZE]
                
                for digest, point_list in batch:
                    publish_date = metadata_map[digest]
                    point_ids = [p.id for p in point_list]
                    
                    try:
                        client.set_payload(
                            collection_name=collection_name,
                            payload={"publish_date": publish_date},
                            points=point_ids,
                        )
                        updated_count += len(point_ids)
                    except Exception as e:
                        log.error(
                            "Failed to update %d points for digest %s: %s",
                            len(point_ids),
                            digest[:20],
                            e,
                        )
                        continue
                
                log.info(
                    "Updated %d points in %d documents (total: %d/%d scrolled)",
                    sum(len(pl) for _, pl in batch),
                    len(batch),
                    updated_count,
                    total_scrolled,
                )
        elif dry_run and digest_to_points:
            for digest, point_list in digest_to_points.items():
                publish_date = metadata_map[digest]
                log.info(
                    "[DRY RUN] Would update %d points with publish_date=%s for digest=%s",
                    len(point_list),
                    publish_date,
                    digest[:30],
                )
            updated_count += sum(len(pl) for pl in digest_to_points.values())
        
        # Check if we've reached the end
        offset = next_offset
        if next_offset is None:
            break
    
    if missing_digests:
        log.warning(
            "No metadata found for %d digests in Qdrant (may be from other collections)",
            len(missing_digests),
        )
    if skipped_digests:
        log.info(
            "Skipped %d digests with no publish_date in source data",
            len(skipped_digests),
        )
    
    log.info(
        "Metadata update complete: %d points updated out of %d scrolled",
        updated_count,
        total_scrolled,
    )
    return updated_count


def check_qdrant_connection(qdrant_url: str) -> QdrantClient:
    """Verify Qdrant connection and collection existence."""
    client = QdrantClient(url=qdrant_url)
    
    try:
        collections = client.get_collections()
        log.info("Connected to Qdrant at %s", qdrant_url)
        
        # Verify collection exists
        collection_names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in collection_names:
            log.error(
                "Collection '%s' not found. Available collections: %s",
                COLLECTION_NAME,
                collection_names,
            )
            sys.exit(2)
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        log.info(
            "Collection '%s' has %d points",
            COLLECTION_NAME,
            collection_info.points_count,
        )
        
        return client
    except Exception as e:
        log.error("Cannot connect to Qdrant at %s: %s", qdrant_url, e)
        log.error("Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(3)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Handle "all" phases recursively
    if args.phase_name == "all":
        total_updated = 0
        for phase_name in PHASES_BY_NAME:
            print(f"\n{'='*60}")
            print(f"Updating metadata for phase {phase_name}")
            print(f"{'='*60}")
            sys.argv[2] = phase_name
            result = main()
            if result != 0:
                return result
        return 0

    # Check Qdrant connection
    client = check_qdrant_connection(args.qdrant_url)

    # Load metadata from deduped file
    deduped_path = (
        PROCESSED_DIR / "deduped" / str(args.collection_id) / f"{args.phase_name}.jsonl"
    )
    
    log.info("Loading document metadata from %s", deduped_path)
    metadata_map = load_document_metadata(deduped_path)
    
    if not metadata_map:
        log.error("No document metadata found in %s", deduped_path)
        return 4
    
    # Count documents with publish_date
    docs_with_date = sum(1 for d in metadata_map.values() if d is not None)
    docs_without_date = len(metadata_map) - docs_with_date
    log.info(
        "Documents with publish_date: %d, without: %d, total: %d",
        docs_with_date,
        docs_without_date,
        len(metadata_map),
    )

    if args.dry_run:
        log.info("DRY RUN MODE - No actual updates will be made")

    # Update Qdrant points
    log.info("Updating Qdrant points with publish_date...")
    updated_count = update_qdrant_metadata(
        client=client,
        collection_name=COLLECTION_NAME,
        metadata_map=metadata_map,
        dry_run=args.dry_run,
    )

    result = {
        "points_updated": updated_count,
        "collection_id": args.collection_id,
        "phase": args.phase_name,
        "qdrant_collection": COLLECTION_NAME,
        "documents_processed": len(metadata_map),
        "documents_with_date": docs_with_date,
        "documents_without_date": docs_without_date,
        "dry_run": args.dry_run,
    }
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print("DRY RUN COMPLETE - No changes were made")
    else:
        print("Success!")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())