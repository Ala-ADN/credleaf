"""Chunk deduplicated documents with Chonkie and upsert stable Qdrant points.

Chonkie supplies semantic chunking and the local encoder/collection setup. This
module owns page/month/content point identities so batch sizes, capture repeats,
and reruns cannot multiply chunks or overwrite another source's provenance.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import uuid

from chonkie import QdrantHandshake, SemanticChunker
from normalize.dedupe import DEDUPE_VERSION, canonical_url, content_hash, observation_period
from document_time import assign_document_time

log = logging.getLogger(__name__)

MAX_QDRANT_PAYLOAD_BYTES = 30 * 1024 * 1024


def stable_chunk_id(collection_name: str, metadata: dict, text: str) -> str:
    """A page/month/content identity, independent of capture, batch, and order."""
    identity = [DEDUPE_VERSION, collection_name, metadata.get("collection_id"),
                canonical_url(metadata.get("capture_url") or metadata.get("seed_url") or ""),
                observation_period(metadata), metadata.get("language"), content_hash(text)]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(identity, ensure_ascii=False)))


def _write_stable_chunks(handshake, chunks, id_namespace=None):
    """Use public encoder/client methods instead of batch-position-based IDs."""
    from qdrant_client.models import PointStruct
    points = []
    for chunk in chunks:
        metadata = chunk.metadata
        points.append(PointStruct(
            id=stable_chunk_id(id_namespace or handshake.collection_name, metadata, chunk.text),
            vector=[float(value) for value in handshake.embedding_model.embed(chunk.text)],
            payload={**metadata, "text": chunk.text,
                     "start_index": chunk.start_index, "end_index": chunk.end_index,
                     "token_count": chunk.token_count},
        ))
    handshake.client.upsert(collection_name=handshake.collection_name, points=points, wait=True)


def embed_to_qdrant(
    normalized_path: Path,
    qdrant_url: str,
    collection_name: str,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 256,
    show_progress: bool = True,
    id_namespace: str | None = None,
    control=None,
) -> int:
    """Load normalized documents, chunk semantically, and write to Qdrant.

    Args:
        normalized_path: Path to normalized JSONL (from normalize_phase).
        qdrant_url: Qdrant server URL (e.g., "http://localhost:6333").
        collection_name: Qdrant collection name (created if doesn't exist).
        embedding_model: Sentence-transformer model for embeddings.
        batch_size: Number of chunks to buffer before writing to Qdrant.
        show_progress: Whether to show progress during processing.

    Returns:
        Number of chunks written to Qdrant.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Initialize Qdrant handshake with embedding model
    handshake = QdrantHandshake(
        url=qdrant_url,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    log.info(
        "Initialized Qdrant connection: %s (collection: %s, model: %s)",
        qdrant_url,
        collection_name,
        embedding_model,
    )

    from qdrant_client.models import FieldCondition, Filter, MatchValue
    legacy = handshake.client.count(
        collection_name=collection_name, exact=True,
        count_filter=Filter(must_not=[FieldCondition(key="dedupe_version", match=MatchValue(value=DEDUPE_VERSION))]),
    ).count
    if legacy:
        raise ValueError(
            f"Collection {collection_name!r} has {legacy} legacy points. Rebuild into a fresh "
            "collection with --collection-name credleaf_dedup_v2; legacy IDs cannot be repaired by upserting."
        )

    # Initialize semantic chunker
    chunker = SemanticChunker()
    log.info("Initialized SemanticChunker")

    # Load and process normalized documents
    doc_count = 0
    chunk_count = 0
    batch_chunks = []
    batch_bytes = 0
    seen_chunks = set()

    def flush_batch() -> None:
        if not batch_chunks:
            return
        _write_stable_chunks(handshake, batch_chunks, id_namespace=id_namespace)
        batch_chunks.clear()
        nonlocal batch_bytes
        batch_bytes = 0

    log.info("Processing normalized documents from %s", normalized_path)
    with normalized_path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            if control:
                if control.requested():
                    flush_batch()
                control.wait(doc_count, None)
            line = line.strip()
            if not line:
                continue

            doc = assign_document_time(json.loads(line))
            doc_count += 1

            # Skip documents that weren't successfully extracted
            if doc.get("fetch_status") != "ok":
                log.debug("Skipping doc %d (status=%s)", doc_count, doc.get("fetch_status"))
                continue

            text = (doc.get("text") or "").strip()
            if not text:
                log.debug("Skipping doc %d (empty text)", doc_count)
                continue

            # Semantically chunk the text
            chunks = chunker.chunk(text)
            if not chunks:
                continue

            document_hash = content_hash(text)
            article_id = canonical_url(doc.get("capture_url") or doc.get("seed_url") or "")
            period = observation_period(doc)

            # Enrich chunks with original document metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Build metadata dict with all provenance info
                metadata = {
                    "collection_id": doc.get("collection_id"),
                    "seed_id": doc.get("seed_id"),
                    "seed_url": doc.get("seed_url"),
                    "phase": doc.get("phase"),
                    "capture_phase": doc.get("capture_phase"),
                    "event_date": doc.get("event_date"),
                    "event_date_source": doc.get("event_date_source"),
                    "capture_timestamp": doc.get("capture_timestamp"),
                    "capture_url": doc.get("capture_url"),
                    "digest": doc.get("digest"),
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                    "title": doc.get("title"),
                    "language": doc.get("language"),
                    "extraction_mode": doc.get("extraction_mode"),
                    "word_count": doc.get("word_count", 0),
                    "char_count": doc.get("char_count", 0),
                    "publish_date": doc.get("publish_date"),
                    "dedupe_version": DEDUPE_VERSION,
                    "article_id": article_id,
                    "content_hash": document_hash,
                    "observation_period": period,
                }

                # Update chunk metadata in-place
                # (chonkie's Chunk objects support metadata)
                chunk.metadata = metadata

            # Buffer chunks and flush when we hit size or payload thresholds
            for chunk in chunks:
                chunk_text = getattr(chunk, "text", "")
                if not chunk_text.strip():
                    continue
                point_id = stable_chunk_id(id_namespace or collection_name, chunk.metadata, chunk_text)
                if point_id in seen_chunks:
                    continue
                chunk_bytes = len(chunk_text.encode("utf-8"))
                if chunk_bytes >= MAX_QDRANT_PAYLOAD_BYTES:
                    log.warning(
                        "Skipping oversized chunk for doc %d (%d bytes)",
                        doc_count,
                        chunk_bytes,
                    )
                    continue
                seen_chunks.add(point_id)
                batch_bytes += chunk_bytes
                batch_chunks.append(chunk)
                chunk_count += 1
                if len(batch_chunks) >= batch_size or batch_bytes >= MAX_QDRANT_PAYLOAD_BYTES:
                    flush_batch()

            if show_progress and doc_count % 100 == 0:
                log.info("Processed %d documents, wrote %d chunks", doc_count, chunk_count)

    flush_batch()

    log.info(
        "Processing complete: %d documents, %d chunks written to Qdrant",
        doc_count,
        chunk_count,
    )
    return chunk_count
