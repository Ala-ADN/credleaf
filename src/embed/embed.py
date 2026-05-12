"""Pipeline to chunk normalized documents and write to Qdrant using chonkie.

Chonkie handles:
- Semantic chunking (preserves meaning across splits)
- Embedding generation (local, no API required)
- Qdrant integration (automatic vector storage)

We focus on:
- Loading normalized JSONL
- Enriching chunks with original metadata (collection_id, seed_id, phase, etc.)
- Writing to Qdrant via chonkie's QdrantHandshake
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from chonkie import QdrantHandshake, SemanticChunker

log = logging.getLogger(__name__)


def embed_to_qdrant(
    normalized_path: Path,
    qdrant_url: str,
    collection_name: str,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    show_progress: bool = True,
) -> int:
    """Load normalized documents, chunk semantically, and write to Qdrant.

    Args:
        normalized_path: Path to normalized JSONL (from normalize_phase).
        qdrant_url: Qdrant server URL (e.g., "http://localhost:6333").
        collection_name: Qdrant collection name (created if doesn't exist).
        embedding_model: Sentence-transformer model for embeddings.
        show_progress: Whether to show progress during processing.

    Returns:
        Number of chunks written to Qdrant.
    """
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

    # Initialize semantic chunker
    chunker = SemanticChunker()
    log.info("Initialized SemanticChunker")

    # Load and process normalized documents
    doc_count = 0
    chunk_count = 0

    log.info("Processing normalized documents from %s", normalized_path)
    with normalized_path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            doc_count += 1

            # Skip documents that weren't successfully extracted
            if doc.get("fetch_status") != "ok":
                log.debug("Skipping doc %d (status=%s)", doc_count, doc.get("fetch_status"))
                continue

            text = doc.get("text", "").strip()
            if not text:
                log.debug("Skipping doc %d (empty text)", doc_count)
                continue

            # Semantically chunk the text
            chunks = chunker.chunk(text)
            if not chunks:
                continue

            # Enrich chunks with original document metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Build metadata dict with all provenance info
                metadata = {
                    "collection_id": doc.get("collection_id"),
                    "seed_id": doc.get("seed_id"),
                    "seed_url": doc.get("seed_url"),
                    "phase": doc.get("phase"),
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
                }

                # Update chunk metadata in-place
                # (chonkie's Chunk objects support metadata)
                chunk.metadata = metadata

            # Write chunks to Qdrant (handshake batches automatically)
            handshake.write(chunks)
            chunk_count += len(chunks)

            if show_progress and doc_count % 100 == 0:
                log.info("Processed %d documents, wrote %d chunks", doc_count, chunk_count)

    log.info(
        "Processing complete: %d documents, %d chunks written to Qdrant",
        doc_count,
        chunk_count,
    )
    return chunk_count
