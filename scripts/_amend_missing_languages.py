"""
Amend missing languages in existing extracted content
**DEPRECATED** in favor of re-running normalization with the updated extract.detect_language()
"""


from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from config import COVID_COLLECTIONS, PHASES_BY_NAME, PROCESSED_DIR
from normalize.extract import detect_language


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def amend_languages_in_file(jsonl_path: Path) -> tuple[int, int]:
    """Amend missing languages in a JSONL file.
    
    Returns (total_docs, amended_count)
    """
    if not jsonl_path.exists():
        logger.warning(f"File does not exist: {jsonl_path}")
        return 0, 0
    
    amended = 0
    total = 0
    updated_docs = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                total += 1
                doc = json.loads(line)
                
                # Detect language if missing and text exists
                if doc.get("language") is None and doc.get("text"):
                    try:
                        detected_lang = detect_language(doc["text"])
                        doc["language"] = detected_lang
                        amended += 1
                    except Exception as e:
                        logger.warning(f"Failed to detect language for digest {doc.get('digest')}: {e}")
                
                updated_docs.append(doc)
        
        # Write back updated documents
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for doc in updated_docs:
                f.write(json.dumps(doc) + "\n")
        
        logger.info(f"Processed {jsonl_path.name}: {amended}/{total} documents amended")
        return total, amended
        
    except Exception as e:
        logger.error(f"Error processing {jsonl_path}: {e}")
        return total, 0


def main() -> int:
    """Process all normalized JSONL files."""
    normalized_dir = PROCESSED_DIR / "normalized"
    
    if not normalized_dir.exists():
        logger.error(f"Normalized directory does not exist: {normalized_dir}")
        return 1
    
    total_docs = 0
    total_amended = 0
    
    # Iterate through collections
    for collection_id in COVID_COLLECTIONS:
        collection_dir = normalized_dir / str(collection_id)
        
        if not collection_dir.exists():
            logger.info(f"Collection directory does not exist: {collection_dir}")
            continue
        
        logger.info(f"\nProcessing collection {collection_id}...")
        
        # Iterate through phases
        for phase_name in PHASES_BY_NAME:
            jsonl_path = collection_dir / f"{phase_name}.jsonl"
            
            if jsonl_path.exists():
                docs, amended = amend_languages_in_file(jsonl_path)
                total_docs += docs
                total_amended += amended
            else:
                logger.debug(f"Phase file does not exist: {jsonl_path}")
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total documents processed: {total_docs}")
    logger.info(f"Total documents amended: {total_amended}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
