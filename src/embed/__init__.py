"""Local embedding pipeline using chonkie for Qdrant integration.

Chonkie handles:
- Semantic chunking
- Embedding generation (local, no API)
- Automatic Qdrant writes

We provide utilities for:
- GPU detection
- Pipeline orchestration
"""

from .embed import embed_to_qdrant
from .model import detect_device

__all__ = [
    "embed_to_qdrant",
    "detect_device",
]

