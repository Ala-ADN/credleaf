"""Stream a private Qdrant collection to aligned JSONL and NumPy files."""
import json
from pathlib import Path

import numpy as np


def export_qdrant_corpus(client, collection_name, output_dir, *, expected_count, batch_size=256, control=None):
    """Keep only one scroll page and vector block in RAM; never collect the corpus.

    The caller owns an immutable collection during export. The NumPy header is
    written once, followed by contiguous float32 blocks in metadata row order.
    """
    if batch_size < 1 or expected_count < 1:
        raise ValueError("Export count and batch size must be positive")
    if client.count(collection_name=collection_name, exact=True).count != expected_count:
        raise ValueError("Embedding count changed before export")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, vector_path = output / "metadata.jsonl", output / "embeddings.npy"
    if metadata_path.exists() or vector_path.exists():
        raise ValueError("Refusing to overwrite an existing embedding export")
    count, dimension, offset = 0, None, None
    try:
        with metadata_path.open("x", encoding="utf-8") as metadata, vector_path.open("xb") as vectors:
            while True:
                if control:
                    control.wait(count, expected_count)
                points, next_offset = client.scroll(collection_name=collection_name, limit=batch_size,
                    offset=offset, with_payload=True, with_vectors=True)
                rows = []
                for point in points:
                    vector = point.vector
                    if isinstance(vector, dict):
                        if len(vector) != 1:
                            raise ValueError("Expected one dense vector per point")
                        vector = next(iter(vector.values()))
                    if not isinstance(vector, list) or not vector:
                        raise ValueError(f"No usable dense vector for {point.id}")
                    dimension = dimension or len(vector)
                    if len(vector) != dimension:
                        raise ValueError(f"Inconsistent vector dimension for {point.id}")
                    rows.append(vector)
                if rows:
                    block = np.asarray(rows, dtype=np.float32)
                    if not np.isfinite(block).all() or not np.any(block != 0, axis=1).all():
                        raise ValueError("Embedding export contains nonfinite or zero vectors")
                    if count + len(points) > expected_count:
                        raise ValueError("Embedding count exceeded expected export size")
                    if count == 0:
                        np.lib.format.write_array_header_2_0(vectors, dict(
                            descr=np.dtype(np.float32).str, fortran_order=False,
                            shape=(expected_count, dimension)))
                    block.tofile(vectors)
                    for point in points:
                        metadata.write(json.dumps(dict(point.payload or {}, point_id=str(point.id)),
                                                  ensure_ascii=False, allow_nan=False) + "\n")
                    count += len(points)
                if next_offset is None:
                    break
                if next_offset == offset:
                    raise ValueError("Embedding scroll cursor did not advance")
                offset = next_offset
        if count != expected_count or client.count(collection_name=collection_name, exact=True).count != count:
            raise ValueError(f"Embedding export count mismatch: {count} != {expected_count}")
        return dict(chunks=count, dimension=dimension, export_batch_size=batch_size)
    except BaseException:
        metadata_path.unlink(missing_ok=True)
        vector_path.unlink(missing_ok=True)
        raise
