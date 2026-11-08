"""Uploaded as a standalone Kaggle script. No credentials or local endpoints."""
from contextlib import closing
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack_sources(bundle, destination):
    """Reconstruct the checksummed JSON bundle without archive auto-extraction."""
    for name, source in json.loads(Path(bundle).read_text()).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError("Invalid source bundle path: " + name)
        path = Path(destination) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.encode("utf-8"))


def main(input_root=Path("/kaggle/input"), output=Path("/kaggle/working")):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    candidates = list(Path(input_root).rglob("job.json"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one Credleaf input shard")
    base = candidates[0].parent
    spec = json.loads(candidates[0].read_text())
    for name, expected in spec["files"].items():
        if sha256(base / name) != expected:
            raise ValueError("Input checksum mismatch: " + name)
    print(f"Validated shard {spec['shard']}: {spec['documents']} documents and source bundle", flush=True)
    print("Installing the embedding runtime", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "chonkie[st,qdrant]==1.6.5", "sentence-transformers==5.5.1",
                    "qdrant-client==1.19.0", "datasketch==1.10.0"], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU assigned; enable GPU and verify Kaggle account/quota")
    print("CUDA device: " + torch.cuda.get_device_name(0), flush=True)
    from importlib.metadata import version
    from unittest.mock import patch
    from chonkie import QdrantHandshake
    from chonkie.embeddings import SentenceTransformerEmbeddings
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from sentence_transformers import SentenceTransformer

    output = Path(output)
    with tempfile.TemporaryDirectory(prefix="credleaf-") as temporary:
        temporary = Path(temporary)
        unpack_sources(base / "source.json", temporary / "source")
        sys.path.insert(0, str(temporary / "source"))
        import embed.embed as embedding
        from embed.export import export_qdrant_corpus
        print("Loading encoder and semantic chunker", flush=True)
        encoder = SentenceTransformer(spec["model"], revision=spec["model_revision"], device="cuda")
        handshake_model = SentenceTransformerEmbeddings(encoder)
        chunker = embedding.SemanticChunker(embedding_model=SentenceTransformerEmbeddings(
            SentenceTransformer(spec["chunker_model"], revision=spec["chunker_revision"], device="cpu")))
        with closing(QdrantClient(path=str(temporary / "qdrant"))) as client:
            handshake = QdrantHandshake(client=client, collection_name="shard", embedding_model=handshake_model)
            written_chunks = 0

            def write_batch(_handshake, chunks, id_namespace=None):
                nonlocal written_chunks
                # Batch the same encoder rather than invoking it once per chunk.
                vectors = encoder.encode([chunk.text for chunk in chunks],
                    batch_size=spec["encode_batch_size"], convert_to_numpy=True, show_progress_bar=False)
                points = [PointStruct(id=embedding.stable_chunk_id(id_namespace, chunk.metadata, chunk.text),
                    vector=vector.tolist(), payload=dict(chunk.metadata, text=chunk.text,
                        start_index=chunk.start_index, end_index=chunk.end_index, token_count=chunk.token_count))
                    for chunk, vector in zip(chunks, vectors)]
                client.upsert(collection_name="shard", points=points, wait=True)
                if written_chunks == 0:
                    print(f"First GPU batch encoded and stored: {len(points)} chunks", flush=True)
                written_chunks += len(points)

            with patch.object(embedding, "QdrantHandshake", return_value=handshake), \
                    patch.object(embedding, "SemanticChunker", return_value=chunker), \
                    patch.object(embedding, "_write_stable_chunks", side_effect=write_batch):
                print(f"Embedding {spec['documents']} documents", flush=True)
                count = embedding.embed_to_qdrant(base / "documents.jsonl", "unused", "shard",
                    embedding_model=spec["model"], batch_size=spec["write_batch_size"],
                    id_namespace="credleaf_snapshot_v1")
            if count:
                print(f"Exporting {count} chunks", flush=True)
                stats = export_qdrant_corpus(client, "shard", output, expected_count=count,
                                             batch_size=spec["write_batch_size"])
            else:
                # Empty eligible shards are legal; the aggregate must be nonempty.
                import numpy as np
                (output / "metadata.jsonl").write_text("")
                np.save(output / "embeddings.npy", np.empty((0, handshake.dimension), dtype=np.float32))
                stats = dict(chunks=0, dimension=handshake.dimension)
    result = dict(format=1, job_id=spec["job_id"], shard=spec["shard"], model=spec["model"],
        model_revision=spec["model_revision"], input_sha256=spec["files"]["documents.jsonl"],
        chunker_model=spec["chunker_model"], chunker_revision=spec["chunker_revision"],
        source_sha256=spec["files"]["source.json"], documents=spec["documents"], **stats,
        runtime={p: version(p) for p in ("torch", "chonkie", "sentence-transformers", "qdrant-client")},
        device=torch.cuda.get_device_name(0),
        files={name: sha256(output / name) for name in ("metadata.jsonl", "embeddings.npy")})
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Completed shard {spec['shard']}: {stats['chunks']} chunks", flush=True)


if __name__ == "__main__":
    main()
