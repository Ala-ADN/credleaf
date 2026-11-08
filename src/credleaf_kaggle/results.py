"""Validate aligned exports and publish them through the existing artifact contract."""
from contextlib import closing
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid

from pipeline.artifacts import ArtifactStore, file_hash, write_json
from credleaf_kaggle.jobs import job_lock, load_job, save_job


def validate_shard(directory, spec):
    import numpy as np
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    expected = dict(format=1, job_id=spec["job_id"], shard=spec["shard"], model=spec["model"],
        model_revision=spec["model_revision"], input_sha256=spec["files"]["documents.jsonl"],
        chunker_model=spec["chunker_model"], chunker_revision=spec["chunker_revision"],
        source_sha256=spec["files"].get("source.json", spec["files"].get("source.zip")), documents=spec["documents"])
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Downloaded result belongs to different input/code/model/job")
    if set(result.get("files", {})) != {"metadata.jsonl", "embeddings.npy"}:
        raise ValueError("Missing embedding export checksums")
    for name, expected_hash in result["files"].items():
        if (directory / name).is_symlink() or file_hash(directory / name) != expected_hash:
            raise ValueError("Downloaded checksum mismatch: " + name)
    vectors = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if vectors.dtype != np.dtype("float32") or vectors.ndim != 2 or vectors.shape != (result["chunks"], result["dimension"]):
        raise ValueError("Embedding shape/dtype does not match result")
    if result["dimension"] < 1:
        raise ValueError("Invalid embedding dimension")
    for start in range(0, len(vectors), 256):
        block = vectors[start:start + 256]
        if not np.isfinite(block).all() or not np.any(block != 0, axis=1).all():
            raise ValueError("Embedding export contains nonfinite or zero vectors")
    with (directory / "metadata.jsonl").open() as stream:
        count = sum(1 for line in stream if line.strip())
    if count != len(vectors):
        raise ValueError("Metadata/vector row count mismatch")
    return result


def index_rows(job, directory, database):
    """Bound memory; deduplicate stable point IDs across shard boundaries."""
    from embed.embed import stable_chunk_id
    from normalize.dedupe import DEDUPE_VERSION, canonical_url, content_hash, observation_period
    database.execute("CREATE TABLE points (id TEXT PRIMARY KEY, shard INTEGER, row_no INTEGER)")
    database.execute("CREATE TABLE documents (identity TEXT PRIMARY KEY)")
    dimension, total = None, 0
    for shard in job["shards"]:
        folder = directory / f"shard-{shard['index']:05d}"
        spec = shard["spec"]
        for name, expected in spec["files"].items():
            if file_hash(folder / "upload" / name) != expected:
                raise ValueError("Local input changed after submission")
        database.execute("DELETE FROM documents")
        def identity(row):
            return json.dumps([row.get("collection_id"),
                canonical_url(row.get("capture_url") or row.get("seed_url") or ""),
                observation_period(row), row.get("language"), row.get("content_hash")], ensure_ascii=False)
        with (folder / "upload/documents.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                row["content_hash"] = content_hash(row["text"])
                database.execute("INSERT OR IGNORE INTO documents VALUES (?)", (identity(row),))
        result = validate_shard(folder / "download", spec)
        if dimension is not None and result["dimension"] != dimension:
            raise ValueError("Shards have incompatible embedding dimensions")
        dimension = result["dimension"]
        with (folder / "download/metadata.jsonl").open() as source:
            for row_no, line in enumerate(source):
                row = json.loads(line)
                point_id = row.get("point_id")
                if row.get("dedupe_version") != DEDUPE_VERSION or not row.get("text"):
                    raise ValueError("Invalid chunk provenance")
                if point_id != stable_chunk_id("credleaf_snapshot_v1", row, row["text"]):
                    raise ValueError("Chunk point ID does not match its provenance")
                if not database.execute("SELECT 1 FROM documents WHERE identity=?", (identity(row),)).fetchone():
                    raise ValueError("Chunk provenance is outside its input shard")
                database.execute("INSERT OR IGNORE INTO points VALUES (?,?,?)", (point_id, shard["index"], row_no))
        total += result["chunks"]
        database.commit()
    unique = database.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    if unique < 1:
        raise ValueError("No nonempty embeddings were returned")
    return unique, dimension, total - unique


def write_embedding(directory, output, *, qdrant_url="http://localhost:6333", qdrant_prefix="credleaf", batch_size=128):
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", qdrant_prefix) or batch_size < 1:
        raise ValueError("Invalid Qdrant prefix/batch size")
    directory = Path(directory)
    job = load_job(directory)
    if not all(s["status"] == "downloaded" for s in job["shards"]):
        raise ValueError("All shards must be downloaded before import")
    with tempfile.TemporaryDirectory(prefix="index-", dir=directory) as temporary, \
            closing(sqlite3.connect(str(Path(temporary) / "rows.sqlite"))) as database:
        count, dimension, duplicates = index_rows(job, directory, database)
        bytes_needed = count * dimension * 4 + sum((directory / f"shard-{s['index']:05d}/download/metadata.jsonl").stat().st_size for s in job["shards"])
        # A new local Qdrant collection also needs space for vectors, payloads and indexing.
        if shutil.disk_usage(job["root"]).free < bytes_needed * 3:
            raise ValueError("Insufficient local disk for merged artifact plus Qdrant import")
        def produce(out):
            (out / "input").mkdir()
            vectors = np.lib.format.open_memmap(out / "input/embeddings.npy", mode="w+", dtype=np.float32,
                                                shape=(count, dimension))
            name = f"{qdrant_prefix}_kaggle_{job['job_id'][:12]}_{uuid.uuid4().hex[:10]}"
            with closing(QdrantClient(url=qdrant_url, timeout=120)) as client:
                if client.collection_exists(name):
                    raise ValueError("Refusing to overwrite an existing collection")
                client.create_collection(name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
                try:
                    written, points = 0, []
                    with (out / "input/metadata.jsonl").open("w") as target:
                        for shard in job["shards"]:
                            folder = directory / f"shard-{shard['index']:05d}/download"
                            block = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
                            with (folder / "metadata.jsonl").open() as source:
                                for row_no, line in enumerate(source):
                                    row = json.loads(line)
                                    chosen = database.execute("SELECT shard,row_no FROM points WHERE id=?", (row["point_id"],)).fetchone()
                                    if chosen != (shard["index"], row_no):
                                        continue
                                    vectors[written] = block[row_no]
                                    target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                    points.append(PointStruct(id=row["point_id"], vector=block[row_no].tolist(),
                                                              payload={k: v for k, v in row.items() if k != "point_id"}))
                                    written += 1
                                    if len(points) >= batch_size:
                                        client.upsert(name, points=points, wait=True)
                                        points.clear()
                        if points:
                            client.upsert(name, points=points, wait=True)
                    vectors.flush()
                    if written != count or client.count(name, exact=True).count != count:
                        raise ValueError("Imported Qdrant count mismatch")
                    stats = dict(collection=name, qdrant_url=qdrant_url, chunks=count, dimension=dimension,
                        embedding_model=job["options"]["model"], model_revision=job["model_revision"],
                        point_id_namespace="credleaf_snapshot_v1", backend="kaggle", job_id=job["job_id"],
                        shards=len(job["shards"]), cross_shard_duplicates=duplicates,
                        remote_runtimes=[s["result"]["runtime"] for s in job["shards"]],
                        chunker_model=job["chunker_model"], chunker_revision=job["chunker_revision"],
                        source_sha256=job["source"], kaggle_adapter=job["adapter"])
                    write_json(out / "qdrant.json", stats)
                    return stats
                except BaseException:
                    try:
                        client.delete_collection(name)
                    except Exception:
                        logging.exception("Could not remove unpublished collection %s", name)
                    raise
        return produce(Path(output))


def publish(directory, *, qdrant_url="http://localhost:6333", qdrant_prefix="credleaf", batch_size=128):
    directory = Path(directory)
    with job_lock(directory):
        job = load_job(directory)
        if "inputs" in job["options"]:
            raise ValueError("This job belongs to the embed task; resume it in Airflow and publish with publish_corpus")
        store = ArtifactStore(job["root"])
        if job["status"] == "published":
            store.verify_tree(job["published"])
            return job["published"]
        if not all(s["status"] == "downloaded" for s in job["shards"]):
            raise ValueError("All shards must be downloaded before publishing")
        code = dict(source=job["source"], kaggle_adapter=job["adapter"],
                    remote_runtimes=[s["result"]["runtime"] for s in job["shards"]])
        config = dict(corpus_layout="collection-v1", embed_model=job["options"]["model"],
            embed_model_revision=job["model_revision"], embed_backend="kaggle", kaggle_job_id=job["job_id"],
            embed_chunker_model=job["chunker_model"], embed_chunker_revision=job["chunker_revision"],
            embed_batch_size=job["options"]["write_batch_size"], qdrant_url=qdrant_url, qdrant_prefix=qdrant_prefix)
        ref, _ = store.build("embed", None, config, job["inputs"], code,
            lambda out: write_embedding(directory, out, qdrant_url=qdrant_url,
                                        qdrant_prefix=qdrant_prefix, batch_size=batch_size))
        bundle = dict(job["bundle"], artifacts=[r for r in job["bundle"]["artifacts"] if r["stage"] != "embed"] + [ref],
                      implementation=code, embedding_source_snapshot_id=job["options"]["source_snapshot_id"])
        def produce_snapshot(out):
            write_json(out / "snapshot.json", bundle)
            return dict(artifacts=len(bundle["artifacts"]), backend="kaggle", job_id=job["job_id"])
        published, _ = store.build("publish", None, dict(kind="corpus", parameters=config),
            [r for r in bundle["artifacts"] if r.get("id")], code, produce_snapshot)
        job.update(status="published", embedding=ref, published=published)
        save_job(directory, job)
        return published
