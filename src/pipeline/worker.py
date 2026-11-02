"""Structured stage adapter, executed only in the project's ML interpreter."""
from __future__ import annotations

import argparse
from contextlib import closing
from importlib.metadata import version, PackageNotFoundError
import json
import logging
from pathlib import Path
import subprocess
import sys
import uuid

from config import PHASES_BY_NAME
from pipeline.artifacts import ArtifactStore, file_hash, write_json
from pipeline.timing import Progress


def execute(request):
    stage, c = request["stage"], request["config"]
    root, out = Path(request["root"]), Path(request["output"])
    code_root = Path(request["code_root"])
    cid = request["collection_id"]
    store = ArtifactStore(root)
    refs = request["inputs"]
    paths = [store.payload(ref) for ref in refs]
    if stage == "harvest":
        from ingest import ArchiveItClient, harvest_all_phases

        phases = list(PHASES_BY_NAME.values()) if c["capture_phase"] == "all" else [PHASES_BY_NAME[c["capture_phase"]]]
        with ArchiveItClient(timeout=c["harvest_timeout"]) as client:
            _, stats = harvest_all_phases(client, cid, phases, output_dir=out / "captures" / str(cid),
                include_inactive=not c["harvest_active_only"], max_seeds=c["harvest_max_seeds"],
                max_captures_per_seed=c["harvest_max_captures_per_seed"],
                max_captures_per_url=c["harvest_max_captures_per_url"], match_type=c["harvest_match_type"],
                max_seed_error_fraction=c["harvest_max_seed_error_fraction"],
                progress=Progress(request.get("progress_path"), unit="seeds (budget upper bound)"))
        if not sum(value.captures_kept for value in stats.values()):
            raise ValueError("Harvest quality gate failed: zero captures retained; check CDX query scope and date coverage")
        return {key: value.as_dict() for key, value in stats.items()}
    if stage == "normalize":
        from ingest import ArchiveItClient
        from normalize import normalize_collection

        phases = None  # Capture scope is carried by the input artifact, not filenames.
        with ArchiveItClient(timeout=c["normalize_timeout"]) as client:
            _, stats = normalize_collection(client, cid, phases, raw_root=paths[0] / "captures",
                output_root=out / "normalized", cache_root=root / "data/cache",
                use_cache=c["normalize_use_cache"], max_docs=c["normalize_max_docs"],
                progress=Progress(request.get("progress_path")))
        ok = sum(value["extracted"] for value in stats["normalization"].values())
        if ok < c["normalize_min_ok"]:
            raise ValueError(f"Extraction quality gate failed: {ok} < {c['normalize_min_ok']} successful documents")
        failed = sum(value["fetch_failed"] + value["extract_failed"] for value in stats["normalization"].values())
        total = sum(value["total"] for value in stats["normalization"].values())
        if failed / max(1, total) > c["normalize_max_error_fraction"]:
            raise ValueError(f"Extraction quality gate failed: {failed}/{total} fetch/extraction failures")
        return stats
    if stage == "dedupe":
        from normalize.collection import dedupe_collection

        return dedupe_collection(paths[0] / "normalized" / str(cid), out / "deduped" / str(cid),
            max_docs=c["dedupe_max_docs"], threshold=c["dedupe_threshold"])
    if stage == "credibility":
        from credibility import build_registry
        from ingest import ArchiveItClient

        with ArchiveItClient(timeout=c["credibility_timeout"]) as client:
            registry = build_registry(client, max_seeds_per_collection=c["credibility_max_seeds"])
        if len(registry.low) + len(registry.mixed) < c["credibility_min_seed_domains"]:
            raise ValueError("Registry quality gate failed: insufficient seed domains")
        registry.save(out / "credibility.json")
        return {tier: len(getattr(registry, tier)) for tier in ("low", "mixed", "authoritative")}
    if stage == "georef":
        from credibility import CredibilityRegistry
        from georeferencing import ToponymExtractor, georef_phase
        from georeferencing.gazetteer import Gazetteer

        gazetteer_path = root / "data/cache/geonames/gazetteer.pkl"
        if file_hash(gazetteer_path) != c["gazetteer_sha256"]:
            raise ValueError("Gazetteer changed after planning; trigger a new run")
        registry = CredibilityRegistry.load(paths[1] / "credibility.json")
        gazetteer = Gazetteer.load_pickle(gazetteer_path)
        extractor = None
        stats = {}
        source = paths[0] / "deduped" / str(cid) / "documents.jsonl"
        if source.stat().st_size:
            extractor = ToponymExtractor(model_name=c["georef_model"], device=c["georef_device"])
        _, value = georef_phase(source, extractor, gazetteer, registry,
            output_path=out / "georef" / str(cid) / "documents.jsonl", resume=False,
            max_docs=c["georef_max_docs"], resume_key=c["gazetteer_sha256"])
        stats["collection"] = value.as_dict()
        failed = sum(value["ner_failed"] for value in stats.values())
        emitted = sum(value["ner_failed"] + value["ok"] + value["no_text"] + value["no_toponyms"] for value in stats.values())
        if failed / max(1, emitted) > c["georef_max_error_fraction"]:
            raise ValueError(f"NER quality gate failed: {failed}/{emitted} documents failed")
        return stats
    if stage == "embed":
        return embed_snapshot(request, paths)
    if stage == "semantic":
        model = store.load(refs[0])["config"]["embed_model"]
        collection = json.loads((paths[0] / "qdrant.json").read_text())["collection"]
        argv = [sys.executable, str(code_root / "scripts/analyze_semantics.py"),
            "--collection", collection,
            "--input-dir", str(paths[0] / "input"), "--registry", str(paths[1] / "credibility.json"),
            "--output-dir", str(out), "--gif", str(out / "semantic_comparison.gif"),
            "--collection-ids", *map(str, c["collection_ids"]), "--language", c["language"],
            "--min-words", str(c["min_words"]), "--min-topic-size", str(c["semantic_min_topic_size"]),
            "--min-chunks", str(c["semantic_min_chunks"]), "--seed", str(c["semantic_seed"]), "--embedding-model", model]
        subprocess.run(argv, check=True)
        return json.loads((out / "metadata.json").read_text())
    if stage == "topic":
        import numpy as np
        import pandas as pd
        from threadpoolctl import threadpool_limits
        from credibility import CredibilityRegistry
        from analysis.semantic_analysis import prepare_semantic_corpus
        from analysis.topic_evolution import fit_topic_slices, set_window_ends, evolution_graph, export_evolution

        raw = pd.read_json(paths[0] / "input/metadata.jsonl", lines=True, convert_dates=False)
        vectors = np.load(paths[0] / "input/embeddings.npy", mmap_mode="r")
        data, vectors, windows = prepare_semantic_corpus(raw, vectors,
            CredibilityRegistry.load(paths[1] / "credibility.json"), collection_ids=c["collection_ids"],
            language=c["language"], min_words=c["min_words"])
        with threadpool_limits(limits=4):
            slices = set_window_ends(fit_topic_slices(data, vectors, windows,
                min_docs=c["topic_min_docs"], min_topic_size=c["topic_min_topic_size"], seed=c["topic_seed"]), "M")
            _, nodes, edges = evolution_graph(slices, threshold=c["topic_threshold"])
        metadata = dict(config=c, retained_chunks=len(data), snapshots=len(nodes), edges=len(edges))
        export_evolution(out / "topics", slices, nodes, edges, metadata=metadata)
        data.drop(columns=["_vector_row"], errors="ignore").to_json(out / "chunk_provenance.jsonl",
            orient="records", lines=True, date_format="iso")
        return metadata
    if stage == "animation":
        labels = out / "labels.json"
        write_json(labels, request["resources"]["labels"])
        subprocess.run([sys.executable, str(code_root / "scripts/animate_topics.py"),
            "--input-dir", str(paths[0]), "--gif", str(out / "semantic_evolution.gif"), "--labels", str(labels),
            "--max-topics", str(c["animation_max_topics"]), "--threshold", str(c["animation_threshold"])], check=True)
        return json.loads((out / "semantic_evolution.json").read_text())
    raise ValueError(f"Unsupported worker stage: {stage}")


def embed_snapshot(request, paths):
    """One exclusive collection per attempt; publication references it only after verification."""
    import numpy as np
    from qdrant_client import QdrantClient
    from embed import embed_to_qdrant
    from analysis.semantic_analysis import load_qdrant_corpus

    c, out = request["config"], Path(request["output"])
    name = f"{c['qdrant_prefix']}_{request['artifact_key'][:16]}_{uuid.uuid4().hex[:12]}"
    combined = out / ".input.jsonl"
    with combined.open("w", encoding="utf-8") as target:
        for base, ref in zip(paths, request["inputs"]):
            with (base / "deduped" / str(ref["collection_id"]) / "documents.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
    with closing(QdrantClient(url=c["qdrant_url"], timeout=120)) as client:
        if client.collection_exists(name):
            raise ValueError("Refusing to modify an existing Qdrant collection")
        try:
            count = embed_to_qdrant(combined, c["qdrant_url"], name, embedding_model=c["embed_model"],
                                    batch_size=c["embed_batch_size"], show_progress=True, id_namespace="credleaf_snapshot_v1")
            if count < 1 or client.count(collection_name=name, exact=True).count != count:
                raise ValueError("Embedding count validation failed")
            metadata, vectors = load_qdrant_corpus(client, name)
            if len(metadata) != count or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) == 0).any():
                raise ValueError("Invalid embedding export")
            (out / "input").mkdir()
            metadata.to_json(out / "input/metadata.jsonl", orient="records", lines=True)
            np.save(out / "input/embeddings.npy", vectors)
            stats = dict(collection=name, qdrant_url=c["qdrant_url"], chunks=count, dimension=int(vectors.shape[1]),
                         embedding_model=c["embed_model"], point_id_namespace="credleaf_snapshot_v1",
                         torch=version("torch"), chonkie=version("chonkie"))
            write_json(out / "qdrant.json", stats)
        except BaseException:
            # Only this attempt's new collection can be deleted. A hard worker
            # crash may leave an unreferenced orphan, never a published target.
            try:
                if client.collection_exists(name):
                    client.delete_collection(name)
            except Exception:
                logging.exception("Could not remove unpublished collection %s", name)
            raise
        finally:
            combined.unlink(missing_ok=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    request = json.loads(args.request.read_text())
    try:
        stats = execute(request)
        write_json(args.result, dict(format=1, stage=request["stage"], status="success", stats=stats))
    except BaseException as error:
        write_json(args.result, dict(format=1, stage=request["stage"], status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
