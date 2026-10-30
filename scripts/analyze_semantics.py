"""Analyze saved Qdrant embeddings and render source-tier comparison plots.

Run from the repository root with ``uv run python scripts/analyze_semantics.py``.
Use --input-dir to reuse a metadata.jsonl + embeddings.npy export without Qdrant.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.semantic_analysis import (
    build_semantic_report, export_semantic_report, load_qdrant_corpus, prepare_semantic_corpus,
)
from analysis.topic_evolution import make_topic_model
from config import PROCESSED_DIR
from credibility.registry import CredibilityRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="credleaf")
    parser.add_argument("--input-dir", type=Path, help="Offline metadata.jsonl + embeddings.npy export")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "semantic_analysis")
    parser.add_argument("--gif", type=Path, default=PROCESSED_DIR / "semantic_analysis/semantic_comparison.gif")
    parser.add_argument("--min-topic-size", type=int, default=50)
    parser.add_argument("--min-chunks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_chunks < 2 or args.min_topic_size < 6:
        parser.error("min-chunks must be >= 2 and min-topic-size >= 6")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        raw = pd.read_json(args.input_dir / "metadata.jsonl", lines=True, convert_dates=False)
        vectors = np.load(args.input_dir / "embeddings.npy", mmap_mode="r")
    else:
        from qdrant_client import QdrantClient
        with QdrantClient(url=args.qdrant_url, timeout=120) as client:
            raw, vectors = load_qdrant_corpus(client, args.collection)
        input_dir = args.output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        raw.to_json(input_dir / "metadata.jsonl", orient="records", lines=True)
        np.save(input_dir / "embeddings.npy", vectors)

    registry_path = PROCESSED_DIR / "credibility.json"
    registry = CredibilityRegistry.load(registry_path)
    data, embeddings, windows = prepare_semantic_corpus(raw, vectors, registry)
    print(f"Retained {len(data):,}/{len(raw):,} chunks; {len(windows)} monthly windows", flush=True)
    print(data.credibility_tier.value_counts().to_string(), flush=True)
    print("Fitting global BERTopic on existing normalized embeddings…", flush=True)
    # A bounded BLAS pool prevents excessive parallel overhead on large hosts.
    with threadpool_limits(limits=4):
        model = make_topic_model(len(data), min_topic_size=args.min_topic_size, seed=args.seed)
        labels, _ = model.fit_transform(data.text.tolist(), embeddings)
        data["global_topic"] = labels
        topic_info = model.get_topic_info()
        topic_info.to_json(args.output_dir / "topics.json", orient="records", indent=2)
        data.to_json(args.output_dir / "chunks.jsonl", orient="records", lines=True, date_format="iso")
        np.save(args.output_dir / "normalized_embeddings.npy", embeddings)
        print(f"Found {(topic_info.Topic >= 0).sum()} topics; outliers: {(data.global_topic == -1).mean():.1%}", flush=True)
        report = build_semantic_report(data, embeddings, labels, windows, min_chunks=args.min_chunks)

    metadata = dict(
        purpose="retrospective descriptive semantic EDA; no truth classification or causal inference",
        collection=args.collection, loaded_chunks=len(raw), retained_chunks=len(data),
        language="en", primary_collections=[13559, 13529], frequency="M", min_words=8,
        deduplication="exact source URL + whitespace-normalized text within month",
        timestamp="valid publish_date in UTC, otherwise capture_timestamp; analysis phase assigned from event time",
        weighting="chunks, not articles or source-balanced observations",
        embedding_dimension=embeddings.shape[1], embedding_model="unverified; CLI default BAAI/bge-m3",
        normalization="L2 raw embeddings; no per-feature standardization",
        topics="one global BERTopic fit across all retained tiers and dates; outlier label -1",
        min_topic_size=args.min_topic_size, min_chunks=args.min_chunks, seed=args.seed,
        entropy="bits among assigned chunks; normalized by log2(global non-outlier topic count)",
        cohesion="chunk-weighted cosine to topic centroid within tier/period",
        topic_spread="unweighted mean pairwise cosine distance of occupied topic centroids",
        dispersion="mean pairwise cosine distance of all retained chunks within tier/period",
        nearest_neighbour="exact same-phase cosine to authoritative chunks, excluding same host; minimum 3 candidates",
        registry_sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(data[["point_id", "text", "timestamp"]].to_json(date_format="iso").encode()).hexdigest(),
        packages={p: version(p) for p in ["bertopic", "hdbscan", "umap-learn", "numpy", "pandas", "scikit-learn", "matplotlib", "pillow"]},
    )
    export_semantic_report(args.output_dir, report, metadata=metadata)
    from analysis.semantic_animation import create_semantic_animation
    from matplotlib.animation import PillowWriter
    import matplotlib.pyplot as plt
    labels_by_topic = {int(row.Topic): ", ".join(row.Representation[:2])
                       for row in topic_info.itertuples() if row.Topic >= 0}
    story = create_semantic_animation(data, embeddings, report, labels_by_topic,
                                      min_chunks=args.min_chunks, seed=args.seed)
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    report.phases.merge(report.neighbour_summary, left_on=["period", "tier"],
                        right_on=["phase", "tier"], how="left").to_csv(args.gif.with_suffix(".csv"), index=False)
    args.gif.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    story.animation.save(args.gif, writer=PillowWriter(fps=2), dpi=90)
    story.render_frame(story.preview_frame)
    story.figure.savefig(args.gif.with_suffix(".png"), dpi=120, facecolor=story.figure.get_facecolor())
    plt.close(story.figure)
    print(f"Saved {args.gif} ({len(story.windows)} frames) and report tables to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
