"""Source-tier comparisons in the original cosine embedding space.

Global topics provide a common partition across tiers and periods. All metrics
are descriptive, chunk-weighted and retrospective; source tier is not a truth
label. Missing/small groups remain explicit rather than becoming zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from config import COVID_COLLECTIONS, PANDEMIC_PHASES
from credibility.registry import CredibilityRegistry, normalize_host
from analysis.topic_evolution import prepare_corpus
from document_time import document_time

TIERS = ("low", "authoritative", "mixed")


def load_qdrant_corpus(client, collection_name="credleaf"):
    """Read the entire collection, appending each payload with its dense vector."""
    records, vectors = [], []
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection_name, limit=1000,
                                       offset=offset, with_payload=True, with_vectors=True)
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                if len(vector) != 1:
                    raise ValueError("Expected one dense embedding per point")
                vector = next(iter(vector.values()))
            if vector is None or not isinstance(vector, list) or not vector:
                raise ValueError(f"No usable dense embedding for {point.id}")
            vectors.append(np.asarray(vector, dtype=np.float32))
            records.append(dict(point.payload or {}, point_id=str(point.id)))
        if offset is None:
            break
    if not records:
        raise ValueError(f"No points in {collection_name}")
    return pd.DataFrame(records), np.asarray(vectors, dtype=np.float32)


def prepare_semantic_corpus(data, embeddings, registry: CredibilityRegistry):
    """Resolve current tiers, limit the primary corpus, deduplicate and align.

    Dates use valid publication metadata, otherwise capture time. The payload's
    historical phase label never overrides this shared event-time policy.
    """
    data = data.copy().reset_index(drop=True)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or len(data) != len(embeddings):
        raise ValueError("Metadata and embeddings must be aligned")
    dates = pd.Series(pd.to_datetime([document_time(row).timestamp for row in data.to_dict("records")], utc=True))
    keep = (data.collection_id.isin([13559, 13529])
            & dates.ge(pd.Timestamp(PANDEMIC_PHASES[0].start, tz="UTC"))
            & dates.lt(pd.Timestamp(PANDEMIC_PHASES[-1].end, tz="UTC") + pd.Timedelta(days=1)))
    data = data.loc[keep].reset_index(drop=True)
    embeddings = embeddings[keep.to_numpy()]

    def tier(row):
        result = registry.lookup(row.seed_url)
        collection = COVID_COLLECTIONS.get(row.collection_id)
        return collection.credibility_tier if result == "unknown" and collection else result

    data["credibility_tier"] = [tier(row) for row in data.itertuples()]
    data["source_host"] = data.seed_url.map(normalize_host).fillna("unknown")
    data, embeddings, windows = prepare_corpus(data, embeddings)
    data["analysis_phase"] = ""
    for phase in PANDEMIC_PHASES:
        match = (data.timestamp.ge(pd.Timestamp(phase.start))
                 & data.timestamp.lt(pd.Timestamp(phase.end) + pd.Timedelta(days=1)))
        data.loc[match, "analysis_phase"] = phase.name
    return data, embeddings, windows


def _pairwise_distance(vectors):
    """Mean off-diagonal cosine distance in O(n*d), with unit input rows."""
    n = len(vectors)
    if n < 2:
        return np.nan
    total = vectors.sum(axis=0, dtype=np.float64)
    return float(np.clip(1 - (total @ total - n) / (n * (n - 1)), 0, 2))


def tier_metrics(data, embeddings, labels, *, group_column="window", groups=None, min_chunks=20):
    """Dispersion, topic cohesion/spread and entropy for each period × tier.

    Dispersion includes outliers. Cohesion is mean cosine similarity to the
    normalized centroid of the assigned topic within this tier/period. Spread
    is the unweighted mean pairwise distance of its occupied topic centroids.
    Entropy excludes BERTopic -1 and uses the fixed global number of topics for
    normalization; assigned_fraction exposes the excluded mass.
    """
    if min_chunks < 2:
        raise ValueError("min_chunks must be at least two")
    vectors = normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels)
    if len(data) != len(vectors) or len(data) != len(labels):
        raise ValueError("Data, embeddings and labels must be aligned")
    k = len(set(labels) - {-1})
    groups = list(groups) if groups is not None else sorted(data[group_column].unique())
    rows = []
    for group in groups:
        for tier in TIERS:
            ids = np.flatnonzero((data[group_column].eq(group) & data.credibility_tier.eq(tier)).to_numpy())
            assigned = ids[labels[ids] >= 0]
            topics, counts = np.unique(labels[assigned], return_counts=True)
            row = dict(period=group, tier=tier, n_chunks=len(ids), n_assigned=len(assigned),
                       n_sources=data.source_host.iloc[ids].nunique(), n_topics=len(topics),
                       assigned_fraction=len(assigned) / len(ids) if len(ids) else np.nan,
                       status="ok" if len(ids) >= min_chunks else "too_small" if len(ids) else "empty",
                       dispersion=np.nan, cohesion=np.nan, topic_spread=np.nan,
                       entropy_bits=np.nan, entropy_normalized=np.nan, effective_topics=np.nan)
            if len(ids) >= min_chunks:
                row["dispersion"] = _pairwise_distance(vectors[ids])
                if len(assigned) >= min_chunks:
                    probabilities = counts / counts.sum()
                    entropy = float(-(probabilities * np.log2(probabilities)).sum())
                    row.update(entropy_bits=entropy,
                               entropy_normalized=entropy / np.log2(k) if k > 1 else 0.0,
                               effective_topics=2 ** entropy)
                    centroids, similarities = [], []
                    for topic in topics:
                        x = vectors[assigned[labels[assigned] == topic]]
                        centroid = normalize(x.mean(axis=0, keepdims=True))[0]
                        centroids.append(centroid)
                        similarities.extend(x @ centroid)
                    row["cohesion"] = float(np.mean(similarities))
                    row["topic_spread"] = _pairwise_distance(np.asarray(centroids))
            rows.append(row)
    return pd.DataFrame(rows)


def authoritative_neighbours(data, embeddings, *, min_reference=3, batch_size=256):
    """Exact same-phase nearest neighbours, excluding the query's source host.

    Authoritative queries form a within-tier reference baseline. No cross-time
    fallback is used. Point IDs, candidate counts and missing references are
    exported so every distance can be audited. No 2D projection enters here.
    """
    if min_reference < 1 or batch_size < 1:
        raise ValueError("min_reference and batch_size must be positive")
    vectors = normalize(np.asarray(embeddings, dtype=np.float32))
    if len(data) != len(vectors):
        raise ValueError("Data and embeddings must be aligned")
    rows = []
    for phase in PANDEMIC_PHASES:
        queries = np.flatnonzero(data.analysis_phase.eq(phase.name).to_numpy())
        reference = queries[data.credibility_tier.iloc[queries].eq("authoritative").to_numpy()]
        reference_hosts = data.source_host.iloc[reference].to_numpy()
        for start in range(0, len(queries), batch_size):
            ids = queries[start:start + batch_size]
            scores = vectors[ids] @ vectors[reference].T
            allowed = data.source_host.iloc[ids].to_numpy()[:, None] != reference_hosts[None, :]
            scores[~allowed] = -np.inf
            for i, idx in enumerate(ids):
                candidates = int(allowed[i].sum())
                record = dict(point_id=str(data.point_id.iloc[idx]), phase=phase.name,
                              tier=data.credibility_tier.iloc[idx], n_reference=candidates,
                              reference_point_id=None, cosine_distance=np.nan,
                              status="insufficient_reference")
                if candidates >= min_reference:
                    best = int(np.argmax(scores[i]))
                    record.update(reference_point_id=str(data.point_id.iloc[reference[best]]),
                                  cosine_distance=float(np.clip(1 - scores[i, best], 0, 2)), status="ok")
                rows.append(record)
    return pd.DataFrame(rows, columns=["point_id", "phase", "tier", "n_reference",
                                      "reference_point_id", "cosine_distance", "status"])


@dataclass
class SemanticReport:
    monthly: pd.DataFrame
    phases: pd.DataFrame
    neighbours: pd.DataFrame
    neighbour_summary: pd.DataFrame


def build_semantic_report(data, embeddings, labels, windows, *, min_chunks=20, min_reference=3):
    monthly = tier_metrics(data, embeddings, labels, groups=windows, min_chunks=min_chunks)
    phases = tier_metrics(data, embeddings, labels, group_column="analysis_phase",
                          groups=[p.name for p in PANDEMIC_PHASES], min_chunks=min_chunks)
    neighbours = authoritative_neighbours(data, embeddings, min_reference=min_reference)
    summary = neighbours.groupby(["phase", "tier"]).agg(
        n_queries=("point_id", "size"), n_matched=("cosine_distance", "count"),
        mean_distance=("cosine_distance", "mean"), median_distance=("cosine_distance", "median"),
        p90_distance=("cosine_distance", lambda x: x.quantile(.9)),
    ).reset_index()
    for column in ["mean_distance", "median_distance", "p90_distance"]:
        summary.loc[summary.n_matched < min_chunks, column] = np.nan
    return SemanticReport(monthly, phases, neighbours, summary)


def export_semantic_report(directory, report, *, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["monthly", "phases", "neighbours", "neighbour_summary"]:
        getattr(report, name).to_csv(directory / f"{name}.csv", index=False)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
