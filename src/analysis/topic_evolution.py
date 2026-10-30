"""Retrospective topic evolution helpers for notebook 03.

No encoder downloads or supervised learning. Node IDs denote (window, local
topic), never a persistent identity. Word features use a shared vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from document_time import document_time


@dataclass
class TopicSlices:
    nodes: pd.DataFrame
    words: sparse.csr_matrix
    vocabulary: np.ndarray
    centroids: np.ndarray
    membership: pd.DataFrame
    coverage: pd.DataFrame


def prepare_corpus(df, vectors, *, frequency="M", language="en", min_words=8):
    """Keep metadata and raw embeddings aligned through filtering and sorting."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(df) != len(vectors):
        raise ValueError("df and vectors must be aligned; rerun the corrected loading cell.")
    required = {"point_id", "text", "language", "seed_url"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing payload columns: {sorted(missing)}")
    data = df.copy().reset_index(drop=True)
    data["_vector_row"] = np.arange(len(data))
    data["point_id"] = data.point_id.astype(str)
    data["text"] = data.text.fillna("").astype(str).str.strip()
    times = [document_time(row) for row in data.to_dict("records")]
    data["timestamp"] = pd.to_datetime([t.timestamp for t in times], utc=True).tz_convert(None)
    data["event_date_source"] = [t.source for t in times]
    data["analysis_phase"] = [t.phase for t in times]
    keep = (data.timestamp.notna() & data.text.str.split().str.len().ge(min_words)
            & np.isfinite(vectors).all(axis=1) & (np.linalg.norm(vectors, axis=1) > 0))
    if language is not None:
        keep &= data.language.fillna("").str.lower().eq(language.lower())
    data = data.loc[keep].copy()
    if data.empty:
        raise ValueError("No usable chunks; inspect language, dates and min_words.")
    data["window"] = data.timestamp.dt.to_period(frequency).dt.start_time
    # Repeated archive captures of the same source text should not dominate a slice.
    data["_text_key"] = data.text.str.replace(r"\s+", " ", regex=True)
    data = data.sort_values(["timestamp", "point_id"]).drop_duplicates(
        ["window", "seed_url", "_text_key"]
    )
    aligned = normalize(vectors[data._vector_row.to_numpy()]).astype(np.float32)
    data = data.drop(columns=["_text_key"]).reset_index(drop=True)
    windows = pd.period_range(data.window.min(), data.window.max(), freq=frequency).start_time
    return data, aligned, pd.DatetimeIndex(windows)


def make_topic_model(n_docs, *, min_topic_size=20, seed=42):
    """Fresh BERTopic pipeline; supplied vectors are used without re-embedding."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    if n_docs < max(6, min_topic_size):
        raise ValueError("Too few documents for this min_topic_size.")
    return BERTopic(
        embedding_model=None,
        umap_model=UMAP(n_neighbors=min(15, n_docs - 1), n_components=5,
                       min_dist=0.0, metric="cosine", init="random", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, min_samples=5,
                             metric="euclidean", prediction_data=False),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                         max_features=20_000),
        top_n_words=10, calculate_probabilities=False, verbose=False,
    )


def _node_record(data, ids, window, local_topic, prefix, word_row, vocabulary):
    values = word_row.toarray().ravel()
    top = np.argsort(-values, kind="stable")[:10]
    terms = [str(vocabulary[i]) for i in top if values[i] > 0]
    node_id = f"{prefix}:{window.date()}:{local_topic}"
    row = dict(node_id=node_id, window=pd.Timestamp(window), local_topic=int(local_topic),
               size=len(ids), prevalence=len(ids) / len(data),
               top_words=", ".join(terms), label=" · ".join(terms[:3]))
    return row


def _pack(rows, word_rows, centroids, members, coverage, vocabulary, dim):
    columns = ["node_id", "window", "local_topic", "size", "prevalence", "top_words", "label"]
    nodes = pd.DataFrame(rows, columns=columns)
    nodes["feature_row"] = np.arange(len(nodes))
    words = sparse.vstack(word_rows, format="csr") if rows else sparse.csr_matrix((0, len(vocabulary)))
    return TopicSlices(nodes, words, np.asarray(vocabulary),
                       np.asarray(centroids, dtype=np.float32).reshape(-1, dim),
                       pd.DataFrame(members, columns=["point_id", "node_id", "window", "local_topic"]),
                       pd.DataFrame(coverage))


def fit_topic_slices(data, embeddings, windows, *, min_docs=50, min_topic_size=20, seed=42):
    """Independent local clusters; shared raw term probabilities for matching.

    Vocabulary fitting is retrospective. Local c-TF-IDF values from different
    fitted BERTopic models are deliberately NOT compared.
    """
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=2, max_df=0.95, max_features=20_000)
    counts = vectorizer.fit_transform(data.text)
    vocabulary = vectorizer.get_feature_names_out()
    rows, word_rows, centroids, members, coverage = [], [], [], [], []
    for window in windows:
        ids = np.flatnonzero(data.window.eq(window).to_numpy())
        status = dict(window=window, n_docs=len(ids), n_topics=0, outlier_fraction=np.nan,
                      status="too_small" if len(ids) else "empty", observed=False)
        if len(ids) >= max(min_docs, min_topic_size, 6):
            # BERTopic's c-TF-IDF must fit only terms present in this slice:
            # absent global terms have zero frequency and infinite IDF. The
            # separate counts matrix above still provides shared matching axes.
            model = make_topic_model(len(ids), min_topic_size=min_topic_size, seed=seed)
            labels, _ = model.fit_transform(data.text.iloc[ids].tolist(), embeddings[ids])
            labels = np.asarray(labels)
            status["outlier_fraction"] = float(np.mean(labels == -1))
            for topic in sorted(set(labels) - {-1}):
                assigned = ids[labels == topic]
                word_row = normalize(sparse.csr_matrix(counts[assigned].sum(axis=0)), norm="l1")
                if word_row.nnz == 0:
                    continue
                row = _node_record(data.iloc[ids], assigned, window, topic, "topic", word_row, vocabulary)
                rows.append(row)
                word_rows.append(word_row)
                centroids.append(embeddings[assigned].mean(axis=0))
                members.extend((pid, row["node_id"], window, int(topic))
                               for pid in data.point_id.iloc[assigned])
                status["n_topics"] += 1
            status.update(status="ok" if status["n_topics"] else "no_topics",
                          observed=bool(status["n_topics"]))
        coverage.append(status)
    return _pack(rows, word_rows, centroids, members, coverage, vocabulary, embeddings.shape[1])


def fit_term_fields(data, embeddings, windows, *, min_docs=50, max_terms=1200,
                    min_cooccurrence=3, min_npmi=0.15, min_field_terms=3, seed=42):
    """Phylomemetic baseline: binary chunk co-occurrence -> NPMI -> Louvain.

    Fields partition terms, not documents; a chunk can belong to multiple fields.
    This is inspired by phylomemies, not a reproduction of their matching operator.
    """
    vectorizer = CountVectorizer(stop_words="english", binary=True, min_df=3,
                                 max_df=0.8, max_features=max_terms)
    counts = vectorizer.fit_transform(data.text).astype(np.float64)
    vocabulary = vectorizer.get_feature_names_out()
    rows, word_rows, centroids, members, coverage = [], [], [], [], []
    term_graphs = {}
    for window in windows:
        ids = np.flatnonzero(data.window.eq(window).to_numpy())
        status = dict(window=window, n_docs=len(ids), n_topics=0, outlier_fraction=np.nan,
                      status="too_small" if len(ids) else "empty", observed=False)
        if len(ids) >= min_docs:
            x = counts[ids]
            dfreq = np.asarray(x.sum(axis=0)).ravel()
            co = sparse.triu(x.T @ x, k=1).tocoo()
            graph = nx.Graph()
            for i, j, overlap in zip(co.row, co.col, co.data):
                if overlap < min_cooccurrence or overlap >= len(ids):
                    continue
                p = overlap / len(ids)
                npmi = np.log(p / ((dfreq[i] / len(ids)) * (dfreq[j] / len(ids)))) / -np.log(p)
                if npmi >= min_npmi:
                    graph.add_edge(int(i), int(j), weight=float(npmi), cooccurrence=int(overlap))
            nx.set_node_attributes(graph, {i: str(vocabulary[i]) for i in graph}, "term")
            term_graphs[str(window.date())] = graph
            communities = nx.community.louvain_communities(graph, seed=seed, weight="weight") if graph.number_of_edges() else []
            communities = sorted(communities, key=lambda c: (-len(c), min(c)))
            for topic, community in enumerate(communities):
                terms = sorted(community)
                if len(terms) < min_field_terms:
                    continue
                assigned = ids[np.asarray(x[:, terms].sum(axis=1)).ravel() > 0]
                word_row = sparse.csr_matrix((dfreq[terms], ([0] * len(terms), terms)),
                                              shape=(1, len(vocabulary)))
                word_row = normalize(word_row, norm="l1")
                row = _node_record(data.iloc[ids], assigned, window, topic, "field", word_row, vocabulary)
                rows.append(row)
                word_rows.append(word_row)
                centroids.append(embeddings[assigned].mean(axis=0))
                members.extend((pid, row["node_id"], window, topic) for pid in data.point_id.iloc[assigned])
                status["n_topics"] += 1
            status.update(status="ok" if status["n_topics"] else "no_fields",
                          observed=bool(status["n_topics"]))
        coverage.append(status)
    return _pack(rows, word_rows, centroids, members, coverage, vocabulary, embeddings.shape[1]), term_graphs


EDGE_COLUMNS = ["source", "target", "source_window", "target_window", "available_at",
                "similarity", "js_divergence", "embedding_cosine", "prevalence_delta"]


def evolution_graph(slices, *, threshold=0.45, metric="cosine", shift_threshold=0.35):
    """All qualifying adjacent-slice matches; no forced one-to-one assignment.

    Availability is the END of the destination window, since its full contents
    are required. Birth/death are unknown at boundaries or beside unobserved bins.
    """
    if not 0 < threshold <= 1 or metric not in {"cosine", "weighted_jaccard"}:
        raise ValueError("Use 0 < threshold <= 1 and cosine or weighted_jaccard.")
    nodes = slices.nodes.copy()
    coverage = slices.coverage.sort_values("window").reset_index(drop=True)
    windows = list(coverage.window)
    observed = dict(zip(coverage.window, coverage.observed))
    nodes["available_at"] = nodes.window.map(dict(zip(coverage.window, coverage.window_end)))
    graph = nx.DiGraph(threshold=threshold, metric=metric)
    for row in nodes.to_dict("records"):
        graph.add_node(row["node_id"], **row)
    edges = []
    novelty = {}
    for left, right in zip(windows[:-1], windows[1:]):
        if not (observed[left] and observed[right]):
            continue
        a = nodes.loc[nodes.window.eq(left)]
        b = nodes.loc[nodes.window.eq(right)]
        p = slices.words[a.feature_row.to_numpy()]
        q = slices.words[b.feature_row.to_numpy()]
        if a.empty or b.empty:
            continue
        if metric == "cosine":
            similarities = np.clip(cosine_similarity(p, q), 0, 1)
        else:
            # Only topic/field matrices become dense; never the document-term matrix.
            pp, qq = p.toarray(), q.toarray()
            similarities = np.array([[np.minimum(x, y).sum() / np.maximum(x, y).sum()
                                      for y in qq] for x in pp])
        novelty.update(zip(b.node_id, 1 - similarities.max(axis=0)))
        for i, j in zip(*np.where(similarities >= threshold)):
            source, target = a.iloc[i], b.iloc[j]
            end = coverage.loc[coverage.window.eq(right), "window_end"].iloc[0]
            edge = dict(source=source.node_id, target=target.node_id,
                        source_window=left, target_window=right, available_at=end,
                        similarity=float(similarities[i, j]),
                        js_divergence=float(jensenshannon(p[i].toarray().ravel(), q[j].toarray().ravel(), base=2) ** 2),
                        embedding_cosine=float(cosine_similarity(
                            slices.centroids[[source.feature_row]], slices.centroids[[target.feature_row]])[0, 0]),
                        prevalence_delta=float(target.prevalence - source.prevalence))
            edges.append(edge)
            graph.add_edge(edge["source"], edge["target"], **{k: v for k, v in edge.items() if k not in {"source", "target"}})
    nodes["in_degree"] = [graph.in_degree(n) for n in nodes.node_id]
    nodes["out_degree"] = [graph.out_degree(n) for n in nodes.node_id]
    previous = dict(zip(windows[1:], windows[:-1]))
    following = dict(zip(windows[:-1], windows[1:]))
    nodes["left_censored"] = [not observed.get(previous.get(w), False) for w in nodes.window]
    nodes["right_censored"] = [not observed.get(following.get(w), False) for w in nodes.window]
    nodes["birth"] = (nodes.in_degree == 0).astype("boolean").mask(nodes.left_censored)
    nodes["death"] = (nodes.out_degree == 0).astype("boolean").mask(nodes.right_censored)
    nodes["split"] = nodes.out_degree > 1
    nodes["merge"] = nodes.in_degree > 1
    nodes["novelty"] = nodes.node_id.map(novelty)
    nodes["shift_candidate"] = [any(e["js_divergence"] >= shift_threshold
                                    for _, _, e in graph.in_edges(n, data=True)) for n in nodes.node_id]
    nodes["event"] = [" + ".join(name for name in ["birth", "death", "split", "merge", "shift_candidate"]
                                  if pd.notna(row[name]) and row[name]) or "continuation / censored"
                      for row in nodes.to_dict("records")]
    for row in nodes.to_dict("records"):
        graph.nodes[row["node_id"]].update(row)
    return graph, nodes, pd.DataFrame(edges, columns=EDGE_COLUMNS)


def set_window_ends(slices, frequency):
    slices.coverage["window_end"] = slices.coverage.window.dt.to_period(frequency).dt.end_time
    return slices


def animation_controls(fig, names):
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.1, buttons=[
            dict(label="Play", method="animate", args=[None, {"fromcurrent": True, "frame": {"duration": 650, "redraw": True}, "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]),
        ])],
        sliders=[dict(x=0.18, len=0.82, y=-0.1, steps=[
            dict(label=name, method="animate", args=[[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}])
            for name in names])],
    )


def intertopic_animation(data, embeddings, global_topics, windows, topics_over_time, *, top_n=20):
    """One shared PCA basis; all IDs in all frames prevent stale Plotly bubbles."""
    import plotly.graph_objects as go

    labels = np.asarray(global_topics)
    selected = pd.Series(labels[labels >= 0]).value_counts().head(top_n).index.tolist()
    if not selected:
        return go.Figure().update_layout(title="No non-outlier topics to map")
    reducer = PCA(n_components=2, random_state=42).fit(embeddings)
    coordinates = reducer.transform(embeddings)
    anchors = {t: coordinates[labels == t].mean(axis=0) for t in selected}
    word_lookup = {(int(r.Topic), pd.Timestamp(r.Timestamp)): r.Words for r in topics_over_time.itertuples()}
    frames, all_positions = [], list(anchors.values())
    for window in windows:
        x, y, sizes, texts, hover = [], [], [], [], []
        total = int(data.window.eq(window).sum())
        for topic in selected:
            ids = np.flatnonzero((labels == topic) & data.window.eq(window).to_numpy())
            position = coordinates[ids].mean(axis=0) if len(ids) else anchors[topic]
            all_positions.append(position)
            words = str(word_lookup.get((topic, pd.Timestamp(window)), ""))
            x.append(position[0]); y.append(position[1]); sizes.append(len(ids))
            texts.append(f"T{topic}: {escape(', '.join(words.split(', ')[:3]))}" if len(ids) else "")
            hover.append(f"Topic {topic}<br>{escape(words)}<br>Chunks: {len(ids)}<br>Share: {len(ids) / max(total, 1):.1%}")
        trace = go.Scatter(x=x, y=y, ids=[str(t) for t in selected], mode="markers+text",
                           text=texts, textposition="top center", hovertext=hover, hoverinfo="text",
                           marker=dict(size=sizes, sizemode="area", color=selected, colorscale="Turbo",
                                       cmin=min(selected), cmax=max(selected) + 1, opacity=0.8, line_width=1))
        frames.append(go.Frame(name=str(window.date()), data=[trace],
                               layout=go.Layout(title=f"Intertopic map — {window.date()}")))
    largest = max(max(f.data[0].marker.size) for f in frames)
    for frame in frames:
        frame.data[0].marker.sizeref = 2 * max(largest, 1) / 65**2
    fig = go.Figure(data=frames[0].data, frames=frames)
    positions = np.asarray(all_positions)
    for axis, dim in [("xaxis", 0), ("yaxis", 1)]:
        lo, hi = positions[:, dim].min(), positions[:, dim].max()
        pad = max((hi - lo) * 0.15, 0.02)
        fig.update_layout(**{axis: dict(range=[lo - pad, hi + pad], autorange=False,
                                       title=f"Shared embedding PC{dim + 1}")})
    fig.update_layout(title=frames[0].layout.title, template="plotly_white", height=700,
                      margin=dict(b=120), showlegend=False)
    animation_controls(fig, [f.name for f in frames])
    return fig


def plot_evolution(nodes, edges, coverage, *, title="Topic evolution"):
    """Time flows left to right; edge widths encode similarity, hover shows words."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if nodes.empty:
        return fig.update_layout(title=f"{title}: no fields/topics; inspect coverage")
    positions = {}
    for window, group in nodes.groupby("window", sort=True):
        for rank, row in enumerate(group.sort_values(["size", "node_id"], ascending=[False, True]).itertuples()):
            positions[row.node_id] = (window, rank - (len(group) - 1) / 2)
    for row in edges.itertuples():
        x0, y0 = positions[row.source]; x1, y1 = positions[row.target]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                line=dict(width=1 + 4 * row.similarity, color="rgba(90,100,120,0.35)"),
                                hovertext=f"{row.source} → {row.target}<br>similarity={row.similarity:.3f}<br>JS divergence={row.js_divergence:.3f}",
                                hoverinfo="text", showlegend=False))
    for event, group in nodes.groupby("event", sort=True):
        fig.add_trace(go.Scatter(
            x=[positions[n][0] for n in group.node_id], y=[positions[n][1] for n in group.node_id],
            mode="markers", name=event,
            marker=dict(size=8 + 25 * np.sqrt(group["size"] / nodes["size"].max())),
            hovertext=[f"{escape(r.node_id)}<br>{escape(r.top_words)}<br>{r.event}<br>Chunks: {r.size}<br>Share: {r.prevalence:.1%}"
                       for r in group.itertuples()], hoverinfo="text"))
    for row in coverage.loc[~coverage.observed].itertuples():
        fig.add_vline(x=row.window, line_dash="dot", line_color="lightgrey")
    fig.update_layout(title=f"{title} (time →; vertical rank is for layout only)",
                      xaxis_title="Window start", yaxis=dict(visible=False), template="plotly_white",
                      height=700, hovermode="closest", legend=dict(orientation="h", y=-0.2))
    return fig


def export_evolution(directory, slices, nodes, edges, *, metadata):
    """Portable tables and matrices; row mapping and provenance are explicit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nodes.to_json(directory / "nodes.jsonl", orient="records", lines=True, date_format="iso")
    edges.to_json(directory / "edges.jsonl", orient="records", lines=True, date_format="iso")
    slices.membership.to_json(directory / "membership.jsonl", orient="records", lines=True, date_format="iso")
    slices.coverage.to_json(directory / "coverage.jsonl", orient="records", lines=True, date_format="iso")
    sparse.save_npz(directory / "word_distributions.npz", slices.words)
    np.save(directory / "embedding_centroids.npy", slices.centroids)
    (directory / "vocabulary.json").write_text(json.dumps(slices.vocabulary.tolist(), ensure_ascii=False), encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
