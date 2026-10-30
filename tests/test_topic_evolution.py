"""Scientific invariants for topic evolution (no Qdrant or model download)."""
import json
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from analysis.topic_evolution import (
    TopicSlices, evolution_graph, export_evolution, intertopic_animation,
    plot_evolution, prepare_corpus, set_window_ends,
)


def example_slices():
    windows = pd.date_range("2020-01-01", periods=5, freq="MS")
    # A splits into B/C, which merge into D. X dies, E is born, April is unobserved.
    distributions = [[.5, .5, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
                     [.5, .5, 0], [0, 0, 1], [.5, .5, 0]]
    names = ["A", "X", "B", "C", "D", "E", "F"]
    times = [windows[i] for i in [0, 0, 1, 1, 2, 2, 4]]
    nodes = pd.DataFrame(dict(node_id=names, window=times, feature_row=range(7),
                              local_topic=range(7), size=10, prevalence=.5,
                              label=names, top_words=names))
    coverage = pd.DataFrame(dict(window=windows, observed=[True, True, True, False, True]))
    return set_window_ends(TopicSlices(nodes, sparse.csr_matrix(distributions),
                                      np.array(["a", "b", "c"]), np.array(distributions),
                                      pd.DataFrame(columns=["point_id", "node_id"]), coverage), "M")


def test_many_to_many_events_and_censored_gaps():
    graph, nodes, edges = evolution_graph(example_slices(), threshold=.65)
    nodes = nodes.set_index("node_id")
    assert set(graph.edges) == {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    assert nodes.loc["A", "split"] and nodes.loc["D", "merge"]
    assert nodes.loc["X", "death"] and nodes.loc["E", "birth"]
    assert pd.isna(nodes.loc["A", "birth"])
    assert pd.isna(nodes.loc["D", "death"])
    assert pd.isna(nodes.loc["F", "birth"]) and pd.isna(nodes.loc["F", "death"])
    assert nodes.loc["E", "novelty"] == 1
    assert edges.similarity.between(0, 1).all()
    assert edges.js_divergence.between(0, 1).all()
    assert (edges.available_at > edges.target_window).all()
    assert (edges.source_window < edges.target_window).all()


def test_threshold_sensitivity_and_weighted_overlap():
    slices = example_slices()
    assert evolution_graph(slices, threshold=.9)[2].empty
    # Weighted Jaccard(.5,.5,0 vs 1,0,0) = .5/1.5 = 1/3.
    edges = evolution_graph(slices, threshold=.3, metric="weighted_jaccard")[2]
    assert len(edges) == 4
    np.testing.assert_allclose(edges.similarity, 1 / 3)
    assert evolution_graph(slices, threshold=.4, metric="weighted_jaccard")[2].empty


def test_filter_sort_deduplicate_preserves_embedding_alignment():
    data = pd.DataFrame(dict(
        point_id=["late", "early", "duplicate", "bad-date", "french", "zero"],
        text=["same usable english text"] * 6,
        publish_date=["2020-03-01", "2020-01-01", "2020-01-02", "invalid", "2020-01-01", "2020-01-01"],
        language=["en", "en", "en", "en", "fr", "en"], seed_url="https://example.org",
    ), index=[5, 9, 20, 23, 30, 50])
    embeddings = np.eye(6, dtype=np.float32)
    embeddings[-1] = 0
    cleaned, aligned, windows = prepare_corpus(data, embeddings, min_words=2)
    assert cleaned.point_id.tolist() == ["early", "late"]
    np.testing.assert_array_equal(aligned, embeddings[[1, 0]])
    assert len(windows) == 3  # February is retained as an explicit gap.
    with pytest.raises(ValueError, match="aligned"):
        prepare_corpus(data.iloc[:3], embeddings)


def test_notebook_loader_pagination_and_date_mask(tmp_path):
    path = Path(__file__).resolve().parents[1] / "notebooks" / "03_embeddings_eda.ipynb"
    source = "".join(json.loads(path.read_text())["cells"][4]["source"])
    # Supply a tiny two-page Qdrant response; no server or registry files needed.
    def point(i, date):
        return SimpleNamespace(id=i, vector=[float(i), 1.0], payload={
            "seed_url": "https://example.org", "publish_date": date,
        })
    class Client:
        def scroll(self, **kwargs):
            if kwargs["offset"] is None:
                return [point(1, "2020-01-10"), point(2, "2010-01-01")], 2
            return [point(3, "2020-12-31T23:00:00Z"), point(4, "bad-date")], None

    class Registry:
        def lookup(self, url):
            return "mixed"

    namespace = dict(np=np, pd=pd, client=Client(), CredibilityRegistry=Registry,
                     PROCESSED_DIR=tmp_path, TIER_COLORS={"mixed": "gold"},
                     PANDEMIC_PHASES=[SimpleNamespace(start="2020-01-01", end="2020-12-31")])
    tree = ast.parse(source)
    tree.body = [statement for statement in tree.body if not isinstance(statement, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "notebook loading cell", "exec"), namespace)
    assert namespace["df"].point_id.tolist() == [1, 3]
    np.testing.assert_array_equal(namespace["vectors"], [[1.0, 1.0], [3.0, 1.0]])
    assert namespace["df"].index.tolist() == [0, 1]


def test_animation_has_absent_topics_and_fixed_geometry():
    windows = pd.date_range("2020-01-01", periods=3, freq="MS")
    data = pd.DataFrame({"window": [windows[0], windows[0], windows[2], windows[2]]})
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [.8, .2, 0], [.9, .1, 0]])
    words = pd.DataFrame(dict(Topic=[0, 1, 0], Timestamp=[windows[0], windows[0], windows[2]],
                              Words=["vaccine, safety", "lockdown, schools", "vaccine, booster"]))
    figure = intertopic_animation(data, embeddings, [0, 1, 0, 0], windows, words)
    assert len(figure.frames) == 3
    assert figure.layout.xaxis.autorange is False
    assert figure.layout.yaxis.autorange is False
    assert list(figure.frames[0].data[0].ids) == list(figure.frames[-1].data[0].ids)
    assert list(figure.frames[1].data[0].marker.size) == [0, 0]
    assert list(figure.frames[-1].data[0].marker.size) == [2, 0]
    assert "booster" in figure.frames[-1].data[0].hovertext[0]
    figure.to_json()  # Validate Plotly serialization as well as Python construction.


def test_empty_graph_and_portable_export(tmp_path):
    slices = example_slices()
    graph, nodes, edges = evolution_graph(slices)
    export_evolution(tmp_path, slices, nodes, edges, metadata={"purpose": "test"})
    restored = pd.read_json(tmp_path / "nodes.jsonl", lines=True)
    assert restored.node_id.tolist() == nodes.node_id.tolist()
    np.testing.assert_allclose(sparse.load_npz(tmp_path / "word_distributions.npz").toarray(), slices.words.toarray())
    assert json.loads((tmp_path / "vocabulary.json").read_text()) == ["a", "b", "c"]
    plot_evolution(nodes, edges, slices.coverage).to_json()
    slices.nodes = slices.nodes.iloc[:0]
    slices.words = slices.words[:0]
    slices.centroids = slices.centroids[:0]
    slices.coverage["observed"] = False
    graph, nodes, edges = evolution_graph(slices)
    assert not len(graph) and nodes.empty and edges.empty
    plot_evolution(nodes, edges, slices.coverage).to_json()


def test_local_topic_vocabulary_handles_terms_absent_from_other_months(monkeypatch):
    """Real c-TF-IDF must stay finite when monthly vocabularies are disjoint."""
    pytest.importorskip("bertopic")
    from bertopic.dimensionality import BaseDimensionalityReduction
    from sklearn.cluster import KMeans
    import analysis.topic_evolution as module

    original = module.make_topic_model
    def model_without_manifold(n_docs, **kwargs):
        model = original(n_docs, **kwargs)
        model.umap_model = BaseDimensionalityReduction()
        model.hdbscan_model = KMeans(n_clusters=2, n_init=1, random_state=42)
        return model
    monkeypatch.setattr(module, "make_topic_model", model_without_manifold)
    windows = pd.date_range("2020-01-01", periods=2, freq="MS")
    texts = ["vaccine clinical immune antibody", "hospital patient medical treatment",
             "school teacher classroom student", "economy inflation market finance"]
    rows, vectors = [], []
    for i, text in enumerate(texts):
        for j in range(6):
            rows.append(dict(point_id=f"{i}-{j}", window=windows[i // 2], text=text))
            vectors.append(np.eye(4)[i])
    slices = module.fit_topic_slices(pd.DataFrame(rows), np.asarray(vectors), windows,
                                     min_docs=6, min_topic_size=6)
    assert slices.coverage.observed.all()
    assert slices.coverage.n_topics.tolist() == [2, 2]
    assert np.isfinite(slices.words.data).all()
    np.testing.assert_allclose(slices.words.sum(axis=1), 1)
    # Both months still occupy the same exported vocabulary, with no overlap.
    left = np.asarray(slices.words[:2].sum(axis=0)).ravel()
    right = np.asarray(slices.words[2:].sum(axis=0)).ravel()
    assert not np.any((left > 0) & (right > 0))


def test_notebook_topic_cells_execute_offline(tmp_path, monkeypatch):
    """Execute the actual added cells with separable synthetic text/embeddings."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    pytest.importorskip("bertopic")
    import plotly.io as pio

    monkeypatch.setattr(pio, "show", lambda *args, **kwargs: None)
    terms = ["vaccine injection immunity antibody clinical trial booster dose patient treatment",
             "school classroom teacher education student learning homework lesson campus course",
             "economy business trade bank inflation market finance currency investment budget"]
    rng = np.random.default_rng(42)
    rows, vectors = [], []
    for month in range(1, 4):
        for topic, text in enumerate(terms):
            for i in range(30):
                rows.append(dict(point_id=f"{month}-{topic}-{i}", text=f"{text} record{i}",
                                 publish_date=f"2020-{month:02d}-10", language="en",
                                 seed_url=f"https://example.org/{topic}", credibility_tier="mixed"))
                vector = rng.normal(0, .015, 12)
                vector[topic] += 1
                vectors.append(vector)
    namespace = dict(Path=Path, np=np, pd=pd, df=pd.DataFrame(rows),
                     vectors=np.asarray(vectors), COLLECTION_NAME="synthetic")
    path = Path(__file__).resolve().parents[1] / "notebooks" / "03_embeddings_eda.ipynb"
    notebook = json.loads(path.read_text())
    for i, cell in enumerate(notebook["cells"][16:], start=16):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"notebook cell {i}", "exec"), namespace)
            if "TOPIC_CONFIG =" in source:
                namespace["OUT_DIR"] = tmp_path
    assert namespace["n_global_topics"] >= 2
    assert len(namespace["topic_graph"]) >= 6
    assert len(namespace["field_graph"]) >= 6
    assert namespace["topic_slices"].coverage.observed.all()
    assert namespace["field_slices"].coverage.observed.all()
    assert namespace["fig_hierarchy"] is not None
    assert (tmp_path / "intertopic.html").exists()
    assert (tmp_path / "topics" / "embedding_centroids.npy").exists()
