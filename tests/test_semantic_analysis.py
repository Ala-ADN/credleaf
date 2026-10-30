"""Numerical and sampling invariants for source-tier semantic comparisons."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.semantic_analysis import (
    authoritative_neighbours, build_semantic_report, export_semantic_report,
    load_qdrant_corpus, prepare_semantic_corpus, tier_metrics,
)
from analysis.semantic_animation import create_semantic_animation
from credibility.registry import CredibilityRegistry


def corpus():
    return pd.DataFrame(dict(
        point_id=list("abcdef"), window=pd.to_datetime(["2020-02-01"] * 5 + ["2021-02-01"]),
        analysis_phase=["P1_global_onset"] * 5 + ["P3_vaccine_rollout"],
        credibility_tier=["low", "low", "authoritative", "authoritative", "authoritative", "low"],
        source_host=["low.org", "low.org", "one.org", "one.org", "two.org", "low.org"],
        global_topic=[0, 1, 0, 0, -1, 0],
    ))


def test_metrics_use_raw_cosine_and_fixed_entropy_denominator():
    data = corpus()
    vectors = np.array([[5, 0], [0, 3], [1, 0], [1, 0], [0, 1], [1, 0]])
    windows = pd.date_range("2020-02-01", "2020-04-01", freq="MS")
    metrics = tier_metrics(data, vectors, data.global_topic, groups=windows, min_chunks=2)
    low = metrics.loc[metrics.tier.eq("low")].iloc[0]
    assert low.dispersion == pytest.approx(1)  # orthogonal chunks, regardless of norms
    assert low.cohesion == pytest.approx(1)
    assert low.topic_spread == pytest.approx(1)
    assert low.entropy_bits == pytest.approx(1)
    assert low.entropy_normalized == pytest.approx(1)
    assert low.effective_topics == pytest.approx(2)
    auth = metrics.loc[metrics.tier.eq("authoritative")].iloc[0]
    assert auth.assigned_fraction == pytest.approx(2 / 3)
    assert auth.dispersion == pytest.approx(2 / 3)  # includes the outlier
    assert auth.entropy_bits == 0 and auth.n_topics == 1
    assert np.isnan(auth.topic_spread)  # one topic is not zero spread
    assert metrics.loc[metrics.period.eq(windows[1]), "status"].eq("empty").all()
    assert metrics.loc[metrics.period.eq(windows[1]), "dispersion"].isna().all()
    with pytest.raises(ValueError, match="aligned"):
        tier_metrics(data, vectors[:-1], data.global_topic)


def test_neighbours_exclude_same_host_and_never_cross_phase():
    data = corpus()
    vectors = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]])
    neighbours = authoritative_neighbours(data, vectors, min_reference=1, batch_size=2).set_index("point_id")
    assert neighbours.loc["a", "cosine_distance"] == 0
    assert neighbours.loc["a", "reference_point_id"] in ["c", "d"]
    # Both points at one.org are excluded from the authoritative self-baseline.
    assert neighbours.loc["c", "reference_point_id"] == "e"
    assert neighbours.loc["c", "n_reference"] == 1
    assert neighbours.loc["c", "cosine_distance"] == 1
    assert neighbours.loc["f", "status"] == "insufficient_reference"
    assert pd.isna(neighbours.loc["f", "cosine_distance"])
    # A larger reference minimum must remain missing, not silently fall back.
    assert authoritative_neighbours(data, vectors, min_reference=4).cosine_distance.isna().all()


def test_preparation_refreshes_tiers_filters_scope_and_preserves_alignment():
    data = pd.DataFrame(dict(
        point_id=["late", "first", "duplicate", "other-collection", "outside", "french"],
        text=["eight words about vaccines immunity medicine clinical trials health"] * 6,
        publish_date=["2020-03-02", "2020-02-02", "2020-02-03", "2020-02-02", "2018-01-01", "2020-02-02"],
        language=["en"] * 5 + ["fr"], seed_url="https://health.example.org",
        credibility_tier="low", collection_id=[13529, 13529, 13529, 4887, 13529, 13529],
        phase="P6_transition_end",
    ))
    registry = CredibilityRegistry(authoritative=["example.org"])
    clean, vectors, windows = prepare_semantic_corpus(data, np.eye(6), registry)
    assert clean.point_id.tolist() == ["first", "late"]
    np.testing.assert_array_equal(vectors, np.eye(6)[[1, 0]])
    assert clean.credibility_tier.eq("authoritative").all()
    assert clean.analysis_phase.eq("P1_global_onset").all()  # publication, not capture phase
    assert len(windows) == 2
    assert data.credibility_tier.eq("low").all()  # input was not mutated


def test_full_qdrant_scroll_and_named_vectors_remain_aligned():
    class Client:
        def scroll(self, **kwargs):
            if kwargs["offset"] is None:
                return [SimpleNamespace(id="a", payload={"text": "first"}, vector={"dense": [1, 0]})], "next"
            return [SimpleNamespace(id="b", payload={"text": "second"}, vector=[0, 1])], None
    data, vectors = load_qdrant_corpus(Client())
    assert data.point_id.tolist() == ["a", "b"]
    np.testing.assert_array_equal(vectors, np.eye(2))


def test_report_exports_and_animation_handles_empty_month(tmp_path):
    from matplotlib.animation import PillowWriter
    from PIL import Image

    data = corpus()
    vectors = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 0]])
    windows = pd.date_range("2020-02-01", periods=3, freq="MS")
    report = build_semantic_report(data, vectors, data.global_topic, windows, min_chunks=2, min_reference=1)
    export_semantic_report(tmp_path, report, metadata={"purpose": "test"})
    assert pd.read_csv(tmp_path / "monthly.csv").shape == report.monthly.shape
    assert report.neighbour_summary.loc[report.neighbour_summary.n_matched.lt(2), "median_distance"].isna().all()
    story = create_semantic_animation(data, vectors, report, {0: "vaccine", 1: "school"}, min_chunks=2)
    bounds = [(ax.get_xlim(), ax.get_ylim()) for ax in story.figure.axes]
    story.render_frame(0)
    assert len(story.figure.axes[0].collections[0].get_offsets()) == 2
    story.render_frame(1)
    assert len(story.figure.axes[0].collections[0].get_offsets()) == 0
    assert [(ax.get_xlim(), ax.get_ylim()) for ax in story.figure.axes] == bounds
    gif = tmp_path / "semantic.gif"
    story.animation.save(gif, writer=PillowWriter(fps=2), dpi=35)
    with Image.open(gif) as image:
        assert image.n_frames == 3
        assert image.info["loop"] == 0
    plt.close(story.figure)
