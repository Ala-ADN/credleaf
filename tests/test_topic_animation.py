"""Topic lineage identity, article counting, and progressive visual disclosure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.topic_animation import prepare_topic_tree, create_topic_tree_animation


def example_tree(max_tracks=12):
    windows = pd.date_range("2020-01-01", periods=5, freq="MS")
    nodes = pd.DataFrame(dict(node_id=list("ABCDEXF"),
                              window=[windows[i] for i in [0, 1, 1, 2, 2, 2, 4]],
                              label=["Immunity", "Antibodies", "Schools", "Boosters", "Classrooms", "Emerging", "Later"]))
    edges = pd.DataFrame(dict(source=["A", "A", "B", "C", "D"], target=["B", "C", "D", "E", "F"],
                              similarity=[.9, .8, .9, .9, .99]))
    # Repeated chunks/captures of one page count once; a fragment is not a new page.
    membership = pd.DataFrame(dict(point_id=list("abcdefghij"), node_id=list("AABBCDDEFX")))
    provenance = pd.DataFrame(dict(point_id=list("abcdefghij"),
                                    capture_url=["https://a.org/p", "https://a.org/p#fragment",
                                                 "https://b.org/p", "https://c.org/p", "https://b.org/p",
                                                 "https://d.org/p", "https://e.org/p", "https://f.org/p",
                                                 "https://g.org/p", "https://h.org/p"]))
    coverage = pd.DataFrame(dict(window=windows, observed=[True, True, True, False, True]))
    return prepare_topic_tree(nodes, edges, membership, provenance, coverage,
                               max_tracks=max_tracks, names={"A": "Immunity", "C": "Schools", "X": "Emerging"})


def test_tree_branches_keep_identity_and_count_articles_not_chunks():
    tree = example_tree()
    n = tree.nodes.set_index("node_id")
    assert n.loc["A", "articles"] == 1
    assert n.loc["B", "articles"] == 2
    assert n.loc["C", "articles"] == 1  # shares a page with B; count in both topics
    assert n.loc["D", "track_id"] == "A"
    assert n.loc["E", "track_id"] == "C"
    assert tree.tracks.set_index("track_id").loc["C", "parent_track"] == "A"
    assert pd.isna(tree.tracks.set_index("track_id").loc["F", "parent_track"])
    assert not ((tree.edges.source == "D") & (tree.edges.target == "F")).any()  # do not bridge April
    assert tree.counts.loc[tree.counts.window.eq(pd.Timestamp("2020-04-01")), "articles"].isna().all()
    assert tree.counts.loc[tree.counts.window.eq(pd.Timestamp("2020-01-01")) & tree.counts.track_id.eq("C"), "articles"].iloc[0] == 0
    small = example_tree(max_tracks=2)
    assert len(small.tracks) <= 2
    assert set(small.tracks.parent_track.dropna()) <= set(small.tracks.track_id)


def test_animation_names_and_branches_emerge_together_with_fixed_axes(tmp_path):
    from matplotlib.animation import PillowWriter
    tree = example_tree()
    story = create_topic_tree_animation(tree)
    ax = story.figure.axes[0]
    bounds = [(a.get_xlim(), a.get_ylim()) for a in story.figure.axes]
    story.render_frame(0)
    names = {text.get_text(): text.get_visible() for text in ax.texts}
    assert names["Immunity"] and not names["Schools"] and not names["Emerging"]
    assert len(ax.collections[2].get_offsets()) == 0
    story.render_frame(1)
    assert len(ax.collections[2].get_offsets()) == 1  # a new branch
    assert next(t for t in ax.texts if t.get_text() == "Schools").get_visible()
    story.render_frame(2)
    assert next(t for t in ax.texts if t.get_text() == "Emerging").get_visible()
    assert [(a.get_xlim(), a.get_ylim()) for a in story.figure.axes] == bounds
    assert any(t.get_text() == "Schools" for a in story.figure.axes[1:] for t in a.texts)
    path = tmp_path / "topics.gif"
    story.animation.save(path, writer=PillowWriter(fps=2), dpi=25)
    with Image.open(path) as image:
        assert image.n_frames == 5 and image.info["loop"] == 0
    plt.close(story.figure)
