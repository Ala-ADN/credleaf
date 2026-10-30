"""Growing topic lineages with matching, named article-volume plots.

Independent monthly topics have no intrinsic persistent identity. For display,
each node chooses its strongest eligible predecessor, and that parent's best
child continues its lineage. Other children become branches. This is a forest
projection of the many-to-many graph, not an inferred causal genealogy.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urldefrag

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd

PAPER, INK, MUTED = "#faf8f3", "#203c42", "#687d80"


@dataclass
class TopicTree:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    tracks: pd.DataFrame
    counts: pd.DataFrame
    windows: pd.DatetimeIndex
    total_tracks: int


def prepare_topic_tree(nodes, edges, membership, provenance, coverage, *,
                       threshold=.55, max_tracks=12, names=None):
    """Select substantial lineages and every ancestor needed to join branches.

    An article is a distinct captured source-page URL, without its fragment.
    Repeated captures/chunks count once per topic/month. A page may contribute
    to several topics, so these counts are not an additive corpus partition.
    """
    if not 0 < threshold <= 1 or max_tracks < 1:
        raise ValueError("Use 0 < threshold <= 1 and max_tracks >= 1")
    nodes, edges = nodes.copy(), edges.copy()
    nodes["window"] = pd.to_datetime(nodes.window)
    windows = pd.DatetimeIndex(pd.to_datetime(coverage.window).sort_values().unique())
    if nodes.empty or windows.empty:
        raise ValueError("No monthly topics to animate")
    if nodes.node_id.duplicated().any() or provenance.point_id.duplicated().any():
        raise ValueError("Node IDs and provenance point IDs must be unique")
    pages = provenance[["point_id"]].copy()
    urls = provenance.get("capture_url", pd.Series(index=provenance.index, dtype=str))
    urls = urls.fillna("").astype(str).str.strip()
    if "seed_url" in provenance:
        urls = urls.where(urls.ne(""), provenance.seed_url.fillna("").astype(str))
    pages["article_id"] = urls.map(lambda url: urldefrag(url)[0])
    joined = membership.merge(pages, on="point_id", how="left", validate="many_to_one")
    if joined.article_id.isna().any() or joined.article_id.eq("").any():
        raise ValueError("Every topic membership needs a source-page URL")
    articles = joined.groupby("node_id").article_id.nunique()
    nodes["articles"] = nodes.node_id.map(articles).fillna(0).astype(int)
    node_times = nodes.set_index("node_id").window
    # Never join across a missing calendar window, even if an input edge does.
    positions = {window: i for i, window in enumerate(windows)}
    valid = edges.source.isin(node_times.index) & edges.target.isin(node_times.index)
    edges = edges.loc[valid & edges.similarity.ge(threshold)].copy()
    edges = edges.loc[
        edges.target.map(node_times).map(positions).sub(edges.source.map(node_times).map(positions)).eq(1)
    ].sort_values(["similarity", "source", "target"], ascending=[False, True, True])
    edges = edges.drop_duplicates("target")  # one primary parent per node
    parent = edges.set_index("target").source.to_dict()
    continuation = edges.drop_duplicates("source").set_index("source").target.to_dict()
    node_track, track_parent, root_nodes = {}, {}, {}
    for row in nodes.sort_values(["window", "node_id"]).itertuples():
        previous = parent.get(row.node_id)
        if previous and continuation[previous] == row.node_id:
            node_track[row.node_id] = node_track[previous]
        else:
            node_track[row.node_id] = row.node_id
            track_parent[row.node_id] = node_track[previous] if previous else None
            root_nodes[row.node_id] = row
    nodes["track_id"] = nodes.node_id.map(node_track)
    statistics = nodes.groupby("track_id").agg(article_months=("articles", "sum"), months=("window", "size"))
    statistics["score"] = statistics.article_months * np.sqrt(statistics.months)
    selected = set()
    for track in statistics.sort_values(["score", "article_months"], ascending=False).index:
        chain, current = set(), track
        while current and current not in selected:
            chain.add(current)
            current = track_parent[current]
        if len(selected | chain) <= max_tracks:
            selected.update(chain)
        if len(selected) == max_tracks:
            break
    # Order siblings together. This fixes the layout before the animation starts.
    ordered = []
    def visit(track):
        ordered.append(track)
        children = [child for child in selected if track_parent[child] == track]
        for child in sorted(children, key=lambda t: (root_nodes[t].window, t)):
            visit(child)
    for root in sorted((t for t in selected if track_parent[t] is None),
                       key=lambda t: (root_nodes[t].window, t)):
        visit(root)
    names = names or {}
    tracks = pd.DataFrame([
        dict(track_id=track, parent_track=track_parent[track],
             name=names.get(track, root_nodes[track].label.replace(" · ", " / ")),
             first_window=root_nodes[track].window, row=i,
             article_months=int(statistics.loc[track, "article_months"]))
        for i, track in enumerate(ordered)
    ])
    nodes = nodes.loc[nodes.track_id.isin(selected)].copy()
    edges = edges.loc[edges.source.isin(nodes.node_id) & edges.target.isin(nodes.node_id)].copy()
    grid = pd.MultiIndex.from_product([windows, ordered], names=["window", "track_id"])
    counts = nodes.groupby(["window", "track_id"]).articles.sum().reindex(grid, fill_value=0).reset_index()
    counts = counts.merge(tracks[["track_id", "name"]], on="track_id", validate="many_to_one")
    # A zero means absent while clustering was observed; unobserved windows are gaps.
    observed = dict(zip(pd.to_datetime(coverage.window), coverage.observed))
    counts.loc[~counts.window.map(observed).fillna(False), "articles"] = np.nan
    return TopicTree(nodes, edges, tracks, counts, windows, len(statistics))


@dataclass
class TopicTreeStory:
    figure: object
    animation: FuncAnimation
    render_frame: object
    windows: pd.DatetimeIndex


def create_topic_tree_animation(tree: TopicTree):
    """Grow the forest and its article curves together, with names in both plots."""
    n = len(tree.tracks)
    if not n:
        raise ValueError("No selected topic lineages")
    fig = plt.figure(figsize=(18, 10.5), dpi=100, facecolor=PAPER)
    fig.text(.035, .948, "A growing tree of topics", fontsize=29, weight="bold", color=INK)
    fig.text(.035, .909, "Independent monthly topics  →  closest predecessor  →  continuations and new branches",
             fontsize=13, color=MUTED)
    month_text = fig.text(.965, .946, "", fontsize=22, weight="bold", color=INK, ha="right")
    progress = fig.text(.965, .909, "", fontsize=11, color=MUTED, ha="right")
    fig.text(.22, .85, "TOPIC LINEAGES", fontsize=11, weight="bold", color=INK)
    fig.text(.765, .85, "ARTICLES PER TOPIC / MONTH", fontsize=11, weight="bold", color=INK)
    fig.text(.765, .826, "Same count scale in every named plot", fontsize=9, color=MUTED)
    ax = fig.add_axes([.22, .17, .47, .63], facecolor=PAPER)
    dates = mdates.date2num(tree.windows)
    ax.set(xlim=(dates[0] - 12, dates[-1] + 20), ylim=(-.65, n - .3))
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", labelsize=10, colors=MUTED)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=.12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cursor = ax.axvline(dates[0], color=MUTED, alpha=.4, linestyle=":")
    ax.set_xlabel("Event month (publication date, otherwise capture time)", fontsize=9, color=MUTED, labelpad=10)
    palette = ["#276f8e", "#5c659d", "#b36d32", "#bd4f61", "#4b8658", "#617d3a",
               "#a95840", "#8a5d91", "#427e82", "#92575d", "#717137", "#4d6b91"]
    colors = {row.track_id: palette[i % len(palette)] for i, row in enumerate(tree.tracks.itertuples())}
    ranks = tree.tracks.set_index("track_id").row.to_dict()
    node_lookup = tree.nodes.set_index("node_id")
    patches = []
    for edge in tree.edges.itertuples():
        source, target = node_lookup.loc[edge.source], node_lookup.loc[edge.target]
        x0, x1 = mdates.date2num(source.window), mdates.date2num(target.window)
        y0, y1 = ranks[source.track_id], ranks[target.track_id]
        curve = MplPath([(x0, y0), (x0 + (x1-x0)*.5, y0), (x1-(x1-x0)*.5, y1), (x1, y1)],
                        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        patch = PathPatch(curve, facecolor="none", edgecolor=colors[target.track_id], linewidth=2, alpha=.8)
        ax.add_patch(patch)
        patches.append((patch, target.window))
    dots = ax.scatter([], [], zorder=4, linewidths=.6, edgecolors=PAPER)
    starts = ax.scatter([], [], s=78, facecolors=PAPER, edgecolors=INK, linewidths=1.2, zorder=5)
    branch_starts = ax.scatter([], [], s=80, marker="D", facecolors=PAPER, edgecolors=INK, linewidths=1.2, zorder=5)
    labels = {}
    volume_lines, volume_titles, volume_counts, volume_cursors = {}, {}, {}, {}
    ymax = max(2, tree.counts.articles.max() * 1.13)
    for row in tree.tracks.itertuples():
        labels[row.track_id] = ax.text(dates[0] - 32, row.row, row.name, ha="right", va="center",
                                       fontsize=10, color=colors[row.track_id], clip_on=False, weight="medium")
        bottom = .18 + (n - 1 - row.row) * (.63 / n)
        volume_ax = fig.add_axes([.765, bottom, .20, .63 / n * .64], facecolor=PAPER)
        volume_ax.set(xlim=ax.get_xlim(), ylim=(0, ymax))
        volume_ax.set_yticks([0, int(ymax / 1.13)])
        volume_ax.tick_params(axis="both", labelsize=7, colors=MUTED, length=2)
        volume_ax.xaxis.set_major_locator(mdates.YearLocator())
        volume_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if row.row != n - 1:
            volume_ax.tick_params(labelbottom=False)
        volume_ax.grid(axis="y", alpha=.1)
        for spine in volume_ax.spines.values():
            spine.set_visible(False)
        volume_lines[row.track_id], = volume_ax.plot([], [], color=colors[row.track_id], linewidth=1.8)
        volume_titles[row.track_id] = volume_ax.text(0, 1.04, row.name, transform=volume_ax.transAxes,
                                                   fontsize=8.5, color=colors[row.track_id])
        volume_counts[row.track_id] = volume_ax.text(1, 1.04, "", transform=volume_ax.transAxes,
                                                   ha="right", fontsize=8.5, color=MUTED)
        volume_cursors[row.track_id] = volume_ax.axvline(dates[0], color=MUTED, alpha=.3, linestyle=":")
    fig.legend(handles=[
        Line2D([], [], marker="o", color="none", markeredgecolor=INK, markerfacecolor=PAPER, label="First observed lineage"),
        Line2D([], [], marker="D", color="none", markeredgecolor=INK, markerfacecolor=PAPER, label="New branch from a parent topic"),
        Line2D([], [], marker="o", color=INK, markersize=4, label="Monthly topic; dot area reflects article count"),
    ], loc="lower left", bbox_to_anchor=(.215, .075), ncol=2, frameon=False, fontsize=9, labelcolor=MUTED)
    fig.text(.035, .043,
             f"Showing {n} substantial lineages and their ancestors ({tree.total_tracks} total). "
             "One strongest parent per topic; weaker matches and merges omitted for readability.", fontsize=9, color=MUTED)
    fig.text(.035, .022,
             "Articles = distinct captured page URLs per topic/month, including index pages. A page can appear in several topics. "
             "Gaps = unobserved clustering; first appearance need not mean a new real-world topic.", fontsize=8.5, color=MUTED)
    nodes = tree.nodes.copy()
    nodes["plot_x"] = mdates.date2num(nodes.window)
    nodes["plot_y"] = nodes.track_id.map(ranks)

    def render_frame(index):
        window = tree.windows[index]
        month_text.set_text(window.strftime("%B %Y"))
        visible = nodes.loc[nodes.window.le(window)]
        now = nodes.loc[nodes.window.eq(window)]
        progress.set_text(f"{len(visible):,} topic snapshots revealed · {len(now)} in this month")
        for patch, appeared in patches:
            patch.set_visible(appeared <= window)
        dots.set_offsets(visible[["plot_x", "plot_y"]].to_numpy())
        dots.set_sizes(12 + visible.articles.to_numpy() * 2)
        dots.set_facecolors([colors[t] for t in visible.track_id])
        roots = visible.loc[visible.node_id.eq(visible.track_id)]
        branch_ids = set(tree.tracks.loc[tree.tracks.parent_track.notna(), "track_id"])
        starts.set_offsets(roots.loc[~roots.track_id.isin(branch_ids), ["plot_x", "plot_y"]].to_numpy())
        branch_starts.set_offsets(roots.loc[roots.track_id.isin(branch_ids), ["plot_x", "plot_y"]].to_numpy())
        x = mdates.date2num(window)
        cursor.set_xdata([x, x])
        for row in tree.tracks.itertuples():
            appeared = window >= row.first_window
            labels[row.track_id].set_visible(appeared)
            volume_titles[row.track_id].set_visible(appeared)
            part = tree.counts.loc[tree.counts.track_id.eq(row.track_id) & tree.counts.window.le(window)]
            volume_lines[row.track_id].set_data(part.window, part.articles)
            value = part.articles.iloc[-1]
            volume_counts[row.track_id].set_text(f"{value:.0f}" if appeared and pd.notna(value) else "—")
            volume_cursors[row.track_id].set_xdata([x, x])
        return (dots, starts, branch_starts)

    render_frame(0)
    animation = FuncAnimation(fig, render_frame, frames=len(tree.windows), interval=650, repeat=True, blit=False)
    return TopicTreeStory(fig, animation, render_frame, tree.windows)
