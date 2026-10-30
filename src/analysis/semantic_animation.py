"""A fixed-coordinate, monthly semantic story for the README."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from analysis.semantic_analysis import TIERS, SemanticReport
from config import PANDEMIC_PHASES

PAPER, INK, MUTED = "#faf8f3", "#203c42", "#687d80"
COLORS = {"low": "#b34b58", "authoritative": "#286e91", "mixed": "#b58b39"}
LABELS = {"low": "Low-credibility sources", "authoritative": "Authoritative sources", "mixed": "Mixed sources"}


@dataclass
class SemanticStory:
    figure: Figure
    animation: FuncAnimation
    windows: pd.DatetimeIndex
    render_frame: Callable
    preview_frame: int


def create_semantic_animation(data, embeddings, report: SemanticReport, topic_names, *,
                              min_chunks=20, max_points=800, seed=42):
    """Use one PCA basis and fixed scales for every tier and frame.

    Point clouds are deterministically capped for readability; every metric uses
    all retained observations. Topic labels are the leading global clusters in
    this month/tier, with global words. Mixed sources appear in metric panels.
    """
    if min_chunks < 2 or max_points < 1:
        raise ValueError("min_chunks >= 2 and max_points >= 1 are required")
    vectors = normalize(np.asarray(embeddings, dtype=np.float32))
    if len(data) != len(vectors) or min(vectors.shape) < 2:
        raise ValueError("At least two aligned chunks and embedding dimensions are required")
    windows = pd.DatetimeIndex(sorted(report.monthly.period.unique()))
    if windows.empty:
        raise ValueError("No monthly metrics to animate")
    with threadpool_limits(limits=4):
        reducer = PCA(n_components=2, svd_solver="randomized", random_state=seed)
        coordinates = reducer.fit_transform(vectors)
    bounds = []
    for dim in range(2):
        lo, hi = coordinates[:, dim].min(), coordinates[:, dim].max()
        pad = max((hi - lo) * .08, .02)
        bounds.append((lo - pad, hi + pad))

    fig = plt.figure(figsize=(14, 8), dpi=100, facecolor=PAPER)
    fig.text(.055, .945, "How the archive's semantic footprint changes", fontsize=23,
             color=INK, weight="bold")
    fig.text(.055, .904, "Monthly English chunks  /  shared topics  /  source credibility", fontsize=12, color=MUTED)
    date_text = fig.text(.945, .904, "", ha="right", fontsize=15, weight="bold", color=INK)
    artists, counts, empty_labels, topic_texts = {}, {}, {}, {}
    for tier, left in zip(("low", "authoritative"), (.065, .55)):
        fig.text(left, .847, LABELS[tier], fontsize=14, weight="bold", color=COLORS[tier])
        counts[tier] = fig.text(left + .39, .849, "", ha="right", fontsize=10, color=MUTED)
        ax = fig.add_axes([left, .47, .39, .34], facecolor=PAPER)
        ax.set(xlim=bounds[0], ylim=bounds[1],
               xlabel=f"Shared PC1 ({reducer.explained_variance_ratio_[0]:.0%})",
               ylabel=f"PC2 ({reducer.explained_variance_ratio_[1]:.0%})")
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
        ax.grid(alpha=.12)
        for spine in ax.spines.values():
            spine.set_color("#d9dcd5")
        artists[tier] = ax.scatter([], [], s=10, color=COLORS[tier], alpha=.26, linewidths=0)
        empty_labels[tier] = ax.text(.5, .5, "", transform=ax.transAxes, ha="center", color=MUTED, fontsize=12)
        topic_texts[tier] = fig.text(left, .385, "", fontsize=9, color=COLORS[tier], linespacing=1.6)

    fig.text(.055, .355, "Measured in the full embedding space", fontsize=13, weight="bold", color=INK)
    metric_lines, cursors = {}, []
    for column, left, title in [("dispersion", .065, "Semantic dispersion"),
                                 ("entropy_normalized", .385, "Topic entropy")]:
        ax = fig.add_axes([left, .13, .265, .16], facecolor=PAPER)
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=16)
        y_max = (max(.1, float(report.monthly.dispersion.max()) * 1.1)
                 if column == "dispersion" and report.monthly.dispersion.notna().any() else 1.05)
        ax.set(ylim=(0, y_max), xlim=(windows[0] - pd.Timedelta(days=15), windows[-1] + pd.Timedelta(days=15)))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.grid(alpha=.15)
        for spine in ax.spines.values():
            spine.set_color("#d9dcd5")
        for tier in TIERS:
            metric_lines[(column, tier)], = ax.plot([], [], color=COLORS[tier], linewidth=1.7,
                                                   marker="o", markersize=2.3)
        cursors.append(ax.axvline(windows[0], color=MUTED, alpha=.4, linestyle=":"))
        ax.set_ylabel("Mean cosine distance" if column == "dispersion" else "Normalized entropy", fontsize=8, color=MUTED)

    nn_ax = fig.add_axes([.72, .13, .225, .16], facecolor=PAPER)
    nn_ax.set_title("Nearest authoritative chunk", loc="left", fontsize=11, color=INK, pad=16)
    valid_nn = report.neighbour_summary.p90_distance.dropna()
    nn_ax.set(ylim=(0, max(.2, valid_nn.max() * 1.1) if len(valid_nn) else 1),
              ylabel="Median cosine distance")
    nn_ax.yaxis.label.set(fontsize=8, color=MUTED)
    nn_ax.set_xticks(range(3), ["Low", "Auth.", "Mixed"])
    nn_ax.tick_params(labelsize=8, colors=MUTED)
    nn_ax.grid(axis="y", alpha=.15)
    for spine in nn_ax.spines.values():
        spine.set_color("#d9dcd5")
    bars = nn_ax.bar(range(3), [0] * 3, color=[COLORS[t] for t in TIERS], width=.55)
    bar_labels = [nn_ax.text(i, .02, "", ha="center", fontsize=8, color=MUTED) for i in range(3)]
    phase_text = fig.text(.72, .305, "", fontsize=8, color=MUTED)

    for tier, left in zip(TIERS, (.065, .26, .47)):
        fig.text(left, .069, "● " + LABELS[tier], fontsize=9, color=COLORS[tier])
    fig.text(.055, .031,
             f"Dots: up to {max_points} chunks per panel. Metrics: all chunks; gaps: fewer than " + str(min_chunks) + ".  "
             "PCA and topics fitted once on the full corpus (retrospective).", fontsize=8, color=MUTED)
    fig.text(.055, .011,
             "Source labels do not judge individual claims. Neighbours: same event phase, different source host. "
             "Corpus sampling and long pages affect these comparisons.", fontsize=8, color=MUTED)

    # Precompute samples so redraws and GIF export display identical points.
    samples, top_topics = {}, {}
    for i, window in enumerate(windows):
        for tier in ("low", "authoritative"):
            ids = np.flatnonzero((data.window.eq(window) & data.credibility_tier.eq(tier)).to_numpy())
            rng = np.random.default_rng(seed + i)
            samples[(window, tier)] = rng.choice(ids, min(len(ids), max_points), replace=False)
            topics = data.global_topic.iloc[ids]
            top_topics[(window, tier)] = topics[topics >= 0].value_counts().head(2)

    def render_frame(index):
        window = windows[index]
        date_text.set_text(window.strftime("%B %Y"))
        for tier in ("low", "authoritative"):
            row = report.monthly.loc[report.monthly.period.eq(window) & report.monthly.tier.eq(tier)].iloc[0]
            counts[tier].set_text(f"{row.n_chunks:,} chunks · {row.n_sources} hosts")
            ids = samples[(window, tier)] if row.n_chunks >= min_chunks else []
            artists[tier].set_offsets(coordinates[ids] if len(ids) else np.empty((0, 2)))
            empty_labels[tier].set_text("" if len(ids) else "No retained chunks" if not row.n_chunks else "Too few chunks to compare")
            top = top_topics[(window, tier)]
            text = "\n".join(f"{count / row.n_chunks:.0%}   T{topic}: {str(topic_names.get(topic, 'unlabelled'))[:58]}"
                              for topic, count in top.items()) if len(ids) else ""
            topic_texts[tier].set_text(text)
        for (column, tier), line in metric_lines.items():
            part = report.monthly.loc[report.monthly.tier.eq(tier) & report.monthly.period.le(window)].sort_values("period")
            line.set_data(part.period, part[column])
        for cursor in cursors:
            cursor.set_xdata([window, window])
        # A month crossing a phase boundary uses the phase of its final study day.
        end = min(window + pd.offsets.MonthEnd(), pd.Timestamp(PANDEMIC_PHASES[-1].end))
        phase = next((p for p in PANDEMIC_PHASES if pd.Timestamp(p.start) <= end <= pd.Timestamp(p.end)), None)
        phase_text.set_text((phase.name.replace("_", " ") if phase else "No study phase") + " · same phase")
        for i, tier in enumerate(TIERS):
            part = report.neighbour_summary.loc[report.neighbour_summary.phase.eq(phase.name if phase else "")
                                                & report.neighbour_summary.tier.eq(tier)]
            value = part.median_distance.iloc[0] if len(part) else np.nan
            bars[i].set_height(value if np.isfinite(value) else 0)
            bar_labels[i].set_y(value + .01 if np.isfinite(value) else .02)
            bar_labels[i].set_text(f"{value:.2f}" if np.isfinite(value) else "n/a")
        return (*artists.values(), *metric_lines.values(), *bars)

    volume = report.monthly.loc[report.monthly.tier.isin(["low", "authoritative"])].groupby("period").n_chunks.min()
    preview = int(windows.get_loc(volume.idxmax()))
    render_frame(0)
    animation = FuncAnimation(fig, render_frame, frames=len(windows), interval=500, blit=False, repeat=True)
    return SemanticStory(fig, animation, windows, render_frame, preview)
