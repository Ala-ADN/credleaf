"""A country-level, two-panel story of geographic attention in the archive.

Country colour is the share of mapped articles mentioning that country, counting
an article once per country. It measures corpus attention, not infection counts.
The two tiers and every frame use the same fixed percentage scale.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from config import COVID_TIMELINE_EVENTS, PANDEMIC_PHASES, PandemicEvent, Phase

MAP_TIERS = ("low", "authoritative")
TIER_LABELS = {"low": "Misinformation", "authoritative": "Authoritative"}
TIER_COLORS = {"low": "#b34b58", "authoritative": "#286e91"}
PAPER, INK, MUTED = "#faf8f3", "#203c42", "#687d80"
NO_MENTIONS = "#e1e4df"
SHARE_BOUNDS = (0, 1, 5, 10, 25, 50, 100)
SHARE_COLORS = ("#f1f1f1", "#d6d6d6", "#b5b5b5", "#929292", "#6d6d6d", "#414141")


def tier_colormap(color: str) -> ListedColormap:
    """Use the same intensity steps in each tier's own hue."""
    background, foreground = np.array(to_rgb(PAPER)), np.array(to_rgb(color))
    return ListedColormap([
        background + strength * (foreground - background)
        for strength in (.15, .29, .44, .62, .80, 1)
    ])


def trailing_window(df: pd.DataFrame, end: object, weeks: int = 4) -> pd.DataFrame:
    """Include all of the final day and exactly ``weeks * 7`` calendar days."""
    if weeks < 1:
        raise ValueError("weeks must be positive")
    stop = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    return df.loc[df["event_dt"].ge(stop - pd.Timedelta(weeks=weeks)) & df["event_dt"].lt(stop)]


@dataclass(frozen=True)
class CountryAttention:
    shares: pd.Series
    mapped_articles: int
    total_articles: int


def country_attention(
    docs: pd.DataFrame,
    mentions: pd.DataFrame,
    tier: str,
    country_codes: Collection[str],
) -> CountryAttention:
    """Deduplicate article/country pairs; exclude other tiers and unrenderable codes."""
    article_ids = docs.loc[docs["credibility_tier"].eq(tier), "article_id"].drop_duplicates()
    pairs = mentions.loc[
        mentions["tier"].eq(tier)
        & mentions["article_id"].isin(article_ids)
        & mentions["country_code"].isin(country_codes),
        ["article_id", "country_code"],
    ].drop_duplicates()
    mapped = pairs["article_id"].nunique()
    shares = pairs.groupby("country_code").size().div(mapped or 1).mul(100)
    return CountryAttention(shares.sort_values(ascending=False), mapped, len(article_ids))


def recent_events(
    events: Sequence[PandemicEvent], end: object, limit: int = 3,
) -> tuple[PandemicEvent, ...]:
    """Only reveal milestones that have occurred by the frame's displayed date."""
    past = sorted(
        (event for event in events if pd.Timestamp(event.date) <= pd.Timestamp(end)),
        key=lambda event: event.date, reverse=True,
    )
    return tuple(past[:limit])


@dataclass
class GeographicStory:
    figure: Figure
    animation: FuncAnimation
    frame_ends: pd.DatetimeIndex
    render_frame: Callable[[int], tuple]


def create_georeferencing_animation(
    docs: pd.DataFrame,
    mentions: pd.DataFrame,
    *,
    events: Sequence[PandemicEvent] = COVID_TIMELINE_EVENTS,
    phases: Sequence[Phase] = PANDEMIC_PHASES,
    window_weeks: int = 4,
    min_articles: int = 3,
    interval: int = 350,
) -> GeographicStory:
    """Build the map once; update country fills, dates and timeline events in place.

    Natural Earth's 110m country boundaries are downloaded by Cartopy on first
    use, then cached locally. No live Wikipedia requests are made at render time.
    """
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shapereader

    if not phases or window_weeks < 1 or min_articles < 1:
        raise ValueError("phases, a positive window and a positive minimum are required")
    events = tuple(sorted(events, key=lambda event: event.date))
    start = min([pd.Timestamp(phases[0].start), *(pd.Timestamp(e.date) for e in events)])
    end = pd.Timestamp(phases[-1].end)
    frame_ends = pd.date_range(start, end, freq="W-SUN")
    if frame_ends.empty or frame_ends[-1] != end:
        frame_ends = frame_ends.union(pd.DatetimeIndex([end]))

    # Work on local copies. Never change the frames used elsewhere in the EDA.
    def dated_copy(frame, tier_column):
        frame = frame.copy()
        if frame.empty:
            frame = frame.reindex(columns=frame.columns.union(
                ["event_dt", tier_column, "article_id", "country_code", "event_date_source"]
            ))
            frame["event_dt"] = pd.Series(dtype="datetime64[ns]")
            return frame
        dates = frame["event_dt"] if "event_dt" in frame else frame["event_date"]
        frame["event_dt"] = pd.to_datetime(dates, errors="coerce", utc=True).dt.tz_localize(None)
        return frame.loc[
            frame[tier_column].isin(MAP_TIERS)
            & frame["event_dt"].ge(start)
            & frame["event_dt"].lt(end + pd.Timedelta(days=1))
        ].copy()

    docs = dated_copy(docs, "credibility_tier").drop_duplicates("article_id")
    mentions = dated_copy(mentions, "tier")
    boundary_path = shapereader.natural_earth("110m", "cultural", "admin_0_countries")
    countries = [r for r in shapereader.Reader(boundary_path).records() if r.attributes["ISO_A2_EH"] != "AQ"]
    country_codes = {r.attributes["ISO_A2_EH"] for r in countries} - {"-99"}
    windows = [trailing_window(docs, date, window_weeks) for date in frame_ends]
    mention_windows = [trailing_window(mentions, date, window_weeks) for date in frame_ends]
    attention = [
        {tier: country_attention(d, m, tier, country_codes) for tier in MAP_TIERS}
        for d, m in zip(windows, mention_windows)
    ]

    norm = BoundaryNorm(SHARE_BOUNDS, len(SHARE_COLORS), clip=True)
    tier_cmaps = {tier: tier_colormap(color) for tier, color in TIER_COLORS.items()}
    fig = plt.figure(figsize=(16, 7.4), facecolor=PAPER, dpi=100)
    fig.text(
        .045, .944,
        "Geographic attention of misinformation vs authoritative sources",
        fontsize=23, weight="bold", color=INK,
    )
    date_text = fig.text(.967, .946, "", ha="right", color=MUTED, fontsize=13)

    artists, counters, empty_labels = {}, {}, {}
    for tier, left in zip(MAP_TIERS, (.025, .515)):
        fig.text(left + .02, .855, TIER_LABELS[tier], fontsize=15,
                 weight="bold", color=TIER_COLORS[tier])
        counters[tier] = fig.text(left + .445, .858, "", ha="right", fontsize=9, color=MUTED)
        ax = fig.add_axes([left, .33, .46, .50], projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor(PAPER)
        ax.spines["geo"].set_edgecolor("#dedfd9")
        ax.spines["geo"].set_linewidth(.5)
        # Keep one cached collection per panel. No graticule or labels cover
        # the map, and the country fill uses the tier's own colour.
        artists[tier] = ax.add_geometries(
            [country.geometry for country in countries], ccrs.PlateCarree(),
            facecolor=NO_MENTIONS, edgecolor=PAPER, linewidth=.4, zorder=2,
        )
        empty_labels[tier] = ax.text(
            .5, .5, "", transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=MUTED, zorder=5,
        )

    # A compact grayscale key explains intensity without competing with the maps.
    fig.text(.045, .275, "Mapped articles mentioning each country (%)", fontsize=9, color=MUTED)
    cax = fig.add_axes([.345, .267, .32, .012])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(SHARE_COLORS)),
        cax=cax, orientation="horizontal", boundaries=SHARE_BOUNDS,
        ticks=SHARE_BOUNDS, spacing="uniform",
    )
    colorbar.ax.set_xticklabels([str(value) for value in SHARE_BOUNDS])
    colorbar.ax.tick_params(labelsize=8, colors=MUTED, length=0, pad=4)
    colorbar.outline.set_visible(False)
    fig.legend(
        handles=[Patch(facecolor=NO_MENTIONS, label="No mentions")],
        loc="center left", bbox_to_anchor=(.70, .271), frameon=False,
        fontsize=8, labelcolor=MUTED,
    )

    timeline = fig.add_axes([.055, .156, .89, .047], facecolor=PAPER)
    timeline.set_xlim(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10))
    timeline.set_ylim(0, 1)
    timeline.set_yticks([])
    timeline.grid(False)
    for spine in timeline.spines.values():
        spine.set_visible(False)
    timeline.xaxis.set_major_locator(mdates.YearLocator())
    timeline.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    timeline.tick_params(axis="x", labelsize=9, colors=MUTED, length=0, pad=3)
    timeline.plot([start, end], [.55, .55], color="#dcded7", linewidth=1.5, zorder=1)
    progress, = timeline.plot([], [], color="#888e86", linewidth=1.5, zorder=2)
    event_dates = pd.DatetimeIndex([e.date for e in events])
    event_marks = timeline.scatter([], [], s=14, color="#888e86", edgecolor=PAPER,
                                   linewidth=.5, zorder=3)
    cursor, = timeline.plot([], [], "o", markersize=4, color=INK, zorder=4)
    # Only the arriving events appear here; the latest remains between milestones.
    # Two events in the same week share this small caption under the timeline.
    event_text = fig.text(.5, .071, "", ha="center", va="center", fontsize=10,
                          color=MUTED, linespacing=1.65)

    def render_frame(index: int) -> tuple:
        frame_end = frame_ends[index]
        date_text.set_text(frame_end.strftime("%d %b %Y"))
        for tier in MAP_TIERS:
            data = attention[index][tier]
            enough = data.mapped_articles >= min_articles
            artists[tier].set_facecolors([
                tier_cmaps[tier](norm(share)) if enough and share > 0 else NO_MENTIONS
                for country in countries
                for share in [data.shares.get(country.attributes["ISO_A2_EH"], 0)]
            ])
            counters[tier].set_text(f"{data.mapped_articles:,} / {data.total_articles:,} articles mapped")
            empty_labels[tier].set_text(
                "" if enough else (
                    "No mapped articles" if not data.mapped_articles
                    else f"Fewer than {min_articles} mapped articles"
                )
            )
            empty_labels[tier].set_visible(not enough)

        previous_end = frame_ends[index - 1] if index else start - pd.Timedelta(days=1)
        arriving = [event for event in events if previous_end < pd.Timestamp(event.date) <= frame_end]
        visible = arriving or recent_events(events, frame_end, limit=1)
        event_text.set_text("\n".join(
            f"{pd.Timestamp(event.date):%d %b %Y}  ·  {event.title}" for event in visible
        ))
        progress.set_data([start, frame_end], [.55, .55])
        cursor.set_data([frame_end], [.55])
        occurred = event_dates[event_dates <= frame_end]
        event_marks.set_offsets(np.column_stack((mdates.date2num(occurred), np.full(len(occurred), .55))))
        return ()

    render_frame(0)
    # Hold new milestones long enough to read. Pillow merges identical repeated
    # frames into a longer GIF delay, including a pause before the loop restarts.
    frame_order = []
    for index, frame_end in enumerate(frame_ends):
        previous = frame_ends[index - 1] if index else start - pd.Timedelta(days=1)
        has_event = any(previous < pd.Timestamp(event.date) <= frame_end for event in events)
        repeats = 8 if index in (0, len(frame_ends) - 1) else (4 if has_event else 1)
        frame_order.extend([index] * repeats)
    animation = FuncAnimation(fig, render_frame, frames=frame_order, interval=interval,
                              init_func=lambda: (), blit=False, repeat_delay=1800, cache_frame_data=False)
    return GeographicStory(fig, animation, frame_ends, render_frame)
