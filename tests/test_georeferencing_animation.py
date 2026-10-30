"""Check the map's denominator, tier isolation and temporal context."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.georeferencing_animation import (
    TIER_COLORS, country_attention, create_georeferencing_animation, recent_events, trailing_window,
)
from config import COVID_TIMELINE_EVENTS, PandemicEvent, Phase


def test_country_shares_count_articles_once_and_isolate_tiers():
    docs = pd.DataFrame({
        "article_id": ["a", "a", "b", "c", "d", "mixed"],
        "credibility_tier": ["low"] * 5 + ["mixed"],
    })
    mentions = pd.DataFrame([
        ("a", "low", "US"), ("a", "low", "US"), ("a", "low", "GB"),
        ("b", "low", "US"), ("c", "low", "XX"),
        ("d", "authoritative", "US"), ("mixed", "mixed", "US"),
        ("outside-window", "low", "US"),
    ], columns=["article_id", "tier", "country_code"])
    result = country_attention(docs, mentions, "low", {"US", "GB"})
    assert result.total_articles == 4
    assert result.mapped_articles == 2
    assert result.shares.to_dict() == {"US": 100.0, "GB": 50.0}
    # The denominator is mapped articles, and co-occurring countries can sum >100%.
    assert result.shares.sum() == 150


def test_rolling_window_includes_final_day_but_no_future_articles():
    frame = pd.DataFrame({"event_dt": pd.to_datetime([
        "2020-01-05 23:59", "2020-01-06 00:00", "2020-02-02 23:59", "2020-02-03 00:00",
    ])})
    assert trailing_window(frame, "2020-02-02").index.tolist() == [1, 2]
    assert len(frame) == 4
    with pytest.raises(ValueError):
        trailing_window(frame, "2020-02-02", weeks=0)


def test_event_cards_do_not_reveal_future_milestones():
    before, current, future = COVID_TIMELINE_EVENTS[:3]
    assert recent_events((future, before, current), current.date) == (current, before)
    assert recent_events(COVID_TIMELINE_EVENTS, "2019-12-30") == ()
    assert recent_events(COVID_TIMELINE_EVENTS, "2023-05-05")[0].title == "WHO ends global emergency"
    dates = [event.date for event in COVID_TIMELINE_EVENTS]
    assert dates == sorted(set(dates))
    for event in COVID_TIMELINE_EVENTS:
        pd.Timestamp(event.date)
        assert event.source_url.startswith("https://en.wikipedia.org/wiki/")


@pytest.fixture
def country_boundaries(monkeypatch):
    """Small offline boundary fixture; exercise projection and rendering too."""
    import cartopy.io.shapereader as shp
    from shapely.geometry import box
    records = [SimpleNamespace(
        geometry=box(-5, 42, 8, 51),
        attributes={"ISO_A2_EH": "FR", "NAME": "France"},
    ), SimpleNamespace(
        geometry=box(5, 58, 10, 68),
        attributes={"ISO_A2_EH": "NO", "NAME": "Norway"},
    )]
    monkeypatch.setattr(shp, "natural_earth", lambda *args: "unused")
    monkeypatch.setattr(shp, "Reader", lambda *args: SimpleNamespace(records=lambda: records))


def test_render_preserves_inputs_and_includes_last_partial_week(country_boundaries):
    docs = pd.DataFrame([
        (f"a{i}", "low", "2020-01-05 23:59", "publish") for i in range(3)
    ] + [
        (f"b{i}", "authoritative", "2020-01-05 23:59", "publish") for i in range(3)
    ] + [("other", "mixed", "2020-01-05 23:59", "capture")],
        columns=["article_id", "credibility_tier", "event_date", "event_date_source"])
    mentions = pd.DataFrame([
        (f"a{i}", "low", "2020-01-05 23:59", "FR") for i in range(3)
    ] + [
        (f"b{i}", "authoritative", "2020-01-05 23:59", "NO") for i in range(3)
    ], columns=["article_id", "tier", "event_date", "country_code"])
    original = docs.copy(deep=True)
    final_event = PandemicEvent("2020-01-10", "Final-day event", "", "https://en.wikipedia.org/wiki/Test")
    story = create_georeferencing_animation(
        docs, mentions, events=(final_event,), phases=(Phase("P1_test", "2020-01-01", "2020-01-10"),),
    )
    assert list(story.frame_ends) == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")]
    assert list(story.animation.new_frame_seq()).count(1) == 8  # Read the closing milestone before looping.
    assert not any("Final-day event" in text.get_text() for text in story.figure.texts)
    story.render_frame(1)
    story.figure.canvas.draw()
    texts = [text.get_text() for text in story.figure.texts]
    event_label = next(text for text in story.figure.texts if "Final-day event" in text.get_text())
    assert event_label.get_position()[1] < story.figure.axes[-1].get_position().y0
    assert texts.count("3 / 3 articles mapped") == 2
    assert not any("PANDEMIC MILESTONES" in text for text in texts)
    from cartopy.mpl.feature_artist import FeatureArtist
    from matplotlib.colors import to_rgba
    countries = next(artist for artist in story.figure.axes[0].collections if isinstance(artist, FeatureArtist))
    np.testing.assert_allclose(countries.get_facecolors(), [to_rgba(TIER_COLORS["low"]), to_rgba("#e1e4df")])
    countries = next(artist for artist in story.figure.axes[1].collections if isinstance(artist, FeatureArtist))
    np.testing.assert_allclose(countries.get_facecolors(), [to_rgba("#e1e4df"), to_rgba(TIER_COLORS["authoritative"])])
    legend_colors = story.figure.axes[2].collections[-1].get_facecolors()
    np.testing.assert_allclose(legend_colors[:, 0], legend_colors[:, 1])
    np.testing.assert_allclose(legend_colors[:, 1], legend_colors[:, 2])
    pd.testing.assert_frame_equal(docs, original)
    plt.close(story.figure)


def test_empty_windows_render_without_nan_labels(country_boundaries):
    story = create_georeferencing_animation(
        pd.DataFrame(), pd.DataFrame(), events=(),
        phases=(Phase("P1_test", "2020-01-01", "2020-01-03"),),
    )
    story.render_frame(0)
    story.figure.canvas.draw()
    texts = [text.get_text() for ax in story.figure.axes for text in ax.texts]
    assert texts.count("No mapped articles") == 2
    assert not any("nan" in text.lower() for text in texts)
    plt.close(story.figure)
