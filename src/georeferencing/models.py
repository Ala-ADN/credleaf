from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

NerStatus = Literal["ok", "no_text", "no_toponyms", "ner_failed"]
EventDateSource = Literal["publish", "capture"]
CredTier = Literal["low", "authoritative", "mixed", "unknown"]
EntityLabel = Literal["GPE", "LOC"]


class RawMention(BaseModel):
    """One toponym surface form found by the NER, before gazetteer resolution."""

    surface: str
    label: EntityLabel
    start_char: int
    end_char: int
    context: str  # ~40-token window around the mention for disambiguation


class GeoEntry(BaseModel):
    """One row from the GeoNames gazetteer."""

    geonameid: int
    name: str
    lat: float
    lon: float
    country_code: str  # ISO2
    population: int
    feature_class: str  # P=populated place, A=admin region, etc.


class GeoMention(BaseModel):
    """One resolved toponym, attached to a single article."""

    toponym: str  # original surface form (first occurrence)
    geonameid: int
    name: str  # canonical name from gazetteer
    lat: float
    lon: float
    country_code: str
    feature_class: str
    mention_count: int  # times this geonameid appears in the article
    confidence: float  # disambiguation confidence in [0, 1]


class GeoDoc(BaseModel):
    """One georeferenced article. Written one-per-line into the georef JSONL."""

    # Composite key + provenance (joins back to the normalized JSONL)
    article_id: str  # f"{collection_id}:{seed_id}:{capture_timestamp}"
    collection_id: int
    seed_id: int
    seed_url: str
    phase: str
    capture_timestamp: str
    publish_date: str | None = None

    # Time axis for the weekly animation
    event_date: str  # ISO YYYY-MM-DD; publish_date if present, else capture date
    event_week: str  # ISO year-week "YYYY-Www"
    event_date_source: EventDateSource

    # Tier + lightweight content metadata
    credibility_tier: CredTier
    language: str | None = None
    word_count: int = 0

    # NER + gazetteer outcome
    mentions: list[GeoMention] = []
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    dispersion_km: float | None = None  # mention-count-weighted mean haversine to centroid
    country_entropy: float | None = None  # Shannon entropy over country distribution (bits)
    unique_countries: int = 0
    ner_status: NerStatus
