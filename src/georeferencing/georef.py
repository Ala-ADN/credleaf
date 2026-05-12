"""Drive a normalized JSONL through NER + gazetteer -> GeoDoc JSONL.

Append-only output with a `*.progress` sidecar listing already-processed
article IDs so long runs can resume after interruption.

Mirrors the shape of `src.normalize.normalize.normalize_phase`:
    georef_phase(normalized_path, extractor, gazetteer, registry, ...)
returns (output_path, GeorefStats).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from config import PROCESSED_DIR
from credibility import CredibilityRegistry
from normalize import NormalizedDoc

from .extract import ToponymExtractor
from .gazetteer import Gazetteer
from .models import GeoDoc, GeoMention

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


@dataclass
class GeorefStats:
    total: int = 0
    skipped_lang: int = 0
    skipped_status: int = 0
    skipped_resume: int = 0
    no_text: int = 0
    no_toponyms: int = 0
    ok: int = 0
    ner_failed: int = 0
    publish_date_used: int = 0
    capture_date_used: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "skipped_lang": self.skipped_lang,
            "skipped_status": self.skipped_status,
            "skipped_resume": self.skipped_resume,
            "no_text": self.no_text,
            "no_toponyms": self.no_toponyms,
            "ok": self.ok,
            "ner_failed": self.ner_failed,
            "publish_date_used": self.publish_date_used,
            "capture_date_used": self.capture_date_used,
            "error_count": len(self.errors),
        }


def georef_phase(
    normalized_path: Path,
    extractor: ToponymExtractor,
    gazetteer: Gazetteer,
    registry: CredibilityRegistry,
    output_path: Path | None = None,
    max_docs: int | None = None,
    resume: bool = True,
    language_filter: str | None = "en",
) -> tuple[Path, GeorefStats]:
    """Read normalized JSONL, georef each doc, append to GeoDoc JSONL."""
    docs = list(_load_normalized(normalized_path))
    if not docs:
        log.warning("no docs in %s", normalized_path)
        return output_path or Path(), GeorefStats()

    if output_path is None:
        cid = docs[0].collection_id
        phase = docs[0].phase
        output_path = PROCESSED_DIR / "georef" / str(cid) / f"{phase}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress")

    processed: set[str] = set()
    if resume and progress_path.exists():
        processed = {
            line.strip()
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        log.info("resume: %d articles already processed", len(processed))

    stats = GeorefStats()
    log.info(
        "georef start: %d docs from %s -> %s",
        len(docs), normalized_path, output_path,
    )

    mode = "a" if resume else "w"
    with output_path.open(mode, encoding="utf-8") as out_fh, \
            progress_path.open(mode, encoding="utf-8") as prog_fh:
        emitted = 0
        for doc in docs:
            if max_docs is not None and emitted >= max_docs:
                break
            stats.total += 1

            article_id = _article_id(doc)
            if article_id in processed:
                stats.skipped_resume += 1
                continue
            if doc.fetch_status != "ok":
                stats.skipped_status += 1
                continue
            if language_filter and (doc.language or "").lower() != language_filter:
                stats.skipped_lang += 1
                continue

            try:
                geo_doc = _process_doc(doc, extractor, gazetteer, registry, stats)
            except Exception as e:  # NER / resolution failure shouldn't kill the batch
                stats.ner_failed += 1
                stats.errors.append((article_id, str(e)))
                geo_doc = _make_doc(
                    doc,
                    registry,
                    mentions=[],
                    ner_status="ner_failed",
                    stats=stats,
                )

            out_fh.write(geo_doc.model_dump_json() + "\n")
            prog_fh.write(article_id + "\n")
            emitted += 1
            if emitted % 50 == 0:
                out_fh.flush()
                prog_fh.flush()
                log.info("georef progress: %d emitted, stats=%s", emitted, stats.as_dict())

    log.info("georef done: %s", stats.as_dict())
    return output_path, stats


# ----- internals -----


def _load_normalized(path: Path) -> Iterable[NormalizedDoc]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield NormalizedDoc.model_validate(json.loads(line))


def _article_id(doc: NormalizedDoc) -> str:
    return f"{doc.collection_id}:{doc.seed_id}:{doc.capture_timestamp}"


def _process_doc(
    doc: NormalizedDoc,
    extractor: ToponymExtractor,
    gazetteer: Gazetteer,
    registry: CredibilityRegistry,
    stats: GeorefStats,
) -> GeoDoc:
    text = doc.text or ""
    if not text.strip():
        stats.no_text += 1
        return _make_doc(doc, registry, mentions=[], ner_status="no_text", stats=stats)

    raw = extractor.extract(text)
    mentions = gazetteer.resolve_mentions(raw)
    if not mentions:
        stats.no_toponyms += 1
        return _make_doc(doc, registry, mentions=[], ner_status="no_toponyms", stats=stats)

    stats.ok += 1
    return _make_doc(doc, registry, mentions=mentions, ner_status="ok", stats=stats)


def _make_doc(
    doc: NormalizedDoc,
    registry: CredibilityRegistry,
    *,
    mentions: list[GeoMention],
    ner_status: str,
    stats: GeorefStats,
) -> GeoDoc:
    event_date, event_week, source = _event_date(doc)
    if source == "publish":
        stats.publish_date_used += 1
    else:
        stats.capture_date_used += 1
    centroid_lat, centroid_lon, dispersion_km = _centroid_and_dispersion(mentions)
    return GeoDoc(
        article_id=_article_id(doc),
        collection_id=doc.collection_id,
        seed_id=doc.seed_id,
        seed_url=doc.seed_url,
        phase=doc.phase,
        capture_timestamp=doc.capture_timestamp,
        publish_date=doc.publish_date,
        event_date=event_date,
        event_week=event_week,
        event_date_source=source,
        credibility_tier=registry.lookup(doc.seed_url),
        language=doc.language,
        word_count=doc.word_count,
        mentions=mentions,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        dispersion_km=dispersion_km,
        country_entropy=_country_entropy(mentions),
        unique_countries=len({m.country_code for m in mentions if m.country_code}),
        ner_status=ner_status,
    )


# ---- date handling ----

def _parse_publish_date(value: str | None) -> date | None:
    if not value:
        return None
    # trafilatura may emit YYYY-MM-DD or full ISO datetime; tolerate both.
    try:
        return datetime.fromisoformat(value.split("T", 1)[0]).date()
    except ValueError:
        return None


def _parse_capture_date(value: str) -> date | None:
    # capture_timestamp is YYYYMMDDhhmmss
    if not value or len(value) < 8:
        return None
    try:
        return datetime.strptime(value[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _iso_week(d: date) -> str:
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _event_date(doc: NormalizedDoc) -> tuple[str, str, str]:
    """Return (ISO date string, ISO year-week, source) for this doc."""
    pub = _parse_publish_date(doc.publish_date)
    if pub is not None:
        return pub.isoformat(), _iso_week(pub), "publish"
    cap = _parse_capture_date(doc.capture_timestamp) or date(1970, 1, 1)
    return cap.isoformat(), _iso_week(cap), "capture"


# ---- spatial metrics ----

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _centroid_and_dispersion(
    mentions: list[GeoMention],
) -> tuple[float | None, float | None, float | None]:
    """Mention-count weighted centroid and mean great-circle distance to it.

    Centroid via averaging unit vectors so antipodal/wraparound mentions don't
    cancel into a fake (0, 0) point.
    """
    if not mentions:
        return None, None, None

    total_w = 0.0
    sx = sy = sz = 0.0
    for m in mentions:
        w = float(m.mention_count)
        rlat = math.radians(m.lat)
        rlon = math.radians(m.lon)
        sx += w * math.cos(rlat) * math.cos(rlon)
        sy += w * math.cos(rlat) * math.sin(rlon)
        sz += w * math.sin(rlat)
        total_w += w

    if total_w == 0:
        return None, None, None
    sx /= total_w
    sy /= total_w
    sz /= total_w
    norm = math.sqrt(sx * sx + sy * sy + sz * sz) or 1e-12
    sx, sy, sz = sx / norm, sy / norm, sz / norm
    centroid_lat = math.degrees(math.asin(max(-1.0, min(1.0, sz))))
    centroid_lon = math.degrees(math.atan2(sy, sx))

    if len(mentions) == 1:
        return centroid_lat, centroid_lon, 0.0

    weighted_dist = 0.0
    for m in mentions:
        weighted_dist += m.mention_count * _haversine_km(
            centroid_lat, centroid_lon, m.lat, m.lon
        )
    return centroid_lat, centroid_lon, weighted_dist / total_w


def _country_entropy(mentions: list[GeoMention]) -> float | None:
    """Shannon entropy (bits) over the mention-count-weighted country distribution."""
    if not mentions:
        return None
    weights: dict[str, float] = {}
    total = 0.0
    for m in mentions:
        if not m.country_code:
            continue
        weights[m.country_code] = weights.get(m.country_code, 0.0) + m.mention_count
        total += m.mention_count
    if total == 0 or not weights:
        return None
    h = 0.0
    for w in weights.values():
        p = w / total
        h -= p * math.log2(p)
    return h
