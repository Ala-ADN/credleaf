"""GeoNames-backed gazetteer: surface form -> coordinates.

Loads `cities1000.txt` (~150k populated places ≥1000 inhabitants) plus
`countryInfo.txt` (so country names like "China" resolve via the capital's
coordinates). Indexes by canonical name and by every alternate name in the
inline `alternatenames` column.

Disambiguation when multiple candidates share a name:
  1. Country-context boost: if the article's surrounding context mentions
     another country (e.g. "Manchester reports cases across England"), prefer
     candidates inside that country and bump confidence.
  2. Population tie-break (CLAVIN-style): the largest-population candidate
     wins. This resolves Paris-FR over Paris-TX, Cambridge-UK over Cambridge-MA,
     etc., which is the right default for international news copy.

Persisted as a pickled snapshot at `data/cache/geonames/gazetteer.pkl` for
fast reload (~0.5s vs ~5s parsing the TSVs from scratch).
"""
from __future__ import annotations

import logging
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import GeoEntry, GeoMention, RawMention

log = logging.getLogger(__name__)

# Single-hit confidence; ambiguous hits use the log-pop heuristic clamped here.
MIN_AMBIG_CONFIDENCE = 0.3
MAX_AMBIG_CONFIDENCE = 0.95
COUNTRY_CONTEXT_BOOST = 0.3


class Gazetteer:
    """In-memory gazetteer. Build via `from_geonames`, persist via `save_pickle`."""

    def __init__(
        self,
        entries: list[GeoEntry],
        by_name: dict[str, list[int]],
        by_alt: dict[str, list[int]],
        country_to_geonameid: dict[str, int],
        country_iso_by_name: dict[str, str],
    ):
        self.entries: dict[int, GeoEntry] = {e.geonameid: e for e in entries}
        self.by_name = by_name  # casefolded name -> list[geonameid]
        self.by_alt = by_alt
        self.country_to_geonameid = country_to_geonameid  # ISO2 -> capital geonameid
        # Casefolded country name (and ISO code) -> ISO2. Used for context detection.
        self.country_iso_by_name = country_iso_by_name

    # ----- construction -----

    @classmethod
    def from_geonames(cls, geonames_dir: Path) -> "Gazetteer":
        cities_path = geonames_dir / "cities1000.txt"
        country_info_path = geonames_dir / "countryInfo.txt"
        if not cities_path.exists():
            raise FileNotFoundError(
                f"missing {cities_path}. run `uv run python -m georeferencing.setup_gazetteer`."
            )
        if not country_info_path.exists():
            raise FileNotFoundError(
                f"missing {country_info_path}. run `uv run python -m georeferencing.setup_gazetteer`."
            )

        log.info("parsing %s", cities_path)
        entries = list(_iter_cities1000(cities_path))
        log.info("loaded %d city entries", len(entries))

        log.info("parsing %s", country_info_path)
        country_rows = list(_iter_country_info(country_info_path))
        # ISO2 -> (country_name, capital_name)
        country_by_iso = {iso: (name, capital_name) for iso, name, capital_name in country_rows}

        # Index cities first so we can look up capitals when synthesizing country entries.
        by_name: dict[str, list[int]] = {}
        by_alt: dict[str, list[int]] = {}
        by_name_in_country: dict[tuple[str, str], list[int]] = {}
        for e in entries:
            by_name.setdefault(e.name.casefold(), []).append(e.geonameid)
            by_name_in_country.setdefault(
                (e.country_code, e.name.casefold()), []
            ).append(e.geonameid)
        # Re-stream to populate alt names without inflating GeoEntry payloads.
        for geonameid, alts in _iter_cities1000_alts(cities_path):
            for alt in alts:
                key = alt.casefold().strip()
                if not key or len(key) < 2:
                    continue
                by_alt.setdefault(key, []).append(geonameid)

        # Map a country mention ("China", "France", "CN", "FR") to a representative
        # geonameid by looking up its capital in cities1000.
        country_to_geonameid: dict[str, int] = {}
        country_iso_by_name: dict[str, str] = {}
        for iso, (country_name, capital_name) in country_by_iso.items():
            if not country_name:
                continue
            country_iso_by_name[country_name.casefold()] = iso
            country_iso_by_name[iso.casefold()] = iso
            if not capital_name:
                continue
            # Prefer a capital that is in the same country; fall back to a
            # plain name match (handles cases where country_code on the city
            # row differs from the listed country, e.g. dependent territories).
            in_country = by_name_in_country.get((iso, capital_name.casefold()), [])
            candidates = in_country or by_name.get(capital_name.casefold(), [])
            if not candidates:
                continue
            # When several capitals share a name, pick the most populous.
            entries_by_id = {e.geonameid: e for e in entries}
            top = max(candidates, key=lambda gid: entries_by_id[gid].population)
            country_to_geonameid[iso] = top
            by_name.setdefault(country_name.casefold(), []).append(top)

        return cls(
            entries=entries,
            by_name=by_name,
            by_alt=by_alt,
            country_to_geonameid=country_to_geonameid,
            country_iso_by_name=country_iso_by_name,
        )

    def save_pickle(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("wrote pickled gazetteer -> %s", path)

    @classmethod
    def load_pickle(cls, path: Path) -> "Gazetteer":
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"unpickled object is {type(obj)}, expected Gazetteer")
        return obj

    # ----- resolution -----

    def resolve(self, surface: str, context: str | None = None) -> tuple[GeoEntry, float] | None:
        """Resolve a surface form to a single GeoEntry + confidence in [0, 1]."""
        key = surface.casefold().strip()
        if not key:
            return None

        candidates: list[int] = self.by_name.get(key) or self.by_alt.get(key) or []
        if not candidates:
            return None

        unique = list({gid for gid in candidates})
        if len(unique) == 1:
            return self.entries[unique[0]], 1.0

        # --- disambiguate ---
        context_isos = self._isos_in_context(context) if context else set()

        # Country-context boost: if any candidate is in a country mentioned in
        # the surrounding text, restrict to those candidates.
        if context_isos:
            in_context = [
                gid for gid in unique
                if self.entries[gid].country_code in context_isos
            ]
            if in_context:
                if len(in_context) == 1:
                    base = self.entries[in_context[0]]
                    return base, min(1.0, MAX_AMBIG_CONFIDENCE + COUNTRY_CONTEXT_BOOST)
                unique = in_context  # narrow further with population

        # Population tie-break.
        ranked = sorted(
            unique, key=lambda gid: self.entries[gid].population, reverse=True
        )
        top, second = ranked[0], ranked[1]
        top_pop = max(self.entries[top].population, 1)
        second_pop = max(self.entries[second].population, 1)
        if top_pop <= second_pop:
            confidence = MIN_AMBIG_CONFIDENCE
        else:
            ratio = math.log(top_pop) - math.log(second_pop)
            confidence = max(
                MIN_AMBIG_CONFIDENCE,
                min(MAX_AMBIG_CONFIDENCE, ratio / max(math.log(top_pop), 1.0)),
            )
        if context_isos:
            confidence = min(1.0, confidence + COUNTRY_CONTEXT_BOOST / 2)

        return self.entries[top], confidence

    def resolve_mentions(self, raw: list[RawMention]) -> list[GeoMention]:
        """Resolve and collapse repeats by geonameid into one GeoMention each."""
        # First pass: resolve each surface; remember (geonameid, surface, confidence)
        by_id: dict[int, dict] = {}
        for m in raw:
            res = self.resolve(m.surface, context=m.context)
            if res is None:
                continue
            entry, conf = res
            slot = by_id.get(entry.geonameid)
            if slot is None:
                by_id[entry.geonameid] = {
                    "entry": entry,
                    "first_surface": m.surface,
                    "count": 1,
                    "confidence_sum": conf,
                }
            else:
                slot["count"] += 1
                slot["confidence_sum"] += conf

        out: list[GeoMention] = []
        for slot in by_id.values():
            entry: GeoEntry = slot["entry"]
            out.append(GeoMention(
                toponym=slot["first_surface"],
                geonameid=entry.geonameid,
                name=entry.name,
                lat=entry.lat,
                lon=entry.lon,
                country_code=entry.country_code,
                feature_class=entry.feature_class,
                mention_count=slot["count"],
                confidence=slot["confidence_sum"] / slot["count"],
            ))
        return out

    # ----- internals -----

    def _isos_in_context(self, context: str) -> set[str]:
        """Find country ISO2 codes for any country names that appear in context."""
        if not context:
            return set()
        ctx = context.casefold()
        found: set[str] = set()
        # The country-name dict is small enough (<300 keys) to scan directly,
        # which avoids tokenizing the context window.
        for name, iso in self.country_iso_by_name.items():
            if len(name) < 3:
                continue
            if name in ctx:
                found.add(iso)
        return found


# ----- GeoNames file parsers -----

# cities1000.txt column order (no header in file):
# 0 geonameid, 1 name, 2 asciiname, 3 alternatenames, 4 lat, 5 lon,
# 6 feature_class, 7 feature_code, 8 country_code, 9 cc2, 10 admin1,
# 11 admin2, 12 admin3, 13 admin4, 14 population, 15 elevation, 16 dem,
# 17 timezone, 18 modification_date


def _iter_cities1000(path: Path) -> Iterable[GeoEntry]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 19:
                continue
            try:
                yield GeoEntry(
                    geonameid=int(parts[0]),
                    name=parts[1],
                    lat=float(parts[4]),
                    lon=float(parts[5]),
                    country_code=parts[8] or "",
                    population=int(parts[14] or 0),
                    feature_class=parts[6] or "",
                )
            except (ValueError, IndexError):
                continue


def _iter_cities1000_alts(path: Path) -> Iterable[tuple[int, list[str]]]:
    """Yield (geonameid, [alternate names]) pairs from the inline alts column."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 19:
                continue
            try:
                gid = int(parts[0])
            except ValueError:
                continue
            asciiname = parts[2]
            alts_csv = parts[3]
            alts: list[str] = []
            if asciiname and asciiname != parts[1]:
                alts.append(asciiname)
            if alts_csv:
                alts.extend(a for a in alts_csv.split(",") if a)
            if alts:
                yield gid, alts


def _iter_country_info(path: Path) -> Iterable[tuple[str, str, str]]:
    """Yield (ISO2, country_name, capital_name) per country.

    countryInfo.txt columns (0-indexed): 0=ISO, 4=Country, 5=Capital.
    Comment lines start with #.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            iso = parts[0].strip()
            name = parts[4].strip()
            capital = parts[5].strip()
            if iso and name:
                yield iso, name, capital
