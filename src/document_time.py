"""One UTC time policy for normalization, deduplication, and analysis."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import re
from numbers import Real
import math
from collections.abc import Mapping, Sequence

from config import PANDEMIC_PHASES, Phase

OUTSIDE_STUDY = "outside_study"
UNDATED = "undated"


def parse_timestamp(value, *, capture=False) -> datetime | None:
    """Accept ISO dates/timestamps and CDX timestamps; never interpret epochs."""
    if value is None or str(value).strip() in {"", "NaT", "nan", "<NA>"}:
        return None
    try:
        # DataFrame loaders can infer compact CDX strings as integers. Only
        # capture timestamps use this notation; numeric epochs stay invalid.
        if capture and isinstance(value, Real) and not isinstance(value, bool):
            if not math.isfinite(value) or not float(value).is_integer():
                return None
            value = str(int(value))
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time())
        elif isinstance(value, str):
            value = value.strip()
            if capture and re.fullmatch(r"\d{14}", value):
                parsed = datetime.strptime(value, "%Y%m%d%H%M%S")
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def phase_for_date(timestamp: datetime | None, phases: Sequence[Phase] | None = None) -> str | None:
    if timestamp is None:
        return None
    day = timestamp.astimezone(timezone.utc).date() if timestamp.tzinfo else timestamp.date()
    for phase in PANDEMIC_PHASES if phases is None else phases:
        if date.fromisoformat(phase.start) <= day <= date.fromisoformat(phase.end):
            return phase.name
    return None


@dataclass(frozen=True)
class DocumentTime:
    timestamp: datetime | None
    source: str | None
    phase: str


def document_time(doc: Mapping) -> DocumentTime:
    """A valid publication date wins, including when it lies outside the study."""
    timestamp = parse_timestamp(doc.get("publish_date"))
    source = "publish_date" if timestamp is not None else None
    if timestamp is None:
        timestamp = parse_timestamp(doc.get("capture_timestamp"), capture=True)
        source = "capture_timestamp" if timestamp is not None else None
    phase = phase_for_date(timestamp) or (OUTSIDE_STUDY if timestamp is not None else UNDATED)
    return DocumentTime(timestamp, source, phase)


def assign_document_time(doc: Mapping) -> dict:
    """Preserve harvest provenance while updating event time and analysis phase."""
    result = dict(doc)
    resolved = document_time(doc)
    result["capture_phase"] = (doc.get("capture_phase")
                               or doc.get("phase")
                               or phase_for_date(parse_timestamp(doc.get("capture_timestamp"), capture=True)))
    result["phase"] = resolved.phase
    result["event_date"] = resolved.timestamp.isoformat() if resolved.timestamp else None
    result["event_date_source"] = resolved.source
    return result
