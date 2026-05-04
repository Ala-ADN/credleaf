from __future__ import annotations

from datetime import datetime
from typing import Literal

from .extract import Extracted

FetchStatus = Literal["ok", "fetch_failed", "extract_failed", "empty"]


class NormalizedDoc(Extracted):
    """One archived document, fetched from Wayback and reduced to clean text.

    Inherits the extracted content fields (title/text/language/publish_date/
    extraction_mode) from Extracted; adds provenance, fetch outcome, and
    derived counters.
    """

    # Provenance
    collection_id: int
    seed_id: int
    seed_url: str
    phase: str
    capture_timestamp: str
    capture_url: str
    digest: str | None = None

    # Fetch + extraction outcome
    fetched_at: datetime
    fetch_status: FetchStatus
    error: str | None = None

    # Derived from extracted text
    word_count: int = 0
    char_count: int = 0

