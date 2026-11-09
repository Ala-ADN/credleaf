"""Client for the public Archive-It partner API and Wayback CDX server.

Endpoints used:
  - https://partner.archive-it.org/api/collection/{id}     (collection metadata)
  - https://partner.archive-it.org/api/seed?collection={id} (seed list, paginated)
  - https://wayback.archive-it.org/{id}/timemap/cdx        (captures per URL)
  - https://wayback.archive-it.org/{id}/{ts}id_/{url}      (raw capture content)
"""
from __future__ import annotations

import logging
import hashlib
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from collections.abc import Iterator
from typing import Any

import httpx

from .models import Capture, Collection, Seed
from .pacing import ArchivePacing, ArchiveUnavailable, CaptureFetchFailed, MALFORMED_RESPONSE_LIMIT

log = logging.getLogger(__name__)

PARTNER_API = "https://partner.archive-it.org/api"
WAYBACK_BASE = "https://wayback.archive-it.org"


def retry_after_seconds(value):
    if not value:
        return 0.
    try:
        return max(0., float(value))
    except ValueError:
        try:
            return max(0., (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.

CDX_FIELDS = (
    "urlkey",
    "timestamp",
    "url",
    "mimetype",
    "status",
    "digest",
    "redirect",
    "robotflags",
    "length",
    "offset",
    "filename",
)


class ArchiveItClient:
    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = "credleaf/0.1 (research; +https://github.com/)",
        request_interval: float = 2.,
        pacing_path=None,
    ) -> None:
        if request_interval < 0:
            raise ValueError("request_interval must be nonnegative")
        self._minimum_interval = self._interval = request_interval
        self._next_request = 0.
        self._successes = 0
        self._pacing = ArchivePacing(pacing_path, request_interval) if pacing_path is not None else None
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ArchiveItClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def get_collection(self, collection_id: int) -> Collection:
        r = self._client.get(f"{PARTNER_API}/collection/{collection_id}")
        r.raise_for_status()
        return Collection.model_validate(r.json())

    def iter_seeds(
        self,
        collection_id: int,
        page_size: int = 20,
        max_seeds: int | None = None,
    ) -> Iterator[Seed]:
        offset = 0
        yielded = 0
        while True:
            r = self._client.get(
                f"{PARTNER_API}/seed",
                params={
                    "collection": collection_id,
                    "limit": page_size,
                    "offset": offset,
                },
            )
            r.raise_for_status()
            page = r.json()
            if not page:
                return
            for item in page:
                yield Seed.model_validate(item)
                yielded += 1
                if max_seeds is not None and yielded >= max_seeds:
                    return
            if len(page) < page_size:
                return
            offset += page_size

    def iter_captures(
        self,
        collection_id: int,
        url: str,
        match_type: str = "exact",
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int | None = None,
        collapse: str | None = None,
    ) -> Iterator[Capture]:
        """Stream CDX rows for a given seed URL.

        match_type: 'exact' | 'prefix' | 'host' | 'domain'
        from_ts/to_ts: 14-digit timestamps (YYYYMMDDhhmmss).
        collapse: CDX `collapse=` field (e.g. 'digest' to drop adjacent
        same-digest rows server-side, 'timestamp:10' for one row per hour).
        """
        # Archive-It's range matcher currently returns empty 200 responses for
        # prefix/host/domain queries with from/to, even for known captures.
        # Keep the requested URL scope and apply dates to the streamed rows.
        local_window = match_type != "exact" and bool(from_ts or to_ts)
        params: dict[str, Any] = {"url": url, "matchType": match_type}
        if from_ts and not local_window:
            params["from"] = from_ts
        if to_ts and not local_window:
            params["to"] = to_ts
        if limit is not None and not local_window:
            params["limit"] = limit
        if collapse and local_window:
            raise ValueError("Server-side collapse cannot precede local date filtering")
        if collapse:
            params["collapse"] = collapse

        emitted = 0
        for attempt in range(4):
            try:
                with self._client.stream(
                    "GET", f"{WAYBACK_BASE}/{collection_id}/timemap/cdx", params=params
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line.strip():
                            continue
                        capture = self._parse_cdx_row(line)
                        if from_ts and capture.timestamp < from_ts.ljust(14, "0"):
                            continue
                        if to_ts and capture.timestamp > to_ts.ljust(14, "9"):
                            continue
                        emitted += 1
                        yield capture
                        if limit is not None and emitted >= limit:
                            return
                return
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = (isinstance(exc, httpx.TransportError)
                             or exc.response.status_code == 429
                             or exc.response.status_code >= 500)
                # Once rows have escaped, report a partial seed rather than
                # silently replaying its output on a retry.
                if not retryable or emitted or attempt == 3:
                    raise
                delay = 2 ** attempt
                log.warning("CDX request failed (%s); retrying in %ss", exc, delay)
                time.sleep(delay)

    @staticmethod
    def _parse_cdx_row(line: str) -> Capture:
        parts = line.split(" ")
        row: dict[str, Any] = dict(zip(CDX_FIELDS, parts, strict=False))
        for k in ("redirect", "robotflags", "digest", "status", "mimetype"):
            if row.get(k) == "-":
                row[k] = None
        for k in ("length", "offset"):
            v = row.get(k)
            if v is not None and v != "-":
                try:
                    row[k] = int(v)
                except ValueError:
                    row[k] = None
        return Capture.model_validate(row)

    def fetch_capture(
        self,
        collection_id: int,
        timestamp: str,
        url: str,
        max_retries: int = 4,
        backoff_base: float = 1.5,
    ) -> bytes:
        """Fetch the raw archived bytes (no Wayback rewriting).

        Retries with exponential backoff on 429 (rate-limit) and 5xx responses.
        Honors the Retry-After header when present.
        """
        snap = f"{WAYBACK_BASE}/{collection_id}/{timestamp}id_/{url}"
        capture_key = hashlib.sha256(snap.encode()).hexdigest()
        failure = self._pacing.capture_failure(capture_key) if self._pacing else dict(attempts=0)
        if failure['attempts'] >= MALFORMED_RESPONSE_LIMIT:
            raise CaptureFetchFailed(f"Capture download failed after {failure['attempts']} malformed responses: {snap}: {failure.get('error')}")
        for attempt in range(max_retries + 1):
            if self._pacing:
                self._interval = self._pacing.acquire()
            wait = self._next_request - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._next_request = time.monotonic() + self._interval
            try:
                r = self._client.get(snap)
            except httpx.TransportError as exc:
                # A reproducibly malformed body is specific to this capture.
                # Do not turn it into an unlimited, archive-wide outage loop.
                if isinstance(exc, httpx.RemoteProtocolError):
                    failure = (self._pacing.capture_failure(capture_key, exc) if self._pacing
                               else dict(attempts=failure['attempts'] + 1))
                    if failure['attempts'] >= MALFORMED_RESPONSE_LIMIT:
                        raise CaptureFetchFailed(f"Capture download failed after {failure['attempts']} malformed responses: {snap}: {exc}") from exc
                if attempt == max_retries:
                    if self._pacing and not isinstance(exc, httpx.RemoteProtocolError):
                        self._pacing.result(transient=True)
                    raise ArchiveUnavailable(f"Archive network unavailable; resume this capture after recovery: {exc}") from exc
                self._next_request = max(self._next_request, time.monotonic() + backoff_base ** (attempt + 1))
                log.warning("Archive transport failure; retry %d/%d: %s", attempt + 1, max_retries, exc)
                continue
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if self._pacing:
                    self._pacing.result(transient=True, retry_after=retry_after_seconds(r.headers.get("Retry-After")))
                    raise ArchiveUnavailable(f"Archive returned {r.status_code}; shared cooldown saved, reschedule capture")
                self._successes = 0
                if r.status_code == 429:
                    self._interval = min(60., max(1., self._interval * 1.5))
                if attempt == max_retries:
                    raise ArchiveUnavailable(f"Archive returned {r.status_code} after retries; resume after cooldown")
                delay = max(self._interval, retry_after_seconds(r.headers.get("Retry-After")), backoff_base ** (attempt + 1))
                log.info(
                    "fetch %s: %d, sleeping %.1fs (attempt %d/%d)",
                    snap, r.status_code, delay, attempt + 1, max_retries,
                )
                self._next_request = max(self._next_request, time.monotonic() + delay)
                continue
            r.raise_for_status()
            if self._pacing:
                self._pacing.capture_succeeded(capture_key)
            self._successes += 1
            if self._pacing:
                self._interval = self._pacing.result()
            if not self._pacing and self._successes >= 20:
                self._interval = max(self._minimum_interval, self._interval * .9)
                self._successes = 0
            log.info("archive fetched: interval=%.2fs retries=%d bytes=%d", self._interval, attempt, len(r.content))
            return r.content
        # unreachable
        raise RuntimeError("retry loop exited without returning")
