"""Client for the public Archive-It partner API and Wayback CDX server.

Endpoints used:
  - https://partner.archive-it.org/api/collection/{id}     (collection metadata)
  - https://partner.archive-it.org/api/seed?collection={id} (seed list, paginated)
  - https://wayback.archive-it.org/{id}/timemap/cdx        (captures per URL)
  - https://wayback.archive-it.org/{id}/{ts}id_/{url}      (raw capture content)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

from .models import Capture, Collection, Seed

log = logging.getLogger(__name__)

PARTNER_API = "https://partner.archive-it.org/api"
WAYBACK_BASE = "https://wayback.archive-it.org"

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
    ) -> None:
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
        params: dict[str, Any] = {"url": url, "matchType": match_type}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        if limit is not None:
            params["limit"] = limit
        if collapse:
            params["collapse"] = collapse

        with self._client.stream(
            "GET", f"{WAYBACK_BASE}/{collection_id}/timemap/cdx", params=params
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                yield self._parse_cdx_row(line)

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
        for attempt in range(max_retries + 1):
            r = self._client.get(snap)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt == max_retries:
                    r.raise_for_status()
                retry_after = r.headers.get("Retry-After")
                delay = (
                    float(retry_after) if retry_after and retry_after.isdigit()
                    else backoff_base ** (attempt + 1)
                )
                log.info(
                    "fetch %s: %d, sleeping %.1fs (attempt %d/%d)",
                    snap, r.status_code, delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.content
        # unreachable
        raise RuntimeError("retry loop exited without returning")
