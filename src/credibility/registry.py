"""Domain → credibility-tier registry.

Three tiers: `low` (curated misinformation seeds), `authoritative` (curated
top-tier health/science sources), and `mixed` (collections that are not
uniformly one or the other — e.g. mainstream media + government).

Lookup walks the domain tree (`news.cdc.gov` → `cdc.gov`) and returns the
first tier match in priority order:

    authoritative > low > mixed > unknown

Authoritative wins on conflict so the curated list trumps a noisy seed list,
and the `mixed` bucket is the catch-all for everything that came in via a
mixed-tier collection but isn't explicitly authoritative or low.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

Tier = Literal["low", "authoritative", "mixed", "unknown"]


def normalize_host(value: str) -> str | None:
    """Normalize a URL or hostname to a bare lowercase domain.

    Strips scheme, port, userinfo, www. prefix, and trailing dots.
    Returns None for inputs that don't yield a parseable host.
    """
    if not value:
        return None
    raw = value.strip().lower()
    if "://" in raw:
        host = urlparse(raw).hostname
    else:
        host = raw.split("/", 1)[0].split(":", 1)[0]
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    host = host.rstrip(".")
    return host or None


def _domain_chain(host: str) -> Iterable[str]:
    """Yield candidate domains from most-specific to least.

    `news.example.co.uk` → `news.example.co.uk`, `example.co.uk`, `co.uk`.
    The TLD-only suffix is yielded but won't match anything realistic;
    callers can stop at the first hit.
    """
    parts = host.split(".")
    for i in range(len(parts) - 1):
        yield ".".join(parts[i:])


class CredibilityRegistry:
    """Three-set domain index: low + authoritative + mixed.

    Lookup priority on conflict: authoritative > low > mixed > unknown.
    """

    __slots__ = ("low", "authoritative", "mixed")

    def __init__(
        self,
        low: Iterable[str] = (),
        authoritative: Iterable[str] = (),
        mixed: Iterable[str] = (),
    ) -> None:
        self.low = frozenset(self._clean(low))
        self.authoritative = frozenset(self._clean(authoritative))
        self.mixed = frozenset(self._clean(mixed))

    @staticmethod
    def _clean(domains: Iterable[str]) -> Iterable[str]:
        for d in domains:
            n = normalize_host(d)
            if n:
                yield n

    def lookup(self, url_or_host: str) -> Tier:
        host = normalize_host(url_or_host)
        if host is None:
            return "unknown"
        for candidate in _domain_chain(host):
            if candidate in self.authoritative:
                return "authoritative"
            if candidate in self.low:
                return "low"
            if candidate in self.mixed:
                return "mixed"
        return "unknown"

    # Convenience for incrementally extending the registry
    def with_low(self, domains: Iterable[str]) -> CredibilityRegistry:
        return CredibilityRegistry(
            low=set(self.low) | set(self._clean(domains)),
            authoritative=self.authoritative,
            mixed=self.mixed,
        )

    def with_authoritative(self, domains: Iterable[str]) -> CredibilityRegistry:
        return CredibilityRegistry(
            low=self.low,
            authoritative=set(self.authoritative) | set(self._clean(domains)),
            mixed=self.mixed,
        )

    def with_mixed(self, domains: Iterable[str]) -> CredibilityRegistry:
        return CredibilityRegistry(
            low=self.low,
            authoritative=self.authoritative,
            mixed=set(self.mixed) | set(self._clean(domains)),
        )

    # Persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "low": sorted(self.low),
                    "authoritative": sorted(self.authoritative),
                    "mixed": sorted(self.mixed),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> CredibilityRegistry:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            low=data.get("low", []),
            authoritative=data.get("authoritative", []),
            mixed=data.get("mixed", []),
        )

    def __repr__(self) -> str:
        return (
            f"CredibilityRegistry(low={len(self.low)}, "
            f"authoritative={len(self.authoritative)}, "
            f"mixed={len(self.mixed)})"
        )
