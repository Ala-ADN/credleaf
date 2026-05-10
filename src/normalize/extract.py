"""HTML cache + main-text extraction via trafilatura.

Two-stage strategy: try article-style extraction first (clean main text + metadata).
For pages that aren't articles (homepages, index pages, listings — which is what
Archive-It captures for these COVID collections), fall back to a cleaned plain-text
dump of the visible page, which still carries headlines and teasers usable for
downstream clustering / KG.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import trafilatura
from pydantic import BaseModel
from fast_langdetect import detect
from trafilatura import load_html, metadata as trafilatura_metadata


MIN_TEXT_LEN = 100


class Extracted(BaseModel):
    """Content fields produced by HTML extraction. Shared with NormalizedDoc."""

    title: str | None = None
    text: str | None = None
    language: str | None = None
    publish_date: str | None = None
    extraction_mode: str | None = None  # "article" | "fallback_html2txt"

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


def cache_path(cache_root: Path, collection_id: int, digest: str) -> Path:
    return cache_root / "html" / str(collection_id) / f"{digest}.html.gz"


def load_cached_html(path: Path) -> bytes | None:
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return f.read()


def save_cached_html(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(content)

def detect_language(text: str) -> str | None:
    if text == "":
        return None
    return str(detect(text[:80], model="auto", k=1)[0]["lang"])



def extract(html: bytes | str, url: str | None = None) -> Extracted | None:
    """Extract text + metadata. Falls back to plain-text dump for non-article pages.

    Returns None only when both article extraction AND the plaintext fallback yield
    nothing meaningful (truly empty / unreadable page).
    """
    # Stage 1: article-style extraction
    raw = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        include_comments=False,
        include_tables=False,
        with_metadata=True,
    )
    if raw:
        data = json.loads(raw)
        text = data.get("text") or data.get("raw_text")
        if text and text.strip():
            return Extracted(
                title=data.get("title"),
                text=text,
                language=detect_language(text),
                publish_date=data.get("date"),
                extraction_mode="article",
            )

    # Stage 2: plaintext fallback for homepage / index / listing pages
    plain = trafilatura.html2txt(html)
    if not plain or len(plain.strip()) < MIN_TEXT_LEN:
        return None

    title: str | None = None
    try:
        meta = trafilatura_metadata.extract_metadata(load_html(html), default_url=url)
        if meta is not None:
            title = meta.title
    except Exception:  # metadata extraction is best-effort
        pass

    return Extracted(
        title=title,
        text=plain.strip(),
        language=detect_language(plain),
        publish_date=None,
        extraction_mode="fallback_html2txt",
    )
