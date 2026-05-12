"""Toponym NER via spaCy transformer model.

Strategy: spaCy `en_core_web_trf` over the article text, keep entities labeled
GPE (countries, cities, states) or LOC (non-GPE places like rivers, mountains),
filter a small demonym stoplist, and attach a short context window for
downstream disambiguation in the gazetteer.

Reuses GPU detection from `src.embed.model.detect_device` so the NER and the
embedding pipeline share the same device-selection convention.
"""
from __future__ import annotations

import logging
from typing import Iterable

from .models import RawMention

log = logging.getLogger(__name__)

# Cap per-doc spaCy input — pages over this get sliced into overlapping windows.
MAX_DOC_CHARS = 100_000
WINDOW_CHARS = 50_000
WINDOW_OVERLAP = 2_000

# 40-token half-window converted to ~250 chars on each side
CONTEXT_HALF_CHARS = 250

# Demonyms that spaCy occasionally tags as GPE — we don't want to geocode these.
DEMONYMS = frozenset({
    "american", "americans", "british", "english", "french", "german", "germans",
    "chinese", "japanese", "indian", "indians", "italian", "italians", "russian",
    "russians", "spanish", "european", "europeans", "asian", "asians", "african",
    "africans", "western", "westerners", "westerner", "easterner", "easterners",
    "northerner", "northerners", "southerner", "southerners",
})


class ToponymExtractor:
    """Wraps a spaCy pipeline. Loads the model once; call `extract` per doc."""

    def __init__(self, model_name: str = "en_core_web_trf", device: str | None = None):
        import spacy

        # If GPU is requested (or auto-detected) ask spaCy to prefer it.
        if device is None:
            from embed.model import detect_device

            device, _ = detect_device()

        if device == "cuda":
            try:
                spacy.prefer_gpu()
            except Exception as e:  # pragma: no cover — GPU init is environment-specific
                log.warning("spacy.prefer_gpu() failed (%s); falling back to CPU", e)

        log.info("loading spaCy model %s on %s", model_name, device)
        self.nlp = spacy.load(model_name)
        self.model_name = model_name
        self.device = device

    def extract(self, text: str) -> list[RawMention]:
        """Return all GPE/LOC mentions, with a short context window each."""
        if not text or not text.strip():
            return []

        if len(text) > MAX_DOC_CHARS:
            return list(self._extract_windowed(text))
        return list(self._extract_one(text, offset=0))

    def extract_batch(self, texts: list[str], batch_size: int = 16) -> list[list[RawMention]]:
        """Process many texts in one nlp.pipe call. Long docs are auto-windowed.

        Order is preserved: result[i] corresponds to texts[i].
        """
        results: list[list[RawMention]] = [[] for _ in texts]
        # Split long docs out: spaCy's pipe is most efficient on uniformly-sized inputs.
        short_indices = [i for i, t in enumerate(texts) if t and len(t) <= MAX_DOC_CHARS]
        short_texts = [texts[i] for i in short_indices]

        if short_texts:
            for idx, doc in zip(short_indices, self.nlp.pipe(short_texts, batch_size=batch_size)):
                results[idx] = list(self._mentions_from_doc(doc, offset=0))

        # Long docs: window each one separately (rare in this corpus).
        for i, t in enumerate(texts):
            if t and len(t) > MAX_DOC_CHARS:
                results[i] = list(self._extract_windowed(t))

        return results

    # ----- internals -----

    def _extract_one(self, text: str, offset: int) -> Iterable[RawMention]:
        doc = self.nlp(text)
        yield from self._mentions_from_doc(doc, offset=offset)

    def _extract_windowed(self, text: str) -> Iterable[RawMention]:
        seen: set[tuple[int, int, str]] = set()
        start = 0
        while start < len(text):
            end = min(start + WINDOW_CHARS, len(text))
            window = text[start:end]
            for m in self._extract_one(window, offset=start):
                key = (m.start_char, m.end_char, m.surface)
                if key in seen:
                    continue
                seen.add(key)
                yield m
            if end == len(text):
                break
            start = end - WINDOW_OVERLAP

    def _mentions_from_doc(self, doc, offset: int) -> Iterable[RawMention]:
        full_text = doc.text
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            surface = _normalize_surface(ent.text)
            if not surface or surface.lower() in DEMONYMS:
                continue
            ctx_start = max(0, ent.start_char - CONTEXT_HALF_CHARS)
            ctx_end = min(len(full_text), ent.end_char + CONTEXT_HALF_CHARS)
            yield RawMention(
                surface=surface,
                label=ent.label_,
                start_char=offset + ent.start_char,
                end_char=offset + ent.end_char,
                context=full_text[ctx_start:ctx_end],
            )


def _normalize_surface(s: str) -> str:
    s = " ".join(s.split())  # collapse whitespace
    if s.lower().startswith("the "):
        s = s[4:]
    return s.strip()
