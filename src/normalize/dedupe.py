"""Conservative, source- and time-aware content deduplication.

Exact hashes bypass approximate search. MinHash is only a candidate generator:
every near duplicate must match a retained representative directly. Removing a
record never adds it to the index, so similarity chains cannot erase new content.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata
from urllib.parse import urlsplit

from datasketch import MinHash, MinHashLSH
from document_time import document_time
from url_utils import canonical_url

DEDUPE_VERSION = 2
K_SHINGLE = 5
N_HASHES = 128
THRESHOLD = .90
WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
NEGATIONS = frozenset({"no", "not", "never", "without", "cannot", "can't", "don't", "doesn't", "isn't", "wasn't", "won't"})


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def observation_period(doc: dict) -> str:
    """Publication month when available, otherwise capture month (UTC)."""
    resolved = document_time(doc)
    return resolved.timestamp.strftime("%Y-%m") if resolved.timestamp else "undated:" + str(doc.get("phase") or "unknown")


def source_scope(doc: dict) -> tuple[str, str, str, str]:
    url = canonical_url(doc.get("capture_url") or doc.get("seed_url") or "")
    host = urlsplit(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    # Unknown sources stay separate: do not erase their provenance by matching.
    host = host or "unknown:" + str(doc.get("seed_id") or doc.get("seed_url") or doc.get("digest"))
    return (str(doc.get("collection_id") or ""), host,
            observation_period(doc), str(doc.get("language") or "unknown").lower())


def remove_repeated_blocks(text: str, *, min_words=20) -> tuple[str, int]:
    """Remove repeated substantive lines within a page; keep the first instance.

    No cross-page boilerplate inference or truncation: unique text, short labels,
    and changed claims remain intact. Extraction emits paragraphs as lines.
    """
    seen, lines, removed = set(), [], 0
    for line in text.splitlines(keepends=True):
        if len(WORDS.findall(line)) >= min_words:
            key = normalize_text(line)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        lines.append(line)
    return "".join(lines).strip(), removed


def get_shingles(text: str | None, k=K_SHINGLE) -> set[bytes]:
    """Stable 64-bit shingle hashes keep long-page indexes reasonably small."""
    tokens = WORDS.findall(normalize_text(text or ""))
    if not tokens:
        return set()
    spans = (tokens[i:i+k] for i in range(max(1, len(tokens) - k + 1)))
    return {hashlib.blake2b(" ".join(span).encode(), digest_size=8).digest() for span in spans}


def to_minhash(shingles: set[bytes]) -> MinHash:
    signature = MinHash(num_perm=N_HASHES, seed=42)
    # Bound temporary arrays for very long bibliography/index pages.
    batch = []
    for shingle in shingles:
        batch.append(shingle)
        if len(batch) == 512:
            signature.update_batch(batch)
            batch.clear()
    if batch:
        signature.update_batch(batch)
    return signature


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def _claim_markers(text: str):
    """Do not collapse near-identical pages that change numbers or negation."""
    text = normalize_text(text)
    tokens = WORDS.findall(text)
    return (tuple(NUMBERS.findall(text)),
            tuple(tuple(tokens[max(0, i-3):i+4]) for i, token in enumerate(tokens) if token in NEGATIONS))


@dataclass
class DedupeResult:
    kept: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def deduplicate_documents(records: list[dict], *, threshold=THRESHOLD) -> DedupeResult:
    """Deduplicate one collection's captures across all supplied phase files.

    Input entries contain ``doc`` plus stable ``origin`` (file:line). Successful
    documents only are compared. Output ordering follows input order, while
    representative selection is independent of that order. Distinct source
    hosts, languages, collections, and observation months are never collapsed.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must satisfy 0 < threshold <= 1")
    prepared = []
    for record in records:
        doc = record["doc"].copy()
        removed = 0
        if doc.get("fetch_status") == "ok" and isinstance(doc.get("text"), str):
            doc["text"], removed = remove_repeated_blocks(doc["text"])
            doc["word_count"] = len(doc["text"].split())
            doc["char_count"] = len(doc["text"])
        prepared.append(dict(record, doc=doc, removed_blocks=removed))

    def quality(index):
        item, doc = prepared[index], prepared[index]["doc"]
        return (doc.get("extraction_mode") != "article", not bool(doc.get("publish_date")),
                -len(doc.get("text") or ""), str(doc.get("capture_timestamp") or "~"), item["origin"])

    exact, indexes, features, marker_sets = {}, {}, {}, {}
    decisions = {}
    for index in sorted(range(len(prepared)), key=quality):
        item = prepared[index]
        doc = item["doc"]
        if doc.get("fetch_status") != "ok" or not (doc.get("text") or "").strip():
            decisions[index] = (index, "not_comparable", None)
            continue
        scope, digest = source_scope(doc), content_hash(doc["text"])
        key = (scope, digest)
        if key in exact:
            decisions[index] = (exact[key], "exact", 1.)
            continue
        shingles = get_shingles(doc["text"])
        markers = _claim_markers(doc["text"])
        signature = to_minhash(shingles) if shingles and threshold < 1 else None
        # Similar templates on different article URLs are not near-duplicates.
        # Exact same-source copies can still share a canonical representative.
        page = canonical_url(doc.get("capture_url") or doc.get("seed_url") or "")
        near_scope = (scope, page)
        if near_scope not in indexes:
            indexes[near_scope] = MinHashLSH(threshold=max(.1, threshold - .1), num_perm=N_HASHES)
        lsh = indexes[near_scope]
        matches = []
        if signature is not None:
            for candidate_key in lsh.query(signature):
                candidate = int(candidate_key)
                if marker_sets[candidate] != markers:
                    continue
                score = jaccard(shingles, features[candidate])
                if score >= threshold:
                    matches.append((-score, quality(candidate), candidate))
        if matches:
            best = min(matches)
            decisions[index] = (best[2], "near", -best[0])
        else:
            decisions[index] = (index, "retained", 1.)
            exact[key] = index
            if signature is not None:
                lsh.insert(str(index), signature)
                features[index], marker_sets[index] = shingles, markers

    result = DedupeResult()
    reasons = Counter()
    for index, item in enumerate(prepared):
        representative, reason, similarity = decisions[index]
        reasons[reason] += 1
        original, retained = item["doc"], prepared[representative]
        digest = content_hash(original.get("text") or "")
        record = dict(origin=item["origin"], representative=retained["origin"],
                      action="keep" if representative == index else "drop", reason=reason,
                      similarity=similarity, content_hash=digest,
                      scope=list(source_scope(original)), removed_blocks=item["removed_blocks"],
                      provenance={k: original.get(k) for k in ["collection_id", "seed_id", "seed_url", "capture_url",
                                                             "capture_timestamp", "capture_phase", "phase", "digest", "publish_date", "event_date_source"]})
        result.audit.append(record)
        if representative == index:
            if reason != "not_comparable":
                original["deduplication"] = dict(version=DEDUPE_VERSION, content_hash=digest,
                                                  observation_period=observation_period(original),
                                                  representative=item["origin"])
            result.kept.append(dict(item, doc=original))
    result.stats = dict(version=DEDUPE_VERSION, threshold=threshold, input_documents=len(records),
                        retained_documents=len(result.kept), removed_documents=len(records)-len(result.kept),
                        exact_duplicates=reasons["exact"], near_duplicates=reasons["near"],
                        not_comparable=reasons["not_comparable"],
                        repeated_blocks_removed=sum(item["removed_blocks"] for item in prepared))
    return result
