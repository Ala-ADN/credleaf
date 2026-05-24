"""Near-Deduplicate normalized documents per collection using MinHash + LSH.

Usage:
	uv run python scripts/_dedupe.py <collection_id> [--max-docs N]

Outputs one JSONL per phase at:
	data/processed/deduped/<collection_id>/<phase>.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from config import COVID_COLLECTIONS, PROCESSED_DIR

K_SHINGLE = 5  # word k-shingle size
N_HASHES = 128  # MinHash signature length
THRESHOLD = 0.90

_word_re = re.compile(r"\w+")
_logger = logging.getLogger(__name__)

def load_jsonl(path: Path, max_docs: int | None = None) -> tuple[list[dict], list[str]]:
    docs, raw_lines = [], []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_docs is not None and i >= max_docs:
                break
            if line := line.strip():
                docs.append(json.loads(line))
                raw_lines.append(line)
    return docs, raw_lines


def get_shingles(text: str | None, k: int = K_SHINGLE) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    tokens = _word_re.findall(text.lower())
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def to_minhash(shingles: set[str]) -> MinHash:
    m = MinHash(num_perm=N_HASHES)
    for s in shingles:
        m.update(s.encode())
    return m


def jaccard(s1: set[str], s2: set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / (len(s1) + len(s2) - len(s1 & s2))


def dedupe_phase(normalized_path: Path, output_path: Path, max_docs: int | None = None) -> None:
    docs, raw_lines = load_jsonl(normalized_path, max_docs=max_docs)
    if not docs:
        print(f"no docs in {normalized_path}")
        return

    ok_positions = [i for i, d in enumerate(docs) if d.get("fetch_status") == "ok"]
    ok_docs = [docs[i] for i in ok_positions]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(ok_docs) < 2:
        output_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        print(f"{normalized_path.name}: <2 ok docs, wrote unchanged")
        _logger.info("%s: duplicate rate 0.00%% (%d/%d ok docs)", normalized_path.name, 0, len(ok_docs))
        return

    shingle_sets = [get_shingles(d.get("text")) for d in ok_docs]
    minhashes = [to_minhash(s) for s in shingle_sets]

    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=N_HASHES)
    for idx, mh in enumerate(minhashes):
        lsh.insert(str(idx), mh)

    dup_pairs = [
    	(idx, j)
    	for idx, mh in enumerate(minhashes)
    	for match in lsh.query(mh)
    	if (j := int(str(match))) > idx and jaccard(shingle_sets[idx], shingle_sets[j]) >= THRESHOLD
	]

    parent = list(range(len(ok_docs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in dup_pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ok_docs)):
        clusters[find(i)].append(i)

    keep = {ok_positions[min(c)] for c in clusters.values()}

    dup_count = len(ok_docs) - len(keep)
    dup_percent = (dup_count / len(ok_docs)) * 100.0 if ok_docs else 0.0

    with output_path.open("w", encoding="utf-8") as fh:
        for i, line in enumerate(raw_lines):
            if docs[i].get("fetch_status") != "ok" or i in keep:
                fh.write(line + "\n")

    _logger.info(
        "%s: duplicate rate %.2f%% (%d/%d ok docs)",
        normalized_path.name,
        dup_percent,
        dup_count,
        len(ok_docs),
    )
    print(f"wrote {output_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("collection_id", type=int, choices=list(COVID_COLLECTIONS))
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    normalized_root = PROCESSED_DIR / "normalized" / str(args.collection_id)
    if not normalized_root.exists():
        print(f"missing normalized dir: {normalized_root}", file=sys.stderr)
        return 2

    phase_files = sorted(normalized_root.glob("*.jsonl"))
    if not phase_files:
        print(f"no JSONL files in {normalized_root}", file=sys.stderr)
        return 3

    output_root = PROCESSED_DIR / "deduped" / str(args.collection_id)
    for path in phase_files:
        dedupe_phase(path, output_root / path.name, max_docs=args.max_docs)
    return 0


if __name__ == "__main__":
    sys.exit(main())