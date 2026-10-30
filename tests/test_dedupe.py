"""Regression tests for content loss, duplicate recall, time, and provenance."""
import importlib.util
import json
from pathlib import Path

import pytest

from normalize.dedupe import (
    canonical_url, content_hash, deduplicate_documents, get_shingles, jaccard,
    observation_period, remove_repeated_blocks,
)


def record(origin, text, **changes):
    doc = dict(collection_id=13559, capture_url="https://news.example/article", seed_url="https://news.example/",
               capture_timestamp="20200310120000", phase="P1_global_onset", publish_date="2020-02-15",
               language="en", fetch_status="ok", text=text, extraction_mode="article", digest=origin)
    doc.update(changes)
    return dict(origin=origin, doc=doc)


def words(n=200):
    return ["word" + chr(97+i//26) + chr(97+i % 26) for i in range(n)]


def test_exact_hashes_catch_unicode_whitespace_and_cross_phase_repeats(monkeypatch):
    # Exact matching must not depend on probabilistic candidate recall.
    from datasketch import MinHashLSH
    monkeypatch.setattr(MinHashLSH, "query", lambda *args: [])
    first = record("p1:1", "Ａ clear statement about public health.")
    later = record("p2:1", "a CLEAR  statement\tabout public health.", phase="P2_pre_vaccine", capture_timestamp="20201210120000")
    result = deduplicate_documents([first, later])
    assert len(result.kept) == 1 and result.stats["exact_duplicates"] == 1
    assert {r["provenance"]["phase"] for r in result.audit} == {"P1_global_onset", "P2_pre_vaccine"}
    assert all(r["representative"] == result.kept[0]["origin"] for r in result.audit)
    assert first["doc"]["text"].startswith("Ａ")  # input untouched


def test_same_text_preserves_sources_languages_collections_and_observation_months():
    text = "same reported fact about the pandemic"
    records = [record("a", text),
               record("source", text, capture_url="https://other.example/article"),
               record("language", text, language="fr"),
               record("collection", text, collection_id=13529),
               record("month", text, publish_date="2020-03-15")]
    assert len(deduplicate_documents(records).kept) == len(records)
    assert observation_period(dict(publish_date="2020-02-01T00:30:00+02:00")) == "2020-01"
    assert observation_period(dict(capture_timestamp="20210402000000")) == "2021-04"


def test_near_duplicates_are_verified_against_retained_representatives_not_chains():
    a = words()
    b = a[:-6] + ["zzzz" + word[-2:] for word in a[-6:]]
    c = a[:-12] + ["zzzz" + word[-2:] for word in a[-12:]]
    texts = [" ".join(tokens) for tokens in [a, b, c]]
    assert jaccard(get_shingles(texts[0]), get_shingles(texts[1])) >= .9
    assert jaccard(get_shingles(texts[1]), get_shingles(texts[2])) >= .9
    assert jaccard(get_shingles(texts[0]), get_shingles(texts[2])) < .9
    rows = [record(name, text) for name, text in zip("ABC", texts)]
    result = deduplicate_documents(rows)
    assert len(result.kept) == 2
    assert result.stats["near_duplicates"] == 1
    assert {r["origin"] for r in result.kept} == {"A", "C"}
    reversed_result = deduplicate_documents(rows[::-1])
    assert {r["origin"] for r in result.kept} == {r["origin"] for r in reversed_result.kept}


def test_candidate_collision_is_not_evidence_of_duplication(monkeypatch):
    from datasketch import MinHashLSH
    def query_all(self, signature):
        return list(self.keys.keys())
    monkeypatch.setattr(MinHashLSH, "query", query_all)
    result = deduplicate_documents([record("a", " ".join(words())),
                                    record("b", " ".join("different" + t for t in words()))])
    assert len(result.kept) == 2


def test_similar_articles_at_different_urls_are_not_deleted_as_near_duplicates():
    base = " ".join(words(240))
    result = deduplicate_documents([
        record("a", base + " town response"),
        record("b", base + " city response", capture_url="https://news.example/different-story"),
    ])
    assert len(result.kept) == 2


@pytest.mark.parametrize("first,last", [("100 cases", "101 cases"), ("will cure disease", "will not cure disease")])
def test_small_changes_to_numbers_and_negation_are_retained(first, last):
    base = " ".join(words(240))
    result = deduplicate_documents([record("a", base + " " + first), record("b", base + " " + last)])
    assert len(result.kept) == 2


def test_repeated_blocks_are_removed_without_truncating_unique_text():
    block = " ".join(words(30))
    text = block + "\n" + block + "\n" + block + " not\nShort label\nShort label\n"
    cleaned, removed = remove_repeated_blocks(text)
    assert removed == 1 and len(cleaned.splitlines()) == 4
    assert block + " not" in cleaned
    assert cleaned.count("Short label") == 2
    result = deduplicate_documents([record("a", text)])
    doc = result.kept[0]["doc"]
    assert doc["word_count"] == len(cleaned.split()) and doc["char_count"] == len(cleaned)
    assert result.stats["repeated_blocks_removed"] == 1


def test_failures_and_empty_text_never_match_empty_minhash_signatures():
    result = deduplicate_documents([record("a", "", fetch_status="empty"),
                                    record("b", None, fetch_status="fetch_failed"),
                                    record("c", "!!!"), record("d", "???")])
    assert len(result.kept) == 4
    assert result.stats["not_comparable"] == 2


def test_page_identity_preserves_query_content_and_removes_tracking():
    assert canonical_url("https://EXAMPLE.org:443/p?id=1&utm_source=a#x") == "https://example.org/p?id=1"
    assert canonical_url("https://example.org/p?id=1") != canonical_url("https://example.org/p?id=2")
    assert content_hash("Full width Ａ  text") == content_hash("full width a\ttext")


def test_collection_run_overwrites_empty_phases_and_preserves_audit_backup(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/_dedupe.py"
    spec = importlib.util.spec_from_file_location("dedupe_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "normalized", tmp_path / "deduped"
    source.mkdir(); output.mkdir()
    a = record("a", "this is exactly the same source text")
    b = record("b", a["doc"]["text"], phase="P2_pre_vaccine", capture_timestamp="20201210120000")
    for name, row in [("P1_global_onset", a), ("P2_pre_vaccine", b)]:
        (source / f"{name}.jsonl").write_text(json.dumps(row["doc"]) + "\n")
    (output / "P2_pre_vaccine.jsonl").write_text("stale data\n")
    (output / "P1_first_wave.jsonl").write_text("legacy data\n")
    summary = module.dedupe_collection(source, output)
    assert summary["retained_documents"] == 1
    assert (output / "P2_pre_vaccine.jsonl").read_text() == ""
    assert not (output / "P1_first_wave.jsonl").exists()
    assert (Path(summary["backup"]) / "P2_pre_vaccine.jsonl").read_text() == "stale data\n"
    assert len((output / "audit/duplicates.jsonl").read_text().splitlines()) == 2
    from config import PHASES_BY_NAME
    assert len(list(output.glob("*.jsonl"))) == len(PHASES_BY_NAME) + 2  # phases + outside/undated, excluding audit
    again = module.dedupe_collection(source, output)
    assert again["retained_documents"] == summary["retained_documents"]
