"""Embedding identities stay stable across batch sizes, reruns, and captures."""
import json
from types import SimpleNamespace

import pytest

import embed.embed as module


def document(text, **changes):
    row = dict(text=text, fetch_status="ok", collection_id=13559, capture_url="https://example.org/a",
               seed_url="https://example.org/", language="en", publish_date="2020-02-15",
               capture_timestamp="20200310120000", digest="old")
    row.update(changes)
    return row


def fake_pipeline(monkeypatch, points):
    class Client:
        def count(self, **kwargs):
            return SimpleNamespace(count=sum(p.payload.get("dedupe_version") != 2 for p in points.values()))
        def upsert(self, *, collection_name, points: list, wait):
            for point in points:
                stored[point.id] = point
    stored = points
    class Handshake:
        def __init__(self, **kwargs):
            self.collection_name = kwargs["collection_name"]
            self.client = Client()
            self.embedding_model = SimpleNamespace(embed=lambda text: [1., float(len(text))])
    class Chunker:
        def chunk(self, text):
            return [SimpleNamespace(text=t.strip(), start_index=0, end_index=len(t), token_count=1)
                    for t in text.split("|")]
    monkeypatch.setattr(module, "QdrantHandshake", Handshake)
    monkeypatch.setattr(module, "SemanticChunker", Chunker)


def test_reruns_and_rebatching_do_not_multiply_chunks_or_overwrite_other_pages(tmp_path, monkeypatch):
    points = {}
    fake_pipeline(monkeypatch, points)
    rows = [document("shared text | shared text | unique text"),
            document("shared text", capture_url="https://example.org/b"),
            document("shared text", publish_date="2020-03-15"),
            document("shared text", digest="new", capture_timestamp="20200410120000")]
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    written = module.embed_to_qdrant(path, "unused", "test", batch_size=1)
    assert written == len(points) == 4
    original_ids = set(points)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[::-1]))
    module.embed_to_qdrant(path, "unused", "test", batch_size=3)
    assert set(points) == original_ids
    assert {point.payload["article_id"] for point in points.values()} == {"https://example.org/a", "https://example.org/b"}
    assert {point.payload["observation_period"] for point in points.values()} == {"2020-02", "2020-03"}


def test_existing_legacy_points_require_a_fresh_collection(tmp_path, monkeypatch):
    points = {"old": SimpleNamespace(payload={"text": "legacy"})}
    fake_pipeline(monkeypatch, points)
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(document("some text")) + "\n")
    with pytest.raises(ValueError, match="fresh collection"):
        module.embed_to_qdrant(path, "unused", "test")
    assert set(points) == {"old"}
