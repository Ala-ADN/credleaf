"""Interrupted embedding skips committed documents and deduplicates partial writes."""
import json
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from embed.checkpoint import EmbeddingCheckpoint
import embed.embed as module
from tests.test_embed_dedupe import document


@pytest.fixture
def embedding(tmp_path, monkeypatch):
    client = QdrantClient(":memory:")
    calls = []
    encoded = []
    class Handshake:
        def __init__(self, collection_name, **kwargs):
            self.collection_name, self.client = collection_name, client
            if not client.collection_exists(collection_name):
                client.create_collection(collection_name, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
            def encode(text):
                encoded.append(text)
                return [1., float(len(text))]
            self.embedding_model = SimpleNamespace(embed=encode)
    class Chunker:
        def chunk(self, text):
            calls.append(text)
            return [SimpleNamespace(text=text, start_index=0, end_index=len(text), token_count=1)]
    monkeypatch.setattr(module, "QdrantHandshake", Handshake)
    monkeypatch.setattr(module, "SemanticChunker", Chunker)
    source = tmp_path / "documents.jsonl"
    source.write_text("".join(json.dumps(document(text, capture_url=f"https://example.org/{i}")) + "\n"
                              for i, text in enumerate(["first document", "second document", "third document"])))
    yield client, calls, encoded, source, Chunker
    client.close()


def test_resume_skips_completed_documents_and_reuses_owned_collection(embedding, tmp_path, monkeypatch):
    client, calls, encoded, source, Chunker = embedding
    original = Chunker.chunk
    def interrupt(self, text):
        if text == "second document":
            raise KeyboardInterrupt()
        return original(self, text)
    monkeypatch.setattr(Chunker, "chunk", interrupt)
    checkpoint = EmbeddingCheckpoint(tmp_path / "checkpoint", "input/code", "fixture")
    name = checkpoint.state["collection"]
    try:
        with pytest.raises(KeyboardInterrupt):
            module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint)
    finally:
        checkpoint.close()
    monkeypatch.setattr(Chunker, "chunk", original)
    checkpoint = EmbeddingCheckpoint(tmp_path / "checkpoint", "input/code", "fixture")
    try:
        assert checkpoint.state["next_line"] == 1
        assert module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint) == 3
        assert calls == ["first document", "second document", "third document"]
        assert client.count(name, exact=True).count == 3
        # A retry after encoding completed only needs a fresh export.
        assert module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint) == 3
        assert len(calls) == 3
    finally:
        checkpoint.close()


def test_vectors_written_before_cursor_commit_are_not_reencoded(embedding, tmp_path, monkeypatch):
    client, calls, encoded, source, _ = embedding
    checkpoint = EmbeddingCheckpoint(tmp_path / "checkpoint", "input/code", "fixture")
    name = checkpoint.state["collection"]
    def power_loss(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(checkpoint, "save", power_loss)
    try:
        with pytest.raises(KeyboardInterrupt):
            module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint)
        assert client.count(name, exact=True).count == 1
    finally:
        checkpoint.close()
    checkpoint = EmbeddingCheckpoint(tmp_path / "checkpoint", "input/code", "fixture")
    try:
        assert checkpoint.state["next_line"] == 0
        assert module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint) == 3
        assert encoded == ["first document", "second document", "third document"]
    finally:
        checkpoint.close()


def test_checkpoint_rejects_changed_identity_and_concurrent_writers(tmp_path):
    checkpoint = EmbeddingCheckpoint(tmp_path, "original", "fixture")
    try:
        with pytest.raises(BlockingIOError):
            EmbeddingCheckpoint(tmp_path, "original", "fixture")
    finally:
        checkpoint.close()
    with pytest.raises(ValueError, match="changed"):
        EmbeddingCheckpoint(tmp_path, "different", "fixture")
    state = json.loads((tmp_path / "embedding.json").read_text())
    state["next_line"] = 1000
    (tmp_path / "embedding.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checksum"):
        EmbeddingCheckpoint(tmp_path, "original", "fixture")


def test_missing_committed_vectors_are_not_silently_skipped(embedding, tmp_path):
    client, _, _, source, _ = embedding
    checkpoint = EmbeddingCheckpoint(tmp_path / "checkpoint", "input/code", "fixture")
    name = checkpoint.state["collection"]
    try:
        assert module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint) == 3
        client.delete_collection(name)
        with pytest.raises(ValueError, match="lost committed"):
            module.embed_to_qdrant(source, "unused", name, checkpoint=checkpoint)
    finally:
        checkpoint.close()
