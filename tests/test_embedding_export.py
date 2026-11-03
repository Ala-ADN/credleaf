import json
from types import SimpleNamespace

import numpy as np
import pytest

from embed.export import export_qdrant_corpus


class Client:
    def __init__(self, vectors):
        self.points = [SimpleNamespace(id=f"point-{i}", vector=v, payload={"text": f"row {i}"})
                       for i, v in enumerate(vectors)]
        self.pages = []

    def count(self, **kwargs):
        return SimpleNamespace(count=len(self.points))

    def scroll(self, *, offset, limit, **kwargs):
        assert limit <= 2
        start = offset or 0
        points = self.points[start:start + limit]
        self.pages.append(len(points))
        end = start + len(points)
        return points, end if end < len(self.points) else None


def test_streamed_export_preserves_alignment_across_pages(tmp_path):
    source = [[1., 2.], {"dense": [3., 4.]}, [5., 6.], [7., 8.], [9., 10.]]
    client = Client(source)
    result = export_qdrant_corpus(client, "snapshot", tmp_path, expected_count=5, batch_size=2)
    vectors = np.load(tmp_path / "embeddings.npy", mmap_mode="r")
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, np.arange(1, 11).reshape(5, 2))
    metadata = [json.loads(line) for line in (tmp_path / "metadata.jsonl").read_text().splitlines()]
    assert [row["point_id"] for row in metadata] == [f"point-{i}" for i in range(5)]
    assert [row["text"] for row in metadata] == [f"row {i}" for i in range(5)]
    assert client.pages == [2, 2, 1]
    assert result == dict(chunks=5, dimension=2, export_batch_size=2)


@pytest.mark.parametrize("bad", [[], None, [0., 0.], [float("nan"), 1.],
                               [float("inf"), 1.], [1.], {"a": [1., 2.], "b": [3., 4.]}])
def test_invalid_vectors_remove_partial_exports(tmp_path, bad):
    client = Client([[1., 2.], [3., 4.], bad])
    with pytest.raises(ValueError):
        export_qdrant_corpus(client, "snapshot", tmp_path, expected_count=3, batch_size=2)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("actual", [1, 3])
def test_truncated_or_growing_scroll_is_rejected(tmp_path, actual):
    client = Client([[1., 2.]] * actual)
    client.count = lambda **kwargs: SimpleNamespace(count=2)
    with pytest.raises(ValueError, match="count"):
        export_qdrant_corpus(client, "snapshot", tmp_path, expected_count=2, batch_size=2)
    assert not list(tmp_path.iterdir())


def test_existing_export_is_preserved(tmp_path):
    (tmp_path / "metadata.jsonl").write_text("existing\n")
    with pytest.raises(ValueError, match="overwrite"):
        export_qdrant_corpus(Client([[1., 2.]]), "snapshot", tmp_path, expected_count=1, batch_size=2)
    assert (tmp_path / "metadata.jsonl").read_text() == "existing\n"
