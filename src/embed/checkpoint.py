"""Durable cursor and ownership for an unpublished embedding collection."""
import fcntl
import json
from pathlib import Path
import uuid

from pipeline.artifacts import digest, write_json


class EmbeddingCheckpoint:
    def __init__(self, directory, identity, prefix):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path = self.directory / "embedding.json"
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                checksum = self.state.pop("sha256", None)
                if checksum != digest(self.state):
                    raise ValueError("Embedding checkpoint checksum mismatch")
                if self.state.get("format") != 1 or self.state["identity"] != identity:
                    raise ValueError("Embedding checkpoint input, configuration or implementation changed")
            else:
                self.state = dict(format=1, identity=identity, collection=f"{prefix}_{uuid.uuid4().hex[:12]}",
                                  next_line=0, documents=0, chunks=0, embedding_complete=False)
                self._write()
        except BaseException:
            self.lock.close()
            raise

    def save(self, next_line, documents, chunks, *, complete=False):
        self.state.update(next_line=next_line, documents=documents, chunks=chunks, embedding_complete=complete)
        self._write()

    def _write(self):
        write_json(self.path, dict(self.state, sha256=digest(self.state)))

    def close(self):
        self.lock.close()
