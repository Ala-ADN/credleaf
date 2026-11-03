"""Durable, input-bound document journal independent of staged artifacts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

from pipeline.artifacts import write_json, sync_directory


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


class DocumentJournal:
    def __init__(self, directory, identity, records):
        self.directory, self.records = Path(directory), records
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            header = self.directory / "identity.json"
            expected = dict(format=1, identity=identity, total=len(records))
            if header.exists():
                if json.loads(header.read_text()) != expected:
                    raise ValueError("Checkpoint input, configuration or implementation changed")
            else:
                write_json(header, expected)
            self.stream = (self.directory / "journal.jsonl").open("a+b")
            sync_directory(self.directory)
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.stream.close()
        self.lock.close()

    def replay(self, output):
        """Validate the committed prefix; discard only an incomplete final write."""
        self.stream.seek(0)
        count, stats, errors, seconds = 0, None, [], 0.
        while True:
            start = self.stream.tell()
            line = self.stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                self.stream.truncate(start)
                self.stream.flush()
                os.fsync(self.stream.fileno())
                break
            row = json.loads(line)
            payload = row["payload"]
            if row["sha256"] != fingerprint(payload):
                raise ValueError("Checkpoint row checksum mismatch")
            if count >= len(self.records) or payload["index"] != count:
                raise ValueError("Checkpoint is not a contiguous input prefix")
            if payload["capture_sha256"] != fingerprint(self.records[count].model_dump(mode="json")):
                raise ValueError("Checkpoint capture ordering changed")
            doc = payload["document"]
            record = self.records[count]
            if (doc["capture_url"] != record.capture.url or doc["capture_timestamp"] != record.capture.timestamp
                    or doc["collection_id"] != record.collection_id or doc["seed_id"] != record.seed_id
                    or doc.get("digest") != record.capture.digest):
                raise ValueError("Checkpoint document provenance mismatch")
            output.write(json.dumps(doc, ensure_ascii=False) + "\n")
            stats, seconds = payload["stats"], payload.get("active_seconds", 0.)
            if doc.get("error"):
                errors.append((doc["capture_timestamp"], doc["error"]))
            count += 1
            if stats["total"] != count:
                raise ValueError("Checkpoint counters do not match document position")
        self.stream.seek(0, 2)
        return count, stats, errors, seconds

    def append(self, index, record, document, stats, active_seconds):
        payload = dict(index=index, capture_sha256=fingerprint(record.model_dump(mode="json")),
                       document=document, stats=stats, active_seconds=active_seconds)
        line = json.dumps(dict(payload=payload, sha256=fingerprint(payload)), ensure_ascii=False).encode() + b"\n"
        self.stream.write(line)
        self.stream.flush()
        os.fsync(self.stream.fileno())
