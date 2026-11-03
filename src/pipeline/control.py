"""Cooperative pause requests and acknowledgements at safe worker boundaries."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from pipeline.artifacts import write_json


class RunControl:
    def __init__(self, path=None, key="worker", progress=None):
        self.path = Path(path) if path else None
        self.key, self.progress, self.paused_seconds = key, progress, 0.

    def requested(self):
        if self.path is None or not self.path.exists():
            return False
        # A malformed existing request must not accidentally resume a worker.
        return bool(json.loads(self.path.read_text()).get("pause_requested"))

    def wait(self, completed=0, total=None, checkpoint=None):
        if not self.requested():
            return
        started = time.monotonic()
        ack = self.path.with_name(self.key + ".json")
        request_id = json.loads(self.path.read_text()).get("request_id")
        state = dict(status="paused", completed=completed, total=total,
            checkpoint=str(checkpoint) if checkpoint else None,
            paused_at=datetime.now(timezone.utc).isoformat(), safe_boundary="document" if checkpoint else "stage",
            request_id=request_id)
        write_json(ack, state)
        while self.requested():
            current_id = json.loads(self.path.read_text()).get("request_id")
            if current_id != request_id:
                request_id = current_id
                state["request_id"] = current_id
                write_json(ack, state)
            if self.progress:
                self.progress(completed, total, paused=True,
                              paused_seconds=self.paused_seconds + time.monotonic() - started)
            time.sleep(1)
        self.paused_seconds += time.monotonic() - started
        if self.progress:
            self.progress(completed, total if self.key.startswith("normalize__") else None,
                          paused=False, paused_seconds=self.paused_seconds, force=True)
        write_json(ack, dict(status="running", resumed_at=datetime.now(timezone.utc).isoformat()))
