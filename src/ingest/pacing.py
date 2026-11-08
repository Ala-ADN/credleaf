"""Cross-worker Wayback pacing, independent of scientific checkpoints."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

import httpx

from pipeline.artifacts import write_json


class ArchiveUnavailable(httpx.TransportError):
    """Transient archive interruption: keep the current capture uncommitted."""


class ArchivePacing:
    def __init__(self, path, minimum_interval=2.):
        self.path = Path(path)
        self.minimum = minimum_interval

    @contextmanager
    def state(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = (json.loads(self.path.read_text()) if self.path.exists() else
                     dict(interval=self.minimum, next_request=0., blocked_until=0., successes=0, cooldown=300.))
            value['interval'] = max(self.minimum, value['interval'])
            yield value
            write_json(self.path, value)

    def acquire(self):
        while True:
            with self.state() as value:
                now = time.time()
                if value['blocked_until'] > now:
                    raise ArchiveUnavailable(f"Archive cooldown until {value['blocked_until']:.0f}; reschedule without fetching")
                wait = value['next_request'] - now
                if wait <= 0:
                    value['next_request'] = now + value['interval']
                    return value['interval']
            # Short waits keep normal service shutdown within its grace period.
            time.sleep(min(wait, 1.))

    def result(self, *, transient=False, retry_after=0.):
        with self.state() as value:
            if transient:
                value['successes'] = 0
                value['interval'] = min(60., max(8., value['interval'] * 1.5))
                delay = max(value['cooldown'], retry_after)
                value['blocked_until'] = max(value['blocked_until'], time.time() + delay)
                value['cooldown'] = min(1800., value['cooldown'] * 2)
            else:
                value['successes'] += 1
                if value['successes'] >= 50:
                    value['interval'] = max(self.minimum, value['interval'] * .9)
                    value['cooldown'] = max(300., value['cooldown'] / 2)
                    value['successes'] = 0
            return value['interval']
