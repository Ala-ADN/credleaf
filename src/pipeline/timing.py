"""Advisory task ETAs, separate from immutable scientific artifacts.

Only successful computations train history. Estimates exclude queue time and
retries; a new machine/configuration starts without a fabricated baseline.
"""
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time
import uuid

from pipeline.artifacts import ArtifactStore, digest, write_json

log = logging.getLogger(__name__)
LINEAR = {"harvest", "normalize", "georef", "embed", "publish"}


def workload(root, stage, config, inputs):
    """Cheap inventory hints only; normal artifact verification stays authoritative."""
    if stage == "harvest":
        cap = config.get("harvest_max_seeds")
        return {"unit": "seed_budget" if cap is not None else "run", "total": cap or 1}
    if not inputs:
        return {"unit": "run", "total": 1}
    files = []
    store = ArtifactStore(root)
    for ref in inputs:
        if ref.get("id"):
            manifest = json.loads((store.directory(ref["id"]) / "manifest.json").read_text())
            files.extend(manifest["files"])
    if stage == "normalize":
        counts = [item["records"] for item in files if item["path"].endswith(".jsonl")]
        cap = config.get("normalize_max_docs")
        return {"unit": "captures", "total": max(1, sum(min(n, cap) if cap is not None else n for n in counts))}
    # Bytes account for text length; avoid pretending a nonlinear fit scales
    # linearly with chunk count. Such stages require a similarly sized sample.
    return {"unit": "input_bytes", "total": max(1, sum(f["bytes"] for f in files))}


def timing_key(stage, cid, config, runtime):
    config = dict(config)
    # These caps change workload, not the algorithm. Per-seed depth stays in
    # the harvest key because its effect is not represented by seed count.
    if stage == "harvest" and config.get("harvest_max_seeds") is not None:
        config.pop("harvest_max_seeds")
    return digest(dict(version=1, stage=stage, cid=cid, config=config, runtime=runtime,
                       host=platform.node(), machine=platform.machine(), cpus=os.cpu_count()))[:32]


def history_estimate(directory, work, linear=False):
    durations = []
    for path in sorted(Path(directory).glob("*.json"), reverse=True)[:100]:
        try:
            sample = json.loads(path.read_text())
            if sample["work"]["unit"] != work["unit"]:
                continue
            ratio = work["total"] / sample["work"]["total"]
            duration = float(sample["duration_seconds"])
            if not math.isfinite(duration) or duration <= 0 or not math.isfinite(ratio) or ratio <= 0:
                continue
            if linear or .9 <= ratio <= 1.1:
                durations.append(duration * ratio if linear else duration)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
    if not durations:
        return None
    middle = statistics.median(durations)
    # This is a planning range, not a statistical confidence interval.
    return dict(total_seconds=middle, low_seconds=min(min(durations), middle * .5),
                high_seconds=max(max(durations), middle * 2), samples=len(durations), source="history")


def estimate(elapsed, baseline=None, progress=None, terminal=None, now=None):
    now = now or datetime.now(timezone.utc)
    wall_elapsed = elapsed
    paused_seconds = (progress or {}).get("paused_seconds", 0.)
    elapsed = (progress or {}).get("processing_seconds", max(0., elapsed - paused_seconds))
    result = dict(elapsed_seconds=round(elapsed, 3), remaining_seconds=None,
                  wall_elapsed_seconds=round(wall_elapsed, 3), paused_seconds=paused_seconds,
                  estimated_finish_at=None, source="calibrating", updated_at=now.isoformat(),
                  excludes="queue time and retries")
    if terminal:
        result.update(source="actual", remaining_seconds=0 if terminal in {"success", "cached", "reused", "off"} else None)
        return result
    if progress and progress.get("paused"):
        result.update(source="paused", progress=progress)
        return result
    remaining = None
    if baseline:
        remaining = baseline["total_seconds"] - elapsed
        result.update(baseline)
        result["remaining_range_seconds"] = [max(0, baseline["low_seconds"] - elapsed),
                                              max(0, baseline["high_seconds"] - elapsed)]
        if remaining <= 0:
            remaining = None
            result["source"] = "overrunning_baseline"
    if progress:
        done, total = progress.get("completed"), progress.get("total")
        result["progress"] = progress
        # Extrapolate only after several items. Discovery budgets may be upper
        # bounds and heterogeneous seeds make harvest ETA particularly rough.
        if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
            if done >= total:
                remaining = None
                result["source"] = "finalizing"
            elif done >= 3 and elapsed > 0:
                remaining = elapsed / done * (total - done)
                result.update(source="live_throughput", total_seconds=elapsed + remaining,
                              remaining_range_seconds=[remaining * .5, remaining * 2])
    if remaining is not None and remaining > 0:
        result.update(remaining_seconds=max(.1, round(remaining, 1)),
                      estimated_finish_at=(now + timedelta(seconds=remaining)).isoformat())
    return result


def format_eta(value):
    if not value:
        return "waiting / no estimate"
    seconds = value.get("remaining_seconds")
    source = value["source"]
    if source == "actual":
        return f"{value['elapsed_seconds'] / 60:.1f} min elapsed"
    if seconds is None:
        return source.replace("_", " ")
    duration = "<1 min" if seconds < 60 else f"~{seconds / 60:.1f} min"
    return f"{duration} remaining ({source}; excludes queue/retries)"


class Tracker:
    """Single writer while a task runs; heartbeat is joined before final writes."""
    def __init__(self, root, stage, cid, config, runtime, inputs, report, path, *, hint=None,
                 progress_path=None, interval=30, learn=True):
        self.root, self.stage = Path(root), stage
        self.label = stage if cid is None else f"{stage}__{cid}"
        self.report, self.path = report, Path(path)
        self.progress_path, self.interval = progress_path, interval
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.work = {"unit": "run", "total": 1}
        self.history = None
        self.baseline = None
        self.learn = learn
        try:
            self.work = workload(root, stage, config, inputs)
            self.history = self.root / "data/pipeline/timings" / timing_key(stage, cid, config, runtime)
            self.baseline = history_estimate(self.history, self.work, stage in LINEAR) if learn else None
            if hint is not None and learn:
                self.baseline = dict(total_seconds=hint, low_seconds=hint * .5, high_seconds=hint * 2,
                                     source="manual", samples=0)
        except Exception:
            log.warning("ETA baseline unavailable", exc_info=True)

    def _refresh(self, terminal=None):
        progress = None
        if self.progress_path:
            try:
                progress = json.loads(Path(self.progress_path).read_text())
                if not isinstance(progress, dict):
                    progress = None
            except (OSError, ValueError):
                pass
        try:
            elapsed = time.monotonic() - self.started
            self.report["eta"] = estimate(elapsed, self.baseline, progress, terminal)
            self.report["duration_seconds"] = self.report["eta"]["elapsed_seconds"]
            write_json(self.path, self.report)
            print(f"ETA {self.label}: {format_eta(self.report['eta'])}", flush=True)
        except Exception:
            log.warning("ETA report unavailable", exc_info=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            self._refresh()

    def start(self):
        self._refresh()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True, name="stage-eta")
        self.thread.start()
        return self

    def finish(self, status):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self._refresh(status)
        if self.learn and status == "success" and self.history:
            try:
                write_json(self.history / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
                           dict(work=self.work, duration_seconds=self.report["duration_seconds"]))
            except Exception:
                log.warning("Could not save ETA sample", exc_info=True)


class Progress:
    """Optional worker sidecar. Telemetry errors never reject scientific output."""
    def __init__(self, path=None, unit="documents"):
        self.path = Path(path) if path else None
        self.unit = unit
        self.last = -math.inf
        self.extra = {}

    def set_context(self, **fields):
        self.extra.update(fields)

    def __call__(self, completed, total, *, force=False, **extra):
        self.extra.update(extra)
        if self.path and (force or time.monotonic() - self.last >= 5 or completed == total):
            try:
                write_json(self.path, dict(completed=completed, total=total, unit=self.unit, **self.extra,
                                          updated_at=datetime.now(timezone.utc).isoformat()))
                self.last = time.monotonic()
            except Exception:
                log.warning("Could not save worker progress", exc_info=True)
