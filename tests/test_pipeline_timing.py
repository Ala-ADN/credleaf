"""ETA honesty, history isolation, progress aggregation and heartbeat lifecycle."""
from datetime import datetime, timezone
import importlib
import json
import time

import httpx
import pytest
from jsonschema import ValidationError, validate

from config import PHASES_BY_NAME
from ingest import Seed, harvest_all_phases
from pipeline.artifacts import write_json
from pipeline.spec import defaults, schema, stage_config
from pipeline.timing import Progress, Tracker, estimate, history_estimate, timing_key


def test_estimates_have_no_invented_cold_start_or_negative_countdown():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    assert estimate(20)["remaining_seconds"] is None
    baseline = dict(total_seconds=100, low_seconds=50, high_seconds=200, source="manual", samples=0)
    value = estimate(20, baseline, now=now)
    assert value["remaining_seconds"] == 80
    assert value["estimated_finish_at"] == "2026-10-30T00:01:20+00:00"
    overrun = estimate(120, baseline)
    assert overrun["source"] == "overrunning_baseline" and overrun["remaining_seconds"] is None
    assert estimate(50, baseline, {"completed": 2, "total": 20})["source"] == "manual"
    live = estimate(50, baseline, {"completed": 5, "total": 20})
    assert live["source"] == "live_throughput" and live["remaining_seconds"] == 150
    finishing = estimate(50, baseline, {"completed": 20, "total": 20})
    assert finishing["source"] == "finalizing" and finishing["remaining_seconds"] is None
    assert estimate(50, terminal="failed")["remaining_seconds"] is None
    assert estimate(50, terminal="success")["remaining_seconds"] == 0


def test_history_scales_linear_work_but_not_nonlinear_fits(tmp_path):
    write_json(tmp_path / "good.json", dict(work={"unit": "input_bytes", "total": 100}, duration_seconds=20))
    (tmp_path / "corrupt.json").write_text("not json")
    write_json(tmp_path / "zero.json", dict(work={"unit": "input_bytes", "total": 0}, duration_seconds=20))
    assert history_estimate(tmp_path, {"unit": "input_bytes", "total": 1000}) is None
    value = history_estimate(tmp_path, {"unit": "input_bytes", "total": 1000}, linear=True)
    assert value["total_seconds"] == 200 and value["samples"] == 1
    assert history_estimate(tmp_path, {"unit": "input_bytes", "total": 105})["total_seconds"] == 20
    assert history_estimate(tmp_path, {"unit": "documents", "total": 100}) is None


@pytest.mark.parametrize("status, samples", [("success", 1), ("cached", 0), ("reused", 0), ("off", 0), ("failed", 0)])
def test_only_real_successful_work_trains_history(tmp_path, status, samples):
    report = {"stage": "dedupe", "status": "running"}
    tracker = Tracker(tmp_path, "dedupe", 13559, {}, {}, [], report, tmp_path / "step.json").start()
    report["status"] = status
    tracker.finish(status)
    assert len(list((tmp_path / "data/pipeline/timings").glob("*/*.json"))) == samples
    assert json.loads((tmp_path / "step.json").read_text())["eta"]["source"] == "actual"
    assert not tracker.thread.is_alive()


def test_heartbeat_reads_live_worker_progress_and_stops_before_final_report(tmp_path):
    progress = tmp_path / "progress.json"
    Progress(progress)(5, 10)
    report = {"stage": "normalize", "status": "running"}
    tracker = Tracker(tmp_path, "normalize", 13559, {}, {}, [], report, tmp_path / "step.json",
                      progress_path=progress, interval=.01).start()
    try:
        Progress(progress)(8, 10)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            value = json.loads((tmp_path / "step.json").read_text())
            if value["eta"]["progress"]["completed"] == 8:
                break
            time.sleep(.01)
        assert value["eta"]["progress"]["completed"] == 8
    finally:
        report["status"] = "success"
        tracker.finish("success")
    assert report["eta"]["remaining_seconds"] == 0 and not tracker.thread.is_alive()


def test_corrupt_progress_does_not_hide_the_baseline_or_final_elapsed_time(tmp_path):
    progress = tmp_path / "broken.json"
    progress.write_text("{")
    report = {"stage": "normalize", "status": "running"}
    tracker = Tracker(tmp_path, "normalize", None, {}, {}, [], report, tmp_path / "step.json",
                      hint=600, progress_path=progress).start()
    assert report["eta"]["source"] == "manual"
    tracker.finish("success")
    assert report["eta"]["source"] == "actual" and "duration_seconds" in report


def test_normalization_workload_honors_per_window_cap(tmp_path):
    from pipeline.timing import workload
    identifier = "harvest-" + "a" * 32
    write_json(tmp_path / "data/pipeline/artifacts" / identifier / "manifest.json", dict(files=[
        dict(path="captures/13559/P1_global_onset.jsonl", records=10000, bytes=1000000),
        dict(path="captures/13559/P2_pre_vaccine.jsonl", records=100, bytes=10000)]))
    assert workload(tmp_path, "normalize", {"normalize_max_docs": 500}, [{"id": identifier}]) == {
        "unit": "captures", "total": 600}


def test_timing_failure_is_advisory(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError("fixture timing disk failure")
    monkeypatch.setattr("pipeline.timing.write_json", unavailable)
    Progress(tmp_path / "progress.json")(3, 10)
    report = {"status": "running"}
    tracker = Tracker(tmp_path, "dedupe", None, {}, {}, [], report, tmp_path / "step.json").start()
    tracker.finish("success")
    assert not tracker.thread.is_alive()


def test_timing_keys_preserve_depth_runtime_and_scope():
    config = {"harvest_max_seeds": 5, "harvest_max_captures_per_seed": 10}
    original = timing_key("harvest", 13559, config, "cpu-runtime")
    assert original == timing_key("harvest", 13559, config | {"harvest_max_seeds": 100}, "cpu-runtime")
    assert original != timing_key("harvest", 13559, config | {"harvest_max_captures_per_seed": 100}, "cpu-runtime")
    assert original != timing_key("harvest", 13529, config, "cpu-runtime")
    assert original != timing_key("harvest", 13559, config, "gpu-runtime")


def test_eta_hints_are_validated_and_do_not_change_scientific_stage_configs():
    params = defaults()
    hinted = params | {"eta_stage_seconds": {"normalize": 7200}}
    validate(hinted, schema())
    assert stage_config("normalize", params, {}) == stage_config("normalize", hinted, {})
    for hints in ({"normalize": 0}, {"normalize": -1}, {"typo": 300}):
        with pytest.raises(ValidationError):
            validate(params | {"eta_stage_seconds": hints}, schema())


def test_harvest_progress_counts_failed_seeds_and_keeps_budget_upper_bound(tmp_path):
    class Client:
        def iter_seeds(self, *args, **kwargs):
            for i in range(4):
                yield Seed(id=i, collection=13559, url=f"https://example.org/{i}")
        def iter_captures(self, *args, **kwargs):
            raise httpx.ConnectError("fixture failed seed")
    updates = []
    harvest_all_phases(Client(), 13559, list(PHASES_BY_NAME.values()), output_dir=tmp_path,
                       max_seeds=5, max_seed_error_fraction=1,
                       progress=lambda done, total: updates.append((done, total)))
    assert updates == [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]


def test_normalization_progress_uses_collection_cap(tmp_path, monkeypatch):
    module = importlib.import_module("normalize.normalize")
    phases = list(PHASES_BY_NAME)[:2]
    for phase in phases:
        path = tmp_path / "raw/13559" / f"{phase}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((json.dumps(dict(phase=phase)) + "\n") * 4)
    def batch(client, path, output_path, progress, **kwargs):
        assert len(path.read_text().splitlines()) == 8
        progress(0, 3)
        for i in range(1, 4):
            progress(i, 3)
        return output_path, module.NormalizeStats(total=3)
    monkeypatch.setattr(module, "normalize_phase", batch)
    updates = []
    module.normalize_collection(None, 13559, phases, raw_root=tmp_path / "raw", output_root=tmp_path / "out",
                                max_docs=3, progress=lambda done, total: updates.append((done, total)))
    assert updates == [(0, 3), (1, 3), (2, 3), (3, 3)]
