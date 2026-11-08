"""The dashboard must distinguish live estimates from scheduler and stale state."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "airflow/plugins/credleaf_ui/data.py"
spec = importlib.util.spec_from_file_location("dashboard_data", MODULE)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def fixture(tmp_path):
    directory = tmp_path / "data/pipeline/runs/fixture"
    (directory / "steps").mkdir(parents=True)
    plan = dict(parameters=dict(collection_ids=[13529], normalize_max_docs=None),
                kind="corpus", dag_id="credleaf_build_corpus", run_id="test")
    (directory / "plan.json").write_text(json.dumps(plan))
    (directory / "steps/harvest__13529.json").write_text(json.dumps(dict(status="reused",
        files=[dict(path="captures/13529/captures.jsonl", records=10, bytes=100)])))
    (directory / "steps/normalize__13529.json").write_text(json.dumps(dict(status="running", eta=dict(
        remaining_seconds=90, estimated_finish_at="2026-11-02T00:02:00+00:00", updated_at="2026-11-02T00:00:30+00:00",
        progress=dict(completed=3, total=10)))))
    return directory


def test_live_eta_and_coverage(tmp_path):
    fixture(tmp_path)
    value = data.collect(tmp_path, now=datetime(2026, 11, 2, 0, 1, tzinfo=timezone.utc))
    run = value["runs"][0]
    assert run["eta_lower_bound_seconds"] == 90
    assert run["normalization"] == dict(completed=3, total=10)
    assert run["unestimated_steps"] > 0


def test_stale_eta_never_looks_live(tmp_path):
    fixture(tmp_path)
    run = data.collect(tmp_path, now=datetime(2026, 11, 2, 1, tzinfo=timezone.utc))["runs"][0]
    step = next(s for s in run["stages"] if s["stage"] == "normalize")
    assert step["eta"]["stale"] is True
    assert step["eta"]["remaining_seconds"] is None
    assert step["eta"]["estimated_finish_at"] is None


def test_scheduler_failure_overrides_unfinished_worker_report(tmp_path):
    fixture(tmp_path)
    states = {("credleaf_build_corpus", "test"): dict(state="failed", tasks={
        ("collection_flow.normalize", 0): dict(state="failed")})}
    run = data.collect(tmp_path, states, datetime(2026, 11, 2, 0, 1, tzinfo=timezone.utc))["runs"][0]
    assert run["state"] == "failed"
    step = next(s for s in run["stages"] if s["stage"] == "normalize")
    assert step["status"] == "failed" and step["eta"]["remaining_seconds"] is None


def test_catalog_does_not_change_frozen_scientific_identity(tmp_path):
    from pipeline.artifacts import implementation
    for folder in ("src", "scripts", "dags"):
        (tmp_path / folder).mkdir()
    before = implementation(tmp_path)
    catalog = tmp_path / "airflow/catalog"
    catalog.mkdir(parents=True)
    (catalog / "catalog.py").write_text("# independent observation\n")
    (tmp_path / "dags/observability").symlink_to(catalog, target_is_directory=True)
    assert implementation(tmp_path) == before


def test_capture_coverage_uses_dates_instead_of_historical_labels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.bridge import capture_coverage
    source = tmp_path / "captures.jsonl"
    source.write_text("\n".join(json.dumps(dict(phase="incorrect", capture=dict(timestamp=stamp)))
        for stamp in ("20200301000000", "20201201000000")) + "\n")
    store = SimpleNamespace(payload=lambda ref: tmp_path,
                            load=lambda ref: dict(files=[dict(path="captures.jsonl")]))
    counts = capture_coverage(tmp_path, store, dict(id="harvest-fixture"))
    assert counts == {"P1_global_onset": 1, "P2_pre_vaccine": 1}


def test_native_notes_preserve_user_text_and_fit_airflow_limit(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.native import merge_note, START, END
    original = "My research note."
    first = merge_note(original, "ETA: 2 h")
    second = merge_note(first, "ETA: paused")
    assert original in second and "ETA: 2 h" not in second
    assert second.count(START) == second.count(END) == 1
    assert len(merge_note("x"*850, "y"*500)) <= 1000
    assert merge_note("x"*990, "summary") == "x"*990


def test_native_links_use_mapped_task_xcom_route(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.native import task_url
    url = task_url(dict(run_link="/dags/build/runs/example"), dict(task_id="collection_flow.normalize", map_index=1))
    assert url == "/dags/build/runs/example/tasks/collection_flow.normalize/mapped/1/xcom"


def test_native_event_coverage_uses_publication_dates(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.native import artifact_coverage
    directory = tmp_path / "normalized/13559"
    directory.mkdir(parents=True)
    (directory / "documents.jsonl").write_text("\n".join(json.dumps(row) for row in [
        dict(publish_date="2020-03-01", capture_timestamp="20220101000000", phase="incorrect", fetch_status="ok"),
        dict(publish_date="2018-01-01", capture_timestamp="20200301000000", phase="P1_global_onset", fetch_status="fetch_failed")]) + "\n")
    store = SimpleNamespace(payload=lambda ref: tmp_path,
                            load=lambda ref: dict(files=[dict(path="normalized/13559/documents.jsonl")]))
    value = artifact_coverage(tmp_path, store, dict(id="normalize-test"), "normalize")
    assert value["records_by_window"] == {"P1_global_onset": 1, "outside_study": 1}
    assert value["successful_by_window"] == {"P1_global_onset": 1}


def test_native_summary_does_not_turn_stage_eta_into_workflow_eta(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.native import note_summary
    fixture(tmp_path)
    run = data.collect(tmp_path, now=datetime(2026, 11, 2, 0, 1, tzinfo=timezone.utc))["runs"][0]
    note = note_summary(run, dict(free=20*1024**3, total=100*1024**3), "2026-11-02T00:01:00Z")
    assert "at least" in note and "steps unestimated" in note
    assert "Disk free: 20.0 GiB" in note


def test_progress_panel_uses_real_counts_and_explicit_unknown_stale_paused_states(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    from credleaf_ui.progress import payload
    fixture(tmp_path)
    run = data.collect(tmp_path, now=datetime(2026, 11, 2, 0, 1, tzinfo=timezone.utc))["runs"][0]
    value = payload(run, "2026-11-02T00:01:00Z")
    normalized = next(s for s in value["steps"] if s["key"] == "normalize__13529")
    assert normalized["percent"] == 30 and normalized["remaining_seconds"] == 90
    assert value["stages"] == dict(completed=1, total=7)
    assert next(s for s in value["steps"] if s["key"] == "dedupe__13529")["percent"] is None
    assert normalized["task_url"].endswith("/tasks/collection_flow.normalize/mapped/0")
    step = next(s for s in run["stages"] if s["stage"] == "normalize")
    step["progress"] = {}
    step["eta"]["stale"] = True
    stale = next(s for s in payload(run, "now")["steps"] if s["key"] == step["key"])
    assert stale["percent"] is None and stale["remaining_seconds"] is None and not stale["indeterminate"]
    step["eta"]["stale"] = False
    assert next(s for s in payload(run, "now")["steps"] if s["key"] == step["key"])["indeterminate"]
    run["control"] = dict(pause_requested=True)
    paused = next(s for s in payload(run, "now")["steps"] if s["key"] == step["key"])
    assert paused["paused"] and paused["remaining_seconds"] is None and not paused["indeterminate"]


def test_progress_collector_reads_only_requested_run_and_latest_sidecar(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parents[1]))
    directory = fixture(tmp_path)
    report = json.loads((directory / "steps/normalize__13529.json").read_text())
    report["log"] = str(directory / "normalize.attempt-1.log")
    (directory / "steps/normalize__13529.json").write_text(json.dumps(report))
    (directory / "normalize.attempt-1.progress.json").write_text(json.dumps(dict(completed=8, total=10)))
    run = data.collect(tmp_path, plan_paths=[directory / "plan.json"])["runs"][0]
    assert next(s for s in run["stages"] if s["stage"] == "normalize")["progress"]["completed"] == 8
    assert run["normalization"]["completed"] == 8
    assert data.collect(tmp_path, plan_paths=[])["runs"] == []
