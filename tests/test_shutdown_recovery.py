"""Document-bound shutdown and selective Airflow boot recovery."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.control import PipelineInterrupted, RunControl, STOP


def test_real_sigterm_is_observed_at_the_next_safe_boundary(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys,time
from pathlib import Path
from pipeline.control import RunControl, PipelineInterrupted, install_shutdown_handlers
from pipeline.artifacts import write_json
root=Path(sys.argv[1])
install_shutdown_handlers()
control=RunControl(root/'request.json')
(root/'ready').touch()
try:
    for i in range(100):
        control.wait(i,100,root)
        time.sleep(.05)
        write_json(root/'committed.json', {'completed':i+1})
except PipelineInterrupted:
    sys.exit(75)
'''
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)],
        env=dict(os.environ, PYTHONPATH=str(root / "src")))
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "ready").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 75
        ack = json.loads((tmp_path / "worker.json").read_text())
        assert ack["status"] == "interrupted"
        saved = json.loads((tmp_path / "committed.json").read_text()) if (tmp_path / "committed.json").exists() else {"completed": 0}
        assert ack["completed"] == saved["completed"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_service_supervisor_requests_shutdown_and_stops_its_child(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    root = Path(__file__).resolve().parents[1]
    child = '''
import signal,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def stop(*args):
    (root/'child-stopped').touch()
    sys.exit(0)
signal.signal(signal.SIGINT,stop)
(root/'child-ready').touch()
while True: time.sleep(.05)
'''
    supervisor = '''
import sys
from pathlib import Path
from credleaf_ui import lifecycle
root=Path(sys.argv[1])
lifecycle.recover=lambda root: (root/'recovered').touch()
def shutdown(root):
    (root/'shutdown-requested').touch()
    return []
lifecycle.request_shutdown=shutdown
sys.exit(lifecycle.supervise(root,[sys.executable,'-c',sys.argv[2],str(root)],grace_seconds=1))
'''
    process = subprocess.Popen([sys.executable, "-c", supervisor, str(tmp_path), child],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "airflow/plugins")])))
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "child-ready").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (tmp_path / "child-ready").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 0
        assert (tmp_path / "recovered").exists()
        assert (tmp_path / "shutdown-requested").exists()
        assert (tmp_path / "child-stopped").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_shutdown_commits_current_normalization_document_then_resumes(tmp_path, monkeypatch):
    from tests.test_normalization_checkpoints import setup
    # Use the same real journal and extraction fixture as manual pause recovery.
    module, source, records, calls, process, options = setup.__wrapped__(tmp_path, monkeypatch)
    def stop_after_document(client, record, cache, use_cache, stats):
        doc = process(client, record, cache, use_cache, stats)
        if record.seed_id == 0:
            STOP.set()
        return doc
    monkeypatch.setattr(module, "_process", stop_after_document)
    control = RunControl(tmp_path / "control/request.json", "normalize__13559")
    try:
        with pytest.raises(PipelineInterrupted):
            module.normalize_phase(None, source, control=control, **options)
        assert calls == [0]
        ack = json.loads((tmp_path / "control/normalize__13559.json").read_text())
        assert ack["completed"] == 1 and ack["status"] == "interrupted"
    finally:
        STOP.clear()
    module.normalize_phase(None, source, control=control, **options)
    assert calls == list(range(5))
    assert len(options["output_path"].read_text().splitlines()) == 5


def test_shutdown_request_also_stops_a_manually_paused_worker(tmp_path):
    from pipeline.artifacts import write_json
    path = tmp_path / "request.json"
    write_json(path, dict(pause_requested=True, shutdown_requested=True))
    with pytest.raises(PipelineInterrupted):
        RunControl(path).wait(2, 10, tmp_path / "checkpoint")
    assert json.loads(path.read_text())["pause_requested"] is True


def test_boot_recovery_keeps_successes_and_does_not_retry_real_failures(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "airflow/plugins"))
    from credleaf_ui.lifecycle import recovery_tasks
    def task(name, state, index=-1):
        return SimpleNamespace(task_id=name, state=state, map_index=index)
    harvest = task("collection_flow.harvest", "success", 0)
    normalize = task("collection_flow.normalize", "running", 0)
    embed = task("embed", "upstream_failed")
    overview = task("overview", "success")
    guard = task("failure_guard", "failed")
    tasks = [harvest, normalize, embed, overview, guard]
    chosen = recovery_tasks(tasks, {})
    assert normalize in chosen and embed in chosen and overview in chosen and guard in chosen
    assert harvest not in chosen
    normalize.state = "failed"
    assert recovery_tasks(tasks, {(normalize.task_id, 0): dict(status="failed")}) == []
    assert normalize in recovery_tasks(tasks, {(normalize.task_id, 0): dict(status="interrupted")})
    normalize.state = "deferred"
    assert recovery_tasks([normalize], {}) == []
    overview.state = "running"
    assert overview in recovery_tasks([harvest, overview], {})


def test_runner_reports_interruption_without_publishing_partial_artifact(tmp_path, monkeypatch):
    import shutil
    import sys
    from pipeline import runner
    from pipeline.plan import prepare, run_directory
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "scripts", "dags"):
        (tmp_path / folder).symlink_to(root / folder, target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(root / name, tmp_path / name)
    plan = prepare(tmp_path, sys.executable, "corpus", "test", "stop", dict(georef_mode="off"))
    def interrupted(*args, **kwargs):
        raise PipelineInterrupted("fixture shutdown")
    monkeypatch.setattr(runner, "_worker", interrupted)
    with pytest.raises(PipelineInterrupted):
        runner.run_stage(plan, "harvest", [], 13559)
    report = json.loads((run_directory(tmp_path, "test", "stop") / "steps/harvest__13559.json").read_text())
    assert report["status"] == "interrupted" and report["resumable"]
    assert not list((tmp_path / "data/pipeline/artifacts").glob("harvest-*"))
