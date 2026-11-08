"""Stop at worker boundaries and recover interrupted local Airflow runs on boot."""
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from credleaf_ui.data import read_json, task_id
from pipeline.artifacts import implementation, write_json
from pipeline.plan import run_directory, runtime_identity

log = logging.getLogger(__name__)
DAGS = {"credleaf_build_corpus", "credleaf_analyze_corpus"}
ACTIVE = {"running", "queued", "restarting"}
REPORTING = {"overview", "failure_guard"}


def reports(directory, plan):
    result = {}
    for path in (directory / "steps").glob("*.json"):
        report = read_json(path, {})
        stage = report.get("stage")
        if not stage:
            continue
        cid = report.get("collection_id")
        ids = plan.get("parameters", {}).get("collection_ids", [])
        mapped = ids.index(cid) if cid in ids else -1
        result[(task_id(stage, plan.get("kind", "corpus")), mapped)] = report
    return result


def recovery_tasks(tasks, saved, shutdown_tasks=()):
    """Select interruptions, never retry an unrelated scientific failure."""
    stopped = {tuple(key) for key in shutdown_tasks}
    candidates = []
    for task in tasks:
        key = (task.task_id, task.map_index)
        report = saved.get(key, {})
        interrupted = task.state in ACTIVE or report.get("status") == "interrupted" or (
            key in stopped and report.get("status") != "failed")
        if task.state not in {"success", "skipped", "removed"} and interrupted:
            candidates.append(task)
    if not candidates:
        return []
    if any(t.state == "failed" and t not in candidates and t.task_id not in REPORTING for t in tasks):
        return []
    return candidates + [t for t in tasks if t not in candidates and
                         (t.state == "upstream_failed" or t.task_id in REPORTING)]


def recover(root):
    """Only called before this service starts its scheduler or workers."""
    from airflow.models.dag import DagModel
    from airflow.models.dagrun import DagRun
    from airflow.models.taskinstance import TaskInstance, clear_task_instances
    from airflow.utils.session import create_session
    actions = []
    with create_session() as session:
        runs = session.query(DagRun).filter(DagRun.dag_id.in_(DAGS), DagRun.state.in_(["running", "queued", "failed"])).all()
        for run in runs:
            directory = run_directory(root, run.dag_id, run.run_id)
            if (directory / "control/superseded.json").exists():
                continue
            plan = read_json(directory / "plan.json", {})
            request_path = directory / "control/request.json"
            request = read_json(request_path, {})
            tasks = session.query(TaskInstance).filter_by(dag_id=run.dag_id, run_id=run.run_id).all()
            chosen = recovery_tasks(tasks, reports(directory, plan), request.get("shutdown_tasks", []))
            if not chosen and not request.get("shutdown_requested"):
                continue
            try:
                if plan and (implementation(Path(root)) != plan["implementation"]["source"] or
                             implementation(Path(plan["code_root"])) != plan["implementation"]["source"] or
                             runtime_identity(root, plan["python"]) != plan["implementation"]["runtime"]):
                    raise ValueError("Code or Python runtime changed; migrate this run before resuming")
                if chosen:
                    # The previous container is gone; RUNNING instances have no
                    # live process that could acknowledge Airflow's RESTARTING state.
                    for task in chosen:
                        if task.state in ACTIVE:
                            task.state = "failed"
                    clear_task_instances(chosen, session=session)
                    run.state = "queued"
                    run.last_scheduling_decision = None
                model = session.query(DagModel).filter_by(dag_id=run.dag_id).one_or_none()
                if model and "shutdown_dag_was_paused" in request:
                    model.is_paused = bool(request.get("pause_requested") or request["shutdown_dag_was_paused"])
                request.update(shutdown_requested=False, recovered_at=datetime.now(timezone.utc).isoformat())
                request.pop("shutdown_tasks", None)
                request.pop("shutdown_dag_was_paused", None)
                write_json(request_path, request)
                actions.append(dict(dag_id=run.dag_id, run_id=run.run_id, tasks=[(t.task_id, t.map_index) for t in chosen]))
            except Exception as error:
                if model := session.query(DagModel).filter_by(dag_id=run.dag_id).one_or_none():
                    model.is_paused = True
                write_json(directory / "control/recovery.json", dict(status="blocked", error=str(error)))
                log.exception("Recovery needs attention for %s / %s", run.dag_id, run.run_id)
    return actions


def request_shutdown(root):
    from airflow.models.dag import DagModel
    from airflow.models.dagrun import DagRun
    from airflow.models.taskinstance import TaskInstance
    from airflow.utils.session import create_session
    paths = []
    original_pause = {}
    with create_session() as session:
        for run in session.query(DagRun).filter(DagRun.dag_id.in_(DAGS), DagRun.state.in_(["running", "queued"])):
            directory = run_directory(root, run.dag_id, run.run_id)
            path = directory / "control/request.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                previous = read_json(path, {})
                model = session.query(DagModel).filter_by(dag_id=run.dag_id).one_or_none()
                original_pause.setdefault(run.dag_id, bool(model and model.is_paused))
                tasks = session.query(TaskInstance).filter_by(dag_id=run.dag_id, run_id=run.run_id).all()
                write_json(path, dict(previous, shutdown_requested=True,
                    shutdown_at=datetime.now(timezone.utc).isoformat(),
                    shutdown_dag_was_paused=previous.get("shutdown_dag_was_paused", original_pause[run.dag_id]),
                    shutdown_tasks=[(t.task_id, t.map_index) for t in tasks if t.state in ACTIVE]))
                if model:
                    model.is_paused = True
                paths.append(directory)
    return paths


def unsettled(directories):
    return any(read_json(path, {}).get("status") == "running"
               for directory in directories for path in (directory / "steps").glob("*.json"))


def supervise(root, command, *, grace_seconds=120):
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    root = Path(root)
    lock_path = root / "data/pipeline/service.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log.info("Recovered interrupted runs: %s", recover(root))
        child = subprocess.Popen(command, start_new_session=True)
        observer = None
        try:
            if os.environ.get("CREDLEAF_OBSERVER_ENABLED") == "1":
                observer = subprocess.Popen([sys.executable, "-m", "credleaf_ui.bridge"],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while child.poll() is None and not stopping.wait(1):
                pass
            if stopping.is_set() or child.poll() is not None:
                directories = request_shutdown(root)
                deadline = time.monotonic() + grace_seconds
                while child.poll() is None and unsettled(directories) and time.monotonic() < deadline:
                    time.sleep(.5)
            if child.poll() is None:
                # Standalone handles SIGINT by stopping its component processes.
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            return 0 if stopping.is_set() else child.returncode or 1
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            if observer and observer.poll() is None:
                observer.terminate()
                try:
                    observer.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    observer.kill()
                    observer.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        raise SystemExit("Expected an Airflow command after --")
    raise SystemExit(supervise(Path(os.environ["CREDLEAF_ROOT"]), command))
