"""Airflow integration kept separate from the model worker and artifact store."""
import os
import json
from pathlib import Path

from pipeline import plan, runner
from pipeline.timing import Tracker


def root():
    return Path(os.environ.get("CREDLEAF_ROOT", Path(__file__).resolve().parents[2]))


def prepare(kind, context):
    project = root()
    params = dict(context["params"])
    path = plan.run_directory(project, context["dag"].dag_id, context["run_id"]) / "control/prepare.json"
    report = dict(stage="prepare", status="running")
    tracker = Tracker(project, "prepare", None, {"kind": kind}, None, [], report, path,
                      hint=params.get("eta_stage_seconds", {}).get("prepare")).start()
    try:
        result = plan.prepare(project, os.environ.get("CREDLEAF_PYTHON", str(project / ".venv/bin/python")),
                              kind, context["dag"].dag_id, context["run_id"], params)
        report["status"] = "success"
        return result
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        tracker.finish(report["status"])


def overview(context):
    dag = context["dag"]
    task_ids = context["task"].upstream_task_ids
    returned = context["ti"].get_task_states(dag_id=dag.dag_id, run_ids=[context["run_id"]], task_ids=list(task_ids))
    states = {}
    for key, value in returned.get(context["run_id"], {}).items():
        if key in task_ids:
            states[f"{key}[-1]"] = value
        else:
            for task_id in sorted(task_ids, key=len, reverse=True):
                if key.startswith(task_id + "_") and key[len(task_id)+1:].isdigit():
                    states[f"{task_id}[{key[len(task_id)+1:]}]"] = value
                    break
    present = {key.rsplit("[", 1)[0] for key in states}
    for task_id in task_ids - present:
        states[f"{task_id}[-1]"] = "missing"
    report = runner.write_overview(root(), dag.dag_id, context["run_id"], states)
    if report["artifact_status"] == "failed":
        raise RuntimeError("Artifacts failed or stages were blocked; see the run overview")
    return dict(status=report["status"], snapshot_id=report["snapshot_id"])


def _finalize(context, state):
    """A DAG callback supplies the terminal event; no ORM access in DAG code."""
    dag_id, run_id = context["dag"].dag_id, context["run_id"]
    path = plan.run_directory(root(), dag_id, run_id) / "overview.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    runner.write_overview(root(), dag_id, run_id, previous.get("task_states") or {}, state)


def finalize_success(context):
    _finalize(context, "success")


def finalize_failure(context):
    _finalize(context, "failed")
