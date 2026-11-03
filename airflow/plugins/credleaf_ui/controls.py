"""Authenticated UI/CLI control of cooperative scientific worker boundaries."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import uuid
from pathlib import Path


def control_run(root, dag_id, run_id, pause):
    from airflow.models.dag import DagModel
    from airflow.models.dagrun import DagRun
    from airflow.models.taskinstance import TaskInstance, clear_task_instances
    from airflow.utils.session import create_session
    from pipeline.plan import run_directory
    from pipeline.artifacts import write_json, implementation
    if dag_id not in {"credleaf_build_corpus", "credleaf_analyze_corpus"}:
        raise ValueError("Unsupported workflow")
    directory = run_directory(root, dag_id, run_id)
    if (directory / "control/superseded.json").exists():
        target = json.loads((directory / "control/superseded.json").read_text())["run_id"]
        raise ValueError(f"This run was superseded; use {target}")
    plan = json.loads((directory / "plan.json").read_text())
    if not (Path(plan["code_root"]) / "src/pipeline/control.py").exists():
        raise ValueError("This legacy run needs checkpoint migration before it can pause")
    path = directory / "control/request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with create_session() as session:
            run = session.query(DagRun).filter_by(dag_id=dag_id, run_id=run_id).one_or_none()
            previous = json.loads(path.read_text()) if path.exists() else {}
            recover = run is not None and run.state == "failed" and not pause and previous.get("pause_requested")
            if run is None or (run.state not in {"running", "queued"} and not recover):
                raise ValueError("Only active workflows or interrupted paused workflows can resume")
            if not pause and implementation(Path(root)) != plan["implementation"]["source"]:
                raise ValueError("Scientific code changed; migrate the saved checkpoint to a new plan before resuming")
            if recover:
                acks = [json.loads(p.read_text()) for p in path.parent.glob("*.json") if p.name not in {"prepare.json", "request.json"}]
                if not any(a.get("status") == "paused" and a.get("checkpoint") and a.get("request_id") == previous.get("request_id") for a in acks):
                    raise ValueError("No durable paused checkpoint is available for recovery")
                tasks = session.query(TaskInstance).filter_by(dag_id=dag_id, run_id=run_id).all()
                if any(t.state == "running" for t in tasks):
                    raise ValueError("Wait for the interrupted worker to stop before recovering")
                clear_task_instances([t for t in tasks if t.state != "success"], session=session)
            value = dict(pause_requested=pause, updated_at=datetime.now(timezone.utc).isoformat(),
                         request_id=(previous.get("request_id") if previous.get("pause_requested") == pause
                                     else uuid.uuid4().hex),
                         run_id=run_id, dag_id=dag_id)
            write_json(path, value)
            session.query(DagModel).filter_by(dag_id=dag_id).one().is_paused = pause
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pause", "resume"])
    parser.add_argument("--dag-id", default="credleaf_build_corpus")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(control_run(os.getenv("CREDLEAF_ROOT", "/opt/credleaf"), args.dag_id,
                                args.run_id, args.action == "pause"), indent=2))
