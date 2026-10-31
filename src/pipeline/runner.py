"""Artifact orchestration; never imports a model into the Airflow interpreter."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import signal
import uuid

from pipeline.artifacts import ArtifactStore, digest, file_hash, implementation, lookup, write_json
from pipeline.plan import run_directory, runtime_identity
from pipeline.spec import STAGES, stage_config
from pipeline.timing import Tracker, format_eta


def _key(stage, cid=None):
    return stage if cid is None else f"{stage}__{cid}"


def load_plan(reference):
    path = Path(reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError("Run plan changed after validation")
    return json.loads(path.read_text(encoding="utf-8"))


def _worker(plan, stage, cid, config, inputs, output, log_path):
    request = dict(stage=stage, collection_id=cid, config=config, inputs=inputs, output=str(output),
                   root=plan["root"], code_root=plan["code_root"], resources=plan["resources"], artifact_key=digest([config, inputs, plan["implementation"]]),
                   progress_path=str(log_path.with_suffix(".progress.json")))
    request_path = log_path.with_suffix(".request.json")
    result_path = log_path.with_suffix(".result.json")
    write_json(request_path, request)
    env = dict(os.environ, PYTHONPATH=str(Path(plan["code_root"]) / "src"), MPLBACKEND="Agg",
               CREDLEAF_DATA_DIR=str(output), CREDLEAF_CACHE_DIR=str(Path(plan["root"]) / "data/cache"))
    process = subprocess.Popen([plan["python"], "-u", "-m", "pipeline.worker", "--request", str(request_path),
                                "--result", str(result_path)], env=env, cwd=plan["root"], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", start_new_session=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        if process.wait():
            detail = json.loads(result_path.read_text()).get("error") if result_path.exists() else "No worker result"
            raise RuntimeError(f"{stage} worker exited {process.returncode}: {detail}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        process.stdout.close()
    result = json.loads(result_path.read_text())
    if result.get("format") != 1 or result.get("stage") != stage or result.get("status") != "success" or not isinstance(result.get("stats"), dict):
        raise ValueError("Invalid structured worker result")
    if (implementation(Path(plan["code_root"])) != plan["implementation"]["source"]
            or runtime_identity(Path(plan["root"]), plan["python"]) != plan["implementation"]["runtime"]):
        raise ValueError("Frozen code or Python environment changed during stage execution")
    return result["stats"]


def run_stage(reference, stage, inputs, collection_id=None, attempt=1):
    plan = load_plan(reference)
    root = Path(plan["root"])
    directory = run_directory(root, plan["dag_id"], plan["run_id"])
    key = _key(stage, collection_id)
    report_path = directory / "steps" / f"{key}.json"
    log_path = directory / f"{key}.attempt-{attempt}-{uuid.uuid4().hex[:8]}.log"
    report = dict(stage=stage, collection_id=collection_id, status="running", log=str(log_path),
                  started_at=datetime.now(timezone.utc).isoformat(), attempt=attempt)
    write_json(report_path, report)
    mode = plan["parameters"].get(f"{stage}_mode")
    timing = Tracker(root, stage, collection_id, stage_config(stage, plan["parameters"], plan["resources"]),
                     plan["implementation"]["runtime"], inputs, report, report_path,
                     hint=plan["parameters"].get("eta_stage_seconds", {}).get(stage),
                     progress_path=log_path.with_suffix(".progress.json"), learn=mode == "build").start()
    try:
        if (implementation(root) != plan["implementation"]["source"]
                or implementation(Path(plan["code_root"])) != plan["implementation"]["source"]
                or runtime_identity(root, plan["python"]) != plan["implementation"]["runtime"]):
            raise ValueError("Code or Python environment changed after planning; trigger a new run")
        if stage not in STAGES or (STAGES[stage].scope == "collection") != (collection_id is not None):
            raise ValueError("Invalid stage scope")
        mode = plan["parameters"][f"{stage}_mode"]
        if mode == "off":
            result = dict(id=None, stage=stage, collection_id=collection_id, mode="off")
            report.update(status="off", artifact=result)
        else:
            store = ArtifactStore(root)
            config = stage_config(stage, plan["parameters"], plan["resources"])
            if mode == "reuse":
                result = store.reuse(lookup(plan["base"], stage, collection_id), stage, collection_id,
                                     config, inputs, plan["implementation"])
                report["status"] = "reused"
            else:
                result, cached = store.build(stage, collection_id, config, inputs, plan["implementation"],
                    lambda output: _worker(plan, stage, collection_id, config, inputs, output, log_path),
                    nonce=plan["run_id"] if stage in {"harvest", "credibility"} else None)
                report["status"] = "cached" if cached else "success"
            manifest = store.load(result)
            report.update(artifact=result, files=manifest["files"], stats=manifest["stats"])
        return result
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        timing.finish(report["status"])
        write_json(report_path, report)
        print(json.dumps(report, indent=2), flush=True)


def publish(reference, artifacts):
    plan = load_plan(reference)
    root = Path(plan["root"])
    store = ArtifactStore(root)
    directory = run_directory(root, plan["dag_id"], plan["run_id"])
    report = dict(stage="publish", collection_id=None, status="running", started_at=datetime.now(timezone.utc).isoformat())
    timing = Tracker(root, "publish", None, {"kind": plan["kind"]}, plan["implementation"]["runtime"],
                     artifacts, report, directory / "steps/publish.json",
                     hint=plan["parameters"].get("eta_stage_seconds", {}).get("publish")).start()
    try:
        if implementation(root) != plan["implementation"]["source"]:
            raise ValueError("Code changed after planning; trigger a new run")
        identities = [(ref["stage"], ref["collection_id"]) for ref in artifacts]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate artifact scope in snapshot")
        if plan["kind"] == "corpus":
            expected = [(stage, cid) for stage in ("harvest", "normalize", "dedupe", "georef")
                        for cid in plan["parameters"]["collection_ids"]] + [("credibility", None), ("embed", None)]
        else:
            expected = [(stage, None) for stage in ("semantic", "topic", "animation")]
        if set(identities) != set(expected):
            raise ValueError("Incomplete snapshot")
        for ref in artifacts:
            if (plan["parameters"][ref["stage"] + "_mode"] == "off") != (ref.get("id") is None):
                raise ValueError("Snapshot stage mode differs from its artifact reference")
        bundle = dict(kind=plan["kind"], artifacts=artifacts, implementation=plan["implementation"],
                      collection_ids=plan["parameters"].get("collection_ids", plan["parameters"].get("analysis_collection_ids")),
                      source_snapshot_id=plan["parameters"].get("snapshot_id"))
        def produce(output):
            for ref in artifacts:
                if ref.get("id"):
                    store.load(ref)
            write_json(output / "snapshot.json", bundle)
            return dict(artifacts=len(artifacts))
        result, cached = store.build("publish", None, dict(kind=plan["kind"], parameters={
                                    k: v for k, v in plan["parameters"].items() if k != "eta_stage_seconds"}),
                                artifacts, plan["implementation"], produce)
        report.update(status="cached" if cached else "success", artifact=result)
        print(f"Published {plan['kind']} snapshot: {result['id']}", flush=True)
        return result
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        timing.finish(report["status"])
        write_json(directory / "steps/publish.json", report)


def write_overview(root, dag_id, run_id, task_states=None, dag_state=None):
    """Task execution writes provisional status; the scheduler finalizes actual states."""
    directory = run_directory(root, dag_id, run_id)
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    expected = ["publish"]
    if plan:
        for stage, spec in STAGES.items():
            if spec.dag == plan["kind"]:
                ids = plan["parameters"]["collection_ids"] if spec.scope == "collection" else [None]
                expected += [_key(stage, cid) for cid in ids]
    reports = []
    for key in expected:
        path = directory / "steps" / f"{key}.json"
        try:
            reports.append(json.loads(path.read_text()))
        except (FileNotFoundError, ValueError) as error:
            reports.append(dict(stage=key, status="missing_or_blocked", error=str(error)))
    artifact_ok = bool(plan) and all(r["status"] in {"success", "cached", "reused", "off"} for r in reports)
    failed_tasks = {key: value for key, value in (task_states or {}).items() if value not in {"success", "skipped"}}
    if task_states is None or dag_state is None:
        status = "awaiting_task_states" if artifact_ok and not failed_tasks else "failed"
    else:
        status = "success" if artifact_ok and not failed_tasks and dag_state == "success" else "failed"
    control = directory / "control/prepare.json"
    control_steps = [json.loads(control.read_text())] if control.exists() else []
    overview = dict(run_id=run_id, dag_id=dag_id, status=status, artifact_status="success" if artifact_ok else "failed",
                    task_states=task_states, dag_state=dag_state, steps=reports,
                    control_steps=control_steps,
                    snapshot_id=next((r.get("artifact", {}).get("id") for r in reports if r["stage"] == "publish"), None))
    write_json(directory / "overview.json", overview)
    lines = [f"# {dag_id}: {run_id}", "", f"Status: **{status}**", "",
             "| Step | Artifact status | ETA / elapsed | Outputs |", "| --- | --- | --- | --- |"]
    for report in [*control_steps, *reports]:
        outputs = [f"`{item['path']}` ({item.get('records', '—')} records, {item['bytes']} bytes)" for item in report.get("files", [])]
        identifier = report.get("artifact", {}).get("id")
        if identifier:
            outputs.insert(0, f"Artifact `{identifier}`")
        if report.get("error"):
            outputs.append(report["error"].replace("|", "\\|").replace("\n", " "))
        lines.append(f"| {report['stage']} {report.get('collection_id') or ''} | {report['status']} | {format_eta(report.get('eta'))} | {'<br>'.join(outputs)} |")
    if task_states is not None:
        lines += ["", "Airflow task states:", "", *[f"- `{key}`: {value}" for key, value in sorted(task_states.items())]]
    (directory / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return overview
