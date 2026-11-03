"""Mirror live ETA properties and verified output versions into native Airflow."""
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time
from collections import Counter

from credleaf_ui.data import collect
from credleaf_ui.native import artifact_coverage, merge_note, note_summary, run_coverage, task_url, operational_status

log = logging.getLogger("credleaf.observer")


def capture_coverage(root, store, ref):
    from config import PANDEMIC_PHASES
    from pipeline.artifacts import write_json
    from credleaf_ui.data import read_json
    path = Path(root) / "data/pipeline/observability/coverage" / (ref["id"] + ".json")
    previous = read_json(path)
    if previous is not None:
        return previous
    windows = [(p.name, *p.cdx_window()) for p in PANDEMIC_PHASES]
    counts = Counter()
    for file in store.load(ref)["files"]:
        if not file["path"].endswith("captures.jsonl"):
            continue
        with (store.payload(ref) / file["path"]).open() as source:
            for line in source:
                row = json.loads(line)
                stamp = (row.get("capture") or {}).get("timestamp", "")
                label = next((name for name, start, end in windows if start <= stamp <= end), "outside_windows")
                counts[label] += 1
    result = dict(counts)
    write_json(path, result)
    return result


def sync(root):
    from airflow.assets.manager import AssetManager
    from airflow.models.asset import AssetModel, AssetEvent
    from airflow.models.dagrun import DagRun, DagRunNote
    from airflow.models.taskinstance import TaskInstance, TaskInstanceNote
    from airflow.models.xcom import XComModel
    from airflow.sdk import Asset
    from airflow.utils.session import create_session
    from pipeline.artifacts import ArtifactStore, write_json

    store, states, tasks, dag_runs = ArtifactStore(root), {}, {}, {}
    with create_session() as session:
        for run in session.query(DagRun).filter(DagRun.dag_id.in_(["credleaf_build_corpus", "credleaf_analyze_corpus"])):
            key = run.dag_id, run.run_id
            dag_runs[key] = run
            states[key] = dict(state=run.state, start_date=run.start_date.isoformat() if run.start_date else None,
                              sort_time=(run.start_date or run.logical_date).timestamp() if (run.start_date or run.logical_date) else 0, tasks={})
            for ti in session.query(TaskInstance).filter_by(dag_id=run.dag_id, run_id=run.run_id):
                states[key]["tasks"][(ti.task_id, ti.map_index)] = dict(state=ti.state)
                tasks[(run.dag_id, run.run_id, ti.task_id, ti.map_index)] = ti
        data = collect(root, states)
        existing = {(e.asset_id, e.extra.get("artifact_id")): e for e in session.query(AssetEvent)
                    .join(AssetModel, AssetModel.id == AssetEvent.asset_id).filter(AssetModel.uri.like("credleaf://%"))}
        # Oldest first so the native Assets page's latest event is meaningful.
        for run in reversed(data["runs"]):
            for step in run["stages"]:
                ti = tasks.get((run["dag_id"], run["run_id"], step["task_id"], step["map_index"]))
                ref = step.get("artifact") or {}
                if not ref.get("id") or step["status"] not in {"success", "cached", "reused"}:
                    continue
                if step["stage"] == "harvest":
                    step["capture_coverage"] = capture_coverage(root, store, ref)
                step["coverage"] = artifact_coverage(root, store, ref, step["stage"])
                scope = str(step["collection_id"]) if step["collection_id"] is not None else run["kind"]
                uri = f"credleaf://{step['stage']}/{scope}"
                asset = session.query(AssetModel).filter_by(uri=uri).one_or_none()
                if asset is None:
                    asset = AssetManager.create_assets([Asset(name=f"Credleaf {step['stage']} · {scope}", uri=uri,
                        extra=dict(stage=step["stage"], scope=scope, observer="verified immutable artifacts"))], session=session)[0]
                    session.flush()
                if (asset.id, ref["id"]) in existing:
                    event = existing[(asset.id, ref["id"])]
                    if step["coverage"] and event.extra.get("coverage") != step["coverage"]:
                        # Enrich the existing version; do not emit a fake new materialization.
                        event.extra = dict(event.extra, coverage=step["coverage"])
                    continue
                # Full integrity checks happen only once per published version.
                store.verify_tree(ref)
                manifest = store.load(ref)
                extra = dict(artifact_id=ref["id"], sha256=ref["sha256"],
                    origin_run_id=run["run_id"], origin_dag_id=run["dag_id"],
                    observed_status=step["status"], records=step["records"], bytes=step["bytes"],
                    files=manifest["files"], inputs=manifest["inputs"], stats=manifest["stats"],
                    created_at=manifest["created_at"], path=str(store.payload(ref)), coverage=step["coverage"])
                event = AssetManager.register_asset_change(asset=asset, task_instance=ti, extra=extra, session=session)
                existing[(asset.id, ref["id"])] = event
            db_run = dag_runs.get((run["dag_id"], run["run_id"]))
            refresh = db_run is not None and (run["state"] in {"running", "queued"}
                or (run.get("control", {}).get("pause_requested") and not run.get("superseded_by"))
                or "[pipeline metrics]" not in (db_run.note or ""))
            if not refresh:
                continue
            disk = dict(data["disk"], free_gib=round(data["disk"]["free"] / 1024**3, 2),
                        used_percent=round(data["disk"]["used"] / data["disk"]["total"] * 100, 1),
                        path=str(root), observed_at=data["updated_at"])
            coverage = run_coverage(run)
            summary = merge_note(db_run.note, note_summary(run, disk, data["updated_at"]))
            if db_run.dag_run_note:
                db_run.dag_run_note.content = summary
            else:
                db_run.dag_run_note = DagRunNote(summary)
            entries = [(s, tasks.get((run["dag_id"], run["run_id"], s["task_id"], s["map_index"]))) for s in run["stages"]]
            prepare_id = "prepare_corpus" if run["kind"] == "corpus" else "prepare_analysis"
            entries.append((None, tasks.get((run["dag_id"], run["run_id"], prepare_id, -1))))
            for step, ti in entries:
                if ti is None:
                    continue
                if step:
                    eta = dict(step["eta"], progress=step["progress"], status=operational_status(run, step),
                               airflow_state=ti.state, metrics_url=task_url(run, step))
                    cov = step.get("coverage") or dict(finalized=False, progress=step["progress"], capture_scope=run["config"].get("capture_phase"))
                    if step["stage"] not in {"harvest", "normalize", "dedupe", "georef"}:
                        cov = dict(applicable=False, note="See upstream corpus-stage coverage or prepare task summary")
                else:
                    eta = dict(status=operational_status(run), lower_bound_seconds=run["eta_lower_bound_seconds"],
                               unestimated_steps=run["unestimated_steps"], progress=run["normalization"],
                               updated_at=data["updated_at"], note="Lower bound only; not a total finish estimate")
                    cov = coverage
                for key, value in dict(eta=eta, coverage=cov, disk=disk).items():
                    XComModel.set(key=key, value=value, dag_id=ti.dag_id, run_id=ti.run_id,
                                  task_id=ti.task_id, map_index=ti.map_index, session=session)
                if step:
                    XComModel.set(key="_link_CredleafEtaLink", value=task_url(run, step), dag_id=ti.dag_id,
                                  run_id=ti.run_id, task_id=ti.task_id, map_index=ti.map_index, session=session)
                content = merge_note(ti.note, note_summary(run, disk, data["updated_at"], step))
                if ti.task_instance_note:
                    ti.task_instance_note.content = content
                else:
                    ti.task_instance_note = TaskInstanceNote(content)
        session.flush()
        for asset in session.query(AssetModel).filter(AssetModel.uri.like("credleaf://%")):
            last = session.query(AssetEvent).filter_by(asset_id=asset.id).order_by(AssetEvent.id.desc()).first()
            data["assets"].append(dict(id=asset.id, name=asset.name, uri=asset.uri,
                latest=last.extra if last else {}, updated_at=last.timestamp.isoformat() if last else None,
                versions=session.query(AssetEvent).filter_by(asset_id=asset.id).count()))
    write_json(Path(root) / "data/pipeline/dashboard.json", data)
    return data


def main():
    root = Path(os.environ.get("CREDLEAF_ROOT", "/opt/credleaf"))
    path = root / "data/pipeline/observer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        handler = RotatingFileHandler(path.with_name("observer.log"), maxBytes=2_000_000, backupCount=2)
        log.addHandler(handler)
        log.setLevel(logging.WARNING)
        while True:
            try:
                sync(root)
            except Exception:
                log.exception("Native monitoring refresh failed")
            time.sleep(30)


if __name__ == "__main__":
    main()
