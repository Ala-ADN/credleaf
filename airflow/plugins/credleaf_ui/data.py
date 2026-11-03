"""Read pipeline telemetry without changing scientific artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from urllib.parse import quote


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (AttributeError, ValueError):
        return 0


def task_id(stage, kind):
    if stage in {"harvest", "normalize", "dedupe", "georef"}:
        return "collection_flow." + stage
    if stage == "publish":
        return "publish_corpus" if kind == "corpus" else "publish_analysis"
    return stage


def collect(root, states=None, now=None):
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    states = states or {}
    runs = []
    for plan_path in (root / "data/pipeline/runs").glob("*/plan.json"):
        plan = read_json(plan_path)
        if not plan:
            continue
        p, directory = plan["parameters"], plan_path.parent
        dag_id, run_id = plan["dag_id"], plan["run_id"]
        db = states.get((dag_id, run_id), {})
        overview = read_json(directory / "overview.json", {})
        control = read_json(directory / "control/request.json", {})
        order = ["credibility", "harvest", "normalize", "dedupe", "georef", "embed", "publish"] if plan["kind"] == "corpus" else ["semantic", "topic", "animation", "publish"]
        stages = []
        for stage in order:
            ids = p.get("collection_ids", []) if stage in {"harvest", "normalize", "dedupe", "georef"} else [None]
            for cid in ids:
                key = stage if cid is None else f"{stage}__{cid}"
                report = read_json(directory / "steps" / f"{key}.json", {})
                eta = report.get("eta", {})
                progress = dict(eta.get("progress") or {})
                if report.get("status") == "running" and report.get("log"):
                    live = read_json(directory / (Path(report["log"]).stem + ".progress.json"), {})
                    if live:
                        progress = live
                mapped = p.get("collection_ids", []).index(cid) if cid is not None else -1
                ti = db.get("tasks", {}).get((task_id(stage, plan["kind"]), mapped), {})
                actual_state = ti.get("state")
                status = report.get("status", "waiting")
                if actual_state in {"failed", "upstream_failed", "skipped", "removed"}:
                    status = actual_state
                elif not report and actual_state:
                    status = actual_state
                ack = read_json(directory / "control" / (key + ".json"), {})
                if (status == "running" and control.get("pause_requested") and ack.get("status") == "paused"
                        and ack.get("request_id") == control.get("request_id")):
                    status = "paused"
                age = now.timestamp() - timestamp(eta.get("updated_at"))
                stale = status == "running" and age > 120
                remaining = eta.get("remaining_seconds") if status == "running" and not stale else None
                finish = eta.get("estimated_finish_at") if remaining is not None else None
                files = report.get("files", [])
                stages.append(dict(key=key, stage=stage, collection_id=cid, task_id=task_id(stage, plan["kind"]),
                    map_index=mapped, status=status, airflow_state=actual_state,
                    progress=progress, eta=dict(eta, remaining_seconds=remaining, estimated_finish_at=finish, stale=stale),
                    artifact=report.get("artifact"), files=files, stats=report.get("stats"),
                    error=report.get("error"), duration_seconds=report.get("duration_seconds"),
                    bytes=sum(f.get("bytes", 0) for f in files),
                    records=sum(f.get("records") or 0 for f in files if f.get("path", "").endswith(("documents.jsonl", "captures.jsonl", "metadata.jsonl")))))
        active = [s for s in stages if s["status"] == "running"]
        known = [s["eta"]["remaining_seconds"] for s in active if s["eta"]["remaining_seconds"] is not None]
        unfinished = [s for s in stages if s["status"] not in {"success", "cached", "reused", "off", "skipped"}]
        normalize_total, normalize_done = 0, 0
        for s in stages:
            if s["stage"] != "normalize":
                continue
            raw = next((r for r in stages if r["stage"] == "harvest" and r["collection_id"] == s["collection_id"]), {})
            total = sum(f.get("records") or 0 for f in raw.get("files", []))
            cap = p.get("normalize_max_docs")
            total = min(total, cap) if cap is not None else total
            normalize_total += total
            normalize_done += total if s["status"] in {"success", "cached", "reused"} else s["progress"].get("completed", 0)
        run_link = f"/dags/{quote(dag_id, safe='')}/runs/{quote(run_id, safe='')}"
        runs.append(dict(dag_id=dag_id, run_id=run_id, kind=plan["kind"],
            state=db.get("state") or overview.get("dag_state") or overview.get("status", "imported"),
            started_at=db.get("start_date"), updated_at=max([timestamp(s["eta"].get("updated_at")) for s in stages] or [0]),
            sort_time=db.get("sort_time", plan_path.stat().st_mtime), stages=stages,
            normalization=dict(completed=normalize_done, total=normalize_total),
            eta_lower_bound_seconds=max(known) if known else None,
            unestimated_steps=sum(s["eta"]["remaining_seconds"] is None for s in unfinished),
            snapshot_id=overview.get("snapshot_id"), run_link=run_link,
            config={k: v for k, v in p.items() if k not in {"animation_labels", "eta_stage_seconds"}},
            directory=str(directory)))
        runs[-1]["control"] = control
        runs[-1]["superseded_by"] = read_json(directory / "control/superseded.json", {}).get("run_id")
    disk = shutil.disk_usage(root)
    return dict(updated_at=now.isoformat(), runs=sorted(runs, key=lambda r: r["sort_time"], reverse=True),
                disk=dict(total=disk.total, used=disk.used, free=disk.free), assets=[])
