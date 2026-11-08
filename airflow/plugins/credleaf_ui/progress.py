"""Small, read-only progress responses for the embedded run tab."""
import math

from credleaf_ui.native import operational_status, task_url

COMPLETE = {"success", "cached", "reused"}
OMITTED = {"off", "skipped", "removed"}


def count(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def payload(run, updated_at):
    steps = []
    for step in run["stages"]:
        progress, eta = step.get("progress", {}), step.get("eta", {})
        completed, total = count(progress.get("completed")), count(progress.get("total"))
        status = step["status"]
        percent = (min(100, completed / total * 100)
                   if completed is not None and total is not None and total > 0 else None)
        if status in COMPLETE:
            percent = 100
            if total is not None:
                completed = total
        if status in OMITTED:
            percent = None
        paused = status not in COMPLETE | OMITTED and bool(
            run.get("control", {}).get("pause_requested") or run.get("control", {}).get("shutdown_requested")
            or progress.get("paused") or status == "paused")
        stale = bool(eta.get("stale"))
        steps.append(dict(key=step["key"], label=step["stage"].replace("_", " ").capitalize(),
            collection_id=step["collection_id"], status=status, status_text=operational_status(run, step),
            completed=completed, total=total, percent=percent, unit=progress.get("unit", "items"),
            paused=paused, stale=stale,
            remaining_seconds=None if paused or stale else eta.get("remaining_seconds"),
            indeterminate=percent is None and status == "running" and not paused and not stale,
            task_url=task_url(run, step).removesuffix("/xcom")))
    finished = sum(s["status"] in COMPLETE | OMITTED for s in steps)
    return dict(dag_id=run["dag_id"], run_id=run["run_id"], state=operational_status(run),
                updated_at=updated_at, steps=steps,
                stages=dict(completed=finished, total=len(steps)),
                normalization=run["normalization"] if run["kind"] == "corpus" else None)
