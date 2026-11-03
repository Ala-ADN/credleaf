"""Small summaries for Airflow's own notes, XCom and Asset views."""
from collections import Counter
import json
from pathlib import Path
import re
from urllib.parse import quote

START, END = "[pipeline metrics]", "[/pipeline metrics]"


def merge_note(existing, summary):
    """Replace our section only; never truncate user-authored notes."""
    user = re.sub(re.escape(START) + r".*?" + re.escape(END), "", existing or "", flags=re.S).rstrip()
    separator = "\n\n" if user else ""
    available = 1000 - len(user) - len(separator) - len(START) - len(END) - 2
    if available < 50:
        return existing
    summary = summary if len(summary) <= available else summary[:available - 1] + "…"
    return user + separator + START + "\n" + summary + "\n" + END


def task_url(run, step, tab="xcom"):
    path = run["run_link"] + "/tasks/" + quote(step["task_id"], safe="")
    if step["map_index"] >= 0:
        path += f"/mapped/{step['map_index']}"
    return path + "/" + tab


def duration(seconds):
    if seconds is None:
        return "not estimated"
    if seconds < 60:
        return "<1 min"
    if seconds < 3600:
        return f"{seconds/60:.0f} min"
    return f"{seconds/3600:.1f} h"


def artifact_coverage(root, store, ref, stage):
    """Version-bound date coverage; immutable files are scanned once."""
    from pipeline.artifacts import write_json
    from document_time import document_time
    from credleaf_ui.data import read_json
    from credleaf_ui.bridge import capture_coverage
    if stage == "harvest":
        counts = capture_coverage(root, store, ref)
        return dict(basis="capture_timestamp", records_by_window=counts, records=sum(counts.values()))
    if stage not in {"normalize", "dedupe", "georef"}:
        return None
    path = Path(root) / "data/pipeline/observability/event_coverage" / (ref["id"] + ".json")
    previous = read_json(path)
    if previous is not None:
        return previous
    counts, successful = Counter(), Counter()
    for file in store.load(ref)["files"]:
        if not file["path"].endswith("/documents.jsonl"):
            continue
        with (store.payload(ref) / file["path"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                phase = document_time(row).phase
                counts[phase] += 1
                if row.get("fetch_status") == "ok":
                    successful[phase] += 1
    value = dict(basis="publication_date_else_capture_timestamp", records_by_window=dict(counts),
                 successful_by_window=dict(successful), records=sum(counts.values()))
    write_json(path, value)
    return value


def run_coverage(run):
    inputs, outputs = {}, {}
    for step in run["stages"]:
        if not step.get("coverage"):
            continue
        if step["stage"] == "harvest":
            inputs[str(step["collection_id"])] = step["coverage"]
        elif step["stage"] == "normalize":
            outputs[str(step["collection_id"])] = step["coverage"]
    return dict(capture_scope=run["config"].get("capture_phase"),
                document_cap=run["config"].get("normalize_max_docs"),
                captures=inputs, finalized_normalization=outputs, progress=run["normalization"])


def operational_status(run, step=None):
    if run.get("superseded_by"):
        return "superseded by " + run["superseded_by"]
    if run.get("control", {}).get("pause_requested"):
        if step and step["status"] in {"success", "cached", "reused", "off", "skipped"}:
            return step["status"]
        if step and not step["progress"].get("paused") and step["status"] != "paused":
            return step["status"] + "; workflow paused"
        if run["state"] == "failed":
            return "paused; worker stopped; saved checkpoint can resume"
        if step and step["status"] == "paused":
            return "paused at document boundary"
        if any(s["status"] == "paused" for s in run["stages"]):
            return "paused"
        return "pause requested"
    return step["status"] if step else run["state"]


def note_summary(run, disk, updated_at, step=None):
    lines = ["Status: " + operational_status(run, step)]
    if step:
        p = step["progress"]
        if p.get("completed") is not None:
            lines.append(f"Progress: {p['completed']:,} / {p.get('total') or '?'} {p.get('unit', '')}")
        eta = step["eta"]
        if step["status"] in {"success", "cached", "reused", "off", "skipped"}:
            lines.append("Elapsed: " + duration(step.get("duration_seconds")))
        else:
            lines.append("ETA: " + ("paused" if run.get("control", {}).get("pause_requested") else duration(eta.get("remaining_seconds"))))
        if eta.get("estimated_finish_at") and not run.get("control", {}).get("pause_requested"):
            lines.append("Estimated finish: " + eta["estimated_finish_at"])
        if eta.get("stale"):
            lines.append("Telemetry is stale; remaining time is unavailable.")
    else:
        p = run["normalization"]
        if p["total"]:
            lines.append(f"Normalized: {p['completed']:,} / {p['total']:,} captures")
        lines.append("Whole-run ETA: " + ("paused" if run.get("control", {}).get("pause_requested") else
            "not estimated" if run["eta_lower_bound_seconds"] is None else
            f"at least {duration(run['eta_lower_bound_seconds'])}; {run['unestimated_steps']} steps unestimated"))
        counts = Counter()
        for coverage in run_coverage(run)["captures"].values():
            counts.update(coverage["records_by_window"])
        if counts:
            lines.append("Input captures: " + "; ".join(f"{name.split('_')[0]} {n:,}" for name, n in sorted(counts.items())))
        if run["config"].get("normalize_max_docs") is not None:
            lines.append(f"Prefix cap: {run['config']['normalize_max_docs']:,}/collection; not balanced date coverage.")
    lines += [f"Disk free: {disk['free']/1024**3:.1f} GiB / {disk['total']/1024**3:.1f} GiB",
              "Details: XCom keys eta, coverage, disk. Versions: Assets.", "Observed: " + updated_at]
    return "\n".join(lines)
