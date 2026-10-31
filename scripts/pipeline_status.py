"""Read-only live task ETA view; accepts host or container data paths."""
import argparse
import json
from pathlib import Path
import time

from pipeline.plan import run_directory
from pipeline.spec import STAGES
from pipeline.timing import format_eta


def display(root, dag_id, run_id):
    directory = run_directory(root, dag_id, run_id)
    path = directory / "plan.json"
    plan = json.loads(path.read_text()) if path.exists() else None
    expected = [("prepare", directory / "control/prepare.json")]
    if plan:
        for stage, spec in STAGES.items():
            if spec.dag != plan["kind"]:
                continue
            ids = plan["parameters"]["collection_ids"] if spec.scope == "collection" else [None]
            for cid in ids:
                key = stage if cid is None else f"{stage}__{cid}"
                expected.append((key, directory / "steps" / f"{key}.json"))
    expected.append(("publish", directory / "steps/publish.json"))
    lines = [f"{dag_id}: {run_id}", "Step                         Status                ETA / elapsed"]
    for key, path in expected:
        try:
            report = json.loads(path.read_text())
            value = format_eta(report.get("eta"))
            # The timestamp stays visible: stopped/killed workers must never
            # look like a fresh countdown. Airflow owns actual task states.
            if report["status"] == "running" and report.get("eta"):
                value += f" [as of {report['eta']['updated_at']}]"
            lines.append(f"{key:28} {report['status']:21} {value}")
        except (OSError, ValueError):
            stage = key.split("__", 1)[0]
            hint = plan["parameters"].get("eta_stage_seconds", {}).get(stage) if plan else None
            value = f"~{hint / 60:.1f} min execution once scheduled (manual)" if hint else "no estimate until task starts"
            lines.append(f"{key:28} {'waiting / no report':21} {value}")
    lines += [f"Reports: {directory}", "Queue time and retries are excluded. Consult Airflow for scheduler state."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dag-id", default="credleaf_build_corpus")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            print(display(args.root, args.dag_id, args.run_id), flush=True)
            if not args.watch:
                break
            time.sleep(30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
