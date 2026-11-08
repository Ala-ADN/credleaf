"""CLI entry point; run with PYTHONPATH=src."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from credleaf_kaggle.client import KaggleCLI, credentials
from credleaf_kaggle.jobs import advance, load_job, prepare, retry_failed


def parser():
    p = argparse.ArgumentParser(description="Run Credleaf embeddings on private Kaggle GPU kernels")
    p.add_argument("action", choices=["prepare", "tick", "run", "status", "publish", "retry-failed"])
    p.add_argument("--root", type=Path, default=Path(os.getenv("CREDLEAF_ROOT", Path.cwd())))
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--owner", help="Kaggle username; required when preparing a job")
    p.add_argument("--auth-file", type=Path, help="Local kaggle.json or a plain API token file; never uploaded")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source-snapshot-id")
    source.add_argument("--source-run-id")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--revision", default="main")
    p.add_argument("--shard-documents", type=int, default=2000)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--write-batch-size", type=int, default=128)
    p.add_argument("--accelerator", default="NvidiaTeslaT4")
    p.add_argument("--kernel-timeout", type=int, default=39600)
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--wait-hours", type=float, default=168)
    p.add_argument("--qdrant-url", default=os.getenv("CREDLEAF_QDRANT_URL", "http://localhost:6333"))
    p.add_argument("--qdrant-prefix", default="credleaf")
    p.add_argument("--download-only", action="store_true", help="Keep validated downloads without Qdrant import/publication")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.poll_seconds < 10 or args.wait_hours <= 0:
        raise ValueError("Polling must be at least 10 seconds; wait-hours must be positive")
    if args.action == "prepare" or (args.action == "run" and not (args.work_dir / "state.json").exists()):
        if not args.owner:
            raise ValueError("--owner is required to prepare a new job")
        prepare(args.root, args.work_dir, owner=args.owner, snapshot_id=args.source_snapshot_id,
            source_run_id=args.source_run_id, model=args.model, revision=args.revision,
            shard_documents=args.shard_documents, encode_batch_size=args.encode_batch_size,
            write_batch_size=args.write_batch_size, accelerator=args.accelerator, kernel_timeout=args.kernel_timeout)
    if args.action in {"tick", "run"}:
        cli = KaggleCLI(credentials(args.auth_file))
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            complete = advance(args.work_dir, cli)
            if args.action == "tick" or complete:
                break
            job = load_job(args.work_dir)
            done = sum(s["status"] == "downloaded" for s in job["shards"])
            print(f"Downloaded {done}/{len(job['shards'])} shards", file=sys.stderr, flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("Local wait expired; remote job is retained. Resume with the same work-dir.")
            time.sleep(args.poll_seconds)
    if args.action == "retry-failed":
        retry_failed(args.work_dir, KaggleCLI(credentials(args.auth_file)))
    if args.action == "publish" or (args.action == "run" and not args.download_only):
        from credleaf_kaggle.results import publish
        publish(args.work_dir, qdrant_url=args.qdrant_url, qdrant_prefix=args.qdrant_prefix,
                batch_size=args.write_batch_size)
    job = load_job(args.work_dir)
    print(json.dumps(dict(work_dir=str(args.work_dir.resolve()), status=job["status"],
        shards=len(job["shards"]), downloaded=sum(s["status"] == "downloaded" for s in job["shards"]),
        documents=sum(s["documents"] for s in job["shards"]), published=job.get("published"),
        kernels=[dict(kernel=s["kernel"], status=s["status"], remote_status=s.get("remote_status")) for s in job["shards"]])))


if __name__ == "__main__" and __package__ == "credleaf_kaggle":
    main()
