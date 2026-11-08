"""Durable private upload → GPU run → verified download, one shard at a time."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import uuid

from pipeline.artifacts import ArtifactStore, digest, file_hash, implementation, lookup, write_json
from credleaf_kaggle.client import KaggleCommandError

HERE = Path(__file__).parent
SOURCE_FILES = ("config.py", "document_time.py", "url_utils.py", "normalize/dedupe.py",
                "embed/__init__.py", "embed/embed.py", "embed/export.py", "embed/model.py")


@contextmanager
def job_lock(directory):
    with (Path(directory) / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load_job(directory):
    return json.loads((Path(directory) / "state.json").read_text())


def save_job(directory, job):
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(Path(directory) / "state.json", job)


def source_bundle(root, destination):
    # Kaggle expands uploaded ZIPs, removing the original file needed for its
    # checksum. A plain JSON bundle survives Dataset materialization unchanged.
    sources = {name: (Path(root) / "src" / name).read_bytes().decode("utf-8") for name in SOURCE_FILES}
    sources["normalize/__init__.py"] = ""
    write_json(destination, sources)


def input_bundle(root, snapshot_id=None, source_run_id=None):
    store = ArtifactStore(root)
    if bool(snapshot_id) == bool(source_run_id):
        raise ValueError("Choose exactly one source_snapshot_id or source_run_id")
    if snapshot_id:
        return store.bundle(snapshot_id, "corpus")
    from pipeline.plan import run_directory
    directory = run_directory(root, "credleaf_build_corpus", source_run_id)
    plan = json.loads((directory / "plan.json").read_text())
    refs = []
    for stage in ("harvest", "normalize", "dedupe", "georef", "credibility"):
        for cid in plan["parameters"]["collection_ids"] if stage != "credibility" else [None]:
            key = stage if cid is None else f"{stage}__{cid}"
            path = directory / "steps" / (key + ".json")
            if not path.exists():
                raise ValueError(f"Source run is not ready: missing {key}")
            report = json.loads(path.read_text())
            if report["status"] not in {"success", "cached", "reused", "off"}:
                raise ValueError(f"Source run is not ready: {key} is {report['status']}")
            ref = report["artifact"]
            if ref.get("id"):
                store.verify_tree(ref)
            refs.append(ref)
    return dict(kind="corpus", collection_ids=plan["parameters"]["collection_ids"], artifacts=refs,
                implementation=plan["implementation"], source_run_id=source_run_id)


def prepare(root, directory, *, owner, snapshot_id=None, source_run_id=None, model="BAAI/bge-m3",
            revision="main", shard_documents=2000, encode_batch_size=16, write_batch_size=128,
            accelerator="NvidiaTeslaT4", kernel_timeout=39600, inputs=None, source_root=None):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", owner):
        raise ValueError("Invalid Kaggle username")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        raise ValueError("Kaggle requires a public Hugging Face model ID, e.g. BAAI/bge-m3")
    if min(shard_documents, encode_batch_size, write_batch_size) < 1 or not 600 <= kernel_timeout <= 43200:
        raise ValueError("Batch/shard sizes must be positive; kernel timeout must be 600–43200 seconds")
    options = dict(owner=owner, source_snapshot_id=snapshot_id, source_run_id=source_run_id, model=model,
        requested_revision=revision, shard_documents=shard_documents, encode_batch_size=encode_batch_size,
        write_batch_size=write_batch_size, accelerator=accelerator, kernel_timeout=kernel_timeout)
    if inputs is not None:
        if snapshot_id or source_run_id:
            raise ValueError("Stage inputs cannot be combined with a source snapshot/run")
        options["inputs"] = inputs
        options["source_root"] = str(Path(source_root or root).resolve())
    directory.mkdir(parents=True, exist_ok=True)
    with job_lock(directory):
        if (directory / "state.json").exists():
            job = load_job(directory)
            if job["options"] != options:
                raise ValueError("Job configuration changed; choose a new work directory")
            return job
        if any(p.name != ".lock" for p in directory.iterdir()):
            raise ValueError("Incomplete preparation exists; choose a new work directory")
        bundle = (input_bundle(root, snapshot_id, source_run_id) if inputs is None else
                  dict(kind="corpus", collection_ids=[ref["collection_id"] for ref in inputs], artifacts=inputs))
        store = ArtifactStore(root)
        refs = [lookup(bundle, "dedupe", cid) for cid in bundle["collection_ids"]]
        if any(not store.load(ref)["config"].get("corpus_layout") for ref in refs):
            raise ValueError("Kaggle requires collection-v1 dedupe artifacts")
        from huggingface_hub import HfApi
        info = HfApi(token=False).model_info(model, revision=revision)
        if info.private or info.gated:
            raise ValueError("Use an ungated public encoder; Hugging Face credentials are not uploaded")
        model_revision = info.sha
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision or ""):
            raise ValueError("Could not resolve the encoder to an immutable revision")
        chunker_model = "minishlab/potion-base-32M"
        chunker_revision = HfApi(token=False).model_info(chunker_model).sha
        if not re.fullmatch(r"[0-9a-f]{40}", chunker_revision or ""):
            raise ValueError("Could not resolve the semantic chunker to an immutable revision")
        source_bundle(source_root or root, directory / "source.json")
        shutil.copyfile(HERE / "kernel.py", directory / "kernel.py")
        job = dict(format=1, job_id=uuid.uuid4().hex, root=str(root), options=options, bundle=bundle,
            inputs=refs, source=implementation(Path(source_root or root)), model_revision=model_revision,
            chunker_model=chunker_model, chunker_revision=chunker_revision, shards=[],
            adapter=digest({p.name: file_hash(p) for p in sorted(HERE.glob("*.py"))}), status="prepared")
        target, shard = None, None
        try:
            for ref in refs:
                file = store.payload(ref) / "deduped" / str(ref["collection_id"]) / "documents.jsonl"
                with file.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row.get("fetch_status") != "ok" or not (row.get("text") or "").strip():
                            continue
                        if target is None or shard["documents"] == shard_documents:
                            if target:
                                target.close()
                            index = len(job["shards"])
                            folder = directory / f"shard-{index:05d}"
                            (folder / "upload").mkdir(parents=True)
                            slug = f"credleaf-{job['job_id'][:16]}-{index:05d}"
                            shard = dict(index=index, slug=slug, dataset=f"{owner}/{slug}-input",
                                         kernel=f"{owner}/{slug}", documents=0, status="prepared")
                            job["shards"].append(shard)
                            target = (folder / "upload/documents.jsonl").open("w")
                        target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        shard["documents"] += 1
        finally:
            if target:
                target.close()
        if not job["shards"]:
            raise ValueError("No successfully extracted, nonempty documents to embed")
        for shard in job["shards"]:
            folder = directory / f"shard-{shard['index']:05d}"
            upload = folder / "upload"
            shutil.copyfile(directory / "source.json", upload / "source.json")
            spec = dict(format=1, job_id=job["job_id"], shard=shard["index"], model=model,
                model_revision=model_revision, chunker_model=chunker_model, chunker_revision=chunker_revision,
                documents=shard["documents"], encode_batch_size=encode_batch_size,
                write_batch_size=write_batch_size, files={name: file_hash(upload / name)
                    for name in ("documents.jsonl", "source.json")})
            write_json(upload / "job.json", spec)
            shard["spec"] = spec
            write_json(upload / "dataset-metadata.json", dict(id=shard["dataset"], title=shard["slug"] + " input",
                licenses=[dict(name="other")], description="Private Credleaf embedding input; original authors retain rights. No additional redistribution license is granted."))
            (folder / "kernel").mkdir()
            shutil.copyfile(directory / "kernel.py", folder / "kernel/kernel.py")
            write_json(folder / "kernel/kernel-metadata.json", dict(id=shard["kernel"], title=shard["slug"],
                code_file="kernel.py", language="python", kernel_type="script", is_private=True,
                enable_gpu=True, enable_internet=True, dataset_sources=[shard["dataset"]],
                competition_sources=[], kernel_sources=[], model_sources=[]))
        save_job(directory, job)
        return job


def advance(directory, cli):
    """One polling tick, shared by CLI and Airflow's rescheduling sensor."""
    directory = Path(directory)
    with job_lock(directory):
        job = load_job(directory)
        if job["status"] == "published":
            return True
        pending = next((s for s in job["shards"] if s["status"] != "downloaded"), None)
        if pending is None:
            job["status"] = "downloaded"
            save_job(directory, job)
            return True
        shard, folder = pending, directory / f"shard-{pending['index']:05d}"
        if shard["status"] == "prepared":
            # Retrying a request with an unknown outcome must not create another GPU run.
            shard["status"] = "uploading"
            save_job(directory, job)
            try:
                response = cli.run("datasets", "create", "-p", folder / "upload", "--keep-tabular", timeout=3600)
            except KaggleCommandError as error:
                shard["last_error"] = str(error)
                if error.rejected:
                    shard["status"] = "prepared"
                save_job(directory, job)
                raise
            shard["upload_response"] = response[:2000]
            shard.pop("last_error", None)
            shard["status"] = "uploaded"
            save_job(directory, job)
        if shard["status"] == "uploading":
            cli.run("datasets", "files", shard["dataset"], "--csv")
            shard["status"] = "uploaded"
            save_job(directory, job)
        if shard["status"] == "uploaded":
            # Kaggle indexes uploads asynchronously. Do not submit until it is ready.
            output = cli.run("datasets", "status", shard["dataset"])
            if not re.search(r"\bready\b", output, re.I):
                if re.search(r"\b(error|failed)\b", output, re.I):
                    raise RuntimeError("Kaggle input processing failed: " + output[-500:])
                return False
            shard["status"] = "submitting"
            save_job(directory, job)
            try:
                response = cli.run("kernels", "push", "-p", folder / "kernel", "--accelerator", job["options"]["accelerator"],
                        "--timeout", job["options"]["kernel_timeout"])
            except KaggleCommandError as error:
                shard["last_error"] = str(error)
                if error.rejected:
                    shard["status"] = "uploaded"
                save_job(directory, job)
                raise
            shard["submission_response"] = response[:2000]
            shard.pop("last_error", None)
            shard["status"] = "submitted"
            save_job(directory, job)
        # A lost push response is reconciled by reading this unique kernel, never pushing twice.
        if shard["status"] == "submitting":
            shard["remote_status"] = cli.status(shard["kernel"])
            shard["status"] = "submitted"
            save_job(directory, job)
        if shard["status"] == "submitted":
            status = cli.status(shard["kernel"])
            shard["remote_status"] = status
            job["status"] = "running"
            save_job(directory, job)
            if status in {"error", "cancelled", "canceled"}:
                job["status"] = "failed"
                failure_path = folder / (shard["kernel"].split("/")[-1] + ".failure.log")
                if not failure_path.exists():
                    try:
                        logs = cli.run("kernels", "logs", shard["kernel"], timeout=90)
                        try:
                            events = json.loads(logs)
                            if isinstance(events, list):
                                logs = "".join(e.get("data", "") for e in events if isinstance(e, dict))
                        except ValueError:
                            pass
                        failure_path.write_text(logs)
                    except Exception as error:
                        failure_path.write_text("Could not retrieve Kaggle logs: " + str(error))
                shard["failure_log"] = str(failure_path)
                save_job(directory, job)
                raise RuntimeError(f"Kaggle shard {shard['index']} is {status}: https://www.kaggle.com/code/{shard['kernel']}\n"
                                   + failure_path.read_text()[-6000:])
            if status != "complete":
                return False
            download = folder / "download"
            download.mkdir(exist_ok=True)
            cli.run("kernels", "output", shard["kernel"], "-p", download, "--force",
                    "--file-pattern", r"^(metadata\.jsonl|embeddings\.npy|result\.json)$", timeout=3600)
            from credleaf_kaggle.results import validate_shard
            shard["result"] = validate_shard(download, shard["spec"])
            shard["status"] = "downloaded"
            if all(s["status"] == "downloaded" for s in job["shards"]):
                job["status"] = "downloaded"
            save_job(directory, job)
        return all(s["status"] == "downloaded" for s in job["shards"])


def retry_failed(directory, cli):
    """Explicitly retry a terminal remote failure, keeping successful downloads."""
    directory = Path(directory)
    with job_lock(directory):
        job = load_job(directory)
        shard = next((s for s in job["shards"] if s["status"] != "downloaded"), None)
        if shard is None or shard["status"] not in {"submitted", "submitting"}:
            raise ValueError("No submitted failed shard to retry")
        if cli.status(shard["kernel"]) not in {"error", "cancelled", "canceled"}:
            raise ValueError("Retry is allowed only after Kaggle confirms a terminal failure")
        shard.setdefault("previous_kernels", []).append(shard["kernel"])
        slug = f"{shard['slug']}-retry-{len(shard['previous_kernels'])}"
        shard["kernel"] = job["options"]["owner"] + "/" + slug
        folder = directory / f"shard-{shard['index']:05d}/kernel"
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        metadata.update(id=shard["kernel"], title=slug)
        write_json(folder / "kernel-metadata.json", metadata)
        shard.update(status="uploaded", remote_status=None)
        job["status"] = "prepared"
        save_job(directory, job)
