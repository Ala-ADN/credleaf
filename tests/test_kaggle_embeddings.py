"""Offline remote-job lifecycle, credentials and atomic artifact import contracts."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.artifacts import ArtifactStore, file_hash, lookup, write_json
from tests.test_pipeline import fixture_outputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job(tmp_path, monkeypatch):
    from credleaf_kaggle.jobs import prepare
    root = tmp_path / "project"
    root.mkdir()
    (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, root / name)
    store = ArtifactStore(root)
    row = dict(text="First article with useful scientific content.", fetch_status="ok", collection_id=13559,
        capture_url="https://example.org/a", language="en", publish_date="2020-03-01", capture_timestamp="20200302000000")
    rows = [row, dict(row), dict(row, capture_url="https://example.org/b", text="Another article.")]
    refs, previous = [], None
    for stage, cid in (("credibility", None), ("harvest", 13559), ("normalize", 13559), ("dedupe", 13559)):
        config = dict(corpus_layout="collection-v1", capture_phase="all")
        ref, _ = store.build(stage, cid, config, [previous] if stage in {"normalize", "dedupe"} else [],
            "fixture", lambda out, stage=stage, cid=cid, config=config: fixture_outputs(out, stage, cid, config, rows=rows))
        refs.append(ref)
        if cid:
            previous = ref
    refs += [dict(id=None, stage="georef", collection_id=13559, mode="off"),
             dict(id=None, stage="embed", collection_id=None, mode="off")]
    bundle = dict(kind="corpus", collection_ids=[13559], artifacts=refs)
    def produce(out):
        write_json(out / "snapshot.json", bundle)
        return {}
    snapshot, _ = store.build("publish", None, dict(kind="corpus"), [r for r in refs if r.get("id")], "fixture", produce)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **kw: SimpleNamespace(sha="a"*40, private=False, gated=False))
    directory = tmp_path / "job"
    value = prepare(root, directory, owner="tester", snapshot_id=snapshot["id"], shard_documents=1)
    return root, directory, value


def returned_output(directory, spec):
    from document_time import assign_document_time
    from normalize.dedupe import content_hash
    from embed.embed import stable_chunk_id
    directory.mkdir(exist_ok=True)
    row = json.loads((directory.parent / "upload/documents.jsonl").read_text())
    metadata = dict(assign_document_time(row), content_hash=content_hash(row["text"]), dedupe_version=2)
    metadata["point_id"] = stable_chunk_id("credleaf_snapshot_v1", metadata, row["text"])
    (directory / "metadata.jsonl").write_text(json.dumps(metadata) + "\n")
    np.save(directory / "embeddings.npy", np.array([[1., 2., 3.]], dtype=np.float32))
    result = dict(format=1, job_id=spec["job_id"], shard=spec["shard"], model=spec["model"],
        model_revision=spec["model_revision"], input_sha256=spec["files"]["documents.jsonl"],
        chunker_model=spec["chunker_model"], chunker_revision=spec["chunker_revision"],
        source_sha256=spec["files"]["source.json"], documents=spec["documents"], chunks=1, dimension=3,
        runtime={"torch": "fixture", "chonkie": "1.6.5"},
        files={name: file_hash(directory / name) for name in ("metadata.jsonl", "embeddings.npy")})
    write_json(directory / "result.json", result)


class FakeCLI:
    def __init__(self):
        self.calls = []
        self.remote_status = "running"

    def run(self, *args, **kwargs):
        self.calls.append(tuple(map(str, args)))
        if args[:2] == ("datasets", "status"):
            return "ready"
        if args[:2] == ("kernels", "output"):
            directory = Path(args[args.index("-p") + 1])
            spec = json.loads((directory.parent / "upload/job.json").read_text())
            returned_output(directory, spec)
        return "ok"

    def status(self, kernel):
        return self.remote_status


def download_all(directory, cli):
    from credleaf_kaggle.jobs import advance
    cli.remote_status = "complete"
    for _ in range(20):
        if advance(directory, cli):
            return
    raise AssertionError("Job did not finish")


def test_credentials_are_child_environment_only(tmp_path, monkeypatch):
    from credleaf_kaggle.client import credentials, KaggleCLI
    token = tmp_path / "auth"
    token.write_text("secret-token")
    monkeypatch.setenv("KAGGLE_KEY", "stale-key")
    env = credentials(token)
    assert env["KAGGLE_API_TOKEN"] == "secret-token" and "KAGGLE_KEY" not in env
    token.write_text(json.dumps(dict(username="tester", key="legacy-secret")))
    env = credentials(token)
    assert env["KAGGLE_USERNAME"] == "tester" and env["KAGGLE_KEY"] == "legacy-secret"
    import os
    assert env["KAGGLE_API_TOKEN"] == os.devnull
    captured = []
    def failed(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=1, stdout="bad legacy-secret", stderr="")
    monkeypatch.setattr("subprocess.run", failed)
    with pytest.raises(RuntimeError, match=r"\[redacted\]") as error:
        KaggleCLI(env, executable="kaggle").run("kernels", "status", "tester/job")
    assert "legacy-secret" not in str(error.value)
    assert all("legacy-secret" not in part for part in captured[0])


def test_cli_status_parses_the_status_field_instead_of_owner_or_failure_text(monkeypatch):
    from credleaf_kaggle.client import KaggleCLI
    cli = KaggleCLI(executable="kaggle")
    monkeypatch.setattr(cli, "run", lambda *a, **kw: 'running/job has status "KernelWorkerStatus.ERROR"\nFailure message: "running failed"')
    assert cli.status("running/job") == "error"
    for remote, expected in (("CANCEL_REQUESTED", "running"), ("CANCEL_ACKNOWLEDGED", "cancelled"), ("NEW_SCRIPT", "queued")):
        monkeypatch.setattr(cli, "run", lambda *a, status=remote, **kw: f'tester/job has status "KernelWorkerStatus.{status}"')
        assert cli.status("tester/job") == expected


@pytest.mark.parametrize("command,response", [
    (("datasets", "create"), "Upload successful\nDataset creation error: Please select a valid license\n"),
    (("kernels", "push"), "Kernel push error: GPU quota exhausted\n"),
])
def test_cli_zero_exit_rejections_are_errors(monkeypatch, command, response):
    from credleaf_kaggle.client import KaggleCLI, KaggleCommandError
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=response, stderr=""))
    with pytest.raises(KaggleCommandError) as error:
        KaggleCLI(executable="kaggle").run(*command)
    assert error.value.rejected


def test_rejected_dataset_creation_does_not_mark_uploaded_or_poll_a_missing_dataset(job):
    from credleaf_kaggle.client import KaggleCommandError
    from credleaf_kaggle.jobs import advance, load_job
    _, directory, _ = job
    class Rejected(FakeCLI):
        def run(self, *args, **kwargs):
            self.calls.append(tuple(args))
            raise KaggleCommandError("Please select a valid license", rejected=True)
    cli = Rejected()
    with pytest.raises(KaggleCommandError):
        advance(directory, cli)
    shard = load_job(directory)["shards"][0]
    assert shard["status"] == "prepared" and "valid license" in shard["last_error"]
    assert len(cli.calls) == 1 and cli.calls[0][:2] == ("datasets", "create")


def test_retry_requires_terminal_failure_and_retains_prior_downloads(job):
    from credleaf_kaggle.jobs import advance, load_job, retry_failed
    _, directory, _ = job
    cli = FakeCLI()
    cli.remote_status = "complete"
    assert not advance(directory, cli)  # First shard downloaded.
    cli.remote_status = "running"
    assert not advance(directory, cli)  # Second shard running.
    with pytest.raises(ValueError, match="terminal failure"):
        retry_failed(directory, cli)
    cli.remote_status = "error"
    retry_failed(directory, cli)
    value = load_job(directory)
    assert value["shards"][0]["status"] == "downloaded"
    assert value["shards"][1]["kernel"].endswith("-retry-1")
    assert len(value["shards"][1]["previous_kernels"]) == 1


def test_incomplete_source_run_is_rejected_before_upload(tmp_path):
    from credleaf_kaggle.jobs import input_bundle
    from pipeline.plan import run_directory
    directory = run_directory(tmp_path, "credleaf_build_corpus", "paused")
    write_json(directory / "plan.json", dict(parameters=dict(collection_ids=[13559])))
    with pytest.raises(ValueError, match="Source run is not ready"):
        input_bundle(tmp_path, source_run_id="paused")


def test_private_shards_pin_model_and_exclude_credentials(job, monkeypatch):
    from credleaf_kaggle.jobs import prepare
    root, directory, value = job
    assert len(value["shards"]) == 3 and value["model_revision"] == "a"*40
    for shard in value["shards"]:
        folder = directory / f"shard-{shard['index']:05d}"
        metadata = json.loads((folder / "kernel/kernel-metadata.json").read_text())
        assert metadata["is_private"] is True and metadata["enable_gpu"] is True
        assert metadata["dataset_sources"] == [shard["dataset"]]
        assert set(p.name for p in (folder / "upload").iterdir()) == {
            "documents.jsonl", "source.json", "job.json", "dataset-metadata.json"}
    assert prepare(root, directory, owner="tester", snapshot_id=value["options"]["source_snapshot_id"], shard_documents=1) == value
    with pytest.raises(ValueError, match="configuration changed"):
        prepare(root, directory, owner="tester", snapshot_id=value["options"]["source_snapshot_id"], shard_documents=2)


def test_source_transport_survives_kaggle_dataset_mount_and_preserves_python_bytes(job, tmp_path):
    import subprocess
    import sys
    from credleaf_kaggle.kernel import unpack_sources
    from credleaf_kaggle.jobs import SOURCE_FILES
    root, directory, _ = job
    mounted = tmp_path / "mounted"
    shutil.copytree(directory / "shard-00000/upload", mounted)
    # Kaggle expands archives on mount. The transport must contain no archive.
    assert not list(mounted.glob("*.zip"))
    destination = tmp_path / "unpacked"
    unpack_sources(mounted / "source.json", destination)
    for name in SOURCE_FILES:
        assert (destination / name).read_bytes() == (root / "src" / name).read_bytes()
    assert (destination / "normalize/__init__.py").read_text() == ""
    # A fresh interpreter must import the mounted package, without the project's
    # already-imported modules masking a missing helper in the upload bundle.
    result = subprocess.run([sys.executable, "-I", "-c",
        "import sys;sys.path.insert(0,sys.argv[1]);import embed.embed,embed.export;"
        "assert embed.embed.__file__.startswith(sys.argv[1])", str(destination)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_bundle_rejects_paths_outside_workspace(tmp_path):
    from credleaf_kaggle.kernel import unpack_sources
    bundle = tmp_path / "source.json"
    bundle.write_text(json.dumps({"../escape.py": "pass"}))
    with pytest.raises(ValueError, match="Invalid source bundle path"):
        unpack_sources(bundle, tmp_path / "unpacked")


def test_remote_script_entrypoint_produces_a_valid_export_from_mounted_shard(job, tmp_path, monkeypatch):
    import torch
    import sentence_transformers
    import chonkie.embeddings
    import embed.embed as embedding
    from chonkie.embeddings import BaseEmbeddings
    from credleaf_kaggle import kernel
    from credleaf_kaggle.results import validate_shard
    _, directory, value = job
    class Encoder:
        def __init__(self, *args, **kwargs): pass
        def encode(self, texts, **kwargs):
            return np.array([[1., float(len(text))] for text in texts], dtype=np.float32)
    class Embeddings(BaseEmbeddings):
        def __init__(self, encoder): self.encoder = encoder
        @property
        def dimension(self): return 2
        def embed(self, text): return self.encoder.encode([text])[0]
        def get_tokenizer(self): return None
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Encoder)
    monkeypatch.setattr(chonkie.embeddings, "SentenceTransformerEmbeddings", Embeddings)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda n: "test encoder")
    monkeypatch.setattr(kernel.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(embedding, "SemanticChunker", lambda **kwargs: SimpleNamespace(
        chunk=lambda text: [SimpleNamespace(text=text, start_index=0, end_index=len(text), token_count=1)]))
    output = tmp_path / "output"
    output.mkdir()
    kernel.main(directory / "shard-00000/upload", output)
    result = validate_shard(output, value["shards"][0]["spec"])
    assert result["chunks"] == 1 and result["dimension"] == 2


def test_restart_reconnects_without_duplicate_gpu_submission(job):
    from credleaf_kaggle.jobs import advance, load_job, save_job
    _, directory, _ = job
    cli = FakeCLI()
    assert advance(directory, cli) is False
    assert advance(directory, cli) is False
    assert sum(call[:2] == ("kernels", "push") for call in cli.calls) == 1
    value = load_job(directory)
    value["shards"][0]["status"] = "submitting"  # Lost push response, server accepted it.
    save_job(directory, value)
    assert advance(directory, cli) is False
    assert sum(call[:2] == ("kernels", "push") for call in cli.calls) == 1
    download_all(directory, cli)
    assert load_job(directory)["status"] == "downloaded"
    assert sum(call[:2] == ("kernels", "push") for call in cli.calls) == 3


def test_failed_remote_kernel_never_publishes_or_resubmits(job):
    from credleaf_kaggle.jobs import advance
    _, directory, _ = job
    cli = FakeCLI()
    cli.remote_status = "error"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="is error"):
            advance(directory, cli)
    assert sum(call[:2] == ("kernels", "push") for call in cli.calls) == 1


@pytest.mark.parametrize("damage", ["wrong_job", "checksum", "row_alignment", "nonfinite", "provenance"])
def test_download_corruption_or_foreign_provenance_is_rejected(job, damage):
    from credleaf_kaggle.jobs import load_job
    from credleaf_kaggle.results import validate_shard, index_rows
    _, directory, _ = job
    download_all(directory, FakeCLI())
    value = load_job(directory)
    shard = value["shards"][0]
    output = directory / "shard-00000/download"
    result = json.loads((output / "result.json").read_text())
    if damage == "wrong_job": result["job_id"] = "other-job"
    elif damage == "checksum": (output / "metadata.jsonl").write_text("broken")
    elif damage == "row_alignment":
        (output / "metadata.jsonl").write_text("")
        result["files"]["metadata.jsonl"] = file_hash(output / "metadata.jsonl")
    elif damage == "nonfinite":
        np.save(output / "embeddings.npy", np.array([[1., np.nan, 0.]], dtype=np.float32))
        result["files"]["embeddings.npy"] = file_hash(output / "embeddings.npy")
    else:
        row = json.loads((output / "metadata.jsonl").read_text())
        row["capture_url"] = "https://foreign.org/page"
        from embed.embed import stable_chunk_id
        row["point_id"] = stable_chunk_id("credleaf_snapshot_v1", row, row["text"])
        (output / "metadata.jsonl").write_text(json.dumps(row) + "\n")
        result["files"]["metadata.jsonl"] = file_hash(output / "metadata.jsonl")
    write_json(output / "result.json", result)
    with pytest.raises(ValueError):
        if damage == "provenance":
            with sqlite3.connect(":memory:") as database:
                index_rows(value, directory, database)
        else:
            validate_shard(output, shard["spec"])


def test_publish_merges_shards_preserves_alignment_and_is_idempotent(job, monkeypatch):
    from credleaf_kaggle.jobs import load_job
    from credleaf_kaggle.results import publish
    from qdrant_client import QdrantClient
    root, directory, _ = job
    download_all(directory, FakeCLI())
    clients = []
    def client(**kwargs):
        value = QdrantClient(":memory:")
        clients.append(value)
        return value
    monkeypatch.setattr("qdrant_client.QdrantClient", client)
    published = publish(directory)
    value = load_job(directory)
    store = ArtifactStore(root)
    bundle = store.bundle(published, "corpus")
    embed = lookup(bundle, "embed")
    stats = store.load(embed)["stats"]
    assert stats["chunks"] == 2 and stats["cross_shard_duplicates"] == 1
    output = store.payload(embed)
    rows = [json.loads(line) for line in (output / "input/metadata.jsonl").read_text().splitlines()]
    vectors = np.load(output / "input/embeddings.npy")
    assert len(rows) == len(vectors) == 2
    assert len({row["point_id"] for row in rows}) == 2
    np.testing.assert_array_equal(vectors, [[1., 2., 3.], [1., 2., 3.]])
    assert publish(directory) == published and len(clients) == 1
    assert value["status"] == "published"


def test_corpus_embed_task_runs_kaggle_in_worker_then_uses_normal_publication(job, monkeypatch, tmp_path):
    """Exercise real worker subprocesses, pending results, retries and publication offline."""
    from credleaf_kaggle.jobs import prepare
    from pipeline.artifacts import digest
    from pipeline.plan import prepare as prepare_plan, run_directory
    from pipeline.runner import load_plan, publish, run_stage
    root, _, source = job
    plan_ref = prepare_plan(root, sys.executable, "corpus", "credleaf_build_corpus", "remote-embed",
        dict(collection_ids=[13559], georef_mode="off", embed_backend="kaggle",
             kaggle_shard_documents=1, qdrant_url="http://fixture", embed_batch_size=1))
    plan = load_plan(plan_ref)
    from pipeline.spec import stage_config
    config = stage_config("embed", plan["parameters"], plan["resources"])
    artifact_key = digest([config, source["inputs"], plan["implementation"]])
    directory = root / "data/pipeline/checkpoints" / artifact_key / "embed/kaggle"
    # Resolve model commits offline before starting the real worker.
    prepare(root, directory, owner="tester", inputs=source["inputs"], source_root=plan["code_root"],
            shard_documents=1, write_batch_size=1)
    remote = tmp_path / "remote-status"
    remote.write_text("running")
    calls = tmp_path / "cli-calls.jsonl"
    cli = tmp_path / "kaggle"
    cli.write_text(f'''#!{sys.executable}
import json
from pathlib import Path
import sys
args = sys.argv[1:]
with Path({str(calls)!r}).open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[:2] == ["kernels", "status"]:
    print('tester/job has status "' + Path({str(remote)!r}).read_text() + '"')
elif args[:2] == ["kernels", "output"]:
    sys.path.insert(0, {str(ROOT)!r})
    from tests.test_kaggle_embeddings import returned_output
    folder = Path(args[args.index("-p") + 1])
    spec = json.loads((folder.parent / "upload/job.json").read_text())
    returned_output(folder, spec)
else:
    print("ready")
''')
    cli.chmod(0o755)
    environment = dict(os.environ, CREDLEAF_KAGGLE_OWNER="tester", CREDLEAF_KAGGLE_BIN=str(cli),
                       KAGGLE_API_TOKEN="fixture-secret-token")
    refs = source["inputs"]
    assert run_stage(plan_ref, "embed", refs, worker_env=environment) is None
    steps = run_directory(root, plan["dag_id"], plan["run_id"]) / "steps"
    report = json.loads((steps / "embed.json").read_text())
    assert report["status"] == "waiting"
    assert not list((root / "data/pipeline/artifacts").glob("embed-*"))
    assert not (steps / "publish.json").exists()
    remote.write_text("complete")
    # Replace only the external Qdrant service inside the worker, using a wrapper
    # interpreter that otherwise executes the frozen pipeline.worker unchanged.
    wrapper = tmp_path / "worker-python"
    wrapper.write_text(f'''#!{sys.executable}
import runpy
import sys
if sys.argv[1:3] == ["-u", "-m"]:
    import qdrant_client
    client = qdrant_client.QdrantClient
    qdrant_client.QdrantClient = lambda **kwargs: client(":memory:")
    sys.argv = [sys.argv[3], *sys.argv[4:]]
    runpy.run_module("pipeline.worker", run_name="__main__")
else:
    import os
    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    # Keep the signed plan unchanged; substitute the process launcher only.
    import subprocess
    real_popen = subprocess.Popen
    def launch(command, **kwargs):
        if command[1:4] == ["-u", "-m", "pipeline.worker"]:
            command = [str(wrapper), *command[1:]]
        return real_popen(command, **kwargs)
    monkeypatch.setattr("pipeline.runner.subprocess.Popen", launch)
    embedding = None
    for _ in range(3):
        embedding = run_stage(plan_ref, "embed", refs, attempt=2, worker_env=environment)
        if embedding is not None:
            break
    assert embedding is not None
    store = ArtifactStore(root)
    manifest = store.load(embedding)
    assert manifest["implementation"] == plan["implementation"]
    assert manifest["inputs"] == refs and manifest["config"]["embed_backend"] == "kaggle"
    assert manifest["stats"]["chunks"] == 2 and manifest["stats"]["cross_shard_duplicates"] == 1
    assert not (steps / "publish.json").exists()
    from credleaf_kaggle.results import publish as publish_job
    with pytest.raises(ValueError, match="publish_corpus"):
        publish_job(directory)
    before = calls.read_text()
    assert run_stage(plan_ref, "embed", refs, attempt=3, worker_env=environment) == embedding
    assert calls.read_text() == before
    assert sum(json.loads(line)[:2] == ["kernels", "push"] for line in before.splitlines()) == 3
    artifacts = [ref for ref in source["bundle"]["artifacts"] if ref["stage"] != "embed"] + [embedding]
    snapshot = publish(plan_ref, artifacts)
    assert lookup(store.bundle(snapshot, "corpus"), "embed") == embedding
    for base in (directory, steps.parent):
        assert all("fixture-secret-token" not in path.read_text() for path in base.rglob("*.json"))
