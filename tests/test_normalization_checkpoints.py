import importlib
import json
from pathlib import Path
import threading
import time

import pytest

from ingest import Capture, CaptureRecord
from normalize.checkpoint import DocumentJournal
from normalize.extract import Extracted
from pipeline.artifacts import write_json
from pipeline.control import RunControl
from pipeline.timing import estimate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    module = importlib.import_module("normalize.normalize")
    records = [CaptureRecord(collection_id=13559, seed_id=i, seed_url=f"https://example.org/{i}",
        phase="P1_global_onset", capture=Capture(urlkey=str(i), timestamp="20200301000000",
        url=f"https://example.org/{i}", digest=str(i))) for i in range(5)]
    source = tmp_path / "captures.jsonl"
    source.write_text("".join(r.model_dump_json() + "\n" for r in records))
    calls = []
    def process(client, record, cache, use_cache, stats):
        calls.append(record.seed_id)
        stats.from_cache += 1
        stats.extracted += 1
        return module._make_doc(record, "ok", extracted=Extracted(text="verified document"))
    monkeypatch.setattr(module, "_process", process)
    options = dict(output_path=tmp_path / "documents.jsonl", checkpoint_dir=tmp_path / "checkpoint",
                   checkpoint_identity={"input": "same", "implementation": "same"})
    return module, source, records, calls, process, options


def test_interrupted_run_resumes_at_next_document(setup, monkeypatch):
    module, source, records, calls, process, options = setup
    def interrupt(client, record, cache, use_cache, stats):
        if record.seed_id == 2:
            raise KeyboardInterrupt()
        return process(client, record, cache, use_cache, stats)
    monkeypatch.setattr(module, "_process", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.normalize_phase(None, source, **options)
    assert calls == [0, 1]
    # An interrupted final append is safely discarded on recovery.
    with (options["checkpoint_dir"] / "journal.jsonl").open("ab") as stream:
        stream.write(b'{"incomplete":')
    monkeypatch.setattr(module, "_process", process)
    _, stats = module.normalize_phase(None, source, **options)
    assert calls == [0, 1, 2, 3, 4]
    assert stats.total == stats.extracted == 5
    rows = [json.loads(x) for x in options["output_path"].read_text().splitlines()]
    assert [r["seed_id"] for r in rows] == list(range(5))


def test_checkpoint_rejects_input_or_code_changes(setup):
    module, source, records, calls, process, options = setup
    module.normalize_phase(None, source, **options)
    with pytest.raises(ValueError, match="changed"):
        module.normalize_phase(None, source, **(options | {"checkpoint_identity": {"implementation": "different"}}))
    assert len(calls) == 5


def test_complete_corrupt_row_is_never_silently_discarded(setup):
    module, source, records, calls, process, options = setup
    module.normalize_phase(None, source, **options)
    path = options["checkpoint_dir"] / "journal.jsonl"
    path.write_text(path.read_text().replace('verified document', 'tampered document', 1))
    with pytest.raises(ValueError, match="checksum"):
        module.normalize_phase(None, source, **options)


def test_only_one_checkpoint_writer(setup):
    module, source, records, calls, process, options = setup
    a = DocumentJournal(options["checkpoint_dir"], options["checkpoint_identity"], records)
    try:
        with pytest.raises(BlockingIOError):
            DocumentJournal(options["checkpoint_dir"], options["checkpoint_identity"], records)
    finally:
        a.close()


def test_pause_acknowledges_committed_document_and_resumes(setup, tmp_path, monkeypatch):
    module, source, records, calls, process, options = setup
    path = tmp_path / "control/request.json"
    write_json(path, dict(pause_requested=False))
    control = RunControl(path, "normalize__13559")
    errors = []
    def pause_after_first(client, record, cache, use_cache, stats):
        result = process(client, record, cache, use_cache, stats)
        if record.seed_id == 0:
            write_json(path, dict(pause_requested=True))
        return result
    monkeypatch.setattr(module, "_process", pause_after_first)
    def run():
        try:
            module.normalize_phase(None, source, control=control, **options)
        except BaseException as e:
            errors.append(e)
    thread = threading.Thread(target=run)
    thread.start()
    ack = path.with_name("normalize__13559.json")
    for _ in range(200):
        if ack.exists():break
        time.sleep(.01)
    try:
        assert json.loads(ack.read_text())["completed"] == 1
        assert calls == [0]
        assert len((options["checkpoint_dir"] / "journal.jsonl").read_text().splitlines()) == 1
    finally:
        write_json(path, dict(pause_requested=False))
        thread.join(timeout=5)
    assert not thread.is_alive() and not errors
    assert calls == list(range(5))
    assert json.loads(ack.read_text())["status"] == "running"


def test_paused_eta_has_no_finish_and_excludes_pause_time():
    value = estimate(1000, progress=dict(completed=5, total=10, paused=True, processing_seconds=100, paused_seconds=900))
    assert value["source"] == "paused" and value["remaining_seconds"] is None
    assert value["elapsed_seconds"] == 100
    resumed = estimate(1000, progress=dict(completed=5, total=10, processing_seconds=100, paused_seconds=900))
    assert resumed["remaining_seconds"] == 100


def test_verified_legacy_prefix_is_adopted_without_reprocessing(setup, tmp_path, monkeypatch):
    import importlib.util
    from datetime import datetime, timezone, timedelta
    from types import SimpleNamespace
    from pipeline.artifacts import digest, file_hash, implementation
    from pipeline.spec import defaults, stage_config
    module, source, records, calls, process, options = setup
    script = Path(__file__).resolve().parents[1] / "scripts/adopt_normalization_checkpoint.py"
    spec = importlib.util.spec_from_file_location("adopt_checkpoint", script)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    raw = tmp_path / "raw/captures/13559/captures.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(source.read_bytes())
    ref = dict(stage="harvest", collection_id=13559, id="harvest-" + "a"*32, sha256="hash")
    monkeypatch.setattr(tool, "ArtifactStore", lambda root: SimpleNamespace(payload=lambda ref: tmp_path / "raw"))
    params = defaults() | dict(normalize_max_docs=None)
    config = stage_config("normalize", params, {})
    plan = dict(root=str(tmp_path), run_id="resumable", implementation=dict(source=implementation(tmp_path)),
                base=dict(artifacts=[ref]), parameters=params, resources={})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    docs = [module._make_doc(r, "ok", extracted=Extracted(text="preserved legacy text")) for r in records[:2]]
    (handoff / "documents.jsonl").write_text("".join(d.model_dump_json() + "\n" for d in docs))
    write_json(handoff / "handoff.json", dict(records=2, documents_sha256=file_hash(handoff / "documents.jsonl"),
        capture_sha256=file_hash(raw), config=config,
        source_started_at=(datetime.now(timezone.utc)-timedelta(seconds=2)).isoformat()))
    result = tool.adopt(plan_path, handoff, 13559)
    identity = dict(artifact_key=digest([config, [ref], plan["implementation"]]), config=config, capture_sha256=file_hash(raw))
    _, stats = module.normalize_phase(None, source, **(options | dict(checkpoint_dir=Path(result["checkpoint"]), checkpoint_identity=identity)))
    assert calls == [2, 3, 4]
    assert stats.total == 5 and stats.adopted == 2 and stats.extracted == 5
    rows = [json.loads(x) for x in options["output_path"].read_text().splitlines()]
    assert rows[0]["text"] == rows[1]["text"] == "preserved legacy text"


def test_resumed_progress_keeps_committed_count_during_outage(setup,monkeypatch,tmp_path):
    from ingest.archive_it import ArchiveUnavailable
    from pipeline.timing import Progress
    module,source,records,calls,process,options=setup
    def interrupt(client,record,cache,use_cache,stats):
        if record.seed_id==2:raise ArchiveUnavailable('archive cooldown')
        return process(client,record,cache,use_cache,stats)
    monkeypatch.setattr(module,'_process',interrupt)
    with pytest.raises(ArchiveUnavailable):module.normalize_phase(None,source,**options)
    path=tmp_path/'progress.json'
    with pytest.raises(ArchiveUnavailable):module.normalize_phase(None,source,progress=Progress(path),**options)
    assert json.loads(path.read_text())['completed']==2
