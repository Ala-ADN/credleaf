"""Snapshot integrity, scope, transaction and replay contracts (offline)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
from jsonschema import validate, ValidationError

from config import PHASES_BY_NAME
from credibility import CredibilityRegistry
from ingest import Seed, harvest_phase, harvest_all_phases
from pipeline.artifacts import ArtifactStore, digest, file_hash, lookup, write_json
from pipeline.inventory import describe
from pipeline.plan import prepare, run_directory
from pipeline.runner import load_plan, run_stage, publish, write_overview
from pipeline.spec import defaults, required_files, schema, stage_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for folder in ("src", "scripts", "dags"):
        (tmp_path / folder).symlink_to(ROOT / folder, target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def fixture_outputs(output, stage, cid, config, rows=None):
    for name in required_files(stage, cid, config):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            values = rows if rows and path.name == "documents.jsonl" else []
            path.write_text("".join(json.dumps(row) + "\n" for row in values))
        elif path.suffix == ".json":
            write_json(path, dict(low=["example.org"], authoritative=["who.int"], mixed=[])
                       if stage == "credibility" else {})
        else:
            path.write_bytes(b"fixture")
    return {"fixture": True}


def corpus_fixture(root):
    plan_ref = prepare(root, sys.executable, "corpus", "test", "initial",
                       dict(collection_ids=[13559], georef_mode="off", embed_mode="off"))
    plan = load_plan(plan_ref)
    store, refs, previous = ArtifactStore(root), [], None
    for stage, cid in (("credibility", None), ("harvest", 13559), ("normalize", 13559), ("dedupe", 13559)):
        c = stage_config(stage, plan["parameters"], plan["resources"])
        inputs = [previous] if stage in {"normalize", "dedupe"} else []
        ref, _ = store.build(stage, cid, c, inputs, plan["implementation"],
                             lambda out: fixture_outputs(out, stage, cid, c))
        refs.append(ref)
        if cid:
            previous = ref
    refs += [dict(id=None, stage="georef", collection_id=13559, mode="off"),
             dict(id=None, stage="embed", collection_id=None, mode="off")]
    return plan_ref, publish(plan_ref, refs), refs


@pytest.mark.parametrize("kind", ["corpus", "analysis"])
def test_schema_defaults(kind):
    validate(defaults(kind), schema(kind))


@pytest.mark.parametrize("changes", [
    dict(collection_ids=[]), dict(collection_ids=[1]), dict(collection_ids=[13559,13559]),
    dict(harvest_mode="off"), dict(dedupe_threshold=0), dict(harvest_max_seed_error_fraction=1.1),
    dict(harvest_timeout=0), dict(embed_batch_size=0), dict(georef_device="tpu"),
    dict(base_snapshot_id="../../other"), dict(qdrant_url="file:///tmp/x"), dict(unknown=True),
])
def test_invalid_configuration(changes):
    with pytest.raises(ValidationError):
        validate(defaults() | changes, schema())


def test_artifact_transaction_and_cached_retry(tmp_path):
    store, config = ArtifactStore(tmp_path), {"capture_phase":"all"}
    calls = []
    def producer(output):
        calls.append(1)
        return fixture_outputs(output, "harvest", 13559, config)
    ref, cached = store.build("harvest", 13559, config, [], "code", producer)
    again, cached_again = store.build("harvest", 13559, config, [], "code", producer)
    assert ref == again and not cached and cached_again and len(calls) == 1
    before = file_hash(store.directory(ref["id"]) / "manifest.json")
    def fail(output):
        (output / "partial.jsonl").write_text("partial")
        raise RuntimeError("fixture crash")
    with pytest.raises(RuntimeError):
        store.build("harvest", 13559, config, [], "new-code", fail)
    assert file_hash(store.directory(ref["id"]) / "manifest.json") == before
    assert len(list(store.base.iterdir())) == 1


def test_missing_outputs_cannot_publish(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="Missing harvest"):
        store.build("harvest", 13559, {"capture_phase":"all"}, [], "code", lambda output: {})
    assert not list(store.base.iterdir())


def test_reuse_detects_config_code_and_payload_changes(tmp_path):
    store = ArtifactStore(tmp_path)
    ref, _ = store.build("credibility", None, {}, [], "code", lambda out: fixture_outputs(out,"credibility",None,{}))
    assert store.reuse(ref, "credibility", None, {}, [], "code") == ref
    with pytest.raises(ValueError, match="reference scope"):
        store.load(ref | dict(collection_id=4887))
    for config, code in [({"changed":True},"code"), ({},"different-code")]:
        with pytest.raises(ValueError, match="Incompatible"):
            store.reuse(ref, "credibility", None, config, [], code)
    (store.payload(ref) / "credibility.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        store.load(ref)
    with pytest.raises(ValueError):
        store.directory("../escape")


def test_input_changes_during_computation_prevent_publication(tmp_path):
    store = ArtifactStore(tmp_path)
    parent, _ = store.build("credibility", None, {}, [], "code", lambda out: fixture_outputs(out,"credibility",None,{}))
    source = store.payload(parent) / "credibility.json"
    def mutate_input(output):
        source.write_text("{}")
        return fixture_outputs(output,"credibility",None,{})
    with pytest.raises(ValueError, match="missing or changed"):
        store.build("credibility", None, {}, [parent], "code", mutate_input)
    assert len(list(store.base.iterdir())) == 1


def test_invalid_reuse_rejected_before_io(project):
    with pytest.raises(ValueError, match="Reuse requires"):
        prepare(project, sys.executable, "corpus", "test", "bad", dict(georef_mode="off", harvest_mode="reuse"))


def test_plan_is_frozen_and_reuse_checks_upstream(project):
    plan_ref, snapshot, refs = corpus_fixture(project)
    assert (Path(load_plan(plan_ref)["code_root"]) / "src/pipeline/worker.py").is_file()
    with pytest.raises(ValueError, match="configuration/code/runtime changed"):
        prepare(project, sys.executable, "corpus", "test", "initial", dict(collection_ids=[13559],
            georef_mode="off", embed_mode="off", dedupe_threshold=.8))
    reuse = dict(collection_ids=[13559], base_snapshot_id=snapshot["id"], georef_mode="off", embed_mode="off",
                 harvest_mode="reuse", normalize_mode="reuse", dedupe_mode="reuse", credibility_mode="reuse")
    planned = prepare(project, sys.executable, "corpus", "test", "reuse", reuse)
    raw_ref = lookup(ArtifactStore(project).bundle(snapshot), "harvest", 13559)
    assert run_stage(planned,"harvest",[],13559) == raw_ref
    with pytest.raises(ValueError, match="rebuilding its inputs"):
        prepare(project, sys.executable, "corpus", "test", "bad-mix", reuse | dict(harvest_mode="build"))
    with pytest.raises(ValueError, match="config differs"):
        prepare(project, sys.executable, "corpus", "test", "bad-knob", reuse | dict(dedupe_threshold=.8))


def test_changed_plan_is_rejected(project):
    plan_ref,_,_ = corpus_fixture(project)
    value = load_plan(plan_ref)
    value["parameters"]["dedupe_threshold"] = .8
    write_json(Path(plan_ref["path"]), value)
    with pytest.raises(ValueError,match="Run plan changed"):
        load_plan(plan_ref)


class BrokenArchive:
    def iter_seeds(self, *args, **kwargs):
        yield Seed(id=1, collection=13559, url="https://example.org/")
    def iter_captures(self, *args, **kwargs):
        raise httpx.ConnectError("fixture failure")


def test_failed_harvest_preserves_previous_single_and_all_windows(tmp_path):
    path = tmp_path / "P1_global_onset.jsonl"
    path.write_text("previous")
    with pytest.raises(RuntimeError, match="Harvest rejected"):
        harvest_phase(BrokenArchive(),13559,PHASES_BY_NAME["P1_global_onset"],output_path=path)
    assert path.read_text() == "previous"
    with pytest.raises(RuntimeError):
        harvest_all_phases(BrokenArchive(),13559,list(PHASES_BY_NAME.values()),output_dir=tmp_path)
    assert path.read_text() == "previous"


def test_georef_resume_distinguishes_urls_and_rejects_changed_inputs(tmp_path, monkeypatch):
    from georeferencing.georef import georef_phase
    from normalize.models import NormalizedDoc
    from datetime import datetime, timezone
    rows = [NormalizedDoc(collection_id=13559,seed_id=1,seed_url="https://example.org",phase="P1_global_onset",
        capture_timestamp="20200301120000",capture_url=f"https://example.org/{page}",publish_date="2020-03-01",
        fetched_at=datetime.now(timezone.utc),fetch_status="ok",language="en",text=f"Content of {page}") for page in ["a","b"]]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(doc.model_dump_json()+"\n" for doc in rows))
    import importlib
    module = importlib.import_module("georeferencing.georef")
    def process(doc,*args):
        return SimpleNamespace(model_dump_json=lambda: json.dumps(dict(article_id=module._article_id(doc))))
    monkeypatch.setattr(module,"_process_doc",process)
    output = tmp_path / "output.jsonl"
    georef_phase(source,None,None,None,output_path=output,max_docs=1,resume=False)
    georef_phase(source,None,None,None,output_path=output,resume=True)
    assert len(output.read_text().splitlines()) == 2
    source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError,match="inputs/config changed"):
        georef_phase(source,None,None,None,output_path=output,resume=True)


def test_real_dedupe_worker_and_cached_retry(project):
    plan_ref, _, refs = corpus_fixture(project)
    store = ArtifactStore(project)
    plan = load_plan(plan_ref)
    config = stage_config("normalize", plan["parameters"], plan["resources"])
    rows = [dict(collection_id=13559, seed_id=1, seed_url="https://example.org/", capture_url="https://example.org/a",
        phase="P1_global_onset", capture_timestamp="20200301120000", publish_date="2020-02-01", text="A report about public health",
        language="en",fetch_status="ok",extraction_mode="article")]*2
    normal,_ = store.build("normalize",13559,config,[refs[1]],plan["implementation"],
                           lambda out:fixture_outputs(out,"normalize",13559,config,rows=rows), nonce="real-worker-fixture")
    result = run_stage(plan_ref,"dedupe",[normal],13559)
    manifest = store.load(result)
    assert manifest["stats"]["input_documents"] == 2 and manifest["stats"]["exact_duplicates"] == 1
    assert run_stage(plan_ref,"dedupe",[normal],13559,attempt=2) == result
    report = json.loads((Path(plan_ref["path"]).parent / "steps/dedupe__13559.json").read_text())
    assert report["status"] == "cached" and report["eta"]["source"] == "actual"
    from pipeline.timing import timing_key
    key = timing_key("dedupe", 13559, stage_config("dedupe", plan["parameters"], plan["resources"]), plan["implementation"]["runtime"])
    assert len(list((project / "data/pipeline/timings" / key).glob("*.json"))) == 1


def test_normalization_quality_gate_prevents_partial_publication(project, monkeypatch):
    import normalize
    from pipeline.worker import execute
    plan_ref, _, refs = corpus_fixture(project)
    plan = load_plan(plan_ref)
    config = stage_config("normalize", plan["parameters"], plan["resources"])
    store = ArtifactStore(project)
    raw = lookup(store.bundle(publish(plan_ref, refs)), "harvest", 13559)
    def partially_failed(client, cid, phases, *, output_root, **kwargs):
        fixture_outputs(output_root.parent, "normalize", cid, config)
        return output_root / str(cid), dict(normalization={"phase":dict(
            extracted=1, fetch_failed=1, extract_failed=0, total=2)})
    monkeypatch.setattr(normalize,"normalize_collection",partially_failed)
    def produce(output):
        return execute(dict(stage="normalize",collection_id=13559,root=str(project),code_root=plan["code_root"],
                            output=str(output),config=config,inputs=[raw]))
    before=set(store.base.iterdir())
    with pytest.raises(ValueError,match="fetch/extraction failures"):
        store.build("normalize",13559,config,[raw],plan["implementation"],produce,nonce="quality-gate")
    assert set(store.base.iterdir()) == before


def test_overview_reconciles_worker_failure_after_manifest(project):
    plan_ref,snapshot,refs = corpus_fixture(project)
    plan = load_plan(plan_ref)
    directory = run_directory(project,"test","initial")
    for ref in refs:
        key = ref["stage"] if ref["collection_id"] is None else f"{ref['stage']}__{ref['collection_id']}"
        write_json(directory / "steps" / f"{key}.json", dict(stage=ref["stage"],collection_id=ref["collection_id"],
            status="success" if ref["id"] else "off",artifact=ref))
    assert write_overview(project,"test","initial")["status"] == "awaiting_task_states"
    final = write_overview(project,"test","initial",{"publish_corpus[-1]":"failed"},"failed")
    assert final["status"] == "failed" and final["artifact_status"] == "success"
    assert final["snapshot_id"] == snapshot["id"]


def test_analysis_requires_explicit_published_embedding_snapshot(project):
    with pytest.raises(ValueError, match="snapshot_id is required"):
        prepare(project,sys.executable,"analysis","analysis","missing",{})
    _,snapshot,_ = corpus_fixture(project)
    with pytest.raises(ValueError, match="no embed artifact"):
        prepare(project,sys.executable,"analysis","analysis","no-embed",dict(snapshot_id=snapshot["id"]))


def test_legacy_import_copies_without_modifying_sources(project):
    _, original, refs = corpus_fixture(project)
    store = ArtifactStore(project)
    sources = {}
    for ref in refs:
        if not ref["id"]:
            continue
        base = store.payload(ref)
        for source in base.rglob("*"):
            if source.is_file():
                destination = project / "data" / ("raw" if ref["stage"] == "harvest" else "processed") / source.relative_to(base)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                sources[destination] = file_hash(destination)
    spec = importlib.util.spec_from_file_location("import_corpus_cli", ROOT / "scripts/import_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported = module.import_corpus(project, sys.executable, defaults() | dict(collection_ids=[13559],georef_mode="off",embed_mode="off"))
    bundle = store.bundle(imported,"corpus")
    assert bundle["collection_ids"] == [13559]
    for ref in bundle["artifacts"]:
        if ref["id"]:
            stats = store.load(ref)["stats"]
            assert stats["origin"] == "legacy_import" and stats["parameters_verified"] is False
    assert all(file_hash(path) == checksum for path,checksum in sources.items())
    store.bundle(original,"corpus")


def test_semantic_study_scope_is_explicit():
    from analysis.semantic_analysis import prepare_semantic_corpus
    frame = pd.DataFrame([dict(point_id=str(cid),collection_id=cid,seed_url=f"https://site{cid}.org",
        text="Enough words about the pandemic and public health today",language="en",publish_date="2020-03-01")
        for cid in [4887,13559,13529]])
    data,_,_ = prepare_semantic_corpus(frame,np.eye(3),CredibilityRegistry(),collection_ids=[4887])
    assert data.collection_id.tolist() == [4887]


def test_reviewed_labels_are_bound_to_a_reused_graph(project):
    _,base,refs = corpus_fixture(project)
    store=ArtifactStore(project)
    core=dict(collection_ids=[13559],base_snapshot_id=base["id"],georef_mode="off",embed_mode="build",
              harvest_mode="reuse",normalize_mode="reuse",dedupe_mode="reuse",credibility_mode="reuse")
    corpus_plan=prepare(project,sys.executable,"corpus","test","with-embedding",core)
    plan=load_plan(corpus_plan)
    config=stage_config("embed",plan["parameters"],plan["resources"])
    embedding,_=store.build("embed",None,config,[lookup(store.bundle(base),"dedupe",13559)],plan["implementation"],
        lambda out:fixture_outputs(out,"embed",None,config))
    snapshot=publish(corpus_plan,[ref for ref in refs if ref["stage"] != "embed"]+[embedding])
    with pytest.raises(ValueError,match="require a reused topic graph"):
        prepare(project,sys.executable,"analysis","test-analysis","bad-labels",dict(snapshot_id=snapshot["id"],
            topic_mode="build",animation_mode="build",animation_labels={"old-id":"Unrelated label"}))
    topic_plan=prepare(project,sys.executable,"analysis","test-analysis","graph",dict(snapshot_id=snapshot["id"],
        semantic_mode="off",topic_mode="build",animation_mode="off"))
    plan=load_plan(topic_plan)
    config=stage_config("topic",plan["parameters"],plan["resources"])
    node="topic:2020-03-01:1"
    def graph(output):
        fixture_outputs(output,"topic",None,config)
        (output/"topics/nodes.jsonl").write_text(json.dumps(dict(node_id=node))+"\n")
        return {}
    topic,_=store.build("topic",None,config,[embedding,plan["resources"]["registry"]],plan["implementation"],graph)
    analysis=publish(topic_plan,[dict(id=None,stage="semantic",collection_id=None,mode="off"),topic,
                                 dict(id=None,stage="animation",collection_id=None,mode="off")])
    settings=dict(snapshot_id=snapshot["id"],base_analysis_id=analysis["id"],semantic_mode="off",topic_mode="reuse",
                  animation_mode="build",animation_labels={node:"Reviewed name"})
    prepared=prepare(project,sys.executable,"analysis","test-analysis","reviewed",settings)
    assert load_plan(prepared)["resources"]["labels"] == {node:"Reviewed name"}
    with pytest.raises(ValueError,match="Unknown topic IDs"):
        prepare(project,sys.executable,"analysis","test-analysis","unknown",settings|dict(animation_labels={"old-id":"Incorrect"}))


def test_embedding_builds_exclusive_collections_and_freezes_exports(tmp_path, monkeypatch):
    import embed.embed as module
    import qdrant_client
    from pipeline.worker import embed_snapshot
    collections = {}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def collection_exists(self,name): return name in collections
        def count(self,collection_name,**kwargs):
            points = collections.get(collection_name,{})
            count = sum(point.payload.get("dedupe_version") != 2 for point in points.values()) if "count_filter" in kwargs else len(points)
            return SimpleNamespace(count=count)
        def upsert(self,collection_name,points,**kwargs):
            collections[collection_name].update({point.id:point for point in points})
        def scroll(self,collection_name,**kwargs): return list(collections[collection_name].values()),None
        def delete_collection(self,name): del collections[name]
    class Handshake:
        def __init__(self,collection_name,**kwargs):
            collections.setdefault(collection_name,{})
            self.collection_name=collection_name
            self.client=Client()
            self.embedding_model=SimpleNamespace(embed=lambda text:[1.,float(len(text))])
    class Chunker:
        def chunk(self,text): return [SimpleNamespace(text=text,start_index=0,end_index=len(text),token_count=2)]
    monkeypatch.setattr(qdrant_client,"QdrantClient",Client)
    monkeypatch.setattr(module,"QdrantHandshake",Handshake)
    monkeypatch.setattr(module,"SemanticChunker",Chunker)
    store=ArtifactStore(tmp_path)
    config=dict(qdrant_prefix="test",qdrant_url="unused",embed_model="fixture",embed_batch_size=1)
    names=[]
    for index,text in enumerate(["original report","changed report","original report"]):
        row=dict(collection_id=13559,seed_url="https://example.org",capture_url="https://example.org/a",
            capture_timestamp="20200301120000",publish_date="2020-03-01",language="en",fetch_status="ok",text=text)
        deduped,_=store.build("dedupe",13559,{},[],"fixture",lambda out:fixture_outputs(out,"dedupe",13559,{},[row]),nonce=index)
        out=tmp_path/f"export-{index}"
        out.mkdir()
        result=embed_snapshot(dict(root=str(tmp_path),output=str(out),config=config,artifact_key="a"*32,inputs=[deduped]),[store.payload(deduped)])
        names.append(result["collection"])
        assert result["chunks"] == 1 and np.load(out/"input/embeddings.npy").shape == (1,2)
        assert pd.read_json(out/"input/metadata.jsonl",lines=True).text.tolist() == [text]
    assert names[0] != names[1] and all(len(collections[name]) == 1 for name in names)
    assert set(collections[names[0]]) == set(collections[names[2]])
    monkeypatch.setattr("pipeline.worker.uuid.uuid4",lambda:SimpleNamespace(hex="b"*32))
    collision="test_"+"a"*16+"_"+"b"*12
    collections[collision]={"sentinel":"previous"}
    out=tmp_path/"collision"
    out.mkdir()
    with pytest.raises(ValueError,match="existing Qdrant"):
        embed_snapshot(dict(root=str(tmp_path),output=str(out),config=config,artifact_key="a"*32,inputs=[deduped]),[store.payload(deduped)])
    assert collections[collision] == {"sentinel":"previous"}


@pytest.mark.skipif(not shutil.which("uv"), reason="Requires uv")
def test_cpu_container_export_is_frozen(tmp_path):
    for name in ("pyproject.toml","uv.lock"):
        shutil.copyfile(ROOT/name,tmp_path/name)
    destination=tmp_path/"requirements.txt"
    subprocess.run([sys.executable,str(ROOT/"airflow/export_requirements.py"),str(tmp_path),str(destination)],
        env=dict(os.environ,UV_CACHE_DIR=str(tmp_path/"cache"),UV_OFFLINE="1"),check=True)
    text=destination.read_text()
    assert "torch==" in text and "bertopic==" in text
    assert not any(line.startswith(("nvidia-","triton==")) for line in text.splitlines())
