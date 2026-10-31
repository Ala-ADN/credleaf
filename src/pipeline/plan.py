"""Validate complete run plans before any archive requests or model work."""
import json
import os
from pathlib import Path
import subprocess
import shutil

from jsonschema import validate

from pipeline.artifacts import ArtifactStore, digest, file_hash, implementation, lookup, write_json
from pipeline.spec import defaults, schema, stage_config


def run_directory(root, dag_id, run_id):
    return Path(root) / "data/pipeline/runs" / digest([dag_id, run_id])[:24]


def runtime_identity(root, python):
    code = "import json,sys,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"
    result = subprocess.run([str(python), "-c", code], text=True, capture_output=True, check=True,
                            env=dict(os.environ, PYTHONPATH=str(Path(root) / "src")))
    return json.loads(result.stdout)


def prepare(root, python, kind, dag_id, run_id, values):
    root = Path(root).resolve()
    params = defaults(kind) | values
    validate(params, schema(kind))
    store = ArtifactStore(root)
    directory = run_directory(root, dag_id, run_id)
    code = dict(source=implementation(root), runtime=runtime_identity(root, python))
    if (directory / "plan.json").exists():
        previous = json.loads((directory / "plan.json").read_text())
        original = dict(previous["requested_parameters"])
        if original != params or code != previous["implementation"]:
            raise ValueError("Run configuration/code/runtime changed; trigger a new DAG run")
        return dict(path=str(directory / "plan.json"), sha256=file_hash(directory / "plan.json"),
                    collection_ids=previous["parameters"].get("collection_ids", []))

    resources = {}
    base = None
    if kind == "corpus":
        params["collection_ids"] = sorted(params["collection_ids"])
        if params["base_snapshot_id"]:
            base = store.bundle(params["base_snapshot_id"], "corpus")
        if params["georef_mode"] == "build":
            path = root / "data/cache/geonames/gazetteer.pkl"
            if not path.is_file():
                raise ValueError("Prepare the GeoNames gazetteer before building geography, or set georef_mode=off")
            resources["gazetteer_sha256"] = file_hash(path)
        elif params["georef_mode"] == "reuse":
            if not base:
                raise ValueError("Reuse requires base_snapshot_id")
            resources["gazetteer_sha256"] = store.load(lookup(base, "georef", params["collection_ids"][0]))["config"]["gazetteer_sha256"]
        else:
            resources["gazetteer_sha256"] = None
    else:
        if not params["snapshot_id"]:
            raise ValueError("snapshot_id is required; select a published corpus")
        corpus = store.bundle(params["snapshot_id"], "corpus")
        resources["corpus"] = corpus
        resources["embedding"] = lookup(corpus, "embed")
        resources["registry"] = lookup(corpus, "credibility")
        params["analysis_collection_ids"] = params["analysis_collection_ids"] or corpus["collection_ids"]
        params["analysis_collection_ids"] = sorted(params["analysis_collection_ids"])
        if not set(params["analysis_collection_ids"]) <= set(corpus["collection_ids"]):
            raise ValueError("Analysis collections must be a subset of the selected snapshot")
        if all(params[f"{stage}_mode"] == "off" for stage in ("semantic", "topic", "animation")):
            raise ValueError("Select at least one analysis stage")
        if params["animation_mode"] != "off" and params["topic_mode"] == "off":
            raise ValueError("Animation requires a built or reused topic artifact")
        if params["base_analysis_id"]:
            base = store.bundle(params["base_analysis_id"], "analysis")
        labels = params["animation_labels"]
        if labels and (params["topic_mode"] != "reuse" or params["animation_mode"] == "off"):
            raise ValueError("Reviewed animation labels require a reused topic graph and enabled animation")
        resources["labels_sha256"] = digest(labels)
        resources["labels"] = labels

    # Reject contradictory reuse choices before spending time building upstream.
    refs = {}
    scopes = params["collection_ids"] if kind == "corpus" else [None]
    order = ("credibility", "harvest", "normalize", "dedupe", "georef", "embed") if kind == "corpus" else ("semantic", "topic", "animation")
    for stage in order:
        ids = scopes if stage in {"harvest", "normalize", "dedupe", "georef"} else [None]
        for cid in ids:
            mode = params[f"{stage}_mode"]
            if mode != "reuse":
                refs[(stage, cid)] = None
                continue
            if base is None:
                raise ValueError("Reuse requires base_snapshot_id or base_analysis_id")
            dependencies = stage_dependencies(kind, stage, cid, params, resources, refs)
            if any(ref is None for ref in dependencies):
                raise ValueError(f"Cannot reuse {stage} while rebuilding its inputs")
            ref = lookup(base, stage, cid)
            refs[(stage, cid)] = store.reuse(ref, stage, cid, stage_config(stage, params, resources), dependencies, code)
    if kind == "analysis" and resources["labels"]:
        nodes = store.payload(refs[("topic", None)]) / "topics/nodes.jsonl"
        identifiers = {json.loads(line)["node_id"] for line in nodes.read_text().splitlines() if line.strip()}
        if unknown := set(resources["labels"]) - identifiers:
            raise ValueError(f"Unknown topic IDs in animation labels: {sorted(unknown)}")
    plan = dict(format=1, kind=kind, dag_id=dag_id, run_id=run_id, root=str(root), python=str(python),
                requested_parameters=defaults(kind) | values, parameters=params, resources=resources,
                implementation=code, base=base)
    code_root = directory / "code"
    files = [root / "pyproject.toml", root / "uv.lock"]
    files += [path for folder in ("src", "scripts", "dags") for path in (root / folder).rglob("*.py")]
    for source in files:
        if source.is_file():
            destination = code_root / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    if implementation(code_root) != code["source"]:
        raise ValueError("Code changed while freezing the run")
    plan["code_root"] = str(code_root)
    write_json(directory / "plan.json", plan)
    return dict(path=str(directory / "plan.json"), sha256=file_hash(directory / "plan.json"),
                collection_ids=params.get("collection_ids", []))


def stage_dependencies(kind, stage, cid, params, resources, refs):
    if stage == "normalize":
        return [refs[("harvest", cid)]]
    if stage == "dedupe":
        return [refs[("normalize", cid)]]
    if stage == "georef":
        return [refs[("dedupe", cid)], refs[("credibility", None)]]
    if stage == "embed":
        return [refs[("dedupe", value)] for value in params["collection_ids"]]
    if stage in {"semantic", "topic"}:
        return [resources["embedding"], resources["registry"]]
    if stage == "animation":
        return [refs[("topic", None)]]
    return []
