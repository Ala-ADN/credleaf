"""Copy an existing complete legacy corpus into the immutable artifact store."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.artifacts import ArtifactStore, file_hash, write_json
from pipeline.plan import prepare, run_directory
from pipeline.spec import defaults, required_files, stage_config
from pipeline.runner import load_plan, publish, write_overview


def import_corpus(root, python, params):
    run_id = "legacy-import-" + uuid.uuid4().hex
    plan_ref = prepare(root, python, "corpus", "credleaf_import_corpus", run_id, params)
    plan = load_plan(plan_ref)
    store = ArtifactStore(root)
    refs, by_scope = [], {}
    for stage in ("credibility", "harvest", "normalize", "dedupe"):
        for cid in ([None] if stage == "credibility" else plan["parameters"]["collection_ids"]):
            config = stage_config(stage, plan["parameters"], plan["resources"])
            inputs = [] if stage in {"credibility", "harvest"} else [by_scope[("harvest" if stage == "normalize" else "normalize", cid)]]
            files = {}
            for name in required_files(stage, cid, config):
                source = root / "data" / ("raw" if stage == "harvest" else "processed") / name
                if not source.is_file():
                    raise ValueError(f"Legacy import requires {source}; run scripts/migrate_corpus.py first")
                files[name] = source
            hashes = {name: file_hash(path) for name, path in files.items()}
            def copy(output):
                for name, source in files.items():
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    if file_hash(destination) != hashes[name]:
                        raise ValueError("Legacy source changed during import")
                return dict(origin="legacy_import", parameters_verified=False, source_hashes=hashes)
            ref, _ = store.build(stage, cid, config, inputs, plan["implementation"], copy, nonce=hashes)
            refs.append(ref)
            by_scope[(stage, cid)] = ref
            manifest = store.load(ref)
            key = stage if cid is None else f"{stage}__{cid}"
            write_json(run_directory(root, plan["dag_id"], run_id) / "steps" / f"{key}.json",
                       dict(stage=stage, collection_id=cid, status="success", artifact=ref,
                            files=manifest["files"], stats=manifest["stats"]))
    for stage in ("georef", "embed"):
        for cid in (plan["parameters"]["collection_ids"] if stage == "georef" else [None]):
            ref = dict(stage=stage, collection_id=cid, id=None, mode="off")
            refs.append(ref)
            key = stage if cid is None else f"{stage}__{cid}"
            write_json(run_directory(root, plan["dag_id"], run_id) / "steps" / f"{key}.json",
                       dict(stage=stage, collection_id=cid, status="off", artifact=ref))
    result = publish(plan_ref, refs)
    write_overview(root, plan["dag_id"], run_id, {"import": "success"}, "success")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-ids", type=int, nargs="+", default=[13559, 13529])
    parser.add_argument("--parameters", type=Path, help="JSON object declaring known original stage settings")
    args = parser.parse_args()
    params = defaults() | (json.loads(args.parameters.read_text()) if args.parameters else {})
    params.update(collection_ids=args.collection_ids, georef_mode="off", embed_mode="off")
    if any(params[name + "_mode"] != "build" for name in ("harvest", "normalize", "dedupe", "credibility")):
        parser.error("Imported core stages must use build mode")
    print(json.dumps(import_corpus(ROOT, sys.executable, params), indent=2))


if __name__ == "__main__":
    main()
