"""Seed a new resumable plan with a verified legacy worker's completed prefix."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingest import load_jsonl
from normalize.checkpoint import DocumentJournal
from normalize.models import NormalizedDoc
from normalize.normalize import NormalizeStats
from pipeline.artifacts import ArtifactStore, digest, file_hash, implementation, lookup, write_json
from pipeline.spec import stage_config


def adopt(plan_path, handoff_dir, collection_id):
    plan = json.loads(Path(plan_path).read_text())
    root, handoff_dir = Path(plan["root"]), Path(handoff_dir)
    if implementation(root) != plan["implementation"]["source"]:
        raise ValueError("Source changed after preparing the destination plan")
    meta = json.loads((handoff_dir / "handoff.json").read_text())
    documents = handoff_dir / "documents.jsonl"
    if file_hash(documents) != meta["documents_sha256"]:
        raise ValueError("Legacy handoff checksum mismatch")
    store = ArtifactStore(root)
    ref = lookup(plan["base"], "harvest", collection_id)
    raw = store.payload(ref) / "captures" / str(collection_id) / "captures.jsonl"
    if file_hash(raw) != meta["capture_sha256"]:
        raise ValueError("Legacy handoff and destination capture files differ")
    config = stage_config("normalize", plan["parameters"], plan["resources"])
    if config != meta["config"]:
        raise ValueError("Normalization settings changed across handoff")
    key = digest([config, [ref], plan["implementation"]])
    directory = root / "data/pipeline/checkpoints" / key / f"normalize__{collection_id}"
    identity = dict(artifact_key=key, config=config, capture_sha256=meta["capture_sha256"])
    records = list(load_jsonl(raw))
    if config["normalize_max_docs"] is not None:
        records = records[:config["normalize_max_docs"]]
    journal = DocumentJournal(directory, identity, records)
    class Discard:
        def write(self, value): pass
    try:
        count, saved, errors, seconds = journal.replay(Discard())
        stats = NormalizeStats()
        if saved:
            for name in stats.__dataclass_fields__:
                if name != "errors" and name in saved:
                    setattr(stats, name, saved[name])
            stats.errors = errors
        counter = 0
        with documents.open() as stream:
            for i, line in enumerate(stream):
                doc = NormalizedDoc.model_validate_json(line)
                record = records[i]
                if (doc.collection_id, doc.seed_id, doc.capture_url, doc.capture_timestamp, doc.digest) != (
                        record.collection_id, record.seed_id, record.capture.url, record.capture.timestamp, record.capture.digest):
                    raise ValueError(f"Legacy document {i} does not match capture")
                counter += 1
                if i < count:
                    continue
                stats.total += 1
                stats.adopted += 1
                attribute = {"ok": "extracted", "empty": "empty", "fetch_failed": "fetch_failed", "extract_failed": "extract_failed"}[doc.fetch_status]
                setattr(stats, attribute, getattr(stats, attribute) + 1)
                if doc.error:
                    stats.errors.append((doc.capture_timestamp, doc.error))
                elapsed = max(0., (doc.fetched_at - datetime.fromisoformat(meta["source_started_at"])).total_seconds())
                journal.append(i, record, doc.model_dump(mode="json"), stats.as_dict(), elapsed)
        if counter != meta["records"] or stats.total != counter:
            raise ValueError("Legacy handoff count mismatch")
        write_json(directory / "adoption.json", dict(source=meta, destination_run_id=plan["run_id"],
            adopted=counter, fetch_accounting="fetched/from_cache counters cover work after adoption only"))
    finally:
        journal.close()
    return dict(checkpoint=str(directory), adopted=counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--collection-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(adopt(args.plan, args.handoff, args.collection_id), indent=2))
