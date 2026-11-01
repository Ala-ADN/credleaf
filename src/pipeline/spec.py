"""Stage contracts and schemas shared by Airflow and the isolated worker."""
from copy import deepcopy
from dataclasses import dataclass
import os

from config import COVID_COLLECTIONS, PHASES_BY_NAME


@dataclass(frozen=True)
class StageSpec:
    name: str
    scope: str
    dag: str
    pool: str = "default_pool"


STAGES = {s.name: s for s in (
    StageSpec("harvest", "collection", "corpus", "credleaf_archive"),
    StageSpec("normalize", "collection", "corpus", "credleaf_archive"),
    StageSpec("dedupe", "collection", "corpus"),
    StageSpec("credibility", "global", "corpus", "credleaf_archive"),
    StageSpec("georef", "collection", "corpus", "credleaf_models"),
    StageSpec("embed", "global", "corpus", "credleaf_models"),
    StageSpec("semantic", "global", "analysis", "credleaf_models"),
    StageSpec("topic", "global", "analysis", "credleaf_models"),
    StageSpec("animation", "global", "analysis"),
)}


def knob(default, section, description, **schema):
    return dict(default=default, section=section, description=description, **schema)


LIMIT = dict(type=["integer", "null"], minimum=1)
IDENTIFIER = dict(type=["string", "null"], pattern=r"^[a-z_]+-[0-9a-f]{32}$")
CORPUS_SETTINGS = {
    "collection_ids": knob([13559, 13529], "Scope", "Collections in the published snapshot.", type="array",
        items=dict(type="integer", enum=list(COVID_COLLECTIONS)), minItems=1, uniqueItems=True),
    "base_snapshot_id": knob(None, "Scope", "Exact published snapshot to reuse stages from.", **IDENTIFIER),
    "capture_phase": knob("all", "Harvest", "Capture windows included in this snapshot; no implicit merge with live files.",
        type="string", enum=["all", *PHASES_BY_NAME]),
    "qdrant_url": knob(os.getenv("CREDLEAF_QDRANT_URL", "http://localhost:6333"), "Embed", "Qdrant endpoint.", type="string", pattern=r"^https?://\S+$"),
    "qdrant_prefix": knob("credleaf", "Embed", "Prefix for exclusive versioned collections; no existing collection is overwritten.", type="string", pattern=r"^[A-Za-z0-9_-]+$", maxLength=80),
    "harvest_max_seeds": knob(None, "Harvest", "Seed cap per collection.", **LIMIT),
    "harvest_max_captures_per_seed": knob(None, "Harvest", "Capture cap per seed per window.", **LIMIT),
    "harvest_max_captures_per_url": knob(1, "Harvest", "Canonical URL cap; 0 keeps changed revisions.", type="integer", minimum=0),
    "harvest_match_type": knob("host", "Harvest", "CDX URL scope.", type="string", enum=["exact", "prefix", "host", "domain"]),
    "harvest_active_only": knob(False, "Harvest", "Only active seeds.", type="boolean"),
    "harvest_timeout": knob(60., "Harvest", "HTTP request timeout in seconds.", type="number", exclusiveMinimum=0),
    "harvest_max_seed_error_fraction": knob(0., "Harvest", "Maximum tolerated fraction of failed seeds before rejecting the artifact.", type="number", minimum=0, maximum=1),
    "normalize_max_docs": knob(None, "Normalize", "Document cap per collection, isolated to this snapshot.", **LIMIT),
    "normalize_use_cache": knob(True, "Normalize", "Use shared HTML cache.", type="boolean"),
    "normalize_timeout": knob(60., "Normalize", "HTTP timeout in seconds.", type="number", exclusiveMinimum=0),
    "normalize_min_ok": knob(1, "Normalize", "Minimum successfully extracted documents per collection.", type="integer", minimum=0),
    "normalize_max_error_fraction": knob(0., "Normalize", "Maximum fraction of fetch/extraction failures.", type="number", minimum=0, maximum=1),
    "dedupe_max_docs": knob(None, "Dedupe", "Input cap per collection.", **LIMIT),
    "dedupe_threshold": knob(.9, "Dedupe", "Near-duplicate Jaccard threshold.", type="number", exclusiveMinimum=0, maximum=1),
    "credibility_max_seeds": knob(None, "Credibility", "Seed cap per registry collection.", **LIMIT),
    "credibility_timeout": knob(60., "Credibility", "HTTP timeout in seconds.", type="number", exclusiveMinimum=0),
    "credibility_min_seed_domains": knob(1, "Credibility", "Minimum harvested low/mixed domains in the registry.", type="integer", minimum=0),
    "embed_model": knob("BAAI/bge-m3", "Embed", "Encoder ID or immutable local model path; recorded in provenance.", type="string", minLength=1),
    "embed_batch_size": knob(256, "Embed", "Chunks per Qdrant batch.", type="integer", minimum=1),
    "georef_max_docs": knob(None, "Georeference", "Document cap per collection.", **LIMIT),
    "georef_model": knob("en_core_web_trf", "Georeference", "Installed spaCy model.", type="string", minLength=1),
    "georef_device": knob("cpu", "Georeference", "Model device.", type="string", enum=["cpu", "cuda"]),
    "georef_max_error_fraction": knob(0., "Georeference", "Maximum fraction of NER failures.", type="number", minimum=0, maximum=1),
}
ANALYSIS_SETTINGS = {
    "snapshot_id": knob(None, "Scope", "Required published corpus snapshot ID.", **IDENTIFIER),
    "base_analysis_id": knob(None, "Scope", "Published analysis whose stages may be reused.", **IDENTIFIER),
    "analysis_collection_ids": knob(None, "Study", "Subset of snapshot collections; null uses all of them.", type=["array", "null"],
        items=dict(type="integer", enum=list(COVID_COLLECTIONS)), minItems=1, uniqueItems=True),
    "language": knob("en", "Study", "Language filter shared by metrics and topic evolution.", type="string", minLength=1),
    "min_words": knob(8, "Study", "Minimum words per chunk.", type="integer", minimum=1),
    "semantic_min_topic_size": knob(50, "Semantic", "Minimum global topic size.", type="integer", minimum=6),
    "semantic_min_chunks": knob(20, "Semantic", "Minimum chunks per metric group.", type="integer", minimum=2),
    "semantic_seed": knob(42, "Semantic", "Random seed.", type="integer", minimum=0),
    "topic_min_docs": knob(50, "Topic evolution", "Minimum chunks per monthly window.", type="integer", minimum=6),
    "topic_min_topic_size": knob(20, "Topic evolution", "Minimum monthly topic size.", type="integer", minimum=6),
    "topic_threshold": knob(.45, "Topic evolution", "Temporal edge similarity threshold.", type="number", exclusiveMinimum=0, maximum=1),
    "topic_seed": knob(42, "Topic evolution", "Random seed.", type="integer", minimum=0),
    "animation_max_topics": knob(12, "Animation", "Maximum displayed lineages.", type="integer", minimum=1),
    "animation_threshold": knob(.55, "Animation", "Displayed parent similarity threshold.", type="number", exclusiveMinimum=0, maximum=1),
    "animation_labels": knob({}, "Animation", "Reviewed node-ID/name overrides; requires topic_mode=reuse. Empty uses current topic keywords.",
                             type="object", additionalProperties=dict(type="string", minLength=1)),
}
for name, spec in STAGES.items():
    settings = CORPUS_SETTINGS if spec.dag == "corpus" else ANALYSIS_SETTINGS
    optional = name in {"georef", "embed", "semantic", "topic", "animation"}
    settings[f"{name}_mode"] = knob("off" if name in {"topic", "animation"} else "build", name.title(),
        "Build an immutable artifact, reuse an exact compatible artifact, or omit an optional stage.",
        type="string", enum=["build", "reuse", "off"] if optional else ["build", "reuse"])

for settings in (CORPUS_SETTINGS, ANALYSIS_SETTINGS):
    settings["eta_stage_seconds"] = knob({}, "Timing", "Optional initial whole-task durations in seconds. Live progress takes over; estimates exclude queue time/retries.",
        type="object", additionalProperties=False, properties={name: dict(type="number", exclusiveMinimum=0)
            for name in (*STAGES, "prepare", "publish")})


def settings_for(kind):
    return CORPUS_SETTINGS if kind == "corpus" else ANALYSIS_SETTINGS


def defaults(kind="corpus"):
    return {name: deepcopy(spec["default"]) for name, spec in settings_for(kind).items()}


def schema(kind="corpus"):
    return dict(type="object", additionalProperties=False, required=list(settings_for(kind)), properties={
        name: {key: value for key, value in spec.items() if key not in {"default", "description", "section"}}
        for name, spec in settings_for(kind).items()})


def stage_config(stage, params, resources):
    config = {key: value for key, value in params.items() if key.startswith(stage + "_") and not key.endswith("_mode")}
    if stage in {"harvest", "normalize", "dedupe", "georef", "embed"}:
        config["corpus_layout"] = "collection-v1"
    if stage == "harvest":
        config["capture_phase"] = params["capture_phase"]
    if stage == "embed":
        config.update(qdrant_url=params["qdrant_url"], qdrant_prefix=params["qdrant_prefix"])
    if stage == "georef":
        config["gazetteer_sha256"] = resources["gazetteer_sha256"]
    if stage in {"semantic", "topic"}:
        config.update(collection_ids=params["analysis_collection_ids"], language=params["language"], min_words=params["min_words"])
    if stage == "animation":
        config["labels_sha256"] = resources["labels_sha256"]
    return config


def required_files(stage, cid, config):
    if stage == "harvest":
        return [f"captures/{cid}/captures.jsonl"]
    if stage in {"normalize", "dedupe"}:
        folder = "normalized" if stage == "normalize" else "deduped"
        summary = "collection_summary.json" if stage == "normalize" else "dedupe_summary.json"
        files = [f"{folder}/{cid}/documents.jsonl", f"{folder}/{cid}/{summary}"]
        return files + ([f"{folder}/{cid}/audit/duplicates.jsonl"] if stage == "dedupe" else [])
    if stage == "credibility":
        return ["credibility.json"]
    if stage == "georef":
        return [f"georef/{cid}/documents.jsonl"]
    if stage == "embed":
        return ["input/metadata.jsonl", "input/embeddings.npy", "qdrant.json"]
    if stage == "semantic":
        return ["metadata.json", "topics.json", "chunks.jsonl", "normalized_embeddings.npy", "monthly.csv",
            "phases.csv", "neighbours.csv", "neighbour_summary.csv", "semantic_comparison.gif",
            "semantic_comparison.png", "semantic_comparison.csv", "semantic_comparison.json"]
    if stage == "topic":
        return ["chunk_provenance.jsonl", *[f"topics/{name}" for name in ("nodes.jsonl", "edges.jsonl", "membership.jsonl",
            "coverage.jsonl", "word_distributions.npz", "embedding_centroids.npy", "vocabulary.json", "metadata.json")]]
    if stage == "animation":
        return [f"semantic_evolution.{extension}" for extension in ("gif", "png", "csv", "json")]
    if stage == "publish":
        return ["snapshot.json"]
    raise ValueError(f"Unknown stage {stage}")


def legacy_required_files(stage, cid, config):
    """Verify historical snapshots without rewriting their signed manifests."""
    phases = [*PHASES_BY_NAME, "outside_study", "undated"]
    if stage == "harvest":
        names = list(PHASES_BY_NAME) if config["capture_phase"] == "all" else [config["capture_phase"]]
        return [f"captures/{cid}/{name}.jsonl" for name in names]
    if stage in {"normalize", "dedupe"}:
        folder = "normalized" if stage == "normalize" else "deduped"
        summary = "phase_summary.json" if stage == "normalize" else "dedupe_summary.json"
        return ([f"{folder}/{cid}/{name}.jsonl" for name in phases] + [f"{folder}/{cid}/{summary}"]
                + ([f"{folder}/{cid}/audit/duplicates.jsonl"] if stage == "dedupe" else []))
    if stage == "georef":
        return [f"georef/{cid}/{name}.jsonl" for name in PHASES_BY_NAME]
    return required_files(stage, cid, config)
