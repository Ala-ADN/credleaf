"""Build immutable research corpora, then analyze explicit published snapshots."""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys

from airflow.sdk import DAG, Param, get_current_context, task, task_group

ROOT = Path(os.environ.get("CREDLEAF_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
from pipeline.spec import CORPUS_SETTINGS, ANALYSIS_SETTINGS, STAGES
from pipeline import airflow_support as support
from pipeline import runner

COMMON = dict(schedule=None, catchup=False, start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
              max_active_runs=1, max_active_tasks=4, tags=["credleaf", "snapshots"],
              default_args=dict(retries=1, retry_delay=timedelta(minutes=2), execution_timeout=timedelta(hours=24)),
              on_success_callback=support.finalize_success, on_failure_callback=support.finalize_failure)


@task(multiple_outputs=True, retries=0)
def prepare_corpus():
    return support.prepare("corpus", get_current_context())


@task(multiple_outputs=True, retries=0)
def prepare_analysis():
    return support.prepare("analysis", get_current_context())


@task
def selected_collections(plan_ref):
    return plan_ref["collection_ids"]


@task(multiple_outputs=False)
def execute(plan_ref, stage, inputs, collection_id=None):
    return runner.run_stage(plan_ref, stage, list(inputs), collection_id,
                            attempt=get_current_context()["ti"].try_number)


@task_group
def collection_flow(collection_id, plan_ref, registry_ref):
    raw = execute.override(task_id="harvest", pool=STAGES["harvest"].pool)(plan_ref, "harvest", [], collection_id)
    normalized = execute.override(task_id="normalize", pool=STAGES["normalize"].pool,
                                  execution_timeout=timedelta(days=14))(plan_ref, "normalize", [raw], collection_id)
    deduped = execute.override(task_id="dedupe", execution_timeout=timedelta(days=7))(plan_ref, "dedupe", [normalized], collection_id)
    georef = execute.override(task_id="georef", pool=STAGES["georef"].pool)(plan_ref, "georef", [deduped, registry_ref], collection_id)
    return dict(harvest=raw, normalize=normalized, dedupe=deduped, georef=georef)


@task(multiple_outputs=False)
def publish_corpus(plan_ref, collections, registry, embedding):
    refs = [ref for values in collections.values() for ref in values]
    return runner.publish(plan_ref, [*refs, registry, embedding])


@task(multiple_outputs=False)
def analyze(plan_ref, stage, topic_ref=None):
    plan = runner.load_plan(plan_ref)
    inputs = [topic_ref] if stage == "animation" else [plan["resources"]["embedding"], plan["resources"]["registry"]]
    return runner.run_stage(plan_ref, stage, inputs, attempt=get_current_context()["ti"].try_number)


@task(multiple_outputs=False)
def publish_analysis(plan_ref, artifacts):
    return runner.publish(plan_ref, artifacts)


@task(trigger_rule="all_done", retries=0)
def overview():
    return support.overview(get_current_context())


@task(trigger_rule="one_failed", retries=0)
def failure_guard():
    raise RuntimeError("An Airflow task failed; the scheduler will finalize the run overview")


def attach_reporting(dag):
    final = overview()
    guard = failure_guard()
    for operator in dag.tasks:
        if operator not in (final.operator, guard.operator):
            operator >> final
        if operator is not guard.operator:
            operator >> guard


with DAG(
    dag_id="credleaf_build_corpus", description="Build, validate and publish an immutable corpus snapshot",
    params={name: Param(**spec) for name, spec in CORPUS_SETTINGS.items()},
    doc_md="""
## Build corpus

Each collection has an independent harvest → normalize → dedupe → geography chain.
Archive tasks share `credleaf_archive`; transformer tasks share the single-slot
`credleaf_models` pool. The final publication requires every requested stage to succeed.

Stages build immutable artifacts, reuse compatible artifacts from `base_snapshot_id`,
or turn off optional geography/embedding. Reuse checks file hashes, source inputs,
configuration, code and Python package versions before archive/model work starts.
Changing a capture window creates a new scoped snapshot; live files are never merged.

Embedding uses an exclusive new Qdrant collection and saves a verified offline
metadata/vector export. A failed build never replaces a previous published corpus.
The `publish_corpus` return value contains the snapshot ID for the analysis DAG.

Every stage logs an ETA heartbeat and its output inventory, and saves a structured
report. Uncalibrated tasks explicitly say so. `eta_stage_seconds` can seed a
whole-task estimate; live harvest/extraction progress supersedes it. `overview`
is provisional until the scheduler callback reconciles actual task states.
See `airflow/README.md` for setup, importing existing data and reuse examples.
""", **COMMON,
) as build_corpus:
    corpus_plan = prepare_corpus()
    registry = execute.override(task_id="credibility", pool=STAGES["credibility"].pool)(corpus_plan, "credibility", [])
    collection_ids = selected_collections(corpus_plan)
    collections = collection_flow.partial(plan_ref=corpus_plan, registry_ref=registry).expand(collection_id=collection_ids)
    embedding = execute.override(task_id="embed", pool=STAGES["embed"].pool,
                                 execution_timeout=timedelta(days=14))(corpus_plan, "embed", collections["dedupe"])
    publication = publish_corpus(corpus_plan, collections, registry, embedding)
    attach_reporting(build_corpus)


with DAG(
    dag_id="credleaf_analyze_corpus", description="Analyze a pinned corpus snapshot without archive access",
    params={name: Param(**spec) for name, spec in ANALYSIS_SETTINGS.items()},
    doc_md="""
## Analyze a snapshot

Set `snapshot_id` to an ID returned by `credleaf_build_corpus` or the legacy importer.
The selected snapshot must include an embedding export. Analysis reads that export
and its pinned registry, without contacting Archive-It or Qdrant.

`analysis_collection_ids` is a validated subset of the snapshot; null selects all.
Metrics and monthly topic evolution share this study scope, language and word filter.
Enable `topic_mode=build` and `animation_mode=build` to generate the graph and GIF
without a notebook. Sparse windows stay explicit; an empty graph cannot render a tree.
Use `base_analysis_id` with reuse modes to render new display settings from an existing graph.

Outputs are immutable and checksummed. The publication ID identifies a complete
analysis; partial failures leave prior results intact. Scheduler callbacks finalize
the JSON/Markdown overview from actual Airflow task states.
""", **COMMON,
) as analyze_corpus:
    analysis_plan = prepare_analysis()
    semantic = analyze.override(task_id="semantic", pool=STAGES["semantic"].pool)(analysis_plan, "semantic")
    topic = analyze.override(task_id="topic", pool=STAGES["topic"].pool)(analysis_plan, "topic")
    animation = analyze.override(task_id="animation")(analysis_plan, "animation", topic)
    analysis_publication = publish_analysis(analysis_plan, [semantic, topic, animation])
    attach_reporting(analyze_corpus)
