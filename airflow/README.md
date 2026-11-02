# Credleaf snapshot pipelines

Two manually triggered Airflow 3.1.5 DAGs replace the original `credleaf_pipeline`:

- **`credleaf_build_corpus`** builds and publishes an immutable corpus, its registry,
  optional geography, and optional versioned embeddings.
- **`credleaf_analyze_corpus`** analyzes one explicit published corpus snapshot.
  It reads the frozen embedding export and registry without archive or Qdrant access.

Each collection has a mapped TaskGroup containing harvest → normalize → dedupe →
georeference. A failure in one collection does not stop another collection's
normalization/deduplication. Publication requires every requested stage to finish.
An archive pool allows two tasks at once; one model pool serializes transformer
and topic-model work across both DAGs. Neither DAG publishes a mutable “latest”.

## Docker setup

From the repository root:

```bash
export AIRFLOW_UID="$(id -u)"
mkdir -p data notebooks/animation_out
docker compose up -d --build

# Once: populate the persistent GeoNames cache before enabling geography.
docker compose run --rm airflow bash -c \
  '$CREDLEAF_PYTHON -m georeferencing.setup_gazetteer'

docker compose logs -f airflow
```

Open http://localhost:8080 and use the standalone credentials printed in the logs.

```
docker compose exec -T airflow python -c 'import json; print(json.load(open("/opt/airflow/simple_auth_manager_passwords.json.generated"))["admin"])'
```

Enable the two DAGs. Trigger the build DAG, then copy the ID from the
`publish_corpus` return-value XCom into the analysis DAG's `snapshot_id` field.
The build's finalized `overview.json` also contains this ID.

The image includes isolated Airflow/project Python environments, the locked
project/dev dependencies, CPU PyTorch, FFmpeg and `en_core_web_trf` 3.8.0. It exports
`uv.lock` without CUDA pins and selects PyTorch's CPU index. The first build
fetches packages/model weights; the first embedding build fetches its encoder.
`georef_device=cpu` is appropriate for this image. GPU use requires a CUDA image
and GPU runtime configuration.

Startup migrates the local Airflow database and creates `credleaf_archive` (two
slots, configurable with `CREDLEAF_ARCHIVE_SLOTS`) and `credleaf_models` (one slot).
The deployment uses standalone Airflow and SQLite for local development. The
healthcheck checks API availability. Distributed deployment needs separately
managed Airflow services/database and storage visible at the same paths to workers.

Standalone Airflow keeps its scheduler, DAG processor, triggerer and API running
even when no DAG is active. This local Compose setup limits Airflow to eight
workers, reparses DAG files at most every two minutes, suppresses routine DAG
processor polling logs, and rotates the container's stdout logs at 10 MB × three
files. Task INFO logs remain available. The API healthcheck runs every minute.
If you do not need the UI or automatic scheduling, `docker compose stop airflow`
stops its idle CPU use; `docker compose up -d airflow` starts it again. A local
ignored `.env` can hold `AIRFLOW_UID` and `CREDLEAF_ARCHIVE_SLOTS` across shells.

| Container path                     | Persistence                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `/opt/airflow`                     | Named `airflow_state` volume: Airflow database, credentials and service logs                |
| `/opt/credleaf/data`               | Host `./data`: legacy data, immutable artifacts, per-run reports/code copies, shared caches |
| `/opt/credleaf/{src,scripts,dags}` | Read-only development mounts; workers execute a frozen copy made when a run is planned      |
| `/opt/credleaf/notebooks`          | Host notebook files and legacy display labels (not automatically applied to new graphs)     |
| `/opt/credleaf-venv`               | Image's isolated project environment; rebuild after dependency changes                      |

`AIRFLOW_UID` matches container writes to your host account. Keep it exported for
`docker compose run`, or add it to your existing `.env`. `AIRFLOW_PORT` changes
the localhost UI port. Compose automatically sets the Qdrant default to
`http://qdrant:6333`. Host Python defaults to `http://localhost:6333`.

```bash
docker compose exec airflow airflow dags list-import-errors
docker compose ps
docker compose down
```

Stopping containers retains state and artifacts. `docker compose down --volumes`
also deletes the named Airflow state volume; host research data remains intact.

## Stage modes and reuse

Every stage uses `*_mode` rather than the old `*_enabled` Boolean:

- **`build`** creates a new artifact in staging, validates it and atomically
  publishes the complete directory. A retry with the same inputs/configuration
  returns the verified completed artifact instead of repeating work.
- **`reuse`** selects the matching stage from `base_snapshot_id` (corpus) or
  `base_analysis_id` (analysis). File hashes, exact upstream references, stage
  settings and implementation/runtime identity must match. Incompatible reuse
  fails during planning, before archive requests or model work.
- **`off`** omits an optional stage. Corpus harvesting, normalization,
  deduplication and registry are required; geography and embedding are optional.
  At least one analysis stage must be selected. Animation requires topics.

Source stages in build mode use a new generation per DAG run, so an intentional
fresh harvest/registry refresh is not served from an earlier run's cache.
Derived stages are keyed by upstream artifact IDs, settings and implementation.
Code and Python package versions are recorded. Workers use a frozen source copy;
code/runtime changes during a run require a new run. Strict reuse binds each
reused stage to the current code/runtime, so upgrades can require rebuilding
artifacts. Existing published corpus snapshots remain available for analysis
under a new analysis implementation without rebuilding their corpus.

To carry verified core files across an adapter/runtime correction without
fetching or extracting them again, run `scripts/import_corpus.py --snapshot-id
publish-...` in the project interpreter. This copies the published core artifacts,
preserves their parameters and records their original references. It publishes
a new snapshot for the current implementation, with geography and embeddings off.

A small corpus build, isolated from existing research files:

```bash
docker compose exec airflow airflow dags trigger credleaf_build_corpus \
  --conf '{"collection_ids":[13559],"capture_phase":"P2_pre_vaccine","harvest_max_seeds":2,"harvest_max_captures_per_seed":5,"normalize_max_docs":5,"credibility_max_seeds":2,"georef_mode":"off","embed_mode":"off"}'
```

That snapshot contains deduped documents but cannot run semantic analysis until
an embedding artifact is built. To embed an existing snapshot without harvesting
or extracting again, substitute its ID below. Stage settings must match the
original snapshot, including any sampling limits/capture window used above:

```json
{
  "base_snapshot_id": "publish-REPLACE_WITH_32_HEX_CHARACTERS",
  "collection_ids": [13559],
  "capture_phase": "P2_pre_vaccine",
  "harvest_max_seeds": 2,
  "harvest_max_captures_per_seed": 5,
  "normalize_max_docs": 5,
  "credibility_max_seeds": 2,
  "harvest_mode": "reuse",
  "normalize_mode": "reuse",
  "dedupe_mode": "reuse",
  "credibility_mode": "reuse",
  "georef_mode": "off",
  "embed_mode": "build"
}
```

## Controls and outputs

All trigger settings use JSON Schema validation; the planner additionally checks
cross-stage dependencies and scope. Document/seed caps affect only the new artifact.

| Stage / scope  | Controls                                                                                                                                                                                | Artifact files and diagnostics                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Corpus scope   | `collection_ids`, `capture_phase`, `base_snapshot_id`                                                                                                                                   | Exactly the selected collections/capture windows; no implicit merge with legacy files                                                                    |
| Harvest        | `harvest_max_seeds`, `harvest_max_captures_per_seed`, `harvest_max_captures_per_url`, `harvest_match_type`, `harvest_active_only`, `harvest_timeout`, `harvest_max_seed_error_fraction` | `captures/{cid}/captures.jsonl`; capture/seed/error counts                                                                                               |
| Normalize      | `normalize_max_docs`, `normalize_use_cache`, `normalize_timeout`, `normalize_min_ok`, `normalize_max_error_fraction`                                                                    | `normalized/{cid}/documents.jsonl`, `collection_summary.json`; event phases, extraction and date-source coverage                                         |
| Dedupe         | `dedupe_max_docs`, `dedupe_threshold`                                                                                                                                                   | `deduped/{cid}/documents.jsonl`, duplicate provenance audit and summary                                                                                  |
| Credibility    | `credibility_max_seeds`, `credibility_timeout`, `credibility_min_seed_domains`                                                                                                          | `credibility.json`; low/mixed/authoritative counts. Registry builds retain all configured registry collections                                           |
| Geography      | `georef_max_docs`, `georef_model`, `georef_device`, `georef_max_error_fraction`                                                                                                         | `georef/{cid}/documents.jsonl`; NER/mention coverage, pinned gazetteer hash. Artifact builds start fresh; retries cannot append into published geography |
| Embedding      | `qdrant_url`, `qdrant_prefix`, `embed_model`, `embed_batch_size`                                                                                                                        | Exclusive Qdrant collection, `qdrant.json`, offline `input/metadata.jsonl` + `input/embeddings.npy`; count/dimension/finite-vector checks                |
| Analysis scope | `snapshot_id`, `base_analysis_id`, `analysis_collection_ids`, `language`, `min_words`                                                                                                   | Collection subset must belong to the snapshot; null selects all snapshot collections, including 4887 when present                                        |
| Semantic       | `semantic_min_topic_size`, `semantic_min_chunks`, `semantic_seed`                                                                                                                       | Global topics/chunk assignments, normalized vectors, monthly/phase metrics, neighbours, GIF/PNG/CSV and metadata                                         |
| Monthly topics | `topic_min_docs`, `topic_min_topic_size`, `topic_threshold`, `topic_seed`                                                                                                               | `chunk_provenance.jsonl`, `topics/{nodes,edges,membership,coverage}.jsonl`, matrices, vocabulary and metadata                                            |
| Animation      | `animation_max_topics`, `animation_threshold`, `animation_labels`                                                                                                                       | `semantic_evolution.{gif,png,csv,json}` plus the pinned display labels                                                                                   |

Semantic analysis defaults to build; monthly topics and their animation default
to off. Set both `topic_mode=build` and `animation_mode=build` for a complete
notebook-free topic-tree export. Sparse windows remain explicit. A corpus without
any eligible topics can publish coverage tables with animation off; it cannot
render a topic tree. Global topic fitting also requires enough retained chunks
for `semantic_min_topic_size`.

New topic graphs use names derived from their own keywords. Legacy notebook labels
are not reused automatically because topic IDs can describe different content in a
new fit. For reviewed names, select a `base_analysis_id`, set `topic_mode=reuse`
and `animation_mode=build`, and pass `animation_labels` as an object mapping IDs
from that graph's `topics/nodes.jsonl` to names. Unknown IDs are rejected. The
rendered metadata distinguishes automatic names from explicit overrides.

Zero seed-error tolerance, zero fetch/extraction-error tolerance and zero NER-error
tolerance are the defaults. These are configurable quality gates, alongside the
minimum successfully extracted document/domain counts. Failed gates prevent
publication. Outside-study and undated records remain in collection files; analysis selects
the study population using document dates. Embedding and geography consume the
collection file. Document caps apply once per collection.

Qdrant names include the artifact key and an attempt-specific suffix. Builds never
upsert into a previously published collection. A normal failed build removes only
its own unpublished collection; a hard crash may leave an unreferenced orphan.
Published collections remain available, while analysis uses the immutable local
vector export and is independent of later Qdrant changes. Encoder IDs and package
versions are recorded; upstream model aliases can change when rebuilding. Use an
immutable local model path when exact weight identity is required.

## Importing existing research data

Existing manual data is left in place. Import a complete legacy corpus without
network calls or model work:

```bash
docker compose run --rm airflow bash -c \
  '$CREDLEAF_PYTHON scripts/import_corpus.py --collection-ids 13559 13529'
```

The importer copies raw, normalized, deduped and registry files into checksummed
artifacts and prints a published snapshot ID. Run `scripts/migrate_corpus.py` first for phase-partitioned data. The importer
requires canonical collection files, normalization/dedupe summaries and the duplicate audit. It excludes
legacy geography and Qdrant collections, whose cache/model history cannot be
certified. Build new embeddings/geography from the imported deduped data using
reuse modes for the core stages.

`--parameters path.json` declares any known original stage settings. Imports are
explicitly marked `origin=legacy_import, parameters_verified=false`; copying data
does not reconstruct its historical extraction/encoder implementation. The old
live paths remain available to existing notebooks. New DAG outputs are stored in
the artifact store and do not automatically replace notebook/README graphics.

## Reports and recovery

### Per-step ETA

Preparation, every data stage, and publication write an `eta` object in their
task report and log an ETA heartbeat every 30 seconds. Harvest and normalization
also send live seed/document progress (at most once per five seconds). Other
steps use comparable successful historical durations, or an optional initial
estimate supplied in the trigger configuration:

```json
{ "eta_stage_seconds": { "harvest": 1800, "normalize": 7200, "dedupe": 300 } }
```

These are **user-supplied example whole-task durations**, not benchmarks. For a
mapped stage, the value applies independently to each collection. Live throughput
takes over after three completed items. An unseen configuration shows
`calibrating` until it has a useful measurement; it does not invent a finish time.
ETAs include elapsed execution/validation, but exclude queue time and retries.
If a baseline expires, the report says `overrunning baseline`; if all items are
processed but verification/export remains, it says `finalizing`, not zero minutes.

Historical samples live under `data/pipeline/timings/`, separate from scientific
artifacts. Failed, cached, reused and off attempts do not train estimates. Samples
are grouped by stage, collection, configuration, installed runtime and host/CPU
identity. Normalization scales by capped capture count; input-byte scaling is
used for georef/embed/publication;
harvest scales by capped seed budget with the same per-seed depth. Uncapped
harvests use whole-run history. Nonlinear dedupe/topic/semantic/animation estimates
require input sizes within 10% of the current size. The displayed 0.5–2× range
(expanded to include matching observations) is a planning range, not a confidence
interval. Network stalls, cache-hit changes, model downloads and different seed
sizes can make it optimistic. Source changes do not automatically erase timing
history; remove the timing directory to recalibrate after an algorithm/hardware
change. Timing hints never alter individual scientific stage configurations.

Read all per-step ETAs while the DAG runs, including tasks waiting to start:

```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_status.py \
  --run-id YOUR_AIRFLOW_RUN_ID --watch
# Or inside the container:
docker compose exec airflow bash -c \
  '$CREDLEAF_PYTHON scripts/pipeline_status.py --run-id YOUR_AIRFLOW_RUN_ID --watch'
```

Pass `--dag-id credleaf_analyze_corpus` for analysis. Waiting tasks say so; queue
waits are not assigned invented start times. Running rows retain their last
update timestamp, so a killed worker's stale report stays distinguishable.
Airflow remains authoritative for scheduler/task state. The finalized overview
includes actual elapsed time alongside each step's output inventory.

See [the overnight runbook](OVERNIGHT.md) for the recommended existing-capture
extraction plan, calibration, ready-to-edit trigger profiles and morning checks.

```text
data/pipeline/
  artifacts/<stage>-<id>/
    manifest.json          # config, input refs, code/runtime, counts and SHA-256
    files/                 # immutable stage payload
  runs/<dag-and-run-key>/
    plan.json              # frozen effective/requested config and pinned inputs
    code/                  # frozen Python sources used by this run's workers
    steps/<stage>__<cid>.json
    *.request.json         # explicit worker inputs
    *.result.json          # machine-readable worker result; no stdout scraping
    *.progress.json        # advisory live seed/document progress
    *.log                  # streamed worker logs
    control/prepare.json   # preparation timing
    overview.json
    overview.md
  timings/<timing-key>/*.json # successful compute samples only
```

Task return-value XComs carry small artifact references, not records/vectors.
Stage logs and reports expose output inventories, quality diagnostics and hashes.
The overview reads actual upstream task states through Airflow 3's Task SDK.
Its status remains provisional until a DAG success/failure callback records the
terminal result. The failure guard preserves failed DAG status even if a worker
fails after publishing its manifest. Callbacks are not invoked by manually marking
DAG/task states in the UI; such a report may remain provisional. Consult Airflow's
[callback behavior](https://airflow.apache.org/docs/apache-airflow/3.1.5/administration-and-deployment/logging-monitoring/callbacks.html).

Failed tasks can be cleared and retried with unchanged plans. Completed artifacts
are verified and reused on retry; unfinished staging directories are never selected
as inputs. Prior published snapshots remain intact. Trigger a new run for changed
configuration/code/runtime. Keep published artifacts and referenced vector exports;
there is no automatic garbage collection or deletion of historical snapshots.

## Local Python alternative and validation

```bash
uv sync --group dev
uv run python -m spacy download en_core_web_trf
uv run python -m georeferencing.setup_gazetteer
docker compose up -d qdrant
uv venv .venv-airflow --python 3.13
uv pip install --python .venv-airflow/bin/python -r airflow/requirements.txt
bash scripts/airflow.sh standalone
```

The wrapper initializes pools and stores Airflow state in `data/airflow/`.
Override `CREDLEAF_PYTHON` or `AIRFLOW_HOME` if needed. Installation follows
[Airflow's constrained installation guide](https://airflow.apache.org/docs/apache-airflow/3.1.5/start.html).

```bash
uv run pytest
PYTHONPATH=src .venv-airflow/bin/python -m unittest discover -s tests -p test_airflow_dag.py
CREDLEAF_RUN_AIRFLOW_INTEGRATION=1 PYTHONPATH=src \
  .venv-airflow/bin/python -m unittest discover -s tests -p test_airflow_runtime.py
docker compose config --quiet
```

The opt-in integration suite uses temporary fixtures to exercise both DAGs,
publication, reuse, failure isolation and post-publication worker failure without
fetching archives, Qdrant or model weights. The ordinary suite also exercises a real
isolated dedupe worker, hash/config rejection and model-free embedding-store checks.
