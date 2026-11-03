# Native Airflow monitoring

Monitoring uses Airflow 3.1.5's existing UI. There is no separate dashboard,
custom theme, JavaScript application or iframe for metrics.

| Information | Where to find it |
| --- | --- |
| Run status, normalization progress, ETA lower bound, input-window counts, disk free | Run **Note** |
| Task ETA, progress, paused/stale status, disk free | Task **Note** |
| Structured details | Task **XCom**: `eta`, `coverage`, `disk` |
| Run-wide structured summary | **prepare_corpus** / **prepare_analysis** XComs |
| Version coverage, successful-document counts, checksums and producer references | **Assets → asset → event metadata** |
| Worker pause/resume | Run's **Pause / Resume** link |

Open the usual DAG Grid/Graph and select the run or task. The existing Notes
control displays a concise summary. XComs provide the complete values. Reload a
terminal run's page if it is not polling; all samples include an observation time.

The observer refreshes active/paused run metrics every 30 seconds. It replaces
only its `[pipeline metrics]` section in a note, preserving user-written text.
Airflow notes have a 1,000-character limit: if there is not enough room, user text
is preserved and full metrics remain available in XComs. Task return values and
scientific artifacts are never changed by monitoring.

## Meaning of the fields

- `eta`: remaining seconds, estimated finish, progress, estimate source, stale
  flag and scheduler state. A paused run has no countdown. Unknown downstream
  costs stay unknown; the run-level value is a lower bound, not a finish forecast.
- `coverage`: capture counts use capture timestamps. Published normalized and
  deduped document counts use valid publication dates, otherwise capture dates.
  Outside-study and undated records remain explicit. Unfinished normalization is
  marked unfinalized, with progress shown separately.
- `disk`: a timestamped sample of the local Airflow/project filesystem, including
  byte counts, free GiB, used percentage and path. It is not remote-worker storage.

Existing Asset events are enriched with coverage from the same immutable version;
this does not emit a new materialization. New events include coverage when first
registered. Definitions without events have not produced data. Version references,
creation timestamps, original event timestamps and artifact hashes are retained.

## Controls and compatibility

Airflow's standard DAG pause toggle controls scheduling; it cannot checkpoint the
current Python task. The remaining small control page has two ordinary HTML forms,
uses the existing Airflow login and DAG edit permission, and enforces same-origin
requests. See [pause/resume behavior](PAUSE_RESUME.md). The old `/credleaf/` URL
redirects to the native run page, preserving bookmarked links.

The plugin, observer and catalog remain outside the frozen scientific code.
`dags/observability` links to `airflow/catalog` for discovery without changing a
paused run's scientific identity. The observer writes notes, its own XCom keys,
Asset metadata and telemetry under `data/pipeline`. Observer logs rotate at 2 MB
with two backups. It does not modify plans, manifests, checkpoints or documents.

After active scientific work has completed, future image deployments use:

```bash
docker compose up -d --build
```

## Airflow documentation

- [UI overview: native run notes and task tabs](https://airflow.apache.org/docs/apache-airflow/3.1.5/ui.html)
- [XComs: small per-task values](https://airflow.apache.org/docs/apache-airflow/3.1.5/core-concepts/xcoms.html)
- [Asset event metadata: counts and date ranges](https://airflow.apache.org/docs/apache-airflow/3.1.5/authoring-and-scheduling/assets.html#attaching-extra-information-to-an-emitting-asset-event)
- [Metrics infrastructure](https://airflow.apache.org/docs/apache-airflow/3.1.5/administration-and-deployment/logging-monitoring/logging-architecture.html)

Airflow has no documented arbitrary ETA column or host-disk widget in this version.
This implementation uses native metadata fields. Metrics systems such as StatsD
and Prometheus are available if infrastructure time-series dashboards become
necessary.
