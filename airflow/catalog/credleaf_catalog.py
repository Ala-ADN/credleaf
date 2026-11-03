"""External data catalog; observation is independent of scientific task code."""
from datetime import datetime, timezone

from airflow.sdk import DAG, Asset
from airflow.providers.standard.operators.empty import EmptyOperator

definitions = [(stage, str(cid)) for cid in (13529, 13559, 4887)
               for stage in ("harvest", "normalize", "dedupe", "georef")]
definitions += [(stage, "corpus" if stage in {"credibility", "embed"} else "analysis")
                for stage in ("credibility", "embed", "semantic", "topic", "animation")]
definitions += [("publish", "corpus"), ("publish", "analysis")]

with DAG("credleaf_data_catalog", schedule=None, catchup=False,
         start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), tags=["credleaf", "data"],
         description="Versioned Credleaf outputs and coverage in Assets",
         doc_md="""
## Credleaf data catalog

Open **Assets** for version history, date coverage, checksums and producer links.
Run/task **Notes** summarize progress, ETA and disk headroom. The task **XCom**
tab has `eta`, `coverage` and `disk`; the prepare task has the run-wide summary.
Use the run's **Pause / Resume** link for cooperative worker controls.

This DAG declares observed inputs only. The observer emits Asset events after
verifying completed immutable artifacts. Running this DAG does not fetch data,
normalize documents or publish new versions. Definitions without an event have
not produced data yet. Event timestamps record observation; original creation
times and producer references are retained in event metadata.
""") as catalog:
    EmptyOperator(task_id="view_catalog", inlets=[
        Asset(name=f"Credleaf {stage} · {scope}", uri=f"credleaf://{stage}/{scope}")
        for stage, scope in definitions])
