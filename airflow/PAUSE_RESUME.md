# Safe pause and resume

Open the run’s **Pause / Resume** link, then use **Pause safely** or **Resume**. These controls
are different from Airflow's ordinary pause toggle: they send a cooperative
request to workers as well as pausing downstream scheduling.

Normalization finishes the current document, appends its result and counters to
an input-bound journal, flushes it to disk, and acknowledges the pause before
fetching another document. The UI distinguishes **Pausing** from **Paused**.
No ETA countdown runs while paused, and paused time is excluded from throughput.
Resume continues at the next uncommitted document.

The journal is under `data/pipeline/checkpoints`, separate from temporary output
directories. It binds the input order, configuration and implementation, checks
row hashes and provenance, and allows only one writer. After interruption, an
incomplete final write is removed; complete corrupt rows cause an error instead
of being skipped. Completed scientific artifacts remain immutable.

If a paused normalization worker is interrupted or the container restarts,
**Resume** can requeue unfinished tasks from the saved document checkpoint.
Successful tasks are retained. Resume refuses changed scientific code: those
checkpoints require an explicit verified handoff to a new plan.

Embedding pauses between documents after pending writes finish, and export
pauses between pages. These pauses resume in place while the worker stays alive.
Embedding itself does not yet have durable document checkpoints: stopping its
worker restarts the embedding stage. Other computations pause at stage boundaries;
an active topic fit or dedupe may finish before the flow becomes idle. Paused
workers retain their memory and Airflow task timeout continues to apply. Durable
normalization checkpoints still survive a timeout; Resume can requeue that work.

CLI equivalent inside the running Airflow container:

```bash
docker compose exec -T -e PYTHONPATH=/opt/airflow/plugins:/opt/credleaf/src airflow \
  python -m credleaf_ui.controls pause --run-id full-checkpointed-20261102

docker compose exec -T -e PYTHONPATH=/opt/airflow/plugins:/opt/credleaf/src airflow \
  python -m credleaf_ui.controls resume --run-id full-checkpointed-20261102
```

The original `full-normalize-embed-20261102` worker did not support checkpoints.
Its 13,630 complete document rows were frozen, copied to persistent storage and
validated against the original capture prefix before stopping it. A new frozen
plan, `full-checkpointed-20261102`, adopts that verified prefix. The old run appears
failed/stopped in Airflow; it has been superseded by the resumable run. Original
handoff checksums and source implementation are retained in checkpoint metadata
and in the final normalization artifact's statistics.

For adopted rows, fetch/cache counters describe work after adoption; the adopted
count is reported separately. Extraction outcomes and quality-gate counts cover
the entire corpus, including adopted rows.
