#!/usr/bin/env bash
set -euo pipefail
airflow_bin="${CREDLEAF_AIRFLOW_BIN:-airflow}"
"$airflow_bin" db migrate
"$airflow_bin" pools set credleaf_archive "${CREDLEAF_ARCHIVE_SLOTS:-2}" "Archive requests and HTML extraction"
"$airflow_bin" pools set credleaf_models 1 "One transformer or topic model at a time"
if [[ -d /opt/airflow/plugins/credleaf_ui ]]; then
  PYTHONPATH="/opt/airflow/plugins:${PYTHONPATH:-}" python -m credleaf_ui.bridge >/dev/null 2>&1 &
fi
exec "$airflow_bin" standalone
