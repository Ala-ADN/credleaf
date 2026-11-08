#!/usr/bin/env bash
set -euo pipefail
airflow_bin="${CREDLEAF_AIRFLOW_BIN:-airflow}"
airflow_python="${CREDLEAF_AIRFLOW_PYTHON:-python}"
plugins_dir="${AIRFLOW__CORE__PLUGINS_FOLDER:-/opt/airflow/plugins}"
export PYTHONPATH="$plugins_dir:${PYTHONPATH:-}"
"$airflow_bin" db migrate
"$airflow_bin" pools set credleaf_archive "${CREDLEAF_ARCHIVE_SLOTS:-2}" "Archive requests and HTML extraction"
"$airflow_bin" pools set credleaf_models 1 "One transformer or topic model at a time"
if [[ -d "$plugins_dir/credleaf_ui" ]]; then
  export CREDLEAF_OBSERVER_ENABLED=1
fi
exec "$airflow_python" -m credleaf_ui.lifecycle -- "$airflow_bin" standalone
