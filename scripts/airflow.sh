#!/usr/bin/env bash
# Invoke any Airflow command with the repository's paths and isolated runtime.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$project_root/data/airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$project_root/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAG_RUN_CONF_OVERRIDES_PARAMS=True
export CREDLEAF_ROOT="$project_root"
export CREDLEAF_PYTHON="${CREDLEAF_PYTHON:-$project_root/.venv/bin/python}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "standalone" ]]; then
  export CREDLEAF_AIRFLOW_BIN="$project_root/.venv-airflow/bin/airflow"
  exec bash "$project_root/airflow/start.sh"
fi
exec "$project_root/.venv-airflow/bin/airflow" "$@"
