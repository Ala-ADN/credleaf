#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CREDLEAF_PYTHON:-$project_root/.venv/bin/python}" -m credleaf_kaggle --root "$project_root" "$@"
