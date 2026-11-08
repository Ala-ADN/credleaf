# Directory-scoped interactive shortcuts. Keep this outside src/, scripts/ and
# dags/ so shell customization does not change a frozen pipeline's identity.
[[ -n ${ZSH_VERSION-} ]] || return 0

typeset -g _credleaf_shortcuts_root="${${(%):-%N}:A:h:h}"
typeset -gA _credleaf_shortcuts_saved

_credleaf_shortcuts_sync() {
  local current_dir="${PWD:A}"
  local name
  if [[ "$current_dir" == "$_credleaf_shortcuts_root" || "$current_dir" == "$_credleaf_shortcuts_root/"* ]]; then
    [[ ${_credleaf_shortcuts_saved[active]-0} == 1 ]] && return 0
    for name in af ctl; do
      _credleaf_shortcuts_saved["${name}_function_set"]=${+functions[$name]}
      _credleaf_shortcuts_saved["${name}_function"]="${functions[$name]-}"
      _credleaf_shortcuts_saved["${name}_alias_set"]=${+aliases[$name]}
      _credleaf_shortcuts_saved["${name}_alias"]="${aliases[$name]-}"
      unalias "$name" 2>/dev/null || true
    done
    _credleaf_shortcuts_saved[dag_set]=${+DAG}
    _credleaf_shortcuts_saved[dag]="${DAG-}"
    _credleaf_shortcuts_saved[run_set]=${+RUN}
    _credleaf_shortcuts_saved[run]="${RUN-}"
    functions[af]='
      command docker compose --project-directory "$_credleaf_shortcuts_root" exec -T airflow airflow "$@"
    '
    functions[ctl]='
      command docker compose --project-directory "$_credleaf_shortcuts_root" exec -T \
        -e PYTHONPATH=/opt/airflow/plugins:/opt/credleaf/src \
        airflow python -m credleaf_ui.controls "$@"
    '
    typeset -g DAG="${DAG:-credleaf_build_corpus}"
    if [[ ! -v RUN ]]; then
      typeset -g RUN
      RUN="$(command python3 - "$_credleaf_shortcuts_root/data/pipeline/dashboard.json" "$DAG" <<'PY'
import json
import sys
try:
    with open(sys.argv[1]) as stream:
        data = json.load(stream)
    runs = [run for run in data.get("runs", [])
            if run.get("dag_id") == sys.argv[2] and not run.get("superseded_by")]
    active = [run for run in runs if run.get("state") in {"running", "queued"}]
    print((active or runs)[0]["run_id"] if runs else "")
except (OSError, ValueError, KeyError, TypeError):
    print("")
PY
)"
    fi
    _credleaf_shortcuts_saved[active]=1
  elif [[ ${_credleaf_shortcuts_saved[active]-0} == 1 ]]; then
    for name in af ctl; do
      if [[ ${_credleaf_shortcuts_saved["${name}_function_set"]} == 1 ]]; then
        functions[$name]="${_credleaf_shortcuts_saved["${name}_function"]}"
      else
        unfunction "$name" 2>/dev/null || true
      fi
      if [[ ${_credleaf_shortcuts_saved["${name}_alias_set"]} == 1 ]]; then
        aliases[$name]="${_credleaf_shortcuts_saved["${name}_alias"]}"
      fi
    done
    if [[ ${_credleaf_shortcuts_saved[dag_set]} == 1 ]]; then
      typeset -g DAG="${_credleaf_shortcuts_saved[dag]}"
    else
      unset DAG
    fi
    if [[ ${_credleaf_shortcuts_saved[run_set]} == 1 ]]; then
      typeset -g RUN="${_credleaf_shortcuts_saved[run]}"
    else
      unset RUN
    fi
    _credleaf_shortcuts_saved=()
  fi
  return 0
}

autoload -Uz add-zsh-hook
add-zsh-hook -d chpwd _credleaf_shortcuts_sync
add-zsh-hook chpwd _credleaf_shortcuts_sync
_credleaf_shortcuts_sync
