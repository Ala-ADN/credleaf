"""Keep routine Airflow polling out of container logs without hiding task logs."""

from copy import deepcopy

from airflow.config_templates.airflow_local_settings import (
    DEFAULT_LOGGING_CONFIG,
    DEFAULT_REMOTE_CONN_ID,
    REMOTE_TASK_LOG,
)

LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)

# These components emit an INFO line for every idle poll or DAG serialization.
# Preserve their warnings/errors and the default INFO task log configuration.
for name in (
    "airflow.dag_processing.manager",
    "airflow.serialization.serialized_objects",
    "airflow.models.dag",
    "airflow.jobs.triggerer_job_runner",
):
    LOGGING_CONFIG["loggers"][name] = {"level": "WARNING"}
