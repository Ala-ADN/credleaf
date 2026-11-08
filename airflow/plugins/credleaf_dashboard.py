from urllib.parse import quote
from airflow.plugins_manager import AirflowPlugin
from airflow.sdk import BaseOperatorLink
from credleaf_ui.app import app


class CredleafEtaLink(BaseOperatorLink):
    name = "Metrics (XCom)"

    def get_link(self, operator, *, ti_key):
        if not ti_key.dag_id.startswith("credleaf_"):
            return ""
        path = "/dags/" + quote(ti_key.dag_id, safe="") + "/runs/" + quote(ti_key.run_id, safe="")
        path += "/tasks/" + quote(ti_key.task_id, safe="")
        if ti_key.map_index >= 0:
            path += f"/mapped/{ti_key.map_index}"
        return path + "/xcom"


class CredleafDashboardPlugin(AirflowPlugin):
    name = "credleaf_monitoring"
    fastapi_apps = [dict(app=app, url_prefix="/credleaf", name="Workflow controls")]
    external_views = [
        dict(name="Progress", href="/credleaf/progress/{DAG_ID}/{RUN_ID}", destination="dag_run",
             url_route="credleaf-progress"),
        dict(name="Pause / Resume", href="/credleaf/control?dag_id={DAG_ID}&run_id={RUN_ID}", destination="dag_run"),
    ]
    # Keep CredleafEtaLink importable for older serialized runs. New runs use
    # the standard XCom tab directly, without an additional metrics button.
    global_operator_extra_links = []
