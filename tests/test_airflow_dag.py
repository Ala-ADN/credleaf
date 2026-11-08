"""Run with isolated Airflow Python; loading never touches archives or models."""
from importlib.metadata import PackageNotFoundError, version
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

try:
    version("apache-airflow")
    INSTALLED = True
except PackageNotFoundError:
    INSTALLED = False


@unittest.skipUnless(INSTALLED, "Requires the isolated Airflow environment")
class AirflowDagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="credleaf-dag-tests-")
        cls.addClassCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, AIRFLOW_HOME=temporary.name, AIRFLOW__CORE__LOAD_EXAMPLES="False")
        environment.start()
        cls.addClassCleanup(environment.stop)
        from airflow.models.dagbag import DagBag
        cls.bag = DagBag(str(Path(__file__).resolve().parents[1] / "dags"), include_examples=False)

    def test_both_dags_load_and_serialize(self):
        from airflow.serialization.serialized_objects import SerializedDAG
        self.assertEqual(self.bag.import_errors, {})
        self.assertEqual(set(self.bag.dags), {"credleaf_build_corpus", "credleaf_analyze_corpus", "credleaf_data_catalog"})
        for dag in self.bag.dags.values():
            self.assertIn("dag", SerializedDAG.to_dict(dag))
            if dag.dag_id == "credleaf_data_catalog":
                continue
            self.assertEqual(dag.max_active_runs, 1)
            self.assertEqual(dag.max_active_tasks, 4)
            self.assertIsNotNone(dag.on_success_callback)
            self.assertIsNotNone(dag.on_failure_callback)

    def test_collection_chains_are_mapped_as_a_group(self):
        dag = self.bag.dags["credleaf_build_corpus"]
        group = dag.get_task("collection_flow.normalize").task_group
        self.assertEqual(type(group).__name__, "MappedTaskGroup")
        self.assertIn("collection_flow.harvest", dag.get_task("collection_flow.normalize").upstream_task_ids)
        self.assertIn("collection_flow.normalize", dag.get_task("collection_flow.dedupe").upstream_task_ids)
        self.assertEqual(dag.get_task("collection_flow.georef").pool, "credleaf_models")
        self.assertEqual(dag.get_task("embed").pool, "credleaf_models")
        self.assertEqual(dag.get_task("collection_flow.harvest").pool, "credleaf_archive")
        self.assertEqual(dag.get_task("collection_flow.normalize").execution_timeout, timedelta(days=14))
        self.assertEqual(dag.get_task("collection_flow.dedupe").execution_timeout, timedelta(days=7))
        self.assertEqual(dag.get_task("embed").execution_timeout, timedelta(days=14))

    def test_analysis_has_no_archive_work_and_topic_producer_precedes_animation(self):
        dag = self.bag.dags["credleaf_analyze_corpus"]
        self.assertEqual(set(dag.task_ids), {"prepare_analysis", "semantic", "topic", "animation", "publish_analysis", "overview", "failure_guard"})
        self.assertIn("topic", dag.get_task("animation").upstream_task_ids)
        self.assertTrue({"semantic", "animation", "topic"} <= dag.get_task("publish_analysis").upstream_task_ids)

    def test_reporting_covers_every_task_and_does_not_mask_failures(self):
        for dag in self.bag.dags.values():
            if dag.dag_id == "credleaf_data_catalog":
                continue
            self.assertEqual(dag.get_task("overview").trigger_rule, "all_done")
            self.assertEqual(dag.get_task("overview").upstream_task_ids, set(dag.task_ids)-{"overview","failure_guard"})
            self.assertEqual(dag.get_task("failure_guard").trigger_rule, "one_failed")
            self.assertEqual(dag.get_task("failure_guard").upstream_task_ids, set(dag.task_ids)-{"failure_guard"})

    def test_catalog_only_declares_observed_inputs(self):
        dag = self.bag.dags["credleaf_data_catalog"]
        task = dag.get_task("view_catalog")
        self.assertEqual(len(task.inlets), 19)
        self.assertEqual(task.outlets, [])
        self.assertIsNone(dag.schedule)

    def test_embed_has_both_backends_and_reschedules_before_publication(self):
        dag = self.bag.dags["credleaf_build_corpus"]
        wait = dag.get_task("embed")
        publish = dag.get_task("publish_corpus")
        self.assertEqual(wait.mode, "reschedule")
        self.assertEqual(wait.poke_interval, 120)
        self.assertIn("embed", publish.upstream_task_ids)
        self.assertEqual(wait.upstream_task_ids, {"prepare_corpus", "collection_flow.dedupe"})
        self.assertEqual(dag.params["embed_backend"], "local")
        self.assertEqual(dag.params.get_param("embed_backend").schema["enum"], ["local", "kaggle"])
        self.assertNotIn("kaggle_api_token", dag.params)

    def test_embed_local_off_and_reuse_do_not_read_kaggle_credentials(self):
        from pipeline import airflow_support as support
        context = dict(ti=SimpleNamespace(try_number=2))
        for backend, mode in (("local", "build"), ("kaggle", "off"), ("kaggle", "reuse")):
            with self.subTest(backend=backend, mode=mode), \
                 patch.object(support.runner, "load_plan", return_value=dict(parameters=dict(embed_backend=backend, embed_mode=mode))), \
                 patch.object(support.runner, "run_stage", return_value={"id": "fixture"}) as run, \
                 patch("airflow.sdk.BaseHook.get_connection") as connection:
                self.assertEqual(support.embed({}, [], context), {"id": "fixture"})
                connection.assert_not_called()
                run.assert_called_once_with({}, "embed", [], attempt=2, worker_env=None)

    def test_embed_kaggle_waits_for_credentials_and_passes_them_only_to_worker(self):
        from airflow.exceptions import AirflowNotFoundException
        from pipeline import airflow_support as support
        params = dict(embed_backend="kaggle", embed_mode="build", kaggle_connection_id="kaggle_default")
        context = dict(ti=SimpleNamespace(try_number=1))
        with patch.object(support.runner, "load_plan", return_value=dict(parameters=params)), \
             patch.object(support.runner, "run_stage", return_value={"id": "fixture"}) as run, \
             patch("airflow.sdk.BaseHook.get_connection") as connection:
            connection.side_effect = AirflowNotFoundException("missing")
            self.assertIsNone(support.embed({}, [], context))
            connection.side_effect = None
            connection.return_value = SimpleNamespace(login="tester", password="", extra_dejson={})
            self.assertIsNone(support.embed({}, [], context))
            run.assert_not_called()
            connection.return_value.password = "secret-token"
            self.assertEqual(support.embed({}, [], context), {"id": "fixture"})
            environment = run.call_args.kwargs["worker_env"]
            self.assertEqual(environment["KAGGLE_API_TOKEN"], "secret-token")
            self.assertEqual(environment["CREDLEAF_KAGGLE_OWNER"], "tester")
            self.assertNotIn("secret-token", str(run.call_args.args))

    def test_plugin_discovery_does_not_execute_helper_cli(self):
        root = Path(__file__).resolve().parents[1]
        code = """
from airflow import plugins_manager
plugins_manager.ensure_plugins_loaded()
assert not plugins_manager.import_errors, plugins_manager.import_errors
assert 'credleaf_monitoring' in {p.name for p in plugins_manager.plugins}
"""
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True,
            env=dict(os.environ, AIRFLOW__CORE__PLUGINS_FOLDER=str(root / "airflow/plugins"),
                     PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "airflow/plugins")])) )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_params(self):
        from airflow.sdk import Param
        from airflow.exceptions import ParamValidationError
        from pipeline.spec import CORPUS_SETTINGS, defaults
        self.assertEqual(self.bag.dags["credleaf_build_corpus"].params.validate(), defaults())
        with self.assertRaises(ParamValidationError):
            Param(**CORPUS_SETTINGS["harvest_mode"]).resolve("off")


if __name__ == "__main__":
    unittest.main()
