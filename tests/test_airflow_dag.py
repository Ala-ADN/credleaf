"""Run with isolated Airflow Python; loading never touches archives or models."""
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(set(self.bag.dags), {"credleaf_build_corpus", "credleaf_analyze_corpus"})
        for dag in self.bag.dags.values():
            self.assertIn("dag", SerializedDAG.to_dict(dag))
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

    def test_analysis_has_no_archive_work_and_topic_producer_precedes_animation(self):
        dag = self.bag.dags["credleaf_analyze_corpus"]
        self.assertEqual(set(dag.task_ids), {"prepare_analysis", "semantic", "topic", "animation", "publish_analysis", "overview", "failure_guard"})
        self.assertIn("topic", dag.get_task("animation").upstream_task_ids)
        self.assertTrue({"semantic", "animation", "topic"} <= dag.get_task("publish_analysis").upstream_task_ids)

    def test_reporting_covers_every_task_and_does_not_mask_failures(self):
        for dag in self.bag.dags.values():
            self.assertEqual(dag.get_task("overview").trigger_rule, "all_done")
            self.assertEqual(dag.get_task("overview").upstream_task_ids, set(dag.task_ids)-{"overview","failure_guard"})
            self.assertEqual(dag.get_task("failure_guard").trigger_rule, "one_failed")
            self.assertEqual(dag.get_task("failure_guard").upstream_task_ids, set(dag.task_ids)-{"failure_guard"})

    def test_params(self):
        from airflow.sdk import Param
        from airflow.exceptions import ParamValidationError
        from pipeline.spec import CORPUS_SETTINGS, defaults
        self.assertEqual(self.bag.dags["credleaf_build_corpus"].params.validate(), defaults())
        with self.assertRaises(ParamValidationError):
            Param(**CORPUS_SETTINGS["harvest_mode"]).resolve("off")


if __name__ == "__main__":
    unittest.main()
