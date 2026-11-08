"""Opt-in real Airflow metadata recovery; all scientific computation is offline."""
import json
import os
import unittest

from tests import test_airflow_runtime as runtime


@unittest.skipUnless(os.getenv("CREDLEAF_RUN_AIRFLOW_INTEGRATION") == "1", "Opt-in Airflow lifecycle test")
class AirflowLifecycleTests(unittest.TestCase):
    setUpClass = classmethod(runtime.AirflowRuntimeTests.setUpClass.__func__)
    worker = classmethod(runtime.AirflowRuntimeTests.worker.__func__)
    dags = runtime.AirflowRuntimeTests.dags

    def test_shutdown_and_boot_recover_stale_tasks_and_keep_manual_pause(self):
        from airflow.models.dag import DagModel
        from airflow.models.dagrun import DagRun
        from airflow.models.taskinstance import TaskInstance
        from airflow.utils.session import create_session
        from pipeline.artifacts import write_json
        from pipeline.plan import run_directory
        from credleaf_ui.lifecycle import request_shutdown, recover
        dag = self.dags()["build_corpus"]
        for task in dag.tasks:
            task.retries = 0
        run = dag.test(run_conf=dict(georef_mode="off", embed_mode="off"))
        self.assertEqual(str(run.state), "success")
        directory = run_directory(self.root, run.dag_id, run.run_id)
        key = dict(dag_id=run.dag_id, run_id=run.run_id)
        with create_session() as session:
            session.query(DagRun).filter_by(**key).one().state = "running"
            session.query(DagModel).filter_by(dag_id=run.dag_id).one().is_paused = False
            session.query(TaskInstance).filter_by(**key, task_id="collection_flow.normalize", map_index=0).one().state = "running"
        write_json(directory / "steps/normalize__13529.json", dict(stage="normalize", collection_id=13529, status="running"))
        paths = request_shutdown(self.root)
        self.assertIn(directory, paths)
        request_path = directory / "control/request.json"
        self.assertTrue(json.loads(request_path.read_text())["shutdown_requested"])
        with create_session() as session:
            self.assertTrue(session.query(DagModel).filter_by(dag_id=run.dag_id).one().is_paused)
        actions = recover(self.root)
        self.assertEqual(len(actions), 1)
        with create_session() as session:
            task = session.query(TaskInstance).filter_by(**key, task_id="collection_flow.normalize", map_index=0).one()
            self.assertIsNone(task.state)
            self.assertEqual(session.query(TaskInstance).filter_by(**key, task_id="collection_flow.harvest", map_index=0).one().state, "success")
            self.assertFalse(session.query(DagModel).filter_by(dag_id=run.dag_id).one().is_paused)
            self.assertEqual(session.query(DagRun).filter_by(**key).one().state, "queued")
        self.assertFalse(json.loads(request_path.read_text())["shutdown_requested"])
        # A user pause remains a pause across the same service shutdown/recovery.
        write_json(request_path, dict(pause_requested=True))
        with create_session() as session:
            session.query(DagModel).filter_by(dag_id=run.dag_id).one().is_paused = True
        request_shutdown(self.root)
        recover(self.root)
        with create_session() as session:
            self.assertTrue(session.query(DagModel).filter_by(dag_id=run.dag_id).one().is_paused)
        self.assertTrue(json.loads(request_path.read_text())["pause_requested"])
        # A real scientific failure remains failed, even during a later boot.
        write_json(request_path, dict(pause_requested=False))
        write_json(directory / "steps/embed.json", dict(stage="embed", collection_id=None, status="failed", error="bad model"))
        with create_session() as session:
            session.query(DagRun).filter_by(**key).one().state = "failed"
            session.query(TaskInstance).filter_by(**key, task_id="embed").one().state = "failed"
        self.assertEqual(recover(self.root), [])
        with create_session() as session:
            self.assertEqual(session.query(TaskInstance).filter_by(**key, task_id="embed").one().state, "failed")
        # A changed source identity blocks resumption and leaves an audit reason.
        plan_path = directory / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["implementation"]["source"] = "changed"
        write_json(plan_path, plan)
        with create_session() as session:
            session.query(DagRun).filter_by(**key).one().state = "running"
            session.query(TaskInstance).filter_by(**key, task_id="embed").one().state = "success"
            session.query(TaskInstance).filter_by(**key, task_id="collection_flow.normalize", map_index=0).one().state = "running"
        self.assertEqual(recover(self.root), [])
        self.assertIn("Code or Python runtime changed", json.loads((directory / "control/recovery.json").read_text())["error"])
