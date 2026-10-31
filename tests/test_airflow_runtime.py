"""Opt-in offline Airflow integration tests; no archives, Qdrant or model weights.

Run separately in the Airflow environment with CREDLEAF_RUN_AIRFLOW_INTEGRATION=1.
The worker fixture replaces only expensive computation; the real planner,
artifact transactions, dynamic mapping, publications and callbacks run normally.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.getenv("CREDLEAF_RUN_AIRFLOW_INTEGRATION") == "1", "Opt-in Airflow runtime test")
class AirflowRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="credleaf-runtime-"))
        cls.addClassCleanup(shutil.rmtree, cls.root)
        cls.project = Path(__file__).resolve().parents[1]
        for folder in ("src", "scripts", "dags"):
            (cls.root / folder).symlink_to(cls.project / folder, target_is_directory=True)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(cls.project / name, cls.root / name)
        env = patch.dict(os.environ, AIRFLOW_HOME=str(cls.root / "airflow"),
            AIRFLOW__CORE__LOAD_EXAMPLES="False", AIRFLOW__CORE__DAGS_FOLDER=str(cls.project / "dags"),
            CREDLEAF_ROOT=str(cls.root), CREDLEAF_PYTHON=sys.executable)
        env.start()
        cls.addClassCleanup(env.stop)
        from airflow.utils import db
        from airflow.models.pool import Pool
        db.initdb()
        Pool.create_or_update_pool("credleaf_archive",2,"fixture",False)
        Pool.create_or_update_pool("credleaf_models",1,"fixture",False)
        cls.calls = []
        cls.fail_collection = False
        from pipeline import runner
        worker = patch.object(runner,"_worker",side_effect=cls.worker)
        worker.start()
        cls.addClassCleanup(worker.stop)

    @classmethod
    def worker(cls,plan,stage,cid,config,inputs,output,log):
        from pipeline.spec import required_files
        from pipeline.artifacts import write_json
        cls.calls.append((stage,cid))
        if cls.fail_collection and stage == "harvest" and cid == 13559:
            raise RuntimeError("fixture archive failure for one collection")
        for name in required_files(stage,cid,config):
            path=output/name
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix == ".json":
                write_json(path,dict(low=["example.org"],mixed=[],authoritative=["who.int"]) if stage=="credibility" else {})
            else:
                path.write_text("")
        return dict(fixture=True)

    def dags(self):
        return runpy.run_path(str(self.project/"dags/credleaf_pipeline.py"))

    def states(self,run):
        from airflow.utils.session import create_session
        with create_session() as session:
            return {f"{ti.task_id}[{ti.map_index}]":str(ti.state) for ti in run.get_task_instances(session=session)}

    def overview(self,run):
        from pipeline.plan import run_directory
        return json.loads((run_directory(self.root,run.dag_id,run.run_id)/"overview.json").read_text())

    def test_robust_flow(self):
        # Disable retries for injected failures so the test does not wait minutes.
        params=dict(georef_mode="off",embed_mode="build")
        dag=self.dags()["build_corpus"]
        for task in dag.tasks:
            task.retries=0
        first=dag.test(run_conf=params)
        self.assertEqual(str(first.state),"success")
        report=self.overview(first)
        self.assertEqual(report["status"],"success")
        self.assertIsNotNone(report["task_states"])
        snapshot=report["snapshot_id"]
        from pipeline.artifacts import ArtifactStore, file_hash
        store=ArtifactStore(self.root)
        before=file_hash(store.directory(snapshot)/"manifest.json")

        # Reuse performs validation and publication without executing the worker.
        calls_before=len(self.calls)
        reuse=self.dags()["build_corpus"]
        for task in reuse.tasks:
            task.retries=0
        second=reuse.test(run_conf=params | dict(base_snapshot_id=snapshot,harvest_mode="reuse",
            normalize_mode="reuse",dedupe_mode="reuse",credibility_mode="reuse",embed_mode="reuse"))
        self.assertEqual(str(second.state),"success")
        self.assertEqual(len(self.calls),calls_before)

        # Failure of one group's harvest cannot block the other's normalization.
        type(self).fail_collection=True
        failing=self.dags()["build_corpus"]
        for task in failing.tasks:
            task.retries=0
        failed=failing.test(run_conf=params)
        states=self.states(failed)
        self.assertEqual(str(failed.state),"failed")
        # Plans canonicalize the IDs: 13529 is map index 0, 13559 is index 1.
        self.assertEqual(states["collection_flow.normalize[0]"],"success")
        self.assertEqual(states["collection_flow.dedupe[0]"],"success")
        self.assertEqual(states["collection_flow.normalize[1]"],"upstream_failed")
        self.assertEqual(self.overview(failed)["status"],"failed")
        self.assertIsNone(self.overview(failed)["snapshot_id"])
        self.assertEqual(file_hash(store.directory(snapshot)/"manifest.json"),before)
        type(self).fail_collection=False

        # Simulate a worker losing completion after publishing a valid manifest.
        after=self.dags()["build_corpus"]
        for task in after.tasks:
            task.retries=0
        original=after.get_task("publish_corpus").python_callable
        def fail_after_publish(plan_ref,collections,registry,embedding):
            original(plan_ref,collections,registry,embedding)
            raise RuntimeError("fixture worker lost after publication")
        after.get_task("publish_corpus").python_callable=fail_after_publish
        failed_after=after.test(run_conf=params)
        report=self.overview(failed_after)
        self.assertEqual(str(failed_after.state),"failed")
        self.assertEqual(report["artifact_status"],"success")
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["task_states"]["publish_corpus[-1]"],"failed")

        analysis=self.dags()["analyze_corpus"]
        for task in analysis.tasks:
            task.retries=0
        analyzed=analysis.test(run_conf=dict(snapshot_id=snapshot,topic_mode="build",animation_mode="build",
                                            analysis_collection_ids=[13529]))
        self.assertEqual(str(analyzed.state),"success")
        self.assertEqual(self.overview(analyzed)["status"],"success")
        bundle=store.bundle(self.overview(analyzed)["snapshot_id"],"analysis")
        self.assertEqual(bundle["collection_ids"],[13529])


if __name__ == "__main__":
    unittest.main()
