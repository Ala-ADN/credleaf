"""Embedded Airflow progress routes, using an in-process HTTP client."""
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    version("apache-airflow")
    INSTALLED = True
except PackageNotFoundError:
    INSTALLED = False


@unittest.skipUnless(INSTALLED, "Requires the isolated Airflow environment")
class ProgressUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "airflow/plugins"))
        from credleaf_ui.app import app
        from airflow.api_fastapi.core_api.security import get_user
        from fastapi.testclient import TestClient
        cls.app = app
        cls.checked = []
        def check(request):
            cls.checked.append(request.path_params["dag_id"])
        from fastapi import Request
        check.__annotations__["request"] = Request
        app.dependency_overrides[get_user] = lambda: object()
        for route in app.routes:
            if "/progress/" in route.path:
                for dependency in route.dependant.dependencies:
                    if dependency.call is not get_user:
                        app.dependency_overrides[dependency.call] = check
        cls.addClassCleanup(app.dependency_overrides.clear)
        cls.client = TestClient(app)
        cls.addClassCleanup(cls.client.close)

    def test_run_page_and_assets_use_embedded_panel(self):
        response = self.client.get("/progress/credleaf_build_corpus/manual__2026-11-07T12:00:00+00:00")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Stages finished", response.text)
        self.assertEqual(response.headers["x-frame-options"], "SAMEORIGIN")
        self.assertIn("credleaf_build_corpus", self.checked)
        script = self.client.get("/static/progress.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("application/javascript", script.headers["content-type"])
        self.assertEqual(self.client.get("/static/progress.css").status_code, 200)
        from credleaf_dashboard import CredleafDashboardPlugin
        view = next(v for v in CredleafDashboardPlugin.external_views if v["name"] == "Progress")
        self.assertEqual(view["destination"], "dag_run")
        self.assertEqual(view["url_route"], "credleaf-progress")

    def test_api_reads_fresh_progress_with_cached_scheduler_state(self):
        from tests.test_dashboard import fixture, data
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, CREDLEAF_ROOT=temporary):
            root = Path(temporary)
            directory = fixture(root)
            dashboard = data.collect(root)
            (root / "data/pipeline/dashboard.json").write_text(json.dumps(dashboard))
            report = json.loads((directory / "steps/normalize__13529.json").read_text())
            report["log"] = str(directory / "normalize.log")
            (directory / "steps/normalize__13529.json").write_text(json.dumps(report))
            (directory / "normalize.progress.json").write_text(json.dumps(dict(completed=8, total=10)))
            response = self.client.get("/api/progress/credleaf_build_corpus/test")
            self.assertEqual(response.status_code, 200)
            value = response.json()
            normalize = next(s for s in value["steps"] if s["key"] == "normalize__13529")
            self.assertEqual(normalize["percent"], 80)
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertNotIn("directory", value)
            self.assertEqual(self.client.get("/api/progress/credleaf_build_corpus/unknown").status_code, 404)

    def test_progress_endpoints_require_the_existing_login(self):
        from airflow.api_fastapi.core_api.security import get_user
        from fastapi import HTTPException
        async def denied():
            raise HTTPException(401, "Not authenticated")
        with patch.dict(self.app.dependency_overrides, {get_user: denied}):
            self.assertEqual(self.client.get("/progress/credleaf_build_corpus/test").status_code, 401)
            self.assertEqual(self.client.get("/api/progress/credleaf_build_corpus/test").status_code, 401)
