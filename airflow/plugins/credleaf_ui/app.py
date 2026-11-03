"""Native Airflow redirects and the small cooperative pause/resume control."""
import os
from html import escape
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from airflow.api_fastapi.core_api.security import get_user, requires_access_dag
from typing import Literal
from credleaf_ui.data import read_json
from credleaf_ui.native import operational_status

app = FastAPI(dependencies=[Depends(get_user)])


@app.get("/")
def dashboard(run_id: str | None = None):
    value = api()
    runs = [r for r in value["runs"] if r["dag_id"] in {"credleaf_build_corpus", "credleaf_analyze_corpus"}]
    run = next((r for r in runs if r["run_id"] == run_id), None) if run_id else next((r for r in runs if not r.get("superseded_by")), None)
    if run is None:
        raise HTTPException(404, "Workflow run not found")
    return RedirectResponse(run["run_link"], status_code=303)


@app.get("/control", response_class=HTMLResponse)
def control_page(dag_id: str, run_id: str):
    run = next((r for r in api()["runs"] if r["dag_id"] == dag_id and r["run_id"] == run_id), None)
    if run is None:
        raise HTTPException(404, "Workflow run not found")
    from urllib.parse import quote
    paused = run["control"].get("pause_requested")
    unavailable = bool(run.get("superseded_by"))
    pause_disabled = "disabled" if paused or unavailable or run["state"] not in {"running", "queued"} else ""
    resume_disabled = "disabled" if not paused or unavailable or run["state"] not in {"running", "queued", "failed"} else ""
    path = "/credleaf/control/" + quote(dag_id, safe="") + "/" + quote(run_id, safe="")
    progress = run["normalization"]
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Pause / resume</title>
<h1>Pause / resume</h1><p>{escape(dag_id)} / {escape(run_id)}</p>
<p>{escape(operational_status(run))}. Saved normalization: {progress['completed']:,} / {progress['total']:,}.</p>
<form method="post" action="{escape(path)}/pause"><button {pause_disabled}>Pause safely</button></form>
<form method="post" action="{escape(path)}/resume"><button {resume_disabled}>Resume</button></form>
<p>Normalization pauses after saving the current document. Other computations stop at their next supported boundary.</p>
<p><a href="{escape(run['run_link'])}">Back to run</a> · <a href="/assets">Assets</a></p></html>"""
    return HTMLResponse(body, headers={"Cache-Control": "no-store", "X-Frame-Options": "SAMEORIGIN"})


@app.get("/api")
def api():
    root = Path(os.environ.get("CREDLEAF_ROOT", "/opt/credleaf"))
    value = read_json(root / "data/pipeline/dashboard.json")
    if value is None:
        raise HTTPException(503, "Observer is starting; retry shortly")
    for run in value["runs"]:
        run["control"] = read_json(Path(run["directory"]) / "control/request.json", {})
        run["pause_acks"] = {p.stem: read_json(p, {}) for p in (Path(run["directory"]) / "control").glob("*.json")
                             if p.stem not in {"prepare", "request"}}
    return value


@app.post("/control/{dag_id}/{run_id}/{action}", dependencies=[Depends(requires_access_dag("PUT"))])
def control(dag_id: str, run_id: str, action: Literal["pause", "resume"], request: Request):
    origin = request.headers.get("origin")
    expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
    if origin != expected_origin and not request.headers.get("authorization", "").lower().startswith("bearer "):
        raise HTTPException(403, "Control requests must originate from this Airflow interface")
    from credleaf_ui.controls import control_run
    try:
        value = control_run(Path(os.environ.get("CREDLEAF_ROOT", "/opt/credleaf")), dag_id, run_id, action == "pause")
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/credleaf/control?" + urlencode(dict(dag_id=dag_id, run_id=run_id)), status_code=303)
        return value
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(409, str(error)) from error
