"""Publication-first phase assignment and lossless offline phase migration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from document_time import assign_document_time, document_time
from normalize.phases import rephase_normalized, load_phase_records
from analysis.semantic_analysis import prepare_semantic_corpus
from credibility.registry import CredibilityRegistry


@pytest.mark.parametrize("publication,capture,phase,source", [
    ("2020-03-10", "20220105120000", "P1_global_onset", "publish_date"),
    (None, "20201231235959", "P2_pre_vaccine", "capture_timestamp"),
    (None, 20201231235959, "P2_pre_vaccine", "capture_timestamp"),
    ("invalid", "20210201120000", "P3_vaccine_rollout", "capture_timestamp"),
    ("2021-01-01T00:30:00+02:00", "20210601120000", "P2_pre_vaccine", "publish_date"),
    ("2018-01-01", "20210201120000", "outside_study", "publish_date"),
    ("2023-05-05T23:59:59Z", "20230505120000", "P6_transition_end", "publish_date"),
    ("2023-05-06", "20230505120000", "outside_study", "publish_date"),
    ("bad", "bad", "undated", None),
])
def test_shared_policy(publication, capture, phase, source):
    result = document_time(dict(publish_date=publication, capture_timestamp=capture))
    assert result.phase == phase and result.source == source


def test_rephase_preserves_records_and_capture_provenance_with_backups(tmp_path):
    directory = tmp_path / "normalized/13559"
    directory.mkdir(parents=True)
    rows = [dict(collection_id=13559, digest="late", phase="P5_omicron_wave", publish_date="2020-06-10", capture_timestamp="20220105120000"),
            dict(collection_id=13559, digest="early", phase="P5_omicron_wave", publish_date="2020-02-10", capture_timestamp="20220106120000"),
            dict(collection_id=13559, digest="fallback", phase="P2_pre_vaccine", publish_date="invalid", capture_timestamp="20201205120000"),
            dict(collection_id=13559, digest="outside", phase="P5_omicron_wave", publish_date="2018-01-01", capture_timestamp="20220105120000"),
            dict(collection_id=13559, digest="undated", phase="P5_omicron_wave", publish_date=None, capture_timestamp="bad")]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    (directory / "P5_omicron_wave.jsonl").write_text(original)
    summary = rephase_normalized(directory)
    assert summary["documents"] == 5
    assert (Path(summary["backup"]) / "P5_omicron_wave.jsonl").read_text() == original
    p1 = [json.loads(line) for line in (directory / "P1_global_onset.jsonl").open()]
    assert [row["digest"] for row in p1] == ["early", "late"]
    assert all(row["capture_phase"] == "P5_omicron_wave" for row in p1)
    assert (directory / "P5_omicron_wave.jsonl").read_text() == ""
    assert summary["counts"]["outside_study"] == summary["counts"]["undated"] == 1
    assert len(load_phase_records(directory)) == 5
    before = {p.name: p.read_bytes() for p in directory.glob("*.jsonl")}
    rephase_normalized(directory)
    assert {p.name: p.read_bytes() for p in directory.glob("*.jsonl")} == before


def test_semantic_fallback_preserves_vector_alignment_and_excludes_old_publications():
    data = pd.DataFrame(dict(
        point_id=["published", "fallback-invalid", "fallback-missing", "outside", "undated"],
        collection_id=13529, seed_url="https://example.org", language="en", phase="P6_transition_end",
        text="a sufficiently long report about public health and pandemic response",
        publish_date=["2020-02-10", "invalid", None, "2018-01-01", None],
        capture_timestamp=["20220105120000", "20201205120000", "20210205120000", "20201205120000", "bad"],
    ))
    clean, vectors, _ = prepare_semantic_corpus(data, np.eye(5), CredibilityRegistry(mixed=["example.org"]))
    assert clean.point_id.tolist() == ["published", "fallback-invalid", "fallback-missing"]
    assert clean.analysis_phase.tolist() == ["P1_global_onset", "P2_pre_vaccine", "P3_vaccine_rollout"]
    assert clean.event_date_source.tolist() == ["publish_date", "capture_timestamp", "capture_timestamp"]
    np.testing.assert_array_equal(vectors, np.eye(5)[:3])


def test_normalize_collection_routes_publication_phases_and_updates_capture_windows(tmp_path, monkeypatch):
    import normalize.normalize as module
    from normalize import Extracted
    from ingest import Capture, CaptureRecord

    raw, output = tmp_path / "raw", tmp_path / "normalized"
    (raw / "13559").mkdir(parents=True)
    def row(name, phase, captured):
        return CaptureRecord(collection_id=13559, seed_id=1, seed_url="https://example.org/", phase=phase,
                             capture=Capture(urlkey=name, url=f"https://example.org/{name}", timestamp=captured,
                                             status="200", mimetype="text/html", digest=name))
    a = row("a", "P2_pre_vaccine", "20201205120000")
    b = row("b", "P2_pre_vaccine", "20201206120000")
    c = row("c", "P5_omicron_wave", "20220105120000")
    (raw / "13559/P2_pre_vaccine.jsonl").write_text(a.model_dump_json() + "\n" + b.model_dump_json() + "\n")
    (raw / "13559/P5_omicron_wave.jsonl").write_text(c.model_dump_json() + "\n")
    dates = {"a": "2020-03-01", "b": None, "c": "2020-06-01"}
    class Client:
        def fetch_capture(self, collection, timestamp, url):
            return url.rsplit("/", 1)[1].encode()
    monkeypatch.setattr(module, "extract", lambda html, **kwargs: Extracted(text="some extracted public health text", language="en", publish_date=dates[html.decode()]))
    directory, summary = module.normalize_collection(Client(), 13559, ["P2_pre_vaccine", "P5_omicron_wave"],
                                                       raw_root=raw, output_root=output, use_cache=False)
    assert summary["documents"] == 3
    p1 = [json.loads(line) for line in (directory / "P1_global_onset.jsonl").open()]
    assert [r["digest"] for r in p1] == ["a", "c"]
    assert {r["capture_phase"] for r in p1} == {"P2_pre_vaccine", "P5_omicron_wave"}
    dates["a"] = "2021-02-01"
    module.normalize_collection(Client(), 13559, ["P2_pre_vaccine"], raw_root=raw, output_root=output, use_cache=False)
    after = load_phase_records(directory)
    assert len(after) == 3
    assert {r["digest"]: r["phase"] for r in after} == {"a": "P3_vaccine_rollout", "b": "P2_pre_vaccine", "c": "P1_global_onset"}
