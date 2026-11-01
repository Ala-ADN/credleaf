"""Migration must preserve records and let processing ignore phase filenames."""
import json
from pathlib import Path

import pytest

from corpus_storage import (CAPTURES, DOCUMENTS, collection_files, load_records,
                            migrate_collection, write_records)


def test_migration_is_lossless_backed_up_and_idempotent(tmp_path):
    directory = tmp_path / "processed/georef/13559"
    rows = [dict(article_id="same-id", phase="P1_first_wave", mentions=[dict(name="Tunis")]),
            dict(article_id="other", phase="outside_study", publish_date="2018-01-01"),
            dict(article_id="undated", phase="undated", publish_date=None)]
    for row in rows:
        write_records(directory / f"{row['phase']}.jsonl", [row])
    # Preserve duplicate observations too: migration must never deduplicate.
    write_records(directory / "P2_pre_vaccine.jsonl", [rows[0]])
    (directory / "P1_first_wave.jsonl.progress").write_text("same-id\n")
    result = migrate_collection(directory)
    expected = [rows[0], rows[0], rows[1], rows[2]]
    assert load_records(directory) == expected
    assert result["records"] == 4
    assert collection_files(directory) == [directory / DOCUMENTS]
    backup = Path(result["backup"])
    assert (backup / "P1_first_wave.jsonl.progress").read_text() == "same-id\n"
    assert not (directory / "P1_first_wave.jsonl.progress").exists()
    assert sorted(p.name for p in directory.glob("*.jsonl")) == [DOCUMENTS]
    before = (directory / DOCUMENTS).read_bytes()
    assert migrate_collection(directory)["status"] == "unchanged"
    assert (directory / DOCUMENTS).read_bytes() == before


def test_migration_rejects_invalid_or_conflicting_inputs_without_discarding(tmp_path):
    directory = tmp_path / "normalized/13559"
    directory.mkdir(parents=True)
    legacy = directory / "P1_global_onset.jsonl"
    legacy.write_text('{"text":"valid"}\nmalformed\n')
    with pytest.raises(ValueError):
        migrate_collection(directory)
    assert not (directory / DOCUMENTS).exists()
    write_records(legacy, [dict(text="original")])
    write_records(directory / DOCUMENTS, [dict(text="new")])
    with pytest.raises(ValueError, match="Conflicting"):
        migrate_collection(directory)
    assert json.loads(legacy.read_text())["text"] == "original"
    assert load_records(directory) == [dict(text="new")]


def test_interrupted_cleanup_recovers_without_double_loading(tmp_path):
    directory = tmp_path / "normalized/13559"
    row = dict(text="preserved")
    write_records(directory / "P1_global_onset.jsonl", [row])
    write_records(directory / DOCUMENTS, [row])
    assert load_records(directory) == [row]
    assert migrate_collection(directory)["records"] == 1
    assert not (directory / "P1_global_onset.jsonl").exists()


def test_partial_harvest_preserves_unselected_capture_dates(tmp_path):
    from ingest import harvest_all_phases, Capture, Seed, load_jsonl
    from config import PHASES_BY_NAME
    class Client:
        def iter_seeds(self, collection, **kwargs):
            yield Seed(id=1, collection=collection, url="https://example.org/")
        def iter_captures(self, collection, target, **kwargs):
            yield Capture(urlkey="a", url="https://example.org/a", timestamp="20200301120000",
                          digest="new", status="200", mimetype="text/html")
    old = dict(collection_id=13559, seed_id=1, seed_url="https://example.org/", phase="P2_pre_vaccine",
               capture=dict(urlkey="b", url="https://example.org/b", timestamp="20201201120000",
                            digest="old", status="200", mimetype="text/html"))
    write_records(tmp_path / CAPTURES, [old])
    paths, _ = harvest_all_phases(Client(), 13559, [PHASES_BY_NAME["P1_global_onset"]], output_dir=tmp_path)
    assert set(paths.values()) == {tmp_path / CAPTURES}
    assert {r.capture.digest for r in load_jsonl(tmp_path / CAPTURES)} == {"old", "new"}
    assert list(tmp_path.glob("*.jsonl")) == [tmp_path / CAPTURES]


def test_dedupe_cap_applies_once_and_keeps_outside_and_undated(tmp_path):
    from normalize.collection import dedupe_collection
    source, output = tmp_path / "normalized/13559", tmp_path / "deduped/13559"
    rows = [dict(collection_id=13559, phase="P1_global_onset", text="first text", publish_date="2020-03-01"),
            dict(collection_id=13559, text="outside text", publish_date="2018-01-01"),
            dict(collection_id=13559, text="undated text")]
    write_records(source / DOCUMENTS, rows)
    summary = dedupe_collection(source, output)
    assert summary["input_documents"] == 3
    assert {r["phase"] for r in load_records(output)} == {"P1_global_onset", "outside_study", "undated"}
    assert dedupe_collection(source, output, max_docs=2)["input_documents"] == 2


def test_phase_boundaries_can_change_without_repartitioning(tmp_path, monkeypatch):
    import document_time
    from config import Phase
    from normalize.corpus import write_collection, refresh_collection_dates
    write_collection([dict(collection_id=13559, publish_date="2020-03-01", text="report")], tmp_path)
    monkeypatch.setattr(document_time, "PANDEMIC_PHASES", [Phase("new_analysis_phase", "2020-01-01", "2020-12-31")])
    refresh_collection_dates(tmp_path)
    assert load_records(tmp_path)[0]["phase"] == "new_analysis_phase"
    assert list(tmp_path.glob("*.jsonl")) == [tmp_path / DOCUMENTS]


def test_georeferencing_writes_mixed_dates_to_one_collection_file(tmp_path, monkeypatch):
    import georeferencing.georef as module
    from credibility import CredibilityRegistry
    from normalize.models import NormalizedDoc
    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)
    rows = [NormalizedDoc(collection_id=13559, seed_id=1, seed_url="https://example.org/",
                          phase=phase, capture_timestamp=stamp, capture_url="https://example.org/a",
                          fetch_status="ok", text="", language="en", fetched_at="2026-01-01T00:00:00Z")
            for phase, stamp in [("P1_global_onset", "20200301120000"), ("P2_pre_vaccine", "20201201120000")]]
    source = tmp_path / "deduped/13559/documents.jsonl"
    write_records(source, [row.model_dump(mode="json") for row in rows])
    output, stats = module.georef_phase(source, None, None, CredibilityRegistry(), resume=False)
    assert output == tmp_path / "georef/13559/documents.jsonl"
    assert stats.no_text == 2
    assert len(load_records(output.parent)) == 2


def test_historical_signed_artifacts_stay_readable(tmp_path, monkeypatch):
    import pipeline.artifacts as module
    from pipeline.spec import legacy_required_files
    store = module.ArtifactStore(tmp_path)
    config = dict(capture_phase="all")
    def producer(output):
        for name in legacy_required_files("harvest", 13559, config):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        return {}
    with monkeypatch.context() as context:
        context.setattr(module, "required_files", legacy_required_files)
        reference, _ = store.build("harvest", 13559, config, [], "historical-code", producer)
    before = module.file_hash(store.directory(reference["id"]) / "manifest.json")
    assert store.load(reference)["config"] == config
    assert module.file_hash(store.directory(reference["id"]) / "manifest.json") == before
