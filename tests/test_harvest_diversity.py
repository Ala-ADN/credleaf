"""URL diversity must precede per-seed budgets and preserve capture windows."""
from ingest import Capture, Seed, harvest_phase, harvest_all_phases, load_jsonl
from config import PHASES_BY_NAME
from url_utils import canonical_url


def capture(url, digest, stamp="20200301120000", status="200", mimetype="text/html"):
    return Capture(urlkey=url, timestamp=stamp, url=url, digest=digest, status=status, mimetype=mimetype)


class Client:
    def __init__(self, pages):
        self.pages = pages
        self.seed_passes = 0
        self.queries = []
    def iter_seeds(self, collection, **kwargs):
        self.seed_passes += 1
        yield Seed(id=1, collection=collection, url="https://example.org/")
        yield Seed(id=2, collection=collection, url="https://example.org/news")
    def iter_captures(self, collection, target, **kwargs):
        self.queries.append(kwargs)
        assert kwargs["collapse"] is None  # avoid collapsing equal digests across windows/URLs upstream
        for page in self.pages:
            if kwargs["from_ts"] <= page.timestamp <= kwargs["to_ts"]:
                yield page


def test_repeats_do_not_consume_unique_page_budget_or_block_valid_captures(tmp_path):
    client = Client([
        capture("https://example.org/a", "error", status="404"),
        capture("https://example.org/a", "pdf", mimetype="application/pdf"),
        capture("https://example.org/a?utm_source=x#fragment", "one"),
        capture("https://example.org/a", "two"),
        capture("https://example.org/b?id=1", "shared"),
        capture("https://example.org/b?id=2", "shared"),
    ])
    path, stats = harvest_phase(client, 13559, PHASES_BY_NAME["P1_global_onset"],
                                 output_path=tmp_path / "one.jsonl", match_type="host",
                                 max_seeds=1, max_captures_per_seed=3)
    records = list(load_jsonl(path))
    assert len(records) == 3
    assert len({canonical_url(r.capture.url) for r in records}) == 3
    assert stats.captures_skipped_dup_url == 1
    assert stats.captures_skipped_status == stats.captures_skipped_mimetype == 1
    assert client.queries[0]["match_type"] == "host"


def test_url_limit_shared_across_overlapping_seeds_but_not_phases(tmp_path):
    client = Client([capture("https://example.org/a", "same", "20200301120000"),
                     capture("https://example.org/a", "same", "20201201120000")])
    phases = [PHASES_BY_NAME["P1_global_onset"], PHASES_BY_NAME["P2_pre_vaccine"]]
    paths, stats = harvest_all_phases(client, 13559, phases, output_dir=tmp_path)
    assert client.seed_passes == 1
    for phase in phases:
        assert sum(r.phase == phase.name for r in load_jsonl(paths[phase.name])) == 1
        assert stats[phase.name].captures_kept == 1
        assert stats[phase.name].captures_skipped_dup_digest == 1


def test_url_limit_can_be_disabled_to_retain_changed_revisions(tmp_path):
    client = Client([capture("https://example.org/a", "one"),
                     capture("https://example.org/a", "one", "20200302120000"),
                     capture("https://example.org/a", "two", "20200303120000")])
    path, stats = harvest_phase(client, 13559, PHASES_BY_NAME["P1_global_onset"],
                                 output_path=tmp_path / "one.jsonl", max_seeds=1, max_captures_per_url=0)
    assert len(list(load_jsonl(path))) == 2
    assert stats.captures_skipped_dup_digest == 1 and stats.captures_skipped_dup_url == 0
