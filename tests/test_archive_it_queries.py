"""Exercise the actual HTTP query path, including Archive-It's range quirk."""
import httpx
import pytest

from ingest import ArchiveItClient


def row(stamp):
    return f"org,example)/ {stamp} https://example.org/ text/html 200 DIGEST - - 100 0 file.warc.gz\n"


@pytest.mark.parametrize("scope", ["prefix", "host", "domain"])
def test_broad_queries_filter_dates_before_applying_limit(scope):
    def respond(request):
        assert request.url.params["matchType"] == scope
        assert not {"from", "to", "limit"} & set(request.url.params)
        return httpx.Response(200, text=row("20190101000000") + row("20200301000000")
                              + row("20200401000000") + row("20240101000000"))

    with ArchiveItClient() as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        records = list(client.iter_captures(13559, "https://example.org/", match_type=scope,
                       from_ts="20200131000000", to_ts="20230505235959", limit=1))
        all_records = list(client.iter_captures(13559, "https://example.org/", match_type=scope,
                           from_ts="20200131000000", to_ts="20230505235959"))
    assert [r.timestamp for r in records] == ["20200301000000"]
    assert [r.timestamp for r in all_records] == ["20200301000000", "20200401000000"]


def test_exact_query_keeps_server_date_bounds():
    def respond(request):
        assert request.url.params["from"] == "2020"
        assert request.url.params["to"] == "2023"
        return httpx.Response(200, text=row("20200301000000"))

    with ArchiveItClient() as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        assert len(list(client.iter_captures(13559, "https://example.org/", from_ts="2020", to_ts="2023"))) == 1


def test_transient_cdx_failure_retries_before_output(monkeypatch):
    requests = []
    monkeypatch.setattr("ingest.archive_it.time.sleep", lambda _: None)

    def respond(request):
        requests.append(request)
        return httpx.Response(503) if len(requests) == 1 else httpx.Response(200, text=row("20200301000000"))

    with ArchiveItClient() as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        assert len(list(client.iter_captures(13559, "https://example.org/"))) == 1
    assert len(requests) == 2
