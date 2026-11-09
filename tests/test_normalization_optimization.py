"""Offline checks for reused-input selection, pacing, and resumable outages."""
import importlib
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from ingest import ArchiveItClient, Capture, CaptureRecord
from ingest.archive_it import ArchiveUnavailable, retry_after_seconds
from normalize.extract import Extracted
from pipeline.spec import defaults, stage_config


def test_reused_harvest_is_capped_by_canonical_url_and_phase(tmp_path, monkeypatch):
    module = importlib.import_module('normalize.normalize')
    raw = tmp_path / 'raw/13559'
    raw.mkdir(parents=True)
    rows = [CaptureRecord(collection_id=13559, seed_id=1, seed_url='https://example.org/',
        phase=phase, capture=Capture(urlkey='x', timestamp=stamp, url=url, digest=str(i)))
        for i, (phase, stamp, url) in enumerate([
            ('P1_global_onset', '20200302120000', 'https://example.org/a'),
            ('P1_global_onset', '20200301120000', 'https://example.org/a'),
            ('P2_pre_vaccine', '20201201120000', 'https://example.org/a'),
            ('P1_global_onset', '20200301120000', 'https://example.org/b')])]
    (raw / 'captures.jsonl').write_text(''.join(r.model_dump_json()+'\n' for r in rows))
    calls = []
    def process(client, rec, cache, use_cache, stats):
        calls.append(rec.capture.timestamp)
        stats.extracted += 1
        return module._make_doc(rec, 'ok', extracted=Extracted(text='valid text'))
    monkeypatch.setattr(module, '_process', process)
    out, summary = module.normalize_collection(None, 13559, raw_root=raw.parent,
        output_root=tmp_path/'out', max_captures_per_url=1)
    docs = [json.loads(l) for l in (out/'documents.jsonl').read_text().splitlines()]
    assert len(docs) == 3
    assert '20200302120000' not in calls
    assert summary['capture_selection']['input_captures'] == 4
    assert summary['capture_selection']['selected_captures'] == 3
    assert stage_config('normalize', defaults(), {})['capture_max_per_url'] == 1


def clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr('ingest.archive_it.time.monotonic', lambda: now[0])
    monkeypatch.setattr('ingest.archive_it.time.sleep', lambda n: now.__setitem__(0, now[0]+n))
    return now


def test_pacing_adapts_and_honors_retry_after(monkeypatch):
    now = clock(monkeypatch)
    starts = []
    def respond(req):
        starts.append(now[0])
        return httpx.Response(429, headers={'Retry-After':'7'}) if len(starts)==1 else httpx.Response(200, content=b'ok')
    with ArchiveItClient(request_interval=2) as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(respond))
        assert client.fetch_capture(1, '20200301000000', 'https://example.org/') == b'ok'
        client.fetch_capture(1, '20200301000000', 'https://example.org/')
        assert starts == [100., 107., 110.]
        assert client._interval == 3


def test_retry_after_http_date():
    assert 25 < retry_after_seconds(format_datetime(datetime.now(timezone.utc)+timedelta(seconds=30))) <= 30
    assert retry_after_seconds('invalid') == 0


def test_network_outage_leaves_no_committed_failure(tmp_path, monkeypatch):
    clock(monkeypatch)
    module = importlib.import_module('normalize.normalize')
    source=tmp_path/'captures.jsonl'
    rec=CaptureRecord(collection_id=13559, seed_id=1, seed_url='https://example.org/',
        phase='P1_global_onset', capture=Capture(urlkey='x', timestamp='20200301000000', url='https://example.org/'))
    source.write_text(rec.model_dump_json()+'\n')
    calls=[]
    def respond(req):
        calls.append(req)
        raise httpx.ConnectError('DNS unavailable', request=req)
    with ArchiveItClient() as client:
        client._client.close()
        client._client=httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(ArchiveUnavailable):
            module.normalize_phase(client, source, output_path=tmp_path/'out.jsonl',
                checkpoint_dir=tmp_path/'checkpoint', checkpoint_identity={'test':1})
    assert len(calls)==5
    assert (tmp_path/'checkpoint/journal.jsonl').read_bytes()==b''


def test_exhausted_throttling_interrupts_instead_of_recording_failure(monkeypatch):
    clock(monkeypatch)
    calls=[]
    def respond(req):
        calls.append(req)
        return httpx.Response(429)
    with ArchiveItClient() as client:
        client._client.close()
        client._client=httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(ArchiveUnavailable,match='429'):
            client.fetch_capture(1,'20200301000000','https://example.org/')
    assert len(calls)==5


def test_permanent_http_failure_is_not_retried(monkeypatch):
    clock(monkeypatch)
    calls=[]
    def respond(req):
        calls.append(req)
        return httpx.Response(404)
    with ArchiveItClient() as client:
        client._client.close()
        client._client=httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(httpx.HTTPStatusError):
            client.fetch_capture(1,'20200301000000','https://example.org/')
    assert len(calls)==1


def test_shared_cooldown_survives_new_clients_and_does_not_fetch(tmp_path, monkeypatch):
    from ingest.pacing import ArchivePacing
    now=[1000.]
    monkeypatch.setattr('ingest.pacing.time.time',lambda:now[0])
    path=tmp_path/'pacing.json'
    calls=[]
    def respond(req):
        calls.append(req)
        return httpx.Response(429)
    with ArchiveItClient(pacing_path=path) as client:
        client._client.close()
        client._client=httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(ArchiveUnavailable):client.fetch_capture(1,'20200301000000','https://example.org/')
    state=json.loads(path.read_text())
    assert state['blocked_until']==1300 and state['interval']==8
    with ArchiveItClient(pacing_path=path) as second:
        second._client.close()
        second._client=httpx.Client(transport=httpx.MockTransport(respond))
        with pytest.raises(ArchiveUnavailable,match='cooldown'):second.fetch_capture(2,'20200301000000','https://example.org/')
    assert len(calls)==1
    now[0]=1301.
    assert ArchivePacing(path).acquire()==8
    ArchivePacing(path).result(transient=True,retry_after=900.)
    assert json.loads(path.read_text())['blocked_until']==2201


def test_shared_pacing_recovers_gradually(tmp_path,monkeypatch):
    from ingest.pacing import ArchivePacing
    monkeypatch.setattr('ingest.pacing.time.time',lambda:1000.)
    p=ArchivePacing(tmp_path/'pacing.json')
    p.result(transient=True)
    for _ in range(49):p.result()
    assert json.loads(p.path.read_text())['interval']==8
    assert p.result()==7.2


def test_worker_reports_outage_as_pending_not_failure(tmp_path, monkeypatch):
    import pipeline.worker as worker
    from pipeline.artifacts import write_json
    request=tmp_path/'request.json';result=tmp_path/'result.json'
    write_json(request,{'stage':'normalize'})
    monkeypatch.setattr('sys.argv',['worker','--request',str(request),'--result',str(result)])
    monkeypatch.setattr(worker,'install_shutdown_handlers',lambda:None)
    def unavailable(_):raise ArchiveUnavailable('cooldown')
    monkeypatch.setattr(worker,'execute',unavailable)
    worker.main()
    assert json.loads(result.read_text())==dict(format=1,stage='normalize',status='pending',detail='cooldown')


def test_broken_capture_is_committed_as_failed_and_next_capture_continues(tmp_path,monkeypatch):
    module=importlib.import_module('normalize.normalize')
    now=clock(monkeypatch)
    monkeypatch.setattr('ingest.pacing.time.time',lambda:now[0])
    calls=[]
    def respond(req):
        calls.append(str(req.url))
        if str(req.url).endswith('/broken'):
            raise httpx.RemoteProtocolError('incomplete message body',request=req)
        return httpx.Response(200,content=b'valid html')
    monkeypatch.setattr(module,'extract',lambda html,url:Extracted(text='valid document'))
    rows=[CaptureRecord(collection_id=13559,seed_id=1,seed_url='https://example.org/',
        phase='P1_global_onset',capture=Capture(urlkey='x',timestamp='20200301000000',url='https://example.org/'+name))
        for name in ('broken','good')]
    source=tmp_path/'captures.jsonl';source.write_text(''.join(r.model_dump_json()+'\n' for r in rows))
    with ArchiveItClient(pacing_path=tmp_path/'pacing.json') as client:
        client._client.close();client._client=httpx.Client(transport=httpx.MockTransport(respond))
        output,stats=module.normalize_phase(client,source,output_path=tmp_path/'docs.jsonl',
            checkpoint_dir=tmp_path/'checkpoint',checkpoint_identity={'test':1})
    docs=[json.loads(l) for l in output.read_text().splitlines()]
    assert [d['fetch_status'] for d in docs]==['fetch_failed','ok']
    assert len(calls)==4 and stats.total==2 and stats.fetch_failed==1 and stats.extracted==1
    assert '3 malformed responses' in docs[0]['error']
    assert json.loads((tmp_path/'pacing.json').read_text())['blocked_until']==0


def test_capture_retry_limit_survives_worker_restart(tmp_path,monkeypatch):
    from ingest.pacing import CaptureFetchFailed
    now=clock(monkeypatch)
    monkeypatch.setattr('ingest.pacing.time.time',lambda:now[0])
    calls=[]
    def respond(req):
        calls.append(req)
        raise httpx.RemoteProtocolError('truncated response',request=req)
    for error in (ArchiveUnavailable,ArchiveUnavailable,CaptureFetchFailed,CaptureFetchFailed):
        with ArchiveItClient(pacing_path=tmp_path/'pacing.json') as client:
            client._client.close();client._client=httpx.Client(transport=httpx.MockTransport(respond))
            with pytest.raises(error):client.fetch_capture(1,'20200301000000','https://example.org/broken',max_retries=0)
    assert len(calls)==3


def test_recovered_capture_clears_malformed_response_count(tmp_path,monkeypatch):
    now=clock(monkeypatch)
    monkeypatch.setattr('ingest.pacing.time.time',lambda:now[0])
    calls=[]
    def respond(req):
        calls.append(req)
        if len(calls)==1:raise httpx.RemoteProtocolError('truncated response',request=req)
        return httpx.Response(200,content=b'complete')
    with ArchiveItClient(pacing_path=tmp_path/'pacing.json') as client:
        client._client.close();client._client=httpx.Client(transport=httpx.MockTransport(respond))
        assert client.fetch_capture(1,'20200301000000','https://example.org/')==b'complete'
    assert json.loads((tmp_path/'pacing.json').read_text())['capture_failures']=={}
