"""Codex验收发现问题的行为回归，故障HTTP/合成runner不代表真实模型。"""
import http.server,json,os,sys,threading,time,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agrolake_web.client import RecognitionClient,ApiError,ApiConfigError
from agrolake_service.config import ServiceConfig
from agrolake_service.jobs import JobManager
from agrolake_service.server import make_server
from streamlit.testing.v1 import AppTest
from fixtures import WebFixtureRunner
OUT=ROOT/'reports/web/Codex-012-review'/datetime.now().strftime('regression-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)

class FaultHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        mode=self.server.mode
        content=self.server.content
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(content)+(50 if mode in ('timeout','truncated') else 0)))
        self.end_headers()
        if mode=='timeout':
            self.wfile.write(b'{');self.wfile.flush();time.sleep(.3)
        else:self.wfile.write(content)
        self.close_connection=True

class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.out=OUT/self._testMethodName;self.out.mkdir();self.resources=[]
        self.env=patch.dict(os.environ,{});self.env.start()
    def tearDown(self):
        for server,thread,manager in reversed(self.resources):
            server.shutdown();server.server_close();thread.join(5)
            if manager is not None:manager.stop()
        self.env.stop()
    def fault(self,content,mode='normal'):
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),FaultHandler)
        server.mode=mode;server.content=content
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.resources.append((server,thread,None))
        return RecognitionClient(f'http://127.0.0.1:{server.server_address[1]}',timeout=.08)
    def fixture(self,mode='success'):
        cfg=ServiceConfig(db_path=self.out/'db.sqlite',store_root=self.out/'rec-store',runs_root=self.out/'runs',port=0,synthetic_mode=True)
        manager=JobManager(cfg,WebFixtureRunner(mode),auto_start=True)
        server=make_server(cfg,manager);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.resources.append((server,thread,manager))
        os.environ.update(AGROLAKE_API_URL=f'http://127.0.0.1:{server.server_address[1]}',AGROLAKE_DB=str(cfg.db_path),
            AGROLAKE_DATA_DIR=str(self.out),AGROLAKE_STORE=str(self.out/'p0-store'),AGROLAKE_REPORTS=str(self.out/'exports'))
        return manager
    def app(self):
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.assertFalse(at.exception);return at
    def terminal(self,manager):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            rows=manager.list_jobs()
            if rows and rows[0]['status'] not in ('queued','running'):return rows[0]
            time.sleep(.03)
        self.fail('fixture job did not finish')
    def test_body_read_timeout_is_categorized(self):
        with self.assertRaises(ApiError) as caught:self.fault(b'{"jobs":[]}',mode='timeout').list_jobs()
        self.assertEqual(caught.exception.category,'timeout')
    def test_truncated_valid_json_is_rejected(self):
        with self.assertRaises(ApiError) as caught:self.fault(b'{"jobs":[]}',mode='truncated').list_jobs()
        self.assertEqual(caught.exception.category,'protocol')
    def test_non_object_and_invalid_json_values(self):
        for content in (b'[]',b'1',b'null',b'NaN'):
            with self.subTest(content=content),self.assertRaises(ApiError):self.fault(content).health()
    def test_mismatched_or_incomplete_job_response(self):
        for value in ({'job_id':6,'status':'succeeded'},{'job_id':5,'status':'unknown'},
                      {'job_id':5,'status':'succeeded','artifact_checksums':{'mask.tif':'bad'}},
                      {'job_id':True,'status':'queued'}):
            with self.subTest(value=value),self.assertRaises(ApiError):
                self.fault(json.dumps(value).encode()).get_job(5)
        with self.assertRaises(ApiError):self.fault(b'{"jobs":[{"job_id":1}]}').list_jobs()
    def test_nonfinite_timeouts_and_bad_limits_rejected(self):
        for timeout in (0,-1,float('nan'),float('inf')):
            with self.subTest(timeout=timeout),self.assertRaises(ApiConfigError):RecognitionClient(timeout=timeout)
        with self.assertRaises(ApiConfigError):RecognitionClient(max_json_bytes=-1)
        with self.assertRaises(ApiConfigError):RecognitionClient('https://127.0.0.1:8506')
    def test_failure_can_explicitly_start_new_job(self):
        manager=self.fixture('failure');at=self.app();at.button(key='rec_submit').click().run()
        first=self.terminal(manager);at.run()
        at.button(key='rec_submit').click().run()
        self.assertEqual(len(manager.list_jobs()),1)
        at.button(key='rec_new_run').click().run();second=self.terminal(manager)
        self.assertFalse(at.exception);self.assertEqual(len(manager.list_jobs()),2)
        self.assertNotEqual(first['request_id'],second['request_id'])
        self.assertEqual(first['input_sha256'],second['input_sha256'])
    def test_lost_submit_reply_retries_same_request(self):
        manager=self.fixture();at=self.app();original=RecognitionClient.submit_job;seen=[]
        def lose_first(client,payload,**kw):
            seen.append(payload['request_id']);result=original(client,payload,**kw)
            if len(seen)==1:raise ApiError('timeout','SYNTHETIC reply lost after acceptance')
            return result
        with patch.object(RecognitionClient,'submit_job',lose_first):
            at.button(key='rec_submit').click().run()
            self.assertFalse(at.exception);self.assertEqual(len(manager.list_jobs()),1)
            self.assertIsNone(at.session_state['rec_selected_job'])
            at.button(key='rec_retry').click().run()
        self.assertFalse(at.exception);self.assertEqual(len(manager.list_jobs()),1)
        self.assertEqual(len(seen),2);self.assertEqual(seen[0],seen[1])
    def test_lost_new_run_reply_disables_further_new_requests(self):
        manager=self.fixture('failure');at=self.app()
        at.button(key='rec_submit').click().run();self.terminal(manager);at.run()
        original=RecognitionClient.submit_job;seen=[]
        def lose_first(client,payload,**kw):
            seen.append(payload['request_id']);result=original(client,payload,**kw)
            if len(seen)==1:raise ApiError('protocol','SYNTHETIC new-run response truncated')
            return result
        with patch.object(RecognitionClient,'submit_job',lose_first):
            at.button(key='rec_new_run').click().run()
            self.assertFalse(at.exception);self.assertEqual(len(manager.list_jobs()),2)
            self.assertTrue(at.button(key='rec_new_run').disabled)
            at.button(key='rec_retry').click().run()
        self.assertFalse(at.exception);self.assertEqual(len(manager.list_jobs()),2)
        self.assertEqual(seen[0],seen[1])
    def test_missing_checksum_hides_success_artifacts(self):
        manager=self.fixture();at=self.app();at.button(key='rec_submit').click().run();self.terminal(manager)
        original=RecognitionClient.get_job
        def incomplete(client,job_id):
            result=original(client,job_id);result.pop('artifact_checksums',None);return result
        with patch.object(RecognitionClient,'get_job',incomplete):at.run()
        self.assertFalse(at.exception);self.assertTrue(any('校验记录' in e.value for e in at.error))
        self.assertEqual(len(at.get('download_button')),0)
    def test_reconnect_after_offline_page_keeps_p0_usable(self):
        self.fixture();original=RecognitionClient.health;offline=True
        def health(client):
            if offline:raise ApiError('offline','SYNTHETIC service unavailable')
            return original(client)
        with patch.object(RecognitionClient,'health',health):
            at=self.app();self.assertEqual(len(at.tabs),8)
            self.assertTrue(any('不可用' in w.value for w in at.warning))
            offline=False;at.button(key='rec_reconnect').click().run()
        self.assertFalse(at.exception);self.assertIsNotNone(at.button(key='rec_submit'))

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewRegressions))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));print((OUT/'tests.log').read_text(encoding='utf-8'))
    sys.exit(0 if result.wasSuccessful() else 1)
