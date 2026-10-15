"""Codex独立行为回归；所有fixture均为合成，不是农业精度或专家标注。"""
import json,os,sqlite3,sys,threading,time,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/web'))
from agrolake_web.client import RecognitionClient,ApiError
from agrolake_web import review_panel as panel
from agrolake_service import reviews
from agrolake_service.config import ServiceConfig
from agrolake_service.jobs import JobManager
from agrolake_service.server import make_server
from agrolake_service.validation import validate_job_payload
from fixtures import WebFixtureRunner
from streamlit.testing.v1 import AppTest
OUT=ROOT/'reports/reviews/Codex-013-review'/datetime.now().strftime('regression-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)

def record(job=1,field=None,request='test-request'):
    return {'review_id':1,'job_id':job,'field_id':field,'target':'task' if field is None else 'region',
        'decision':'questioned','decision_zh':'存疑待核实','reviewer':'SYNTHETIC','note':'SYNTHETIC',
        'request_id':request,'input_sha256':'a'*64,'model_sha256':'b'*64,'run_seq':1,'created_at':'2026-10-15T00:00:00+00:00'}
def page(row=None):
    rows=[] if row is None else [row]
    return {'job_id':1,'reviews':rows,'latest':{'task':row,'regions':{}},'total':len(rows),
        'pagination':{'after_id':0,'until_id':max((r['review_id'] for r in rows),default=0),
                      'limit':200,'returned':len(rows),'has_more':False,'next_after_id':None}}

class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{});self.env.start();self.resources=[]
        self.out=OUT/self._testMethodName;self.out.mkdir()
    def tearDown(self):
        for s,t,m in reversed(self.resources):s.shutdown();s.server_close();t.join(5);m.stop()
        self.env.stop()
    def fixture(self):
        cfg=ServiceConfig(db_path=self.out/'db.sqlite',store_root=self.out/'store',runs_root=self.out/'runs',port=0,synthetic_mode=True)
        manager=JobManager(cfg,WebFixtureRunner(),auto_start=True)
        server=make_server(cfg,manager);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.resources.append((server,thread,manager))
        os.environ.update(AGROLAKE_API_URL=f'http://127.0.0.1:{server.server_address[1]}',AGROLAKE_DB=str(cfg.db_path),
            AGROLAKE_DATA_DIR=str(self.out),AGROLAKE_STORE=str(self.out/'p0-store'),AGROLAKE_REPORTS=str(self.out/'exports'))
        payload={'filename':'synthetic.tif','content_base64':'U1lOVEhFVElD','source':'SYNTHETIC 软件行为','scene_id':'TEST'}
        job=manager.submit(validate_job_payload(payload))['job_id']
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if manager.get_job(job)['status']=='succeeded':return manager,job
            time.sleep(.03)
        self.fail('fixture did not finish')
    def test_opinion_A_B_A_is_three_events_but_double_click_is_one(self):
        manager,job=self.fixture();at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        label=next(o for o in at.selectbox(key='rec_history').options if o.startswith(f'#{job} '))
        at.selectbox(key='rec_history').set_value(label).run()
        at.text_input(key=f'rec_review_reviewer_task_{job}').set_value('SYNTHETIC 流程署名')
        for note in ['意见A','意见B','意见A']:
            at.text_area(key=f'rec_review_note_task_{job}').set_value(note)
            at.button(key=f'rec_review_save_task_{job}').click().run()
            self.assertFalse(at.exception)
        self.assertEqual(reviews.list_reviews(manager.db_path,job)['total'],3)
        at.button(key=f'rec_review_save_task_{job}').click().run()
        self.assertEqual(reviews.list_reviews(manager.db_path,job)['total'],3)
        self.assertEqual(reviews.list_reviews(manager.db_path,job)['latest']['task']['note'],'意见A')
    def test_cross_job_rows_rejected(self):
        with patch.object(RecognitionClient,'_request_json',return_value=page(record(job=2))):
            with self.assertRaises(ApiError):RecognitionClient().list_reviews(1)
    def test_malformed_latest_rejected(self):
        p=page();p['latest']=['bad']
        with patch.object(RecognitionClient,'_request_json',return_value=p):
            with self.assertRaises(ApiError):RecognitionClient().list_reviews(1)
    def test_submit_reply_must_match_request_and_region(self):
        response=record(field=99,request='wrong');response['created']=True
        with patch.object(RecognitionClient,'_request_json',return_value=response):
            with self.assertRaises(ApiError):RecognitionClient().submit_review(1,field_id=1,decision='questioned',reviewer='SYNTHETIC',note='SYNTHETIC',request_id='expected')
    def test_missing_next_cursor_cannot_claim_complete(self):
        p=page(record());p['pagination'].update(has_more=True,next_after_id=None);p['total']=2
        class Client:
            def list_reviews(self,*a,**kw):return p
        try:rows,complete=panel._fetch_all(Client(),1)
        except ApiError:return
        self.assertFalse(complete)
    def test_csv_formula_after_bom_or_whitespace_is_escaped(self):
        for text in ['=1+1','\ufeff=1+1',' \t=1+1','\n@SUM(A1:A2)']:
            with self.subTest(value=repr(text)):self.assertTrue(panel._csv_safe(text).startswith("'"))
    def test_export_snapshot_excludes_concurrent_later_append(self):
        manager,job=self.fixture();client=RecognitionClient(os.environ['AGROLAKE_API_URL'])
        def submit(key):return client.submit_review(job,decision='questioned',reviewer='SYNTHETIC',note='流程测试',request_id=key)
        for i in range(3):submit(f'snapshot-{i}')
        calls=[]
        class ConcurrentClient:
            def list_reviews(self,job_id,**kw):
                kw['limit']=1;result=client.list_reviews(job_id,**kw);calls.append(result['pagination']['until_id'])
                if len(calls)==1:submit('snapshot-later')
                return result
        rows,complete=panel._fetch_all(ConcurrentClient(),job)
        self.assertTrue(complete);self.assertEqual(len(rows),3)
        self.assertEqual(len(set(calls)),1)
        self.assertEqual(client.list_reviews(job)['total'],4)
    def test_capacity_rejects_new_but_allows_same_id_retry_and_export(self):
        manager,job=self.fixture();client=RecognitionClient(os.environ['AGROLAKE_API_URL'])
        args=dict(decision='questioned',reviewer='SYNTHETIC',note='容量行为测试',request_id='capacity-original')
        client.submit_review(job,**args)
        conn=sqlite3.connect(manager.db_path);conn.row_factory=sqlite3.Row
        try:
            row=dict(conn.execute('SELECT * FROM rec_reviews').fetchone());row.pop('review_id')
            columns=list(row)
            for i in range(499):
                row['request_id']=f'capacity-{i}'
                conn.execute(f'INSERT INTO rec_reviews({",".join(columns)}) VALUES({",".join("?" for _ in columns)})',[row[k] for k in columns])
            conn.commit()
        finally:conn.close()
        with self.assertRaises(ApiError) as caught:client.submit_review(job,**dict(args,request_id='capacity-overflow'))
        self.assertEqual(caught.exception.status,409)
        self.assertFalse(client.submit_review(job,**args)['created'])
        rows,complete=panel._fetch_all(client,job)
        self.assertTrue(complete);self.assertEqual(len(rows),500)
    def test_repeated_opinion_lost_reply_keeps_new_request(self):
        manager,job=self.fixture();at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        label=next(o for o in at.selectbox(key='rec_history').options if o.startswith(f'#{job} '))
        at.selectbox(key='rec_history').set_value(label).run()
        at.text_input(key=f'rec_review_reviewer_task_{job}').set_value('SYNTHETIC')
        for note in ['意见A','意见B']:
            at.text_area(key=f'rec_review_note_task_{job}').set_value(note)
            at.button(key=f'rec_review_save_task_{job}').click().run()
        original=RecognitionClient.submit_review;seen=[]
        def lost(client,job_id,**kw):
            seen.append(kw['request_id']);result=original(client,job_id,**kw)
            if len(seen)==1:raise ApiError('timeout','SYNTHETIC accepted but response lost')
            return result
        with patch.object(RecognitionClient,'submit_review',lost):
            at.text_area(key=f'rec_review_note_task_{job}').set_value('意见A')
            at.button(key=f'rec_review_save_task_{job}').click().run()
            self.assertEqual(reviews.list_reviews(manager.db_path,job)['total'],3)
            at.button(key=f'rec_review_retry_task_{job}').click().run()
        self.assertFalse(at.exception);self.assertEqual(len(seen),2);self.assertEqual(seen[0],seen[1])
        self.assertEqual(reviews.list_reviews(manager.db_path,job)['total'],3)

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewRegressions))
    summary={'tests':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));print((OUT/'tests.log').read_text(encoding='utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
