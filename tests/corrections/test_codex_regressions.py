"""Codex独立整改行为探针。合成fixture不代表真实模型或农业精度。"""
import base64,http.client,json,os,sqlite3,sys,threading,time,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests/web')]
from agrolake_service import corrections,governance,reviews,db,store
from agrolake_service.config import ServiceConfig,MAX_CONTENT_BYTES
from agrolake_service.jobs import JobManager
from agrolake_service.validation import validate_job_payload,ValidationError
from agrolake_service.server import make_server
from fixtures import WebFixtureRunner
OUT=ROOT/'reports/corrections/Codex-015'/datetime.now().strftime('regression-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)

class CorrectionRegressions(unittest.TestCase):
    def setUp(self):
        self.out=OUT/self._testMethodName;self.out.mkdir()
        self.env=patch.dict(os.environ,{'AGROLAKE_STORE':str(self.out/'p0-store')});self.env.start()
        self.cfg=ServiceConfig(db_path=self.out/'db.sqlite',store_root=self.out/'store',runs_root=self.out/'runs',
                               port=0,synthetic_mode=True,model_sha256='f'*64)
        self.runner=WebFixtureRunner();self.manager=JobManager(self.cfg,self.runner,auto_start=True)
        self.parent=self.manager.submit(validate_job_payload({'filename':'synthetic.tif','content_base64':'U1lOVEhFVElD',
            'source':'SYNTHETIC 软件测试','scene_id':'CASE'}))['job_id']
        self.wait(self.parent);self.review=self.problem(self.parent)
    def tearDown(self):self.manager.stop();self.env.stop()
    def wait(self,job):
        end=time.monotonic()+10
        while time.monotonic()<end:
            result=self.manager.get_job(job)
            if result['status'] not in ('queued','running'):
                self.assertEqual(result['status'],'succeeded');return result
            time.sleep(.02)
        self.fail('fixture作业未完成')
    def problem(self,job):
        governance.register_governance(self.cfg,job,{'request_id':f'govern-{job}'})
        return reviews.submit_review(self.cfg.db_path,job,{'decision':'questioned','reviewer':'SYNTHETIC 流程署名',
                'note':'仅软件行为测试','request_id':f'review-{job}'})['record']['review_id']
    def body(self,**changes):
        body={'request_id':'codex-correction','review_id':self.review,'operator':'SYNTHETIC',
              'reason':'软件整改测试','changes':{'scene_id':'CASE-REVISED'}}
        body.update(changes);return body
    def submit(self,body=None,parent=None):
        return corrections.submit_correction(self.manager,parent or self.parent,body or self.body())['record']
    def test_equal_pixel_totals_do_not_verify_pixel_alignment(self):
        record=self.submit(self.body(replacement={'filename':'replacement.tif','content_base64':base64.b64encode(b'REPLACED').decode()}))
        self.wait(record['child_job_id']);result=corrections.get_correction(self.manager,record['correction_id'])
        self.assertFalse(result['comparison']['class_pixel_comparison']['comparable_pixel_space'],
                         '像素总量相等不能证明形状/空间对应')
    def test_missing_artifact_checksums_prevent_completed_comparison(self):
        record=self.submit();child=record['child_job_id'];self.wait(child)
        conn=db.connect(self.cfg.db_path)
        try:
            conn.execute("UPDATE rec_artifacts SET name='MISSING-'||name WHERE job_id=?",(child,));conn.commit()
        finally:conn.close()
        result=corrections.get_correction(self.manager,record['correction_id'])
        self.assertIsNone(result['comparison']['class_pixel_comparison'])
        self.assertIsNone(result['comparison']['nodata']['child'])
    def test_submission_error_does_not_export_runtime_paths(self):
        with patch.object(self.manager,'submit',side_effect=OSError('D:/PRIVATE/input.tif')):
            with self.assertRaises(ValidationError):self.submit()
        result=corrections.list_corrections(self.manager,self.parent)
        self.assertNotIn('D:/PRIVATE',json.dumps(result,ensure_ascii=False))
    def test_lost_reply_confirms_existing_child_even_if_service_model_changed(self):
        original=self.manager.submit
        def lose(normalized):original(normalized);raise OSError('SYNTHETIC ACK LOST')
        with patch.object(self.manager,'submit',side_effect=lose):
            with self.assertRaises(ValidationError):self.submit()
        conn=db.connect(self.cfg.db_path)
        try:child=conn.execute('SELECT MAX(job_id) FROM rec_jobs').fetchone()[0]
        finally:conn.close()
        self.wait(child);before=self.runner.calls;self.cfg.model_sha256='b'*64
        record=self.submit()
        self.assertEqual(record['child_job_id'],child);self.assertEqual(self.runner.calls,before)
    def test_synthetic_context_survives_a_public_child_revision(self):
        first=self.submit();child=first['child_job_id'];self.wait(child)
        conn=db.connect(self.cfg.db_path)
        try:
            # 合成软件fixture模拟修复输入恰为公开摘要，沿链仍须标明人工异常背景。
            conn.execute("UPDATE rec_jobs SET sample_kind='public_adapter_sample' WHERE job_id=?",(child,));conn.commit()
        finally:conn.close()
        review=self.problem(child)
        next_record=self.submit(self.body(request_id='codex-correction-next',review_id=review,
                    changes={'scene_id':'CASE-NEXT'}),parent=child)
        self.assertTrue(next_record['is_synthetic_case'])
    def test_stored_object_size_is_bounded_before_loading(self):
        data=b'SYNTHETIC';sha=store.sha256_bytes(data);target=store.object_path(self.cfg.store_root,sha)
        original=Path.stat
        def oversized(path,*args,**kw):
            result=original(path,*args,**kw)
            if path==target:
                fields=list(result);fields[6]=MAX_CONTENT_BYTES+1;return os.stat_result(fields)
            return result
        with patch.object(Path,'stat',oversized),patch.object(Path,'read_bytes',side_effect=AssertionError('不应分配超限内容')):
            with self.assertRaises(ValidationError):corrections._read_stored_object(self.cfg.store_root,sha)

    def test_blank_optional_metadata_does_not_create_a_revision(self):
        with self.assertRaises(ValidationError):
            self.submit(self.body(changes={'source_link':'   ','license_status':''}))
        self.assertEqual(corrections.list_corrections(self.manager,self.parent)['total'],0)

    def test_child_binding_cannot_point_back_to_parent(self):
        with patch.object(self.manager,'submit',return_value={'job_id':self.parent}):
            with self.assertRaises(ValidationError):self.submit()
        row=corrections.list_corrections(self.manager,self.parent)['corrections'][0]
        self.assertIsNone(row['child_job_id']);self.assertEqual(row['submission_state'],'prepared')

    def test_http_resume_after_reload_and_bounded_upload(self):
        server=make_server(self.cfg,self.manager)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def request(path,body):
            conn=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=5)
            try:
                conn.request('POST',path,json.dumps(body),{'Content-Type':'application/json'})
                response=conn.getresponse();return response.status,json.loads(response.read())
            finally:conn.close()
        try:
            with patch.object(self.manager,'submit',side_effect=OSError('SYNTHETIC NOT ACCEPTED')):
                with self.assertRaises(ValidationError):self.submit()
            record=corrections.list_corrections(self.manager,self.parent)['corrections'][0]
            path=f"/corrections/{record['correction_id']}/resume"
            self.assertEqual(request(path,{'request_id':'wrong-request'})[0],409)
            self.assertEqual(request(path,{'request_id':record['request_id'],'extra':True})[0],400)
            status,first=request(path,{'request_id':record['request_id']})
            self.assertEqual(status,200);child=first['child_job_id'];self.wait(child)
            status,second=request(path,{'request_id':record['request_id']})
            self.assertEqual(status,200);self.assertEqual(second['child_job_id'],child)
            self.cfg.max_content_bytes=3
            body=self.body(request_id='bounded-case',replacement={'filename':'over.tif','content_base64':'MTIzNA=='})
            self.assertEqual(request(f'/jobs/{self.parent}/corrections',body)[0],413)
            self.assertEqual(corrections.list_corrections(self.manager,self.parent)['total'],1)
            self.assertEqual(self.runner.calls,2)
        finally:
            server.shutdown();server.server_close();thread.join(5)

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CorrectionRegressions))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'skipped':len(result.skipped),'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));print((OUT/'tests.log').read_text(encoding='utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
