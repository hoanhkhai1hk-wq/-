"""独立行为探针；合成fixture仅验证软件，不代表真实模型或农业效果。"""
import copy
import json
import os
import sqlite3
import sys
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests/web')]
from agrolake import db as p0_db, disposition
from agrolake_service import db, governance, reviews, store
from agrolake_service.config import ServiceConfig
from agrolake_service.jobs import JobManager
from agrolake_service.server import make_server
from agrolake_service.validation import validate_job_payload
from agrolake_web.client import RecognitionClient, ApiError
from fixtures import WebFixtureRunner

OUT = ROOT/'reports/governance/Codex-014'/datetime.now().strftime('regression-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True, exist_ok=False)

class GovernanceRegressions(unittest.TestCase):
    def setUp(self):
        self.out=OUT/self._testMethodName;self.out.mkdir()
        self.env=patch.dict(os.environ, {'AGROLAKE_STORE':str(self.out/'p0-store'),
                                        'AGROLAKE_DATA_DIR':str(self.out/'p0')})
        self.env.start()
        self.cfg=ServiceConfig(db_path=self.out/'db.sqlite',store_root=self.out/'store',
                               runs_root=self.out/'runs',port=0,synthetic_mode=True)
        self.runner=WebFixtureRunner()
        self.manager=JobManager(self.cfg,self.runner,auto_start=True)
        self.server=make_server(self.cfg,self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.client=RecognitionClient(f'http://127.0.0.1:{self.server.server_address[1]}')
        self.job=self.manager.submit(validate_job_payload({'filename':'synthetic.tif','content_base64':'U1lOVEhFVElD',
            'source':'SYNTHETIC 软件行为','source_link':'https://example.org/fixture','scene_id':'TEST'}))['job_id']
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if self.manager.get_job(self.job)['status']=='succeeded':break
            time.sleep(.03)
        self.assertEqual(self.manager.get_job(self.job)['status'],'succeeded')
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5);self.manager.stop();self.env.stop()
    def register(self):
        return self.client.register_governance(self.job,request_id='codex-register')
    def test_unregistered_GET_does_not_initialize_or_write_P0(self):
        conn=db.connect(self.cfg.db_path)
        try:before=list(conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        finally:conn.close()
        raw_before=self.cfg.db_path.read_bytes()
        result=self.client.get_lineage(self.job)
        self.assertFalse(result['registered'])
        conn=db.connect(self.cfg.db_path)
        try:after=list(conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        finally:conn.close()
        self.assertEqual([r[0] for r in after],[r[0] for r in before], 'GET不得建P0表')
        self.assertEqual(self.cfg.db_path.read_bytes(),raw_before,'GET不得写数据库')
    def test_latest_quality_controls_current_overall(self):
        reg=self.register();vid=reg['version_id']
        conn=db.connect(self.cfg.db_path)
        try:
            for status in ('block','pass'):
                conn.execute('INSERT INTO quality_checks(version_id,rule_id,rule_version,status,reason,checked_at) VALUES (?,?,?,?,?,?)',
                             (vid,'required_source','SYNTHETIC',status,'流程质检历史',db.now()))
            conn.commit()
        finally:conn.close()
        result=self.client.get_lineage(self.job)
        self.assertEqual(result['registration']['current']['quality']['overall'],'pass')
    def test_lineage_with_real_review_rows_is_usable_by_client(self):
        self.register()
        self.client.submit_review(self.job,decision='questioned',reviewer='SYNTHETIC 流程署名',
                                  note='仅接口行为测试',request_id='codex-review')
        result=self.client.get_lineage(self.job)
        self.assertEqual(result['reviews']['total'],1)
        self.assertEqual(result['reviews']['items'][0]['job_id'],self.job)
    def test_client_rejects_cross_object_input_and_version(self):
        self.register();valid=self.client.get_lineage(self.job)
        mutations=[lambda x:x['registration'].__setitem__('input_sha256','c'*64),
                   lambda x:x['registration']['current']['version'].__setitem__('object_id','c'*64),
                   lambda x:x.__setitem__('job',[])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                broken=copy.deepcopy(valid);mutate(broken)
                with patch.object(self.client,'_request_json',return_value=broken):
                    with self.assertRaises(ApiError):self.client.get_lineage(self.job)
    def test_truncated_history_has_no_download(self):
        self.register();initial=self.client.get_lineage(self.job)
        conn=p0_db.connect(self.cfg.db_path)
        try:
            for source in ('SYNTHETIC 处置A','SYNTHETIC 处置B'):
                disposition.set_source(conn,initial['registration']['version_id'],source,
                                       'SYNTHETIC','仅软件边界测试',self.out/'p0-store')
        finally:conn.close()
        with patch.object(governance,'MAX_LINEAGE_HISTORY',1):
            result=self.client.get_lineage(self.job)
        self.assertTrue(result['registration']['current']['history']['truncated'])
        # 不取缺行且宣称完整的下载；只检查行为，不镜像序列化代码。
        from streamlit.testing.v1 import AppTest
        with patch.object(RecognitionClient,'get_lineage',return_value=result):
            app=AppTest.from_string('from agrolake_web.governance_panel import render\n'
                                   'from agrolake_web.client import RecognitionClient\n'
                                   'render(RecognitionClient(), {"job_id":1}, [])').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('download_button')),0,'截断报告必须隐藏完整下载')
    def test_safe_publish_occurs_in_P0_root_before_importer(self):
        original=governance.p0_ingest.import_bytes
        observations=[]
        def checked_import(conn,data,filename,**kw):
            target=store.object_path(Path(kw['store_root']),store.sha256_bytes(data))
            observations.append(target.is_file() and store.sha256_file(target)==store.sha256_bytes(data))
            return original(conn,data,filename,**kw)
        with patch.object(governance.p0_ingest,'import_bytes',side_effect=checked_import):self.register()
        self.assertEqual(observations,[True],'P0 importer只能复用已经安全发布的对象')

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GovernanceRegressions))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'skipped':len(result.skipped),'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));print((OUT/'tests.log').read_text(encoding='utf-8'))
    sys.exit(0 if result.wasSuccessful() else 1)
