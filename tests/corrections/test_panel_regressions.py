"""Codex独立页面行为回归；真实HTTP/SQLite配合合成runner，无农业精度含义。"""
import json,os,re,sqlite3,sys,threading,time,unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent),str(ROOT/'tests/web')]
import test_correction_panel_apptest as helpers
from agrolake_service import corrections,governance,reviews
from agrolake_service.config import ServiceConfig
from agrolake_service.server import make_server
from agrolake_service.validation import ValidationError
from fixtures import WebFixtureRunner
OUT=ROOT/'reports/corrections/Codex-015'/datetime.now().strftime('panel-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)

class PanelTests(unittest.TestCase):
    def setUp(self):
        self.out=OUT/self._testMethodName;self.out.mkdir()
        self.env=patch.dict(os.environ);self.env.start()
        self.cfg=ServiceConfig(db_path=self.out/'db.sqlite',store_root=self.out/'store',runs_root=self.out/'runs',
                               port=0,synthetic_mode=True,model_sha256='f'*64)
        self.manager=helpers.CountingManager(self.cfg,WebFixtureRunner('success'),auto_start=True)
        self.server=make_server(self.cfg,self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        helpers.set_env(self.out,f'http://127.0.0.1:{self.server.server_address[1]}')
        self.parent=helpers.make_job(self.manager,b'INDEPENDENT-PANEL')
        if 'ungoverned' not in self._testMethodName:
            governance.register_governance(self.cfg,self.parent,{'request_id':'independent-governance'})
        self.review=reviews.submit_review(self.cfg.db_path,self.parent,{'request_id':'independent-opinion',
            'decision':'questioned','reviewer':'SYNTHETIC reviewer','note':'软件流程问题非真值'})['record']['review_id']
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(5);self.manager.stop();self.env.stop()
    def app(self):
        at=helpers.new_app();helpers.select_job(at,self.parent)
        self.assertFalse(helpers.exceptions(at));return at
    def prepare(self,at,scene):
        at.text_input(key=f'rec_corr_operator_{self.parent}').set_value('SYNTHETIC operator')
        at.text_area(key=f'rec_corr_reason_{self.parent}').set_value('独立行为验证，非田间业务证据')
        at.text_input(key=f'rec_corr_scene_{self.parent}').set_value(scene)
    def create(self,at,scene):
        self.prepare(at,scene);at.button(key=f'rec_corr_create_{self.parent}').click().run()
        self.assertFalse(helpers.exceptions(at));return self.rows()
    def rows(self):
        conn=sqlite3.connect(self.cfg.db_path);conn.row_factory=sqlite3.Row
        try:return [dict(r) for r in conn.execute('SELECT * FROM rec_corrections ORDER BY correction_id')]
        finally:conn.close()
    def test_a_b_a_produces_three_versions_and_same_click_reuses(self):
        at=self.app();first=self.create(at,'CASE-A')[0]
        helpers.wait_child(self.manager,first['child_job_id']);at.run()
        self.create(at,'CASE-A');self.assertEqual(len(self.rows()),1)
        self.create(at,'CASE-B');self.create(at,'CASE-A')
        rows=self.rows();self.assertEqual(len(rows),3)
        self.assertEqual(len({r['request_id'] for r in rows}),3)
        self.assertEqual([r['revision_no'] for r in rows],[1,2,3])
    def test_http503_prepared_blocks_new_intent(self):
        at=self.app()
        with patch.object(self.manager,'submit',side_effect=OSError('SYNTHETIC SUBMIT UNAVAILABLE')):
            rows=self.create(at,'CASE-UNKNOWN')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['submission_state'],'prepared')
        at.run()
        # 合法实现可以隐藏或禁用创建表单；只在表单仍存在时尝试修改。
        if any(e.key==f'rec_corr_scene_{self.parent}' for e in at.text_input):
            self.prepare(at,'CASE-DIFFERENT');at.run()
        buttons=[b for b in at.button if b.key==f'rec_corr_create_{self.parent}']
        if buttons and not buttons[0].disabled:buttons[0].click().run()
        self.assertFalse(helpers.exceptions(at));self.assertEqual(len(self.rows()),1)
        self.assertEqual(self.rows()[0]['request_id'],rows[0]['request_id'])
    def test_full_reload_resume_only_original_prepared(self):
        body={'request_id':'stored-original-request','review_id':self.review,'operator':'SYNTHETIC',
              'reason':'仅恢复已持久化意图','changes':{'scene_id':'PERSISTED'}}
        with patch.object(self.manager,'submit',side_effect=OSError('SYNTHETIC')):
            with self.assertRaises(ValidationError):corrections.submit_correction(self.manager,self.parent,body)
        row=self.rows()[0];at=self.app()
        self.assertEqual(self.manager.submit_calls,1)
        at.button(key=f"rec_corr_resume_{row['correction_id']}").click().run()
        self.assertFalse(helpers.exceptions(at));child=self.rows()[0]['child_job_id']
        self.assertIsNotNone(child);helpers.wait_child(self.manager,child)
        before=self.manager.submit_calls
        for _ in range(3):at.run()
        self.assertFalse(helpers.exceptions(at));self.assertEqual(self.manager.submit_calls,before)
        self.assertEqual(len(self.rows()),1);self.assertEqual(self.rows()[0]['request_id'],body['request_id'])
    def test_ungoverned_parent_has_no_create_form(self):
        at=self.app()
        self.assertFalse([b for b in at.button if b.key==f'rec_corr_create_{self.parent}'])
        self.assertTrue(any('登记' in str(e.value) for e in [*at.info,*at.warning,*at.caption]))
    def test_nodata_fraction_and_ratio_are_visible(self):
        at=self.app();row=self.create(at,'CASE-DISPLAY')[0]
        helpers.wait_child(self.manager,row['child_job_id']);at.run()
        text=' '.join(str(e.value) for e in [*at.markdown,*at.caption,*at.info,*at.metric])
        self.assertRegex(text,r'2\s*/\s*12')
        self.assertRegex(text,r'16\.6\d*\s*%')
        self.assertFalse(helpers.exceptions(at))

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PanelTests))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary));print((OUT/'tests.log').read_text(encoding='utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
