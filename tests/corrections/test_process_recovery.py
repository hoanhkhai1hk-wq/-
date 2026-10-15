"""实际硬退出/重启恢复测试。合成fixture，不是模型/农业效果证明。"""
import http.client,json,os,sqlite3,sys,time,unittest
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from agrolake_service.processes import ProcessTree
OUT=ROOT/'reports/corrections/Codex-015'/datetime.now().strftime('process-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)

class RecoveryTests(unittest.TestCase):
    def setUp(self):self.resources=[]
    def tearDown(self):
        for tree,log in reversed(self.resources):
            if tree.process.poll() is None:tree.terminate()
            tree.close();log.close()
    def start(self,mode):
        log=(OUT/f'{mode}.log').open('x',encoding='utf-8')
        tree=ProcessTree([sys.executable,str(ROOT/'tests/corrections/process_fixture.py'),
            '--directory',str(OUT),'--mode',mode],cwd=str(OUT),
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8'),stdout=log)
        self.resources.append((tree,log));end=time.monotonic()+15;ready=OUT/f'ready-{mode}.json'
        while time.monotonic()<end:
            if ready.is_file():
                try:return tree,json.loads(ready.read_text(encoding='utf-8'))['port']
                except ValueError:pass
            if tree.process.poll() is not None:self.fail((OUT/f'{mode}.log').read_text(encoding='utf-8'))
            time.sleep(.05)
        self.fail('fixture启动超时')
    def request(self,port,method,path,payload=None):
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=15)
        try:
            conn.request(method,path,None if payload is None else json.dumps(payload).encode(),{'Content-Type':'application/json'})
            response=conn.getresponse();return response.status,json.loads(response.read())
        finally:conn.close()
    def test_hard_exit_after_job_commit_recovers_without_replay(self):
        first,port=self.start('crash')
        status,job=self.request(port,'POST','/jobs',{'filename':'synthetic.tif','content_base64':'U1lOVEhFVElD',
            'source':'SYNTHETIC process test','scene_id':'CASE','request_id':'parent'})
        self.assertEqual(status,202);parent=job['job_id'];end=time.monotonic()+10
        while time.monotonic()<end:
            _,job=self.request(port,'GET',f'/jobs/{parent}')
            if job['status']=='succeeded':break
            time.sleep(.05)
        self.assertEqual(job['status'],'succeeded')
        self.assertEqual(self.request(port,'POST',f'/jobs/{parent}/governance',{'request_id':'govern'})[0],201)
        status,review=self.request(port,'POST',f'/jobs/{parent}/reviews',{'decision':'questioned','reviewer':'SYNTHETIC',
            'note':'仅软件硬退出恢复测试','request_id':'review'})
        self.assertEqual(status,201)
        body={'request_id':'hard-crash-correction','review_id':review['review_id'],'operator':'SYNTHETIC',
              'reason':'仅合成故障测试','changes':{'scene_id':'CASE-REVISED'}}
        with self.assertRaises((http.client.RemoteDisconnected,ConnectionError,OSError)):
            self.request(port,'POST',f'/jobs/{parent}/corrections',body)
        self.assertEqual(first.process.wait(timeout=10),73)
        marker=json.loads((OUT/'crash-boundary.json').read_text(encoding='utf-8'))
        cid,child=marker['correction_id'],marker['child_job_id']
        conn=sqlite3.connect(OUT/'db.sqlite')
        try:
            before=conn.execute('SELECT COUNT(*) FROM rec_jobs').fetchone()[0]
            self.assertEqual(before,2)
            row=conn.execute('SELECT child_job_id,submission_state FROM rec_corrections').fetchone()
            self.assertEqual(row,(None,'prepared'))
        finally:conn.close()
        second,port2=self.start('restart')
        status,detail=self.request(port2,'GET',f'/corrections/{cid}')
        self.assertEqual(status,200);self.assertIsNone(detail['child'])
        self.assertFalse(list(OUT.glob('calls-restart-*.json')))
        status,resumed=self.request(port2,'POST',f'/corrections/{cid}/resume',{'request_id':body['request_id']})
        self.assertEqual(status,200);self.assertEqual(resumed['child_job_id'],child)
        status,detail=self.request(port2,'GET',f'/corrections/{cid}')
        self.assertEqual(detail['child']['status'],'interrupted')
        self.assertIsNone(detail['comparison']['class_pixel_comparison'])
        self.assertIsNone(detail['comparison']['nodata']['child'])
        self.assertFalse(list(OUT.glob('calls-restart-*.json')))
        conn=sqlite3.connect(OUT/'db.sqlite')
        try:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM rec_jobs').fetchone()[0],before)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM rec_corrections').fetchone()[0],1)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM rec_runs').fetchone()[0],1)
        finally:conn.close()
        (OUT/'verification.json').write_text(json.dumps({'passed':True,'exit_code':73,'jobs':2,'corrections':1,
            'child_status':'interrupted','resume_replayed':False,'synthetic':True},indent=2),encoding='utf-8')

if __name__=='__main__':
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary));print((OUT/'tests.log').read_text(encoding='utf-8'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
