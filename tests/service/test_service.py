"""服务行为验收；合成fixture不证明模型精度。所有测试资料留在reports，不删除。"""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import http.client,io,json,os,sqlite3,subprocess,sys,threading,time,unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agrolake_service import config,store,db
from agrolake_service.jobs import JobManager,IdempotencyConflict
from agrolake_service.ownership import OwnershipError
from agrolake_service.runner import CliRunner
from agrolake_service.server import make_server
from agrolake_service.validation import validate_job_payload,ValidationError
from support import payload,artifacts,FixtureRunner,wait_job

OUT=ROOT/'reports/service/DEV-20261015-011'/datetime.now().strftime('unit-%Y%m%d-%H%M%S-%f')
OUT.mkdir(parents=True,exist_ok=False)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory=OUT/self._testMethodName;self.directory.mkdir()
        self.cfg=config.ServiceConfig(db_path=self.directory/'test.sqlite',store_root=self.directory/'store',
            runs_root=self.directory/'runs',port=0,synthetic_mode=True,model_sha256='f'*64,
            max_body_bytes=4096,max_content_bytes=1024)
        self.managers=[];self.servers=[]

    def tearDown(self):
        for server,thread in self.servers:
            server.shutdown();server.server_close();thread.join(5)
        for manager in reversed(self.managers):
            manager.stop()

    def manager(self,runner=None,auto_start=False):
        manager=JobManager(self.cfg,runner or FixtureRunner(),auto_start=auto_start)
        self.managers.append(manager);return manager

    def http(self,auto_start=True,runner=None):
        manager=self.manager(runner,auto_start)
        server=make_server(self.cfg,manager)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.servers.append((server,thread));self.port=server.server_address[1]
        return manager

    def request(self,method,path,body=None,headers=None):
        headers=dict(headers or {})
        if isinstance(body,dict):
            body=json.dumps(body).encode();headers.setdefault('Content-Type','application/json')
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=10)
        try:
            conn.request(method,path,body,headers)
            response=conn.getresponse();raw=response.read()
            return response.status,raw
        finally:
            conn.close()

    def submit(self,manager,**changes):
        return manager.submit(validate_job_payload(payload(**changes)))['job_id']

    def test_validation_rejects_invalid_inputs(self):
        cases=[[],None,payload(filename='x.jpg'),payload(content_base64='%%%'),payload(content_base64=''),
            payload(source=' '),payload(scene_id=''),payload(source=123),payload(request_id='bad\n'),
            payload(command='whoami'),payload(filename='x.tif',source='a'*501)]
        for value in cases:
            with self.subTest(value=str(value)[:80]),self.assertRaises(ValidationError):
                validate_job_payload(value)

    def test_validation_byte_limit_and_unicode(self):
        self.assertEqual(validate_job_payload(payload(filename='中文.TIFF'))['filename'],'中文.TIFF')
        with self.assertRaises(ValidationError) as caught:
            validate_job_payload(payload(b'x'*33),max_content_bytes=32)
        self.assertEqual(caught.exception.status,413)
        self.assertEqual(validate_job_payload(payload(filename='../中文.tif'))['filename'],'中文.tif')

    def test_public_identity_is_pinned(self):
        fake=self.directory/'fake.tif';fake.write_bytes(b'SYNTHETIC')
        cfg=self.cfg.with_overrides(public_sample_path=fake,synthetic_mode=False)
        self.assertEqual(cfg.classify_sample(store.sha256_bytes(fake.read_bytes())),config.SAMPLE_KIND_UPLOAD)
        self.assertEqual(cfg.classify_sample(config.PUBLIC_SAMPLE_SHA256),config.SAMPLE_KIND_PUBLIC)
        self.assertEqual(self.cfg.classify_sample('0'*64),config.SAMPLE_KIND_SYNTHETIC)

    def test_paths_are_independent_of_shell_directory(self):
        relative=Path('reports/service/relative-configuration.sqlite')
        self.assertEqual(self.cfg.with_overrides(db_path=relative).db_path,ROOT/relative)
        for timeout in (0,-1,float('nan'),float('inf')):
            with self.subTest(timeout=timeout),self.assertRaises(ValueError):
                self.cfg.with_overrides(timeout_seconds=timeout)

    def test_content_storage_and_boundary(self):
        data=b'SYNTHETIC object'
        sha,path,created=store.store_bytes(self.cfg.store_root,data)
        self.assertTrue(created)
        self.assertFalse(store.store_bytes(self.cfg.store_root,data)[2])
        self.assertEqual(path.read_bytes(),data)
        with self.assertRaises(store.PathBoundaryError):
            store.ensure_within(self.cfg.store_root,self.cfg.store_root/'../outside')
        path.write_bytes(b'corrupt fixture')
        with self.assertRaises(ValueError):
            store.store_bytes(self.cfg.store_root,data)
        self.assertEqual(path.read_bytes(),b'corrupt fixture')

    def test_failed_publication_preserves_staging(self):
        with patch('agrolake_service.store.os.link',side_effect=OSError('SYNTHETIC disk failure')):
            with self.assertRaises(OSError):
                store.store_bytes(self.cfg.store_root,b'SYNTHETIC disk failure')
        stages=list(self.cfg.store_root.rglob('.staging-*.tmp'))
        self.assertEqual(len(stages),1);self.assertEqual(stages[0].read_bytes(),b'SYNTHETIC disk failure')

    def test_concurrent_content_submission(self):
        manager=self.manager();normalized=validate_job_payload(payload(b'x'*900))
        gate=threading.Barrier(12)
        def run(_):
            gate.wait(10);return manager.submit(normalized)
        with ThreadPoolExecutor(max_workers=12) as pool:
            values=list(pool.map(run,range(12)))
        self.assertEqual(len({v['job_id'] for v in values}),12)
        with sqlite3.connect(self.cfg.db_path) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM rec_inputs').fetchone()[0],1)

    def test_concurrent_idempotency_and_conflict(self):
        manager=self.manager();normalized=validate_job_payload(payload(request_id='same'))
        with ThreadPoolExecutor(max_workers=12) as pool:
            values=list(pool.map(lambda _:manager.submit(normalized),range(12)))
        self.assertEqual(len({v['job_id'] for v in values}),1)
        self.assertEqual(sum(v['created'] for v in values),1)
        with self.assertRaises(IdempotencyConflict):
            self.submit(manager,request_id='same',source='different declaration')

    def test_duplicate_content_keeps_each_source(self):
        manager=self.manager();a=self.submit(manager,source='来源A');b=self.submit(manager,source='来源B')
        self.assertNotEqual(a,b)
        self.assertEqual(manager.get_job(a)['source'],'来源A');self.assertEqual(manager.get_job(b)['source'],'来源B')

    def test_queue_is_serial_and_persists_results(self):
        runner=FixtureRunner();manager=self.manager(runner,True)
        jobs=[self.submit(manager,request_id=str(i)) for i in range(3)]
        for job in jobs:
            result=wait_job(manager,job)
            self.assertEqual(result['status'],'succeeded');self.assertEqual(result['region_count'],1)
            self.assertEqual(result['class_counts'],{'0':0,'1':1,'2':0,'255':0})
            self.assertEqual(result['run']['adapter'],'SYNTHETIC FIXTURE, NOT REAL MODEL')
        self.assertEqual(runner.max_active,1)
        manager.stop();again=self.manager()
        self.assertEqual(again.get_job(jobs[0])['status'],'succeeded')
        self.assertEqual(again.get_fields(jobs[0])[0]['area_unit'],'pixel')
        self.assertEqual(again.read_artifact(jobs[0],'mask.tif'),b'SYNTHETIC MASK FIXTURE')

    def test_failures_do_not_get_success_artifacts(self):
        manager=self.manager(FixtureRunner('failure'),True);job=self.submit(manager)
        result=wait_job(manager,job)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['artifacts'],[])
        self.assertIsNone(manager.read_artifact(job,'mask.tif'))

    def test_success_transaction_rolls_back_on_bad_regions(self):
        manager=self.manager(FixtureRunner('invalid_region'),True);job=self.submit(manager)
        self.assertEqual(wait_job(manager,job)['status'],'failed')
        with sqlite3.connect(self.cfg.db_path) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM rec_regions').fetchone()[0],0)
            self.assertEqual(conn.execute('SELECT count(*) FROM rec_artifacts').fetchone()[0],0)
            self.assertEqual(conn.execute('SELECT count(*) FROM rec_runs WHERE status="succeeded"').fetchone()[0],0)

    def test_second_owner_cannot_change_first_jobs(self):
        first=self.manager();job=self.submit(first)
        with self.assertRaises(OwnershipError):
            JobManager(self.cfg,FixtureRunner())
        self.assertEqual(first.get_job(job)['status'],'queued')

    def test_resource_ownership_also_covers_other_db(self):
        self.manager()
        with self.assertRaises(OwnershipError):
            JobManager(self.cfg.with_overrides(db_path=self.directory/'another.sqlite'),FixtureRunner())

    def test_owner_lock_is_cross_process(self):
        first=self.manager();job=self.submit(first)
        code=('import sys; from pathlib import Path; from agrolake_service.config import ServiceConfig; '
              'from agrolake_service.jobs import JobManager; '
              'm=JobManager(ServiceConfig(db_path=Path(sys.argv[1]),store_root=Path(sys.argv[2]),runs_root=Path(sys.argv[3])),None)')
        result=subprocess.run([sys.executable,'-c',code,str(self.cfg.db_path),str(self.cfg.store_root),str(self.cfg.runs_root)],
            cwd=ROOT,capture_output=True,timeout=15,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        (self.directory/'child.stderr.log').write_bytes(result.stderr)
        self.assertNotEqual(result.returncode,0);self.assertIn(b'OwnershipError',result.stderr)
        self.assertEqual(first.get_job(job)['status'],'queued')

    def test_stop_and_restart_do_not_replay_jobs(self):
        first=self.manager();job=self.submit(first)
        first.stop();again=self.manager()
        self.assertEqual(again.get_job(job)['status'],'interrupted')
        first.stop()  # 重复stop不能误改新实例的记录
        new=self.submit(again)
        self.assertEqual(again.get_job(new)['status'],'queued')
        with self.assertRaises(ValidationError):
            self.submit(first)

    def test_http_end_to_end_fixture_and_download_integrity(self):
        manager=self.http();status,raw=self.request('POST','/jobs',payload(request_id='web'))
        self.assertEqual(status,202);job=json.loads(raw)['job_id']
        self.assertEqual(wait_job(manager,job)['status'],'succeeded')
        self.assertEqual(self.request('GET',f'/jobs/{job}/fields')[0],200)
        self.assertEqual(self.request('GET',f'/jobs/{job}/artifacts/mask.tif'),(200,b'SYNTHETIC MASK FIXTURE'))
        manager.artifact_path(job,'mask.tif').write_bytes(b'tampered synthetic fixture')
        self.assertEqual(self.request('GET',f'/jobs/{job}/artifacts/mask.tif')[0],409)
        self.assertEqual(self.request('POST','/jobs',payload(request_id='web'))[0],200)
        self.assertEqual(self.request('POST','/jobs',payload(request_id='web',scene_id='DIFFERENT'))[0],409)

    def test_http_bad_requests_are_structured(self):
        self.http(False)
        cases=[('POST','/jobs',b'{','application/json',400),('POST','/jobs',b'{}','text/plain',415),
            ('POST','/jobs',b'x'*4097,'application/json',413),('POST','/jobs',b'NaN','application/json',400),
            ('GET','/jobs/9223372036854775808',None,None,404),
            ('GET','/jobs/9999999999999999999999/fields',None,None,404),
            ('GET','/jobs/0',None,None,404),('GET','/jobs/1/artifacts/../../config.py',None,None,404),
            ('GET','/jobs/1/artifacts/%2e%2e',None,None,404)]
        for method,path,body,ctype,expected in cases:
            with self.subTest(path=path,body_length=len(body or b'')):
                headers={'Content-Type':ctype} if ctype else {}
                status,raw=self.request(method,path,body,headers)
                self.assertEqual(status,expected);self.assertIn('error',json.loads(raw))

    def test_http_host_origin_and_body_limits(self):
        self.http(False)
        for origin in ('https://example.org','http://[bad','null','http://127.0.0.1:abc'):
            with self.subTest(origin=origin):
                self.assertEqual(self.request('POST','/jobs',payload(),{'Origin':origin})[0],403)
        self.assertEqual(self.request('GET','/health',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('GET','/health')[0],200)
        self.assertEqual(self.request('POST','/jobs',payload(b'x'*1025))[0],413)
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=5)
        try:
            conn.putrequest('POST','/jobs');conn.putheader('Content-Type','application/json');conn.endheaders()
            response=conn.getresponse();response.read();self.assertEqual(response.status,411)
        finally:
            conn.close()

    def test_port_exclusion_and_network_binding(self):
        manager=self.http(False)
        with self.assertRaises(OSError):
            make_server(self.cfg,manager,port=self.port)
        with self.assertRaises(ValueError):
            make_server(self.cfg,manager,host='0.0.0.0')

    def test_existing_p0_tables_are_unchanged(self):
        from PIL import Image
        from agrolake import db as p0db,ingest,scenes
        conn=p0db.connect(self.cfg.db_path);p0db.init_schema(conn)
        buf=io.BytesIO();Image.new('RGB',(64,48),(30,70,90)).save(buf,format='PNG')
        ingest.import_bytes(conn,buf.getvalue(),'synthetic.png',store_root=self.directory/'p0-store',
                            source='SYNTHETIC P0 compatibility',scene_id='TEST')
        names=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        before={n:conn.execute('SELECT * FROM '+n).fetchall() for n in names};conn.close()
        self.manager()
        conn=p0db.connect(self.cfg.db_path)
        self.assertEqual(before,{n:conn.execute('SELECT * FROM '+n).fetchall() for n in names})
        self.assertEqual(scenes.summary(conn)['total_import_events'],1);conn.close()
        self.assertNotIn('torch',sys.modules)

    def test_runner_rejects_existing_output_without_overwrite(self):
        directory=self.directory/'existing';directory.mkdir();marker=directory/'keep';marker.write_bytes(b'keep')
        runner=CliRunner(Path(sys.executable),ROOT/'scripts/recognition/run_inference.py',self.cfg.model_path,'f'*64)
        result=runner.run(marker,'0'*64,directory,self.directory/'log')
        self.assertFalse(result.success);self.assertEqual(marker.read_bytes(),b'keep')
        self.assertEqual(list(directory.iterdir()),[marker])

    def test_runner_rejects_missing_and_inconsistent_artifacts(self):
        runner=CliRunner(Path(sys.executable),Path(__file__),self.cfg.model_path,'f'*64)
        sha='a'*64
        for case in ('missing','input_hash','model_hash','negative','total','regions','shape','malformed'):
            with self.subTest(case=case):
                directory=self.directory/case
                if case=='missing':
                    directory.mkdir()
                else:
                    run,_=artifacts(directory,sha)
                    if case=='input_hash':run['input']['sha256']='b'*64
                    if case=='model_hash':run['checkpoint']['sha256']='b'*64
                    if case=='negative':run['class_counts']['background_0']=-1
                    if case=='total':run['class_counts']['background_0']=1
                    if case=='regions':run['field_count']=2
                    if case=='shape':run['input']['shape_hw']=['bad',1]
                    (directory/'run.json').write_text('[]' if case=='malformed' else json.dumps(run),encoding='utf-8')
                self.assertFalse(runner._verify_success(Path('unused'),sha,directory,0,[]).success)

    def assert_process_gone(self,pid):
        import ctypes
        from ctypes import wintypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
        k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
        k.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=k.OpenProcess(0x100000,False,pid)
        if handle:
            try:self.assertEqual(k.WaitForSingleObject(handle,5000),0)
            finally:k.CloseHandle(handle)

    @unittest.skipUnless(os.name=='nt','Windows Job Object behavior')
    def test_timeout_terminates_windows_process_tree(self):
        runner=CliRunner(Path(sys.executable),ROOT/'tests/service/process_fixture.py',self.cfg.model_path,'f'*64,2)
        out=self.directory/'output'
        result=runner.run(Path(__file__),'a'*64,out,self.directory/'process.log')
        self.assertFalse(result.success);self.assertIn('超时',result.error)
        pids=json.loads((out/'pids.json').read_text(encoding='utf-8'))
        self.assertEqual(pids['dont_write_bytecode'],'1')
        self.assert_process_gone(pids['parent']);self.assert_process_gone(pids['child'])

    @unittest.skipUnless(os.name=='nt','Windows Job Object behavior')
    def test_stop_cancels_active_windows_worker_tree(self):
        runner=CliRunner(Path(sys.executable),ROOT/'tests/service/process_fixture.py',self.cfg.model_path,'f'*64,120)
        manager=self.manager(runner,True);job=self.submit(manager)
        deadline=time.monotonic()+15;pidsfile=None
        while time.monotonic()<deadline:
            matches=list(self.cfg.runs_root.glob('job-*/cli_output/pids.json'))
            if matches:pidsfile=matches[0];break
            time.sleep(.03)
        self.assertIsNotNone(pidsfile)
        pids=json.loads(pidsfile.read_text(encoding='utf-8'))
        manager.stop();self.assertEqual(manager.get_job(job)['status'],'failed')
        self.assert_process_gone(pids['parent']);self.assert_process_gone(pids['child'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ServiceTests)
    with (OUT/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'skipped':len(result.skipped),'successful':result.wasSuccessful(),'directory':str(OUT)}
    (OUT/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));print((OUT/'tests.log').read_text(encoding='utf-8'))
    sys.exit(0 if result.wasSuccessful() else 1)
