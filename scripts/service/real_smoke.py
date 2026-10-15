"""真实本地HTTP/模型验收。每次独立目录，保留失败，绝不下载/安装/删除。

运行：.venv-p0/Scripts/python.exe scripts/service/real_smoke.py
通过独立serve.py进程和真实CLI验证；结束会停止本脚本启动的服务树。
"""
from collections import Counter
from datetime import datetime
import base64,csv,hashlib,http.client,io,json,os,re,sqlite3,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agrolake_service.config import PUBLIC_SAMPLE_SHA256,EXPECTED_MODEL_SHA256,FIXED_ARTIFACTS
from agrolake_service.processes import ProcessTree


class ServiceProcess:
    def __init__(self,directory,sequence):
        self.logpath=directory/f'service-{sequence}.log'
        self.log=self.logpath.open('x',encoding='utf-8')
        command=[str(ROOT/'.venv-p0/Scripts/python.exe'),str(ROOT/'scripts/service/serve.py'),
            '--port','0','--db',str(directory/'service.sqlite'),
            '--store',str(directory/'store'),'--runs',str(directory/'runs')]
        # 从子目录启动，实际覆盖“非项目cwd”场景。
        self.tree=ProcessTree(command,cwd=str(directory),env=dict(os.environ,
            PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8'),stdout=self.log)
        self.port=None
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                text=self.logpath.read_text(encoding='utf-8')
                found=re.search(r'http://127\.0\.0\.1:(\d+)/',text)
                if found:
                    self.port=int(found[1]);break
                if self.tree.process.poll() is not None:
                    raise RuntimeError('服务启动失败：'+text[-1500:])
                time.sleep(.05)
            if self.port is None:
                raise RuntimeError('服务启动超时')
        except BaseException:
            self.stop();raise

    def request(self,method,path,payload=None):
        body=json.dumps(payload,ensure_ascii=False).encode('utf-8') if payload is not None else None
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=30)
        try:
            conn.request(method,path,body,{'Content-Type':'application/json'} if body else {})
            response=conn.getresponse();data=response.read()
            if response.status>=400:
                raise RuntimeError(f'{method} {path}: {response.status} {data[:1000]!r}')
            return json.loads(data) if 'application/json' in response.getheader('Content-Type','') else data
        finally:
            conn.close()

    def wait_job(self,job_id,terminal=True,timeout=120):
        deadline=time.monotonic()+timeout
        states=[]
        while time.monotonic()<deadline:
            result=self.request('GET',f'/jobs/{job_id}')
            if not states or states[-1]!=result['status']:
                states.append(result['status'])
            if (terminal and result['status'] not in ('queued','running')) or (not terminal and result['status']=='running'):
                return result,states
            time.sleep(.1)
        raise AssertionError(f'作业状态未按期结束：{states}')

    def stop(self):
        try:
            self.tree.terminate()
        finally:
            self.tree.close();self.log.close()


def main():
    directory=ROOT/'reports/service/DEV-20261015-011'/datetime.now().strftime('real-%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True,exist_ok=False)
    checks={};evidence={'directory':str(directory),'checks':checks,'synthetic_tests_are_separate':True}
    service=None
    def check(name,value):
        checks[name]=bool(value)
        if not value:
            raise AssertionError(name)
    try:
        sample=ROOT/'samples/ftw_smoke/inference-img.tif'
        content=sample.read_bytes()
        check('input_sha256_pinned',hashlib.sha256(content).hexdigest()==PUBLIC_SAMPLE_SHA256)
        service=ServiceProcess(directory,1)
        health=service.request('GET','/health')
        check('health_files_present',all(health[k] for k in ('worker_python_ready','cli_script_ready','model_ready')))
        payload={'filename':sample.name,'content_base64':base64.b64encode(content).decode(),
            'source':'Fields of the World公开上游适配样例；时相身份未核实',
            'source_link':'https://github.com/fieldsoftheworld/ftw-baselines/tree/45d7df7ea0c357f2c34182e4a7e0c4625dd43533',
            'license_status':'上游公开示例；许可/署名分项见THIRD_PARTY.md；不是教师无人机数据',
            'scene_id':'FTW-PUBLIC-ADAPTER-DEMO','request_id':'real-public-sample'}
        submitted=service.request('POST','/jobs',payload);job_id=submitted['job_id']
        job,states=service.wait_job(job_id)
        evidence.update(job=job,observed_states=states)
        check('real_job_succeeded',job['status']=='succeeded')
        check('sample_identity',job['sample_kind']=='public_adapter_sample')
        check('source_and_scene_retained',job['source']==payload['source'] and job['scene_id']==payload['scene_id'])
        check('model_identity',job['model_sha256']==EXPECTED_MODEL_SHA256)
        checksums={}
        downloads=directory/'downloads';downloads.mkdir()
        for name in FIXED_ARTIFACTS:
            data=service.request('GET',f'/jobs/{job_id}/artifacts/{name}')
            if isinstance(data,dict):
                # run.json通过JSON响应读取；保存精确下载字节用于哈希另用普通HTTP。
                conn=http.client.HTTPConnection('127.0.0.1',service.port,timeout=10)
                try:
                    conn.request('GET',f'/jobs/{job_id}/artifacts/{name}')
                    response=conn.getresponse();data=response.read();check('run_json_download_http200',response.status==200)
                finally:conn.close()
            (downloads/name).write_bytes(data)
            sha=hashlib.sha256(data).hexdigest();checksums[name]=sha
            check('download_sha_'+name,sha==job['artifact_checksums'][name]['sha256'])
        run=json.loads((downloads/'run.json').read_text(encoding='utf-8'))
        check('strict_634_tensors',run['checkpoint']['load_report']['strict'] is True and
            run['checkpoint']['load_report']['model_tensors_loaded']==634)
        check('full_window_coverage',run['inference']['windows']==6 and run['inference']['coverage_ratio_valid']==1.0)
        check('run_hashes',run['input']['sha256']==PUBLIC_SAMPLE_SHA256 and run['checkpoint']['sha256']==EXPECTED_MODEL_SHA256)
        from PIL import Image
        with Image.open(downloads/'mask.tif') as mask:
            mask.load();counts=Counter(mask.getdata())
            check('mask_shape',(mask.height,mask.width)==(379,578))
            check('mask_class_counts',counts=={0:105910,1:74580,2:38572})
            check('mask_nodata_255',str(mask.tag_v2[42113]).rstrip('\x00')=='255')
            for key,values in run['input']['geotiff_tags'].items():
                actual=mask.tag_v2[int(key)]
                if isinstance(actual,tuple):actual=list(actual)
                check('geotiff_tag_'+key,actual==values)
        with Image.open(downloads/'preview.png') as preview:
            preview.load();check('preview_decodes',preview.width==1754 and preview.height==397)
        with (downloads/'field_pixels.csv').open(encoding='utf-8',newline='') as fh:
            fields=list(csv.DictReader(fh))
        with (downloads/'class_pixels.csv').open(encoding='utf-8',newline='') as fh:
            csv_counts={int(r['class']):int(r['pixels']) for r in csv.DictReader(fh)}
        check('csv_classes_match_mask',all(csv_counts.get(c,0)==counts.get(c,0) for c in (0,1,2,255)))
        check('candidate_regions_pixel_only',len(fields)==722 and sum(int(r['pixels']) for r in fields)==74580)
        api_fields=service.request('GET',f'/jobs/{job_id}/fields')
        check('api_regions_match_csv',api_fields['area_unit']=='pixel' and api_fields['region_count']==722
              and sum(r['pixels'] for r in api_fields['regions'])==74580)
        conn=sqlite3.connect(directory/'service.sqlite')
        try:
            record=conn.execute('SELECT j.source,j.scene_id,r.model_sha256,r.field_count FROM rec_jobs j JOIN rec_runs r USING(job_id) WHERE job_id=?',(job_id,)).fetchone()
            check('sqlite_provenance',record==(payload['source'],payload['scene_id'],EXPECTED_MODEL_SHA256,722))
            check('sqlite_regions',conn.execute('SELECT count(*),sum(pixels) FROM rec_regions WHERE job_id=?',(job_id,)).fetchone()==(722,74580))
        finally:conn.close()
        same=service.request('POST','/jobs',payload)
        check('http_idempotency_no_second_run',same['job_id']==job_id and not same['created'])
        # 非TIFF伪装后缀必须真实失败，保留CLI日志与error.json。
        bad=dict(payload,filename='SYNTHETIC-invalid.tif',source='SYNTHETIC invalid format test',
                 content_base64=base64.b64encode(b'SYNTHETIC not a TIFF').decode(),request_id='bad-format')
        bad_id=service.request('POST','/jobs',bad)['job_id']
        bad_job,bad_states=service.wait_job(bad_id)
        check('invalid_tiff_failed',bad_job['status']=='failed' and not bad_job['artifacts'] and 'CLI' in bad_job['error'])
        evidence.update(invalid_job=bad_job,invalid_observed_states=bad_states)
        # 模拟进程中断，再启动不能自动重放旧任务。
        interrupted_id=service.request('POST','/jobs',dict(bad,request_id='interrupted-format-test'))['job_id']
        service.wait_job(interrupted_id,terminal=False)
        old_port=service.port
        service.stop();service=None
        service=ServiceProcess(directory,2)
        recovered=service.request('GET',f'/jobs/{interrupted_id}')
        check('crash_recovery_no_replay',recovered['status']=='interrupted' and recovered['attempt_count']==1)
        check('completed_job_survives_restart',service.request('GET',f'/jobs/{job_id}')['status']=='succeeded')
        check('fields_survive_restart',service.request('GET',f'/jobs/{job_id}/fields')['region_count']==722)
        after=service.request('GET',f'/jobs/{job_id}/artifacts/mask.tif')
        check('download_survives_restart',hashlib.sha256(after).hexdigest()==checksums['mask.tif'])
        check('api_process_does_not_import_torch','torch' not in sys.modules)
        evidence.update(recovered_job=recovered,download_sha256=checksums,
                        inference_seconds=run['inference']['seconds'],successful=True)
    except BaseException as exc:
        evidence.update(successful=False,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if service is not None:service.stop()
        (directory/'verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'directory':str(directory),'successful':evidence['successful'],'checks':checks,
                      'error':evidence.get('error')},ensure_ascii=False,indent=2))
    return 0 if evidence['successful'] else 1


if __name__=='__main__':
    raise SystemExit(main())
