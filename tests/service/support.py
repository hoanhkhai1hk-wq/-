"""纯合成fixture，绝不代表真实模型、TIFF解码或农业效果。"""
import base64,csv,json,threading,time
from pathlib import Path
from agrolake_service.runner import RunResult


def payload(content=b'SYNTHETIC', **changes):
    result={'filename':'synthetic.tif','content_base64':base64.b64encode(content).decode(),
            'source':'SYNTHETIC 软件测试','scene_id':'TEST','license_status':'人工合成测试'}
    result.update(changes)
    return result


def artifacts(directory,sha,model='f'*64):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    # 两个占位二进制只用于校验服务状态/固定文件，实际格式由real_smoke另外实测。
    (directory/'mask.tif').write_bytes(b'SYNTHETIC MASK FIXTURE')
    (directory/'preview.png').write_bytes(b'SYNTHETIC PREVIEW FIXTURE')
    with (directory/'class_pixels.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.writer(fh);writer.writerow(['class','meaning','pixels'])
        writer.writerows([(0,'background',0),(1,'field_interior',1),(2,'boundary',0),(255,'nodata/unlabelled',0)])
    region={'field_id':1,'pixels':1,'min_row':0,'min_col':0,'max_row':0,'max_col':0}
    with (directory/'field_pixels.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(region));writer.writeheader();writer.writerow(region)
    run={'status':'success','adapter':'SYNTHETIC FIXTURE, NOT REAL MODEL',
         'input':{'sha256':sha,'shape_hw':[1,1]},'checkpoint':{'sha256':model},
         'versions':{'fixture':True},'inference':{'fixture':True},
         'class_counts':{'background_0':0,'field_interior_1':1,'boundary_2':0,'nodata_255':0},'field_count':1}
    (directory/'run.json').write_text(json.dumps(run),encoding='utf-8')
    return run,[region]


class FixtureRunner:
    def __init__(self,mode='success'):
        self.mode=mode; self.calls=0; self.active=0; self.max_active=0
    def run(self,input_path,input_sha256,out_dir,log_path):
        self.calls+=1; self.active+=1;self.max_active=max(self.max_active,self.active)
        try:
            time.sleep(.03)
            if self.mode=='failure':
                return RunResult(False,'failed',str(out_dir),exit_code=7,error='SYNTHETIC failure')
            run,regions=artifacts(out_dir,input_sha256)
            if self.mode=='invalid_region':
                regions[0].pop('pixels')
            return RunResult(True,'succeeded',str(out_dir),exit_code=0,run_json=run,
                class_counts=run['class_counts'],field_count=1,regions=regions,
                model_sha256='f'*64,input_sha256=input_sha256,
                versions=run['versions'],params=run['inference'],adapter=run['adapter'])
        finally:
            self.active-=1


def wait_job(manager,job_id,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        job=manager.get_job(job_id)
        if job['status'] not in ('queued','running'):
            return job
        time.sleep(.03)
    raise AssertionError('job did not reach a terminal state')
