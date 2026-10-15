"""跨进程故障fixture，仅合成软件测试；只退出自身进程，不运行模型。"""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests/web')]
from agrolake_service import corrections
from agrolake_service.config import ServiceConfig
from agrolake_service.jobs import JobManager
from agrolake_service.server import make_server
from fixtures import WebFixtureRunner

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--mode',choices=['crash','restart'],required=True);args=parser.parse_args()
    out=Path(args.directory);os.environ['AGROLAKE_STORE']=str(out/'p0-store')
    class Runner(WebFixtureRunner):
        def run(self,*a,**kw):
            if args.mode=='crash' and self.calls>=1:
                # 子作业保持queued/running，硬退出发生在原manager已提交之后。
                time.sleep(60)
            marker=out/f'calls-{args.mode}-{self.calls+1}.json'
            with marker.open('x',encoding='utf-8') as file:json.dump({'synthetic':True},file)
            return super().run(*a,**kw)
    if args.mode=='crash':
        def crash(db_path,correction_id,child_job_id):
            with (out/'crash-boundary.json').open('x',encoding='utf-8') as file:
                json.dump({'correction_id':correction_id,'child_job_id':child_job_id,'exit_code':73},file)
                file.flush();os.fsync(file.fileno())
            os._exit(73)
        corrections._link_child=crash
    cfg=ServiceConfig(db_path=out/'db.sqlite',store_root=out/'store',runs_root=out/'runs',
                      port=0,synthetic_mode=True,model_sha256='f'*64)
    manager=JobManager(cfg,Runner(),auto_start=True);server=make_server(cfg,manager)
    ready={'port':server.server_address[1],'pid':os.getpid(),'synthetic':True}
    with (out/f'ready-{args.mode}.json').open('x',encoding='utf-8') as file:json.dump(ready,file)
    print(json.dumps(ready),flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:server.server_close();manager.stop()
if __name__=='__main__':main()
