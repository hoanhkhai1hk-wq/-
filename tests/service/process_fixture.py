"""子进程生命周期fixture：派生sleep进程并记录PID，无模型/网络/清理。"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('input','checkpoint','output-dir','expected-input-sha256'):
    p.add_argument('--'+name)
args=p.parse_args()
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
(out/'pids.json').write_text(json.dumps({'parent':os.getpid(),'child':child.pid,
    'dont_write_bytecode':os.environ.get('PYTHONDONTWRITEBYTECODE')}),encoding='utf-8')
time.sleep(120)
