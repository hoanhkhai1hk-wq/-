"""Check this installation; optionally create only the Python 3.11 web venv.

Never reinstall an existing environment or pretend the local recognition lock
includes its inherited Torch installation. No credentials or network API calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


class SetupError(RuntimeError):
    pass


def run(command, *, cwd=ROOT, timeout=120):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"执行失败：{command[0]} ({type(exc).__name__})") from exc
    if result.returncode:
        # Import/pip output only; never inspect environment variables or secrets.
        raise SetupError(f"命令失败（退出码{result.returncode}）：{' '.join(map(str, command[:4]))}\n"
                         + (result.stderr or result.stdout)[-4000:])
    return result.stdout.strip()


def prepare_web(root: Path, install: bool, base_python: str | None):
    web = root / ".venv-p0/Scripts/python.exe"
    if web.is_file():
        return web, False
    if not install:
        raise SetupError("缺少 .venv-p0；运行一键安装.bat 准备网页环境。")
    destination = root / ".venv-p0"
    if destination.exists():
        raise SetupError(".venv-p0 已存在但解释器缺失；保留原目录，请检查安装中断原因后再恢复。")
    base = base_python or sys.executable
    version = run([base, "-B", "-c", "import sys; print('%d.%d'%sys.version_info[:2])"], cwd=root)
    if version != "3.11":
        raise SetupError(f"网页环境需要 Python 3.11，当前基础解释器为 {version}；用 --base-python 指定3.11解释器。")
    lock = root / "requirements-p0.lock.txt"
    if not lock.is_file():
        raise SetupError(f"缺少锁定依赖：{lock}")
    run([base, "-m", "venv", str(destination)], cwd=root)
    if not web.is_file():
        raise SetupError("venv 命令结束但网页解释器不存在，停止安装。")
    print("正在安装新网页环境的锁定依赖；现有识别环境保持。", flush=True)
    run([str(web), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(lock)],
        cwd=root, timeout=1200)
    return web, True


def check(root: Path, web: Path):
    web_code = (
        "import sys,json,importlib.metadata as m; import streamlit,PIL,requests; "
        "assert sys.version_info[:2]==(3,11),'web requires Python 3.11'; "
        "assert m.version('streamlit')=='1.50.0'; assert m.version('Pillow')=='11.3.0'; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'packages':{n:m.version(n) for n in ['streamlit','Pillow','requests']}}))"
    )
    web_info = json.loads(run([str(web), "-B", "-c", web_code], cwd=root))
    run([str(web), "-m", "pip", "check"], cwd=root)
    config_code = (
        "import json; from agrolake_service.config import ServiceConfig,sha256_file; "
        "c=ServiceConfig(); p=c.missing_requirements(); "
        "p += ['模型SHA256不符'] if c.model_path.is_file() and sha256_file(c.model_path)!=c.model_sha256 else []; "
        "print(json.dumps({'problems':p,'worker':str(c.worker_python),"
        "'sample_present':bool(c.public_sample_path and c.public_sample_path.is_file())}))"
    )
    config = json.loads(run([str(web), "-B", "-c", config_code], cwd=root))
    if config["problems"]:
        raise SetupError("网页依赖已通过；识别资料尚未准备：\n- " + "\n- ".join(config["problems"])
                         + "\n识别基础Torch及权重不包含在小包清单中，按 docs/模型适配说明.md 准备。")
    rec_code = (
        "import json,sys,torch,torchvision,numpy,tifffile,PIL,segmentation_models_pytorch; "
        "from recognition import read_image,load_model,predict; "
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__}))"
    )
    # Imports validate the inherited runtime without loading weights or inference.
    recognition = json.loads(run([config["worker"], "-B", "-c", rec_code], cwd=root, timeout=120))
    return {"web": web_info, "recognition": recognition,
            "model_sha256_verified": True, "public_sample_present": config["sample_present"],
            "fresh_machine_install_verified": False, "inference_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="只检查，不安装（默认）")
    action.add_argument("--install-web", action="store_true", help="仅缺少网页环境时创建及安装")
    parser.add_argument("--base-python", help="新建网页环境使用的Python3.11解释器绝对路径")
    args = parser.parse_args(argv)
    try:
        web, created = prepare_web(ROOT, args.install_web, args.base_python)
        result = check(ROOT, web)
        result.update(status="ready", created_web_environment=created)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("本机环境检查通过。运行一键启动.bat 打开网页；新电脑完整安装尚未验收。")
        if not result["public_sample_present"]:
            print("提示：未找到公开演示TIFF；不影响启动，公开样例演示需另行准备。")
        return 0
    except (SetupError, ValueError, KeyError) as exc:
        print(f"准备未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
