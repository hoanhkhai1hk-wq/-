"""识别后端服务启动入口（DEV-20261015-009）。

用法（在项目根目录，显式解释器）：
  .venv-p0/Scripts/python.exe scripts/service/serve.py

默认绑定 127.0.0.1:8506，使用 .venv-p0 运行 API，通过参数数组启动
.venv-recognition 的已审 CLI 子进程做真实识别。

本脚本只使用标准库，不导入 torch。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrolake_service import config as cfg  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.runner import CliRunner  # noqa: E402
from agrolake_service.ownership import OwnershipError
from agrolake_service.server import configure_logging, make_server  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AgroLake 本地识别后端服务（DEV-20261015-009）")
    p.add_argument("--host", default=cfg.DEFAULT_HOST, help="绑定地址，必须为本机（默认 127.0.0.1）")
    p.add_argument("--port", type=int, default=cfg.DEFAULT_PORT, help="端口（默认 8506，0 表示随机）")
    p.add_argument("--db", default=None, help="识别表 SQLite 路径（默认与网页共享 AGROLAKE_DB）")
    p.add_argument("--store", default=None, help="输入内容存储根目录")
    p.add_argument("--runs", default=None, help="作业运行目录根")
    p.add_argument("--worker-python", default=None, help="识别子进程解释器（默认 .venv-recognition）")
    p.add_argument("--cli", default=None, help="已审识别 CLI 脚本路径")
    p.add_argument("--model", default=None, help="模型权重路径")
    p.add_argument("--timeout", type=float, default=600.0, help="单次 CLI 超时秒数")
    p.add_argument("--synthetic", action="store_true",
                   help="把非公开样例的上传标记为合成测试（仅测试/演示部署使用）")
    return p


def build_config(args) -> cfg.ServiceConfig:
    config = cfg.ServiceConfig()
    return config.with_overrides(
        host=args.host,
        port=args.port,
        db_path=Path(args.db) if args.db else None,
        store_root=Path(args.store) if args.store else None,
        runs_root=Path(args.runs) if args.runs else None,
        worker_python=Path(args.worker_python) if args.worker_python else None,
        cli_script=Path(args.cli) if args.cli else None,
        model_path=Path(args.model) if args.model else None,
        timeout_seconds=args.timeout,
    ).with_overrides(synthetic_mode=args.synthetic)


def main(argv=None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = build_config(args)

    problems = config.missing_requirements()
    if problems:
        print("无法启动识别服务，请先解决以下问题：", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 2

    runner = CliRunner(
        worker_python=config.worker_python,
        cli_script=config.cli_script,
        model_path=config.model_path,
        model_sha256=config.model_sha256,
        timeout_seconds=config.timeout_seconds,
    )
    try:
        manager = JobManager(config, runner)
    except OwnershipError as exc:
        print(str(exc),file=sys.stderr)
        return 2
    try:
        server = make_server(config, manager)
    except (OSError,ValueError) as exc:
        manager.stop()
        print(f'识别服务未能监听，请检查端口或绑定地址：{exc}',file=sys.stderr)
        return 2
    manager.start()
    host, port = server.server_address[0], server.server_address[1]
    print(f"识别服务已启动： http://{host}:{port}/  （数据库 {config.db_path}）",flush=True)
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n正在停止识别服务……")
    finally:
        server.shutdown()
        server.server_close()
        manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
