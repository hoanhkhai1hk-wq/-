"""AgroLake 本机一键启动器。

只使用 Python 标准库，不导入 torch。
- start：完整预检 → 复用本项目健康服务或隐藏后台启动缺失服务 → 端口实核 listener/命令行/健康
- status：只读输出两端口状态 JSON，不创建文件、不启动进程

根目录固定为 Path(__file__).resolve().parents[2]；测试通过函数参数注入 root。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
WEB_PORT = 8505
API_PORT = 8506
WEB_HEALTH_PATH = "/_stcore/health"
API_HEALTH_PATH = "/health"
API_SERVICE = "agrolake-recognition"
READY_KEYS = ("worker_python_ready", "cli_script_ready", "model_ready")
DEFAULT_TIMEOUT = 45.0
HTTP_TIMEOUT = 2.0
POLL_INTERVAL = 0.5


class StartupError(RuntimeError):
    """启动前置条件或互斥失败。"""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_paths(root) -> dict:
    root = Path(root)
    return {
        "root": root,
        "p0_python": root / ".venv-p0" / "Scripts" / "python.exe",
        "app": root / "app.py",
        "serve": root / "scripts" / "service" / "serve.py",
        "worker_python": root / ".venv-recognition" / "Scripts" / "python.exe",
        "cli": root / "scripts" / "recognition" / "run_inference.py",
        "model": root / "models" / "ftw-v1" / "3_Class_CCBY_FTW_Pretrained.ckpt",
    }


def load_service_config(root, paths):
    """按代码根路径导入 ServiceConfig；只读文件存在性，不导入 torch。"""
    root_str = str(Path(root))
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    sys.dont_write_bytecode = True  # 不往源码目录写 __pycache__
    from agrolake_service.config import ServiceConfig

    return ServiceConfig()


def check_prerequisites(root, paths) -> tuple[list[str], dict]:
    problems: list[str] = []
    for key, label in (("p0_python", "P0 解释器"), ("app", "网页 app.py"), ("serve", "识别服务 serve.py")):
        if not Path(paths[key]).is_file():
            problems.append(f"缺少{label}：{paths[key]}")
    readiness: dict = {}
    try:
        config = load_service_config(root, paths)
        problems.extend(config.missing_requirements())
        readiness = config.readiness()
    except Exception as exc:  # 导入/配置失败必须明确，不带着残缺环境启动
        problems.append(f"无法读取识别服务配置：{exc}")
    return problems, readiness


def service_command(kind, paths) -> list[str]:
    python = str(paths["p0_python"])
    if kind == "api":
        return [python, "-u", "-B", str(paths["serve"])]
    return [
        python, "-B", "-m", "streamlit", "run", str(paths["app"]),
        "--server.address", HOST,
        "--server.port", str(WEB_PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]


def _launch_flags() -> int:
    if os.name != "nt":
        return 0
    return (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def _query_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def start_service(kind, paths, log_path):
    """隐藏后台创建服务进程；日志文件独占新建，不覆盖旧记录。"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "xb") as log:
        return subprocess.Popen(
            service_command(kind, paths),
            cwd=str(paths["root"]),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=_launch_flags(),
            env=env,
        )


_PS_QUERY = r"""
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8
$port = __PORT__
$list = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
$owners = @($list | Select-Object -ExpandProperty OwningProcess -Unique)
$result = [ordered]@{ port = $port; listening = $false; listener_pid = $null; command_line = $null; error = $null }
if ($owners.Count -gt 1) {
  $result.listening = $true
  $result.error = 'multiple listeners'
} elseif ($owners.Count -eq 1) {
  $result.listening = $true
  $result.listener_pid = [int]$owners[0]
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($owners[0])" -ErrorAction SilentlyContinue
  if ($null -eq $proc) { $result.error = 'process not found' }
  elseif ([string]::IsNullOrWhiteSpace($proc.CommandLine)) { $result.error = 'command line unavailable' }
  else { $result.command_line = [string]$proc.CommandLine }
}
$result | ConvertTo-Json -Compress
"""


def _parse_last_json(text: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def query_port_owner(port: int, timeout: float = 15.0) -> dict:
    """用 Get-NetTCPConnection + Win32_Process 取得监听 PID 与命令行。"""
    script = _PS_QUERY.replace("__PORT__", str(int(port)))
    base = {"port": int(port), "listening": None, "listener_pid": None, "command_line": None}
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, creationflags=_query_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(base, error=f"powershell 调用失败：{exc}")
    if proc.returncode != 0:
        return dict(base, error=f"powershell 退出码 {proc.returncode}")
    data = _parse_last_json(proc.stdout)
    if not isinstance(data, dict):
        return dict(base, error="无法解析端口查询结果")
    return {
        "port": int(data.get("port", port)),
        "listening": data.get("listening") is True,
        "listener_pid": data.get("listener_pid"),
        "command_line": data.get("command_line"),
        "error": data.get("error"),
    }


def _normalized(text) -> str:
    return str(text).replace("\\", "/").casefold()


def command_matches(kind: str, command: str | None, paths) -> bool:
    """Match actual Windows arguments, including the venv's base interpreter."""
    if not command:
        return False
    import ctypes
    from ctypes import wintypes
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    parse = shell.CommandLineToArgvW
    parse.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    parse.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    pointer = parse(command, ctypes.byref(count))
    if not pointer:
        return False
    try:
        tokens = [_normalized(pointer[i]) for i in range(count.value)]
    finally:
        free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(pointer)
    if not tokens:
        return False
    target = _normalized(paths["serve"] if kind == "api" else paths["app"])
    pythons = {
        _normalized(paths["p0_python"]),
        _normalized(Path(paths["p0_python"]).with_name("pythonw.exe")),
    }
    # Windows venv python.exe launches its base executable as the listener.
    # pyvenv.cfg is local installation metadata, not a credential file.
    cfg_path = Path(paths["p0_python"]).parents[1] / "pyvenv.cfg"
    if cfg_path.is_file():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "executable":
                pythons.add(_normalized(value.strip()))
    if tokens[0] not in pythons:
        return False
    remaining = tokens[1:]
    while remaining and remaining[0] in ("-b", "-u"):
        remaining = remaining[1:]
    if kind == "api":
        return bool(remaining) and remaining[0] == target
    return remaining[:4] == ["-m", "streamlit", "run", target]


def http_fetch(url: str, timeout: float = HTTP_TIMEOUT):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status, response.read(4096).decode("utf-8", "replace")
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return None, str(exc)


def probe_web(timeout: float = HTTP_TIMEOUT) -> dict:
    status, body = http_fetch(f"http://{HOST}:{WEB_PORT}{WEB_HEALTH_PATH}", timeout)
    ok = status == 200 and (body or "").strip() == "ok"
    return {"ok": ok, "http_status": status, "detail": (body or "无响应").strip()[:200]}


def probe_api(timeout: float = HTTP_TIMEOUT) -> dict:
    status, body = http_fetch(f"http://{HOST}:{API_PORT}{API_HEALTH_PATH}", timeout)
    if status != 200:
        return {"ok": False, "http_status": status, "detail": (body or "无响应")[:200]}
    try:
        payload = json.loads(body)
    except ValueError:
        return {"ok": False, "http_status": status, "detail": "健康响应不是 JSON"}
    if not isinstance(payload, dict):
        return {"ok": False, "http_status": status, "detail": "健康响应不是对象"}
    ready = {key: payload.get(key) is True for key in READY_KEYS}
    ok = payload.get("service") == API_SERVICE and payload.get("status") == "ok" and all(ready.values())
    return {
        "ok": ok,
        "http_status": status,
        "service": payload.get("service"),
        "status": payload.get("status"),
        "ready": ready,
    }


def _probe(kind: str) -> dict:
    return probe_api() if kind == "api" else probe_web()


def _inspect_ports(paths) -> tuple[dict, list[str]]:
    """两端口完整预检；只要命令行证据不足就报错，绝不结束占用进程。"""
    entries: dict = {}
    errors: list[str] = []
    for kind, port in (("api", API_PORT), ("web", WEB_PORT)):
        owner = query_port_owner(port)
        listening = owner.get("listening")
        entry = {
            "port": port,
            "url": f"http://{HOST}:{port}",
            "listening": bool(listening),
            "listener_pid": owner.get("listener_pid") if listening else None,
            "command_verified": False,
            "reused": False,
            "health": None,
            "log": None,
            "launcher_pid": None,
        }
        error = owner.get("error")
        if listening is None:
            errors.append(f"{port} 端口状态无法确认：{error or '未知错误'}")
        elif listening:
            if error:
                errors.append(f"{port} 已监听但无法取得命令行证据（{error}），不启动也不结束占用进程")
            elif command_matches(kind, owner.get("command_line"), paths):
                entry["command_verified"] = True
            else:
                errors.append(f"{port} 已被其他程序占用（命令不属于本项目），未启动任何服务，也未结束占用进程")
        entries[kind] = entry
    return entries, errors


class LaunchLock:
    """项目 reports 路径上的生命周期锁；释放后不遗留持有中的锁。"""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / ".launch.lock"
        self._fh = None

    def acquire(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+b")
            self._fh.seek(0, os.SEEK_END)
            if self._fh.tell() == 0:
                self._fh.write(b"\0")
                self._fh.flush()
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            raise StartupError("另一个启动器正在运行，请等待其结束后重试") from exc
        return self

    def release(self):
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            self._fh.close()
            self._fh = None


def status(root) -> tuple[int, dict]:
    """只读状态：不创建文件、不启动进程、不打开浏览器。"""
    root = Path(root).resolve()
    paths = project_paths(root)
    entries, errors = _inspect_ports(paths)
    for kind in ("api", "web"):
        probe = _probe(kind)
        entries[kind]["health"] = probe
        if not probe.get("ok"):
            errors.append(f"{kind} 服务健康检查未通过：{probe.get('detail') or '无响应'}")
        if not entries[kind]["command_verified"] and not entries[kind]["listening"]:
            errors.append(f"{kind} 未发现本项目监听进程")
    report = {
        "root": str(root),
        "mode": "status",
        "ok": not errors,
        "ports": entries,
        "logs": None,
        "errors": errors,
        "opened_browser": False,
    }
    return (0 if not errors else 1), report


def start(root, *, timeout: float = DEFAULT_TIMEOUT, open_browser: bool = True) -> tuple[int, dict]:
    root = Path(root).resolve()
    paths = project_paths(root)
    report = {
        "root": str(root),
        "mode": "start",
        "ok": False,
        "partial": False,
        "timeout_seconds": float(timeout),
        "ports": {},
        "logs": {"dir": None, "api": None, "web": None},
        "launched": {},
        "errors": [],
        "opened_browser": False,
    }
    if not math.isfinite(timeout) or timeout <= 0:
        report["errors"] = ["--timeout 必须是大于零的有限秒数"]
        return 1, report
    problems, _ = check_prerequisites(root, paths)
    if problems:
        report["errors"] = problems
        return 1, report

    try:
        lock = LaunchLock(root / "reports" / "local-start").acquire()
    except StartupError as exc:
        report["errors"] = [str(exc)]
        return 1, report
    try:
        return _start_locked(root, paths, report, timeout, open_browser)
    finally:
        lock.release()


def _start_locked(root, paths, report, timeout, open_browser):
    entries, errors = _inspect_ports(paths)
    report["ports"] = entries

    for kind in ("api", "web"):
        entry = entries[kind]
        if entry["command_verified"]:
            probe = _probe(kind)
            entry["health"] = probe
            if probe.get("ok"):
                entry["reused"] = True
            else:
                errors.append(f"{kind} 端口已由本项目监听但健康检查未通过：{probe.get('detail') or '无响应'}")

    if errors:
        report["errors"] = errors
        return 1, report

    to_launch = [kind for kind in ("api", "web") if not entries[kind]["listening"]]
    log_dir = None
    if to_launch:
        try:
            log_dir = Path(root) / "reports" / "local-start" / uuid.uuid4().hex
            log_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            report["errors"] = [f"无法创建本次日志目录：{exc}"]
            return 1, report
        report["logs"]["dir"] = str(log_dir)
        for kind in to_launch:
            log_path = log_dir / f"{kind}.log"
            report["logs"][kind] = str(log_path)
            entries[kind]["log"] = str(log_path)
            try:
                process = start_service(kind, paths, log_path)
            except OSError as exc:
                errors.append(f"{kind} 服务未能创建进程：{exc}；日志：{log_path}")
                continue
            launcher_pid = getattr(process, "pid", None)
            entries[kind]["launcher_pid"] = launcher_pid
            report["launched"][kind] = {"launcher_pid": launcher_pid, "log": str(log_path)}

    pending = [kind for kind in to_launch if kind in report["launched"]]
    deadline = time.monotonic() + max(0.05, float(timeout))
    while pending and time.monotonic() < deadline:
        for kind in tuple(pending):
            entry = entries[kind]
            owner = query_port_owner(entry["port"])
            if not owner.get("listening"):
                continue
            entry["listening"] = True
            if owner.get("listener_pid"):
                entry["listener_pid"] = owner.get("listener_pid")
            if not owner.get("command_line"):
                entry["health"] = {"ok": False, "detail": owner.get("error") or "命令行证据无法取得"}
                continue
            if not command_matches(kind, owner.get("command_line"), paths):
                entry["health"] = {"ok": False, "detail": "端口监听者命令不属于本项目"}
                continue
            entry["command_verified"] = True
            probe = _probe(kind)
            entry["health"] = probe
            if probe.get("ok"):
                pending.remove(kind)
        if pending:
            time.sleep(POLL_INTERVAL)

    for kind in pending:
        detail = (entries[kind].get("health") or {}).get("detail")
        suffix = f"（{detail}）" if detail else ""
        errors.append(
            f"{kind} 服务未在 {float(timeout):g} 秒内达到健康状态{suffix}；"
            f"日志：{entries[kind]['log'] or log_dir}"
        )

    report["errors"] = errors
    report["ok"] = not errors and all(
        (entry.get("health") or {}).get("ok") and entry.get("command_verified")
        for entry in entries.values()
    )
    report["partial"] = (not report["ok"]) and any(
        (entry.get("health") or {}).get("ok") for entry in entries.values()
    )

    if report["ok"] and open_browser:
        webbrowser.open(f"http://{HOST}:{WEB_PORT}")
        report["opened_browser"] = True

    if log_dir is not None:
        try:
            (log_dir / "status.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    return (0 if report["ok"] else 1), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AgroLake 本机一键启动器（标准库）")
    parser.add_argument("command", nargs="?", default="start", choices=("start", "status"))
    parser.add_argument("--no-browser", action="store_true", help="启动成功后不打开浏览器")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="启动等待秒数（默认 45）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "status":
        code, report = status(root)
    else:
        code, report = start(root, timeout=args.timeout, open_browser=not args.no_browser)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
