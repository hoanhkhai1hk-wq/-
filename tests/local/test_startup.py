"""startup.py 的离线合成测试：全部 mock，不调用真实 PowerShell/网络/服务。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[1] / "scripts" / "local"))
import startup  # noqa: E402

REAL_PROBE_API = startup.probe_api
REAL_PROBE_WEB = startup.probe_web


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.terminated = False

    def kill(self):
        self.terminated = True

    def terminate(self):
        self.terminated = True


class StubConfig:
    def __init__(self, missing=()):
        self._missing = list(missing)

    def missing_requirements(self):
        return list(self._missing)

    def readiness(self):
        return {"worker_python_ready": True, "cli_script_ready": True, "model_ready": True}


def ok_probe(detail="ok"):
    return {"ok": True, "http_status": 200, "detail": detail}


def bad_probe(detail="connection refused"):
    return {"ok": False, "http_status": None, "detail": detail}


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(BASE), prefix="_tmp-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        (self.root / "app.py").write_text("", encoding="utf-8")
        (self.root / "scripts" / "service").mkdir(parents=True)
        (self.root / "scripts" / "service" / "serve.py").write_text("", encoding="utf-8")
        (self.root / ".venv-p0" / "Scripts").mkdir(parents=True)
        (self.root / ".venv-p0" / "Scripts" / "python.exe").write_text("", encoding="utf-8")

        self.owners = {}
        self.probes = {"api": bad_probe(), "web": bad_probe()}
        self.started = []
        self.opened = []
        self.queries = 0

        self._patchers = [
            patch.object(startup, "query_port_owner", side_effect=self._owner),
            patch.object(startup, "probe_api", side_effect=lambda *a, **k: self.probes["api"]),
            patch.object(startup, "probe_web", side_effect=lambda *a, **k: self.probes["web"]),
            patch.object(startup, "start_service", side_effect=self._start_service),
            patch.object(startup, "load_service_config", return_value=StubConfig()),
            patch.object(startup.webbrowser, "open", side_effect=self.opened.append),
            patch.object(startup, "POLL_INTERVAL", 0.01),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    # ---- helpers ---------------------------------------------------------

    def command(self, kind, python_name="python.exe"):
        python = self.root / ".venv-p0" / "Scripts" / python_name
        if kind == "api":
            target = self.root / "scripts" / "service" / "serve.py"
            return f'"{python}" -u -B "{target}"'
        target = self.root / "app.py"
        return (
            f'"{python}" -B -m streamlit run "{target}"'
            " --server.address 127.0.0.1 --server.port 8505 --server.headless true"
        )

    def occupy(self, kind, command=None, pid=4000):
        port = startup.API_PORT if kind == "api" else startup.WEB_PORT
        self.owners[port] = {
            "port": port,
            "listening": True,
            "listener_pid": pid,
            "command_line": self.command(kind) if command is None else command,
            "error": None,
        }

    def mark_healthy(self):
        self.probes["api"] = ok_probe()
        self.probes["web"] = ok_probe()

    def _owner(self, port):
        self.queries += 1
        default = {"port": port, "listening": False, "listener_pid": None, "command_line": None, "error": None}
        return dict(default, **self.owners.get(port, {}))

    def _start_service(self, kind, paths, log_path):
        process = FakeProc(9000 + len(self.started))
        self.started.append({"kind": kind, "log": Path(log_path), "proc": process})
        Path(log_path).write_text(f"fake {kind} log\n", encoding="utf-8")
        port = startup.API_PORT if kind == "api" else startup.WEB_PORT
        self.owners[port] = {
            "port": port,
            "listening": True,
            "listener_pid": 4000 + len(self.started),
            "command_line": self.command(kind),
            "error": None,
        }
        return process

    # ---- tests -----------------------------------------------------------

    def test_foreign_program_on_both_ports_starts_nothing(self):
        self.occupy("api", 'C:\\other\\.venv-p0\\Scripts\\python.exe -u -B C:\\other\\scripts\\service\\serve.py')
        self.occupy("web", 'C:\\other\\python.exe -m streamlit run C:\\other\\app.py')
        code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertEqual(self.started, [])
        self.assertEqual(report["launched"], {})
        self.assertIsNone(report["logs"]["dir"])
        self.assertEqual(len(report["errors"]), 2)
        self.assertIn("其他程序占用", " ".join(report["errors"]))
        leftovers = [p.name for p in (self.root / "reports" / "local-start").iterdir()]
        self.assertEqual(leftovers, [".launch.lock"])

    def test_foreign_port_plus_free_port_has_no_partial_launch(self):
        self.occupy("api", 'C:\\other\\python.exe C:\\other\\serve.py')
        code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code, 1)
        self.assertEqual(self.started, [])
        self.assertIsNone(report["logs"]["dir"])
        self.assertFalse(report["ports"]["web"]["listening"])

    def test_healthy_project_services_are_reused_with_zero_popen(self):
        self.occupy("api", pid=111)
        self.occupy("web", pid=222)
        self.mark_healthy()
        code, report = startup.start(self.root, timeout=0.2, open_browser=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.started, [])
        self.assertTrue(report["ports"]["api"]["reused"])
        self.assertTrue(report["ports"]["web"]["reused"])
        self.assertEqual(report["ports"]["api"]["listener_pid"], 111)
        self.assertEqual(report["ports"]["web"]["listener_pid"], 222)
        self.assertIsNone(report["logs"]["dir"])
        self.assertEqual(self.opened, ["http://127.0.0.1:8505"])

    def test_occupied_but_unhealthy_fails_before_launching_other_port(self):
        self.occupy("web", pid=333)
        self.probes["api"] = ok_probe()
        code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code, 1)
        self.assertEqual(self.started, [])
        self.assertIn("健康检查未通过", " ".join(report["errors"]))

    def test_missing_model_fails_in_preflight_without_port_queries(self):
        stub = StubConfig(["未找到模型权重文件：models/ftw-v1/x.ckpt"])
        with patch.object(startup, "load_service_config", return_value=stub):
            code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code, 1)
        self.assertEqual(self.queries, 0)
        self.assertEqual(self.started, [])
        self.assertIn("未找到模型权重文件", " ".join(report["errors"]))

    def test_both_ports_free_launch_and_report_real_listener_pid(self):
        self.mark_healthy()
        code, report = startup.start(self.root, timeout=1.0, open_browser=True)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(s["kind"] for s in self.started), ["api", "web"])
        for kind in ("api", "web"):
            entry = report["ports"][kind]
            self.assertFalse(entry["reused"])
            self.assertTrue(entry["listening"])
            self.assertTrue(entry["command_verified"])
            self.assertEqual(entry["listener_pid"], 4000 + [s["kind"] for s in self.started].index(kind) + 1)
            self.assertEqual(entry["launcher_pid"], 9000 + [s["kind"] for s in self.started].index(kind))
            self.assertNotEqual(entry["listener_pid"], entry["launcher_pid"])
            self.assertTrue(Path(entry["log"]).is_file())
        self.assertIsNotNone(report["logs"]["dir"])
        self.assertTrue((Path(report["logs"]["dir"]) / "status.json").is_file())
        self.assertEqual(self.opened, ["http://127.0.0.1:8505"])
        self.assertTrue(report["opened_browser"])

    def test_launch_timeout_reports_log_and_keeps_process_alive(self):
        self.mark_healthy()
        with patch.object(startup, "start_service", side_effect=self._never_listens):
            code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertFalse(report["partial"])
        self.assertIn("未在", " ".join(report["errors"]))
        self.assertIn(report["logs"]["api"], " ".join(report["errors"]))
        self.assertTrue(Path(report["logs"]["api"]).is_file())
        self.assertFalse(self.started[0]["proc"].terminated)

    def _never_listens(self, kind, paths, log_path):
        process = FakeProc(7000 + len(self.started))
        self.started.append({"kind": kind, "log": Path(log_path), "proc": process})
        Path(log_path).write_text("still starting\n", encoding="utf-8")
        return process

    def test_status_healthy_is_read_only(self):
        self.occupy("api", pid=11)
        self.occupy("web", pid=22)
        self.mark_healthy()
        before = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        code, report = startup.status(self.root)
        after = sorted(p.relative_to(self.root) for p in self.root.rglob("*"))
        self.assertEqual(code, 0)
        self.assertEqual(report["mode"], "status")
        self.assertTrue(report["ok"])
        self.assertFalse(report["opened_browser"])
        self.assertEqual(self.started, [])
        self.assertEqual(before, after)
        self.assertFalse((self.root / "reports" / "local-start").exists())

    def test_status_unhealthy_returns_nonzero_without_launch(self):
        self.occupy("web", pid=22)
        code, report = startup.status(self.root)
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertEqual(self.started, [])
        self.assertEqual(self.opened, [])
        self.assertTrue(report["errors"])

    def test_command_recognition_with_spaces_and_venv_wrapper(self):
        paths = startup.project_paths(Path("D:/AI Workspace/agro"))
        api = '"D:\\AI Workspace\\agro\\.venv-p0\\Scripts\\python.exe" -u -B "D:\\AI Workspace\\agro\\scripts\\service\\serve.py"'
        web = ('"D:\\AI Workspace\\agro\\.venv-p0\\Scripts\\pythonw.exe" -B -m streamlit run '
               '"D:\\AI Workspace\\agro\\app.py" --server.port 8505')
        self.assertTrue(startup.command_matches("api", api, paths))
        self.assertTrue(startup.command_matches("web", web, paths))
        self.assertFalse(startup.command_matches("api", api.replace("D:\\AI Workspace\\agro", "D:\\Other"), paths))
        self.assertFalse(startup.command_matches("web", 'C:\\Python312\\python.exe -m streamlit run D:\\AI Workspace\\agro\\app.py', paths))
        self.assertFalse(startup.command_matches("api", None, paths))
        self.assertFalse(startup.command_matches("api", '"D:\\AI Workspace\\agro\\.venv-p0\\Scripts\\python.exe" -m streamlit run "D:\\AI Workspace\\agro\\app.py"', paths))

    def test_base_interpreter_and_exact_script_arguments(self):
        paths = startup.project_paths(self.root)
        base = 'D:/base python/python.exe'
        cfg = self.root / '.venv-p0/pyvenv.cfg'
        cfg.write_text('executable = ' + base + '\n', encoding='utf-8')
        api = f'"{base}" -u -B "{paths["serve"]}"'
        web = f'"{base}" -B -m streamlit run "{paths["app"]}"'
        self.assertTrue(startup.command_matches('api', api, paths))
        self.assertTrue(startup.command_matches('web', web, paths))
        self.assertFalse(startup.command_matches('api', api.replace('serve.py', 'serve.py.bak'), paths))
        self.assertFalse(startup.command_matches('web', web.replace('app.py', 'app.py.bak'), paths))
        self.assertFalse(startup.command_matches('api', f'"{base}" -c "print(1)" "{paths["serve"]}"', paths))
        self.assertFalse(startup.command_matches('web', f'"{base}" -c "print(1)" -m streamlit run "{paths["app"]}"', paths))

    def test_invalid_timeout_never_starts_or_creates_logs(self):
        for value in (0, -1, float('inf'), float('nan')):
            code, report = startup.start(self.root, timeout=value, open_browser=False)
            self.assertEqual(code, 1)
            self.assertFalse(report['ok'])
        self.assertEqual(self.started, [])
        self.assertFalse((self.root / 'reports/local-start').exists())

    def test_lock_blocks_concurrent_start_then_releases(self):
        lock = startup.LaunchLock(self.root / "reports" / "local-start").acquire()
        try:
            code, report = startup.start(self.root, timeout=0.2, open_browser=False)
        finally:
            lock.release()
        self.assertEqual(code, 1)
        self.assertEqual(self.started, [])
        self.assertIn("另一个启动器", " ".join(report["errors"]))

        self.occupy("api", pid=1)
        self.occupy("web", pid=2)
        self.mark_healthy()
        code2, _ = startup.start(self.root, timeout=0.2, open_browser=False)
        self.assertEqual(code2, 0)

    def test_no_browser_option_does_not_open(self):
        self.mark_healthy()
        code, report = startup.start(self.root, timeout=1.0, open_browser=False)
        self.assertEqual(code, 0)
        self.assertEqual(self.opened, [])
        self.assertFalse(report["opened_browser"])

    def test_each_launch_gets_unique_log_directory(self):
        self.mark_healthy()
        with patch.object(startup, "start_service", side_effect=self._never_listens):
            with patch.object(startup.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)):
                startup.start(self.root, timeout=0.15, open_browser=False)
            with patch.object(startup.uuid, "uuid4", return_value=SimpleNamespace(hex="b" * 32)):
                startup.start(self.root, timeout=0.15, open_browser=False)
        root_dir = self.root / "reports" / "local-start"
        dirs = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
        self.assertEqual(dirs, ["a" * 32, "b" * 32])
        first_log = root_dir / ("a" * 32) / "api.log"
        self.assertEqual(first_log.read_text(encoding="utf-8"), "still starting\n")

    def test_partial_success_keeps_healthy_service_and_reports_failure(self):
        self.mark_healthy()

        def flaky(kind, paths, log_path):
            if kind == "web":
                Path(log_path).write_text("popen failed\n", encoding="utf-8")
                raise OSError("mock spawn failure")
            return self._start_service(kind, paths, log_path)

        with patch.object(startup, "start_service", side_effect=flaky):
            code, report = startup.start(self.root, timeout=1.0, open_browser=False)
        self.assertEqual(code, 1)
        self.assertFalse(report["ok"])
        self.assertTrue(report["partial"])
        self.assertTrue(report["ports"]["api"]["health"]["ok"])
        self.assertEqual(len(self.started), 1)
        self.assertFalse(self.started[0]["proc"].terminated)
        self.assertIn("mock spawn failure", " ".join(report["errors"]))
        self.assertIn(report["logs"]["web"], " ".join(report["errors"]))

    def test_api_health_requires_service_and_all_three_ready_flags(self):
        good = {
            "service": "agrolake-recognition",
            "status": "ok",
            "worker_python_ready": True,
            "cli_script_ready": True,
            "model_ready": True,
        }
        with patch.object(startup, "http_fetch", return_value=(200, json.dumps(good))):
            self.assertTrue(REAL_PROBE_API()["ok"])
        for key in ("worker_python_ready", "cli_script_ready", "model_ready"):
            with patch.object(startup, "http_fetch", return_value=(200, json.dumps(dict(good, **{key: False})))):
                self.assertFalse(REAL_PROBE_API()["ok"])
        with patch.object(startup, "http_fetch", return_value=(200, json.dumps(dict(good, service="other")))):
            self.assertFalse(REAL_PROBE_API()["ok"])
        with patch.object(startup, "http_fetch", return_value=(200, json.dumps(dict(good, status="degraded")))):
            self.assertFalse(REAL_PROBE_API()["ok"])
        with patch.object(startup, "http_fetch", return_value=(200, '[]')):
            self.assertFalse(REAL_PROBE_API()["ok"])

    def test_web_health_requires_200_and_exact_ok_body(self):
        for status, body, expected in ((200, "ok", True), (200, "ok\n", True), (200, "OK", False), (500, "ok", False)):
            with patch.object(startup, "http_fetch", return_value=(status, body)):
                self.assertEqual(REAL_PROBE_WEB()["ok"], expected, (status, body))

    def test_service_commands_use_p0_interpreter_only(self):
        paths = startup.project_paths(self.root)
        api = startup.service_command("api", paths)
        web = startup.service_command("web", paths)
        self.assertEqual(api[0], str(paths["p0_python"]))
        self.assertEqual(web[0], str(paths["p0_python"]))
        self.assertIn(str(paths["serve"]), api)
        self.assertIn("streamlit", web)
        self.assertIn(str(startup.WEB_PORT), web)
        self.assertNotIn("-m", api)


if __name__ == "__main__":
    unittest.main(verbosity=2)
