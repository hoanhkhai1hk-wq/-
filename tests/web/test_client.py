"""agrolake_web.client 客户端行为测试（DEV-20261015-012）。

覆盖：成功JSON/二进制、错误状态、离线、超时、坏JSON、本机地址限制、
响应上限、摘要不符、409冲突。服务端全部为本测试的合成假服务，不代表真实识别。
所有证据写入 reports/web/DEV-20261015-012/ 下的独立目录并保留。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_client.py
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrolake_web import client as C  # noqa: E402
from fixtures import free_closed_port, start_fake_server  # noqa: E402

OUT = ROOT / "reports" / "web" / "DEV-20261015-012" / datetime.now().strftime(
    "unit-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)


def ok_json(obj):
    return 200, {}, json.dumps(obj, ensure_ascii=False).encode("utf-8")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self._servers = []

    def tearDown(self):
        for stop in self._servers:
            stop()

    def serve(self, responder):
        server, url, stop = start_fake_server(responder)
        self._servers.append(stop)
        return url

    # ---- 地址与配置 -------------------------------------------------------

    def test_local_addresses_accepted(self):
        self.assertEqual(C.resolve_base_url("http://127.0.0.1:8506"), "http://127.0.0.1:8506")
        self.assertEqual(C.resolve_base_url("http://localhost:9000"), "http://localhost:9000")
        self.assertEqual(C.resolve_base_url("http://127.0.0.1:8506/"),
                         "http://127.0.0.1:8506")

    def test_remote_and_unsafe_addresses_rejected(self):
        bad = ["http://example.com:8506", "https://10.0.0.5:8506",
               "http://user:pass@127.0.0.1:8506", "http://127.0.0.1:8506/api",
               "http://127.0.0.1:8506?a=1", "http://127.0.0.1:8506#x",
               "ftp://127.0.0.1:8506", "http://127.0.0.1:99999"]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(C.ApiConfigError):
                C.resolve_base_url(value)

    # ---- 正常路径 ---------------------------------------------------------

    def test_job_detail_success_json(self):
        detail = {"job_id": 5, "status": "succeeded", "class_counts": {"1": 10}}

        def responder(method, path, headers, body):
            self.assertEqual(path, "/jobs/5")
            return ok_json(detail)

        url = self.serve(responder)
        self.assertEqual(C.RecognitionClient(url).get_job(5), detail)

    def test_list_jobs_and_protocol_error(self):
        url = self.serve(lambda m, p, h, b: ok_json({"jobs": [{"job_id": 1,'status':'queued'}]}))
        self.assertEqual(C.RecognitionClient(url).list_jobs(), [{"job_id": 1,'status':'queued'}])

        url2 = self.serve(lambda m, p, h, b: ok_json({"nope": []}))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url2).list_jobs()
        self.assertEqual(caught.exception.category, "protocol")

    def test_submit_job_success_and_missing_id(self):
        url = self.serve(lambda m, p, h, b: ok_json({"job_id": 7, "created": True}))
        self.assertEqual(C.RecognitionClient(url).submit_job({"filename": "x.tif"})["job_id"], 7)

        url2 = self.serve(lambda m, p, h, b: ok_json({"created": True}))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url2).submit_job({"filename": "x.tif"})
        self.assertEqual(caught.exception.category, "protocol")

    def test_binary_artifact_and_checksum(self):
        payload = b"\x89PNG\r\n\x1a\nSYNTHETIC-BYTES"
        expected = __import__("hashlib").sha256(payload).hexdigest()

        def responder(method, path, headers, body):
            return 200, {"Content-Type": "image/png"}, payload

        url = self.serve(responder)
        client = C.RecognitionClient(url)
        self.assertEqual(client.get_artifact(1, "preview.png", expected_sha256=expected), payload)
        with self.assertRaises(C.ApiError) as caught:
            client.get_artifact(1, "preview.png", expected_sha256="0" * 64)
        self.assertEqual(caught.exception.category, "checksum")
        with self.assertRaises(C.ApiError) as caught2:
            client.get_artifact(1, "unknown.txt")
        self.assertEqual(caught2.exception.category, "protocol")

    # ---- 错误分类 ---------------------------------------------------------

    def test_error_status_maps_to_http(self):
        url = self.serve(lambda m, p, h, b: (404, {}, json.dumps(
            {"error": "未知作业 9"}, ensure_ascii=False).encode()))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_job(9)
        err = caught.exception
        self.assertEqual(err.category, "http")
        self.assertEqual(err.status, 404)
        self.assertIn("未知作业 9", err.message)

    def test_conflict_409(self):
        url = self.serve(lambda m, p, h, b: (409, {}, json.dumps(
            {"error": "识别产物已变化"}, ensure_ascii=False).encode()))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).get_artifact(1, "mask.tif")
        self.assertEqual(caught.exception.category, "conflict")

    def test_bad_json(self):
        url = self.serve(lambda m, p, h, b: (200, {"Content-Type": "application/json"},
                                             b"not-a-json-body"))
        with self.assertRaises(C.ApiError) as caught:
            C.RecognitionClient(url).health()
        self.assertEqual(caught.exception.category, "badjson")

    def test_response_too_large(self):
        big = b"x" * 5000
        url = self.serve(lambda m, p, h, b: (200, {"Content-Type": "application/json"}, big))
        client = C.RecognitionClient(url, max_json_bytes=100)
        with self.assertRaises(C.ApiError) as caught:
            client.health()
        self.assertEqual(caught.exception.category, "toolarge")

    def test_offline(self):
        port = free_closed_port()
        # 本机对关闭端口的连接在约 2 秒后才返回“积极拒绝”，超时要放宽，
        # 否则会把“连接被拒”误判成超时。这里用 5 秒确保拿到 offline 分类。
        client = C.RecognitionClient(f"http://127.0.0.1:{port}", timeout=5)
        with self.assertRaises(C.ApiError) as caught:
            client.health()
        self.assertEqual(caught.exception.category, "offline")

    def test_timeout(self):
        def responder(method, path, headers, body):
            time.sleep(2.5)
            return ok_json({"status": "ok"})

        url = self.serve(responder)
        client = C.RecognitionClient(url, timeout=0.5)
        with self.assertRaises(C.ApiError) as caught:
            client.health()
        self.assertEqual(caught.exception.category, "timeout")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests)
    with (OUT / "tests.log").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {"tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "successful": result.wasSuccessful(), "directory": str(OUT)}
    (OUT / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print((OUT / "tests.log").read_text(encoding="utf-8"))
    sys.exit(0 if result.wasSuccessful() else 1)
