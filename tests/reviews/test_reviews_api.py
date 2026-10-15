"""人工复核 API 行为验收（DEV-20261015-013）。

真实走 HTTP 接口与独立 SQLite 查询核对：追加/更正保留历史、任务与区域分开、
未知区域/非成功作业、非法与超长请求、并发幂等与冲突、事务失败回滚、分页遍历、
重启持久化。合成 runner 明确不代表真实模型；证据写入
reports/reviews/DEV-20261015-013/ 下的独立目录并保留。

用法：
  .venv-p0/Scripts/python.exe tests/reviews/test_reviews_api.py
"""
from __future__ import annotations

import base64
import http.client
import json
import sqlite3
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agrolake_service import reviews  # noqa: E402
from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from fixtures import WebFixtureRunner  # noqa: E402

OUT = ROOT / "reports" / "reviews" / "DEV-20261015-013" / datetime.now().strftime(
    "api-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)


def review_body(**changes):
    body = {
        "decision": "questioned",
        "reviewer": "流程测试署名",
        "note": "合成测试用原因，不代表真实专家意见",
        "request_id": "req-" + datetime.now().strftime("%H%M%S%f"),
    }
    body.update(changes)
    return body


class ReviewApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = OUT / self._testMethodName
        self.directory.mkdir()
        self.cfg = ServiceConfig(
            db_path=self.directory / "test.sqlite", store_root=self.directory / "store",
            runs_root=self.directory / "runs", port=0, synthetic_mode=True,
            model_sha256="f" * 64,
        )
        self.runner = WebFixtureRunner("success")
        self.manager = JobManager(self.cfg, self.runner, auto_start=True)
        self.server = make_server(self.cfg, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.manager.stop()

    # ---- helpers ----------------------------------------------------------

    def request(self, method, path, body=None, headers=None):
        headers = dict(headers or {})
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(method, path, body, headers)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, raw
        finally:
            conn.close()

    def success_job(self, content=b"SYNTHETIC-OK", **changes):
        norm = validate_job_payload({
            "filename": "ok.tif",
            "content_base64": base64.b64encode(content).decode(),
            "source": "SYNTHETIC 复核测试", "scene_id": "REVIEW",
            "request_id": "job-" + datetime.now().strftime("%H%M%S%f"),
            **changes,
        })
        job_id = self.manager.submit(norm)["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = self.manager.get_job(job_id)["status"]
            if status not in ("queued", "running"):
                assert status == "succeeded", status
                return job_id
            time.sleep(0.03)
        raise AssertionError("作业未结束")

    def failed_job(self):
        norm = validate_job_payload({
            "filename": "fail.tif",
            "content_base64": base64.b64encode(b"SYNTHETIC FAIL").decode(),
            "source": "SYNTHETIC 失败", "scene_id": "FAIL",
            "request_id": "fail-" + datetime.now().strftime("%H%M%S%f"),
        })
        job_id = self.manager.submit(norm)["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = self.manager.get_job(job_id)["status"]
            if status not in ("queued", "running"):
                assert status == "failed", status
                return job_id
            time.sleep(0.03)
        raise AssertionError("失败作业未结束")

    def review_count(self, job_id=None):
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            if job_id is None:
                return conn.execute("SELECT COUNT(*) FROM rec_reviews").fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM rec_reviews WHERE job_id=?", (job_id,)
            ).fetchone()[0]
        finally:
            conn.close()

    # ---- 正常路径与历史 ----------------------------------------------------

    def test_append_correction_keeps_history_and_separates_targets(self):
        job = self.success_job()
        # 任务整体意见
        status, raw = self.request("POST", f"/jobs/{job}/reviews", review_body(
            request_id="t1", decision="questioned", note="整体存疑"))
        self.assertEqual(status, 201)
        first = json.loads(raw)
        self.assertEqual(first["field_id"], None)
        self.assertEqual(first["target"], "task")
        self.assertTrue(first["created"])
        # 区域 1 意见
        status, raw = self.request("POST", f"/jobs/{job}/reviews", review_body(
            request_id="r1a", field_id=1, decision="accepted", note="区域可用"))
        self.assertEqual(status, 201)
        # 区域 1 更正：新请求号，新记录（不覆盖）
        status, raw = self.request("POST", f"/jobs/{job}/reviews", review_body(
            request_id="r1b", field_id=1, decision="rejected", note="复核后不采用"))
        self.assertEqual(status, 201)
        self.assertEqual(self.review_count(job), 3)

        status, raw = self.request("GET", f"/jobs/{job}/reviews")
        self.assertEqual(status, 200)
        payload = json.loads(raw)
        self.assertEqual(payload["total"], 3)
        self.assertEqual(len(payload["reviews"]), 3)
        latest = payload["latest"]
        self.assertEqual(latest["task"]["decision"], "questioned")
        self.assertEqual(latest["regions"]["1"]["decision"], "rejected")
        self.assertEqual(latest["task"]["review_id"] < latest["regions"]["1"]["review_id"], True)
        # 历史里三条都在
        ids = [r["review_id"] for r in payload["reviews"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_snapshot_taken_from_job_not_client(self):
        job = self.success_job()
        status, raw = self.request("POST", f"/jobs/{job}/reviews", review_body(
            request_id="s1",
            input_sha256="0" * 64, model_sha256="1" * 64, run_seq=999,
            created_at="1999-01-01T00:00:00", review_id=12345))
        # 客户端提交的额外字段不被接受（快照/时间/ID 由服务端决定）
        self.assertEqual(status, 400)
        status, raw = self.request("POST", f"/jobs/{job}/reviews", review_body(request_id="s2"))
        self.assertEqual(status, 201)
        rec = json.loads(raw)
        job_rec = self.manager.get_job(job)
        self.assertEqual(rec["input_sha256"], job_rec["input_sha256"])
        self.assertEqual(rec["model_sha256"], job_rec["model_sha256"])
        self.assertEqual(rec["run_seq"], 1)

    def test_idempotent_same_request_and_conflict_on_difference(self):
        job = self.success_job()
        body = review_body(request_id="dup", note="同一意见")
        s1, r1 = self.request("POST", f"/jobs/{job}/reviews", body)
        s2, r2 = self.request("POST", f"/jobs/{job}/reviews", body)
        self.assertEqual((s1, s2), (201, 200))
        self.assertEqual(json.loads(r1)["review_id"], json.loads(r2)["review_id"])
        self.assertFalse(json.loads(r2)["created"])
        self.assertEqual(self.review_count(job), 1)
        # 同号不同内容 -> 409
        s3, r3 = self.request("POST", f"/jobs/{job}/reviews",
                              review_body(request_id="dup", note="不同内容"))
        self.assertEqual(s3, 409)
        # 同号不同区域 -> 409
        s4, _ = self.request("POST", f"/jobs/{job}/reviews",
                             review_body(request_id="dup", field_id=1))
        self.assertEqual(s4, 409)
        self.assertEqual(self.review_count(job), 1)

    def test_unknown_region_and_job_and_failed_job(self):
        job = self.success_job()
        s, _ = self.request("POST", f"/jobs/{job}/reviews",
                            review_body(request_id="x1", field_id=999))
        self.assertEqual(s, 404)
        s, _ = self.request("POST", "/jobs/999999/reviews", review_body(request_id="x2"))
        self.assertEqual(s, 404)
        s, _ = self.request("GET", "/jobs/999999/reviews")
        self.assertEqual(s, 404)
        bad = self.failed_job()
        s, _ = self.request("POST", f"/jobs/{bad}/reviews", review_body(request_id="x3"))
        self.assertEqual(s, 409)
        self.assertEqual(self.review_count(), 0)

    def test_invalid_and_oversized_requests(self):
        job = self.success_job()
        cases = [
            ("非对象", []),
            ("未知字段", review_body(request_id="e1", path="/etc/passwd")),
            ("结论错误", review_body(request_id="e2", decision="maybe")),
            ("复核人空", review_body(request_id="e3", reviewer="   ")),
            ("复核人超长", review_body(request_id="e4", reviewer="A" * 81)),
            ("原因空", review_body(request_id="e5", note="")),
            ("原因超长", review_body(request_id="e6", note="N" * 2001)),
            ("缺请求号", {"decision": "accepted", "reviewer": "x", "note": "y"}),
            ("请求号非法", review_body(request_id="bad\nid")),
            ("区域为布尔", review_body(request_id="e7", field_id=True)),
            ("区域为负", review_body(request_id="e8", field_id=-1)),
            ("区域零", review_body(request_id="e9", field_id=0)),
            ("区域溢出", review_body(request_id="e10", field_id=9223372036854775808)),
            ("区域字符串", review_body(request_id="e11", field_id="1")),
        ]
        for name, body in cases:
            with self.subTest(name=name):
                status, raw = self.request("POST", f"/jobs/{job}/reviews", body)
                self.assertEqual(status, 400, f"{name}: {raw[:120]!r}")
                self.assertIn("error", json.loads(raw))
        # 超长正文（略超 16 KiB 复核正文上限）-> 413
        huge = "N" * 16400
        status, _ = self.request("POST", f"/jobs/{job}/reviews",
                                 review_body(request_id="e12", note=huge))
        self.assertEqual(status, 413)
        self.assertEqual(self.review_count(job), 0)

    def test_host_and_origin_guards_apply(self):
        job = self.success_job()
        s, _ = self.request("POST", f"/jobs/{job}/reviews", review_body(request_id="h1"),
                            {"Host": "evil.example"})
        self.assertEqual(s, 403)
        s, _ = self.request("POST", f"/jobs/{job}/reviews", review_body(request_id="h2"),
                            {"Origin": "https://example.org"})
        self.assertEqual(s, 403)
        self.assertEqual(self.review_count(job), 0)

    # ---- 并发与事务 -------------------------------------------------------

    def test_concurrent_same_request_only_one_row(self):
        job = self.success_job()
        body = review_body(request_id="concurrent", note="并发同一意见")
        gate = threading.Barrier(12)

        def run(_):
            gate.wait(15)
            return reviews.submit_review(self.cfg.db_path, job, body)

        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(run, range(12)))
        self.assertEqual(len({r["record"]["review_id"] for r in results}), 1)
        self.assertEqual(sum(1 for r in results if r["created"]), 1)
        self.assertEqual(self.review_count(job), 1)

    def test_concurrent_different_content_same_request_conflicts(self):
        job = self.success_job()
        gate = threading.Barrier(8)

        def run(i):
            gate.wait(15)
            body = review_body(request_id="race", note=f"内容 {i}")
            try:
                return ("ok", reviews.submit_review(self.cfg.db_path, job, body)["created"])
            except Exception as exc:  # noqa: BLE001
                return ("err", getattr(exc, "status", None))

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(8)))
        exact_one = [r for r in results if r[0] == "ok" and r[1] is True]
        conflicts = [r for r in results if r[0] == "err" and r[1] == 409]
        self.assertEqual(len(exact_one), 1, results)
        self.assertEqual(len(conflicts), 7, results)
        self.assertEqual(self.review_count(job), 1)

    def test_transaction_failure_rolls_back(self):
        job = self.success_job()
        with patch.object(reviews.db, "now", side_effect=RuntimeError("SYNTHETIC boom")):
            with self.assertRaises(RuntimeError):
                reviews.submit_review(self.cfg.db_path, job, review_body(request_id="roll"))
        self.assertEqual(self.review_count(job), 0)

    # ---- 分页与持久化 -----------------------------------------------------

    def test_pagination_traverses_all_records(self):
        job = self.success_job()
        for i in range(5):
            s, _ = self.request("POST", f"/jobs/{job}/reviews",
                                review_body(request_id=f"p{i}", note=f"第{i}条"))
            self.assertEqual(s, 201)
        seen, after = [], None
        pages = 0
        while True:
            query = f"?limit=2" + (f"&after_id={after}" if after is not None else "")
            s, raw = self.request("GET", f"/jobs/{job}/reviews{query}")
            self.assertEqual(s, 200)
            payload = json.loads(raw)
            seen.extend(r["review_id"] for r in payload["reviews"])
            pages += 1
            if not payload["pagination"]["has_more"]:
                break
            after = payload["pagination"]["next_after_id"]
        self.assertEqual(len(seen), 5)
        self.assertEqual(len(set(seen)), 5)
        self.assertGreaterEqual(pages, 3)
        # 非法分页参数
        self.assertEqual(self.request("GET", f"/jobs/{job}/reviews?limit=0")[0], 400)
        self.assertEqual(self.request("GET", f"/jobs/{job}/reviews?limit=99999")[0], 400)
        self.assertEqual(self.request("GET", f"/jobs/{job}/reviews?after_id=-1")[0], 400)

    def test_restart_persists_without_reprocessing(self):
        job = self.success_job()
        self.request("POST", f"/jobs/{job}/reviews", review_body(request_id="k1"))
        calls_before = self.runner.calls
        self.manager.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

        again = JobManager(self.cfg, WebFixtureRunner("success"), auto_start=True)
        self.addCleanup(again.stop)
        self.server2 = make_server(self.cfg, again)
        t2 = threading.Thread(target=self.server2.serve_forever, daemon=True)
        t2.start()
        self.addCleanup(lambda: (self.server2.shutdown(), self.server2.server_close(), t2.join(5)))
        self.port = self.server2.server_address[1]

        status, raw = self.request("GET", f"/jobs/{job}/reviews")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["total"], 1)
        self.assertEqual(calls_before, 1)  # 无额外推理


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewApiTests)
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
