"""识别输入登记单治理与血缘测试（DEV-20261015-014）。

真实走治理模块与 HTTP 接口，配合独立 SQLite 查询核对：
未登记读取不写、登记 P0 对象可重读、并发/跨作业幂等冲突、非成功/坏请求、
对象损坏与路径边界、事务中途注入失败回滚、重启读取、P0 处置后原快照保留/当前状态变化、
人工意见与产物不串任务、明确截断边界。

合成 runner 明确不代表真实模型或真实 TIFF 解码；证据写入
reports/governance/DEV-20261015-014/ 下的独立目录并保留，不删除。

用法：
  .venv-p0/Scripts/python.exe tests/governance/test_governance.py
"""
from __future__ import annotations

import base64
import http.client
import json
import os
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
for p in (str(ROOT), str(ROOT / "tests" / "web"), str(ROOT / "tests" / "service")):
    if p not in sys.path:
        sys.path.insert(0, p)

import agrolake.db as p0_db  # noqa: E402
import agrolake.disposition as p0_disposition  # noqa: E402
import agrolake.ingest as p0_ingest  # noqa: E402
from agrolake_service import governance  # noqa: E402
from agrolake_service import store  # noqa: E402
from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from agrolake_web.client import RecognitionClient  # noqa: E402
from fixtures import WebFixtureRunner  # noqa: E402

OUT = ROOT / "reports" / "governance" / "DEV-20261015-014" / datetime.now().strftime(
    "unit-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)


def rid(prefix="gov"):
    return prefix + "-" + datetime.now().strftime("%H%M%S%f")


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = OUT / self._testMethodName
        self.directory.mkdir()
        self.p0_store = self.directory / "p0_store"
        self._old_env = os.environ.get("AGROLAKE_STORE")
        os.environ["AGROLAKE_STORE"] = str(self.p0_store)  # 隔离 P0 存储根
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
        if self._old_env is None:
            os.environ.pop("AGROLAKE_STORE", None)
        else:
            os.environ["AGROLAKE_STORE"] = self._old_env

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
            return resp.status, resp.read()
        finally:
            conn.close()

    def success_job(self, content=b"SYNTHETIC-MANIFEST-OK", filename="ok.tif", **changes):
        norm = validate_job_payload({
            "filename": filename,
            "content_base64": base64.b64encode(content).decode(),
            "source": "SYNTHETIC 登记测试", "scene_id": "GOV",
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

    def counts(self):
        conn = p0_db.connect(self.cfg.db_path)
        try:
            p0_db.init_schema(conn)
            def n(table):
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            return {t: n(t) for t in ("objects", "assets", "asset_versions",
                                      "import_events", "quality_checks", "text_records",
                                      "rec_governance_links")}
        finally:
            conn.close()

    # ---- 1. 未登记读取不写 -------------------------------------------------

    def test_unregistered_read_no_write(self):
        job_id = self.success_job()
        before = self.counts()
        status, _ = self.request("GET", f"/jobs/{job_id}/lineage")
        self.assertEqual(status, 200)
        after = self.counts()
        self.assertEqual(before, after, "GET 血缘不应写入任何记录")
        self.assertEqual(after["rec_governance_links"], 0)
        self.assertEqual(after["asset_versions"], 0)

    # ---- 2. 登记 P0 目录/对象可重读 ---------------------------------------

    def test_register_and_reread(self):
        job_id = self.success_job()
        status, raw = self.request("POST", f"/jobs/{job_id}/governance",
                                   {"request_id": rid()})
        self.assertEqual(status, 201, raw)
        body = json.loads(raw)
        self.assertTrue(body["created"])
        self.assertRegex(body["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(body["manifest_sha256"], body["input_sha256"])

        # 服务登记单对象可重读且内容为稳定 JSON 登记单。
        obj = store.object_path(self.cfg.store_root, body["manifest_sha256"])
        self.assertTrue(obj.is_file())
        manifest = json.loads(obj.read_text(encoding="utf-8"))
        self.assertEqual(manifest["kind"], governance.MANIFEST_KIND)
        self.assertEqual(manifest["job_id"], job_id)
        self.assertEqual(manifest["input_sha256"], body["input_sha256"])
        self.assertNotIn("\\", obj.read_text(encoding="utf-8"))  # 不含本地路径分隔符

        # P0 文本资产与质检/导入事件真实存在。
        c = self.counts()
        self.assertGreaterEqual(c["assets"], 1)
        self.assertGreaterEqual(c["asset_versions"], 1)
        self.assertGreaterEqual(c["import_events"], 1)
        self.assertGreaterEqual(c["quality_checks"], 1)
        self.assertEqual(c["rec_governance_links"], 1)

        # 同号重试复用，不新增登记/版本/事件。
        status2, raw2 = self.request("POST", f"/jobs/{job_id}/governance",
                                     {"request_id": body["request_id"]})
        self.assertEqual(status2, 200, raw2)
        body2 = json.loads(raw2)
        self.assertFalse(body2["created"])
        self.assertEqual(body2["version_id"], body["version_id"])
        self.assertEqual(self.counts(), c)

        # GET 血缘包含登记内容与两个不同摘要。
        status3, raw3 = self.request("GET", f"/jobs/{job_id}/lineage")
        self.assertEqual(status3, 200)
        lin = json.loads(raw3)
        self.assertTrue(lin["registered"])
        self.assertEqual(lin["registration"]["manifest_sha256"], body["manifest_sha256"])
        self.assertEqual(lin["registration"]["input_sha256"], body["input_sha256"])
        self.assertIn("quality_at_registration", lin["registration"])
        self.assertIn("current", lin["registration"])

    # ---- 3. 并发与跨作业幂等冲突 ------------------------------------------

    def test_concurrent_idempotency_and_cross_job_conflict(self):
        job_a = self.success_job(content=b"JOB-A")
        job_b = self.success_job(content=b"JOB-B")
        same = rid("same")

        def post(job):
            return self.request("POST", f"/jobs/{job}/governance", {"request_id": same})

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(post, (job_a, job_b)))
        statuses = sorted(s for s, _ in results)
        self.assertIn(409, statuses, results)
        self.assertLessEqual(statuses.count(201) + statuses.count(200), 1)
        # 只有一个作业真正建立登记。
        c = self.counts()
        self.assertEqual(c["rec_governance_links"], 1)

        # 同作业另号不能多建登记。
        winner = job_a if json.loads(
            next(r for s, r in results if s in (200, 201))
        )["job_id"] == job_a else job_b
        status, raw = self.request("POST", f"/jobs/{winner}/governance",
                                   {"request_id": rid("other")})
        self.assertEqual(status, 409, raw)
        self.assertEqual(self.counts()["rec_governance_links"], 1)

    # ---- 4. 非成功 / 坏请求 ------------------------------------------------

    def test_non_success_and_bad_request(self):
        job = self.failed_job()
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        self.assertEqual(status, 409, raw)
        status, _ = self.request("POST", "/jobs/999999/governance", {"request_id": rid()})
        self.assertEqual(status, 404)
        status, _ = self.request("GET", "/jobs/999999/lineage")
        self.assertEqual(status, 404)
        # 多余字段与非法请求号。
        good = self.success_job()
        status, _ = self.request("POST", f"/jobs/{good}/governance",
                                 {"request_id": rid(), "asset_id": 1})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", f"/jobs/{good}/governance", {"request_id": "坏 值"})
        self.assertEqual(status, 400)
        status, _ = self.request("POST", f"/jobs/{good}/governance", {})
        self.assertEqual(status, 400)

    # ---- 5. 对象损坏 / 路径边界 -------------------------------------------

    def test_object_corruption_and_path_boundary(self):
        job = self.success_job(content=b"CORRUPT-ME")
        jobrow = self.manager.get_job(job)
        sha = jobrow["input_sha256"]

        # 5a. 输入对象损坏 -> 409，不把损坏当验证成功。
        obj = store.object_path(self.cfg.store_root, sha)
        original = obj.read_bytes()
        obj.write_bytes(original[:-1])  # 截断一个字节
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        self.assertEqual(status, 409, raw)
        obj.write_bytes(original)  # 恢复，便于后续步骤继续（不删除）

        # 5b. run.json 产物损坏 -> 409。
        run_rel = jobrow.get("cli_output_relpath") or ""
        # 从数据库中取 cli_output_relpath（get_job 未直接返回）。
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            run_rel = conn.execute(
                "SELECT cli_output_relpath FROM rec_jobs WHERE job_id=?", (job,)
            ).fetchone()[0]
        finally:
            conn.close()
        run_json = Path(self.cfg.runs_root) / run_rel / "run.json"
        rj_original = run_json.read_bytes()
        run_json.write_bytes(rj_original + b" ")
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        self.assertEqual(status, 409, raw)
        run_json.write_bytes(rj_original)

        # 5c. 路径边界：篡改输入对象相对路径为越界值 -> 409。
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            conn.execute("UPDATE rec_inputs SET object_relpath='../escape' WHERE sha256=?", (sha,))
            conn.commit()
        finally:
            conn.close()
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        self.assertEqual(status, 409, raw)

    # ---- 6. 事务中途注入失败回滚 ------------------------------------------

    def test_transaction_rollback_on_injection(self):
        job = self.success_job(content=b"ROLLBACK")
        before = self.counts()
        real = p0_ingest.import_bytes

        def boom(conn, data, *args, **kwargs):
            real(conn, data, *args, **kwargs)
            raise RuntimeError("injected mid-transaction failure")

        with patch("agrolake.ingest.import_bytes", boom):
            with self.assertRaises(RuntimeError):
                governance.register_governance(self.cfg, job, {"request_id": rid()})

        after = self.counts()
        for table in ("objects", "assets", "asset_versions", "import_events",
                      "quality_checks", "text_records", "rec_governance_links"):
            self.assertEqual(before[table], after[table], f"{table} 未随事务回滚")
        # 登记单服务对象（孤立文件）保留，不删除。
        lin_status, lin_raw = self.request("GET", f"/jobs/{job}/lineage")
        self.assertEqual(lin_status, 200)
        self.assertFalse(json.loads(lin_raw)["registered"])

    # ---- 7. 重启后读取 -----------------------------------------------------

    def test_restart_read(self):
        job = self.success_job(content=b"RESTART")
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        body = json.loads(raw)
        self.server.shutdown(); self.server.server_close(); self.thread.join(5)
        self.manager.stop()

        # 新 manager/服务进程重新打开同一数据库与存储。
        cfg2 = ServiceConfig(
            db_path=self.cfg.db_path, store_root=self.cfg.store_root,
            runs_root=self.cfg.runs_root, port=0, synthetic_mode=True, model_sha256="f" * 64,
        )
        mgr2 = JobManager(cfg2, WebFixtureRunner("success"), auto_start=True)
        srv2 = make_server(cfg2, mgr2)
        t2 = threading.Thread(target=srv2.serve_forever, daemon=True)
        t2.start()
        try:
            port2 = srv2.server_address[1]
            conn = http.client.HTTPConnection("127.0.0.1", port2, timeout=30)
            try:
                conn.request("GET", f"/jobs/{job}/lineage")
                resp = conn.getresponse()
                lin = json.loads(resp.read())
            finally:
                conn.close()
            self.assertTrue(lin["registered"])
            self.assertEqual(lin["registration"]["version_id"], body["version_id"])
            self.assertEqual(lin["integrity"]["input_object_intact"], True)
            self.assertEqual(lin["integrity"]["manifest_p0_object_intact"], True)
        finally:
            srv2.shutdown(); srv2.server_close(); t2.join(5); mgr2.stop()

    # ---- 8. P0 处置后原快照保留 / 当前状态变化 ----------------------------

    def test_disposition_keeps_snapshot_current_changes(self):
        job = self.success_job(content=b"DISPOSE")
        status, raw = self.request("POST", f"/jobs/{job}/governance", {"request_id": rid()})
        body = json.loads(raw)
        version_id = body["version_id"]

        _, raw0 = self.request("GET", f"/jobs/{job}/lineage")
        snap_before = json.loads(raw0)["registration"]["quality_at_registration"]

        conn = p0_db.connect(self.cfg.db_path)
        try:
            p0_db.init_schema(conn)
            p0_disposition.isolate(conn, version_id, "合成测试", "测试隔离")
            p0_disposition.set_source(conn, version_id, "SYNTHETIC 更改来源", "合成测试", "改来源",
                                      store_root=self.p0_store)
        finally:
            conn.close()

        status, raw = self.request("GET", f"/jobs/{job}/lineage")
        lin = json.loads(raw)
        reg = lin["registration"]
        # 登记时快照仍是登记时内容，不被处置覆盖。
        self.assertEqual(reg["quality_at_registration"], snap_before)
        self.assertEqual(reg["current"]["version"]["status"], "isolated")
        self.assertTrue(reg["current"]["diff"]["source_changed"])
        self.assertRegex(reg["manifest_sha256"], r"^[0-9a-f]{64}$")

    # ---- 9. 人工意见与产物不串任务 ----------------------------------------

    def test_reviews_artifacts_no_cross_job(self):
        job_a = self.success_job(content=b"JOB-A-REVIEW")
        job_b = self.success_job(content=b"JOB-B-REVIEW")
        status, _ = self.request("POST", f"/jobs/{job_a}/reviews", {
            "decision": "questioned", "reviewer": "合成署名",
            "note": "合成意见，不代表真实专家意见", "request_id": rid("rev"),
        })
        self.assertEqual(status, 201)
        _, raw_a = self.request("GET", f"/jobs/{job_a}/lineage")
        _, raw_b = self.request("GET", f"/jobs/{job_b}/lineage")
        lin_a = json.loads(raw_a)
        lin_b = json.loads(raw_b)
        self.assertEqual(lin_a["reviews"]["total"], 1)
        self.assertEqual(lin_b["reviews"]["total"], 0)
        self.assertEqual(lin_b["reviews"]["items"], [])
        sha_b = self.manager.get_job(job_b)["input_sha256"]
        for art in lin_b["artifacts"]:
            self.assertNotEqual(art["sha256"], sha_b)  # 产物属于各自任务登记

    # ---- 10. 明确截断边界 --------------------------------------------------

    def test_truncation_boundary(self):
        job = self.success_job(content=b"TRUNCATE")
        for i in range(3):
            status, _ = self.request("POST", f"/jobs/{job}/reviews", {
                "decision": "accepted", "reviewer": "合成署名",
                "note": f"合成意见 {i}", "request_id": rid(f"rev{i}"),
            })
            self.assertEqual(status, 201)
        with patch.object(governance, "MAX_LINEAGE_REVIEWS", 1):
            status, raw = self.request("GET", f"/jobs/{job}/lineage")
        self.assertEqual(status, 200)
        reviews = json.loads(raw)["reviews"]
        self.assertEqual(reviews["total"], 3)
        self.assertEqual(reviews["returned"], 1)
        self.assertTrue(reviews["truncated"])
        self.assertEqual(reviews["limit"], 1)


class GovernanceHttpClientTests(unittest.TestCase):
    """通过 agrolake_web 客户端走真实 HTTP，覆盖回执一致性检查。"""

    def setUp(self):
        self.directory = OUT / self._testMethodName
        self.directory.mkdir()
        self.p0_store = self.directory / "p0_store"
        self._old_env = os.environ.get("AGROLAKE_STORE")
        os.environ["AGROLAKE_STORE"] = str(self.p0_store)
        self.cfg = ServiceConfig(
            db_path=self.directory / "test.sqlite", store_root=self.directory / "store",
            runs_root=self.directory / "runs", port=0, synthetic_mode=True,
            model_sha256="f" * 64,
        )
        self.manager = JobManager(self.cfg, WebFixtureRunner("success"), auto_start=True)
        self.server = make_server(self.cfg, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = RecognitionClient(f"http://127.0.0.1:{self.server.server_address[1]}")

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(5)
        self.manager.stop()
        if self._old_env is None:
            os.environ.pop("AGROLAKE_STORE", None)
        else:
            os.environ["AGROLAKE_STORE"] = self._old_env

    def test_client_register_and_lineage(self):
        norm = validate_job_payload({
            "filename": "client.tif",
            "content_base64": base64.b64encode(b"CLIENT-JOB").decode(),
            "source": "SYNTHETIC 客户端测试", "scene_id": "CLIENT",
            "request_id": rid("job"),
        })
        job_id = self.manager.submit(norm)["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and self.manager.get_job(job_id)["status"] in (
                "queued", "running"):
            time.sleep(0.03)
        self.assertEqual(self.manager.get_job(job_id)["status"], "succeeded")

        result = self.client.register_governance(job_id, request_id=rid("c"))
        self.assertIn("asset_id", result)
        lineage = self.client.get_lineage(job_id)
        self.assertTrue(lineage["registered"])
        self.assertEqual(lineage["job_id"], job_id)
        # 回执不一致检查：服务端会拒绝跨任务请求号。
        other = self.manager.submit(validate_job_payload({
            "filename": "client2.tif",
            "content_base64": base64.b64encode(b"CLIENT-JOB-2").decode(),
            "source": "SYNTHETIC 客户端测试", "scene_id": "CLIENT",
            "request_id": rid("job"),
        }))["job_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and self.manager.get_job(other)["status"] in (
                "queued", "running"):
            time.sleep(0.03)
        self.assertEqual(self.manager.get_job(other)["status"], "succeeded")
        with self.assertRaises(Exception):
            self.client.register_governance(other, request_id=result["request_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
