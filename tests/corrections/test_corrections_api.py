"""整改版本与显式复跑 API 测试（DEV-20261015-015）。

真实走 corrections 模块与 HTTP 接口，配合独立 SQLite 查询核对：
- 正常元数据整改与输入替换；原事实与输入文件不变；新作业只一次；
- 非法/越界/旧意见/accepted/跨 job 拒绝；
- 并发同号/跨 parent 冲突；准备事务中途失败回滚；
- manager 接受子 job 后模拟丢回执/链接失败，同号恢复且不重复推理；
- 重启 GET 不自动提交 prepared；已中断子 job 不重放；
- 父子计数/NoData 分母与治理 ID/人工意见不串；
- 每 root 50 条容量边界后旧号仍可读/确认。

合成 fixture 明确不代表真实模型或真实 TIFF 解码；证据写入
reports/corrections/DEV-20261015-015/ 下的独立目录并保留，不删除。

用法：
  .venv-p0/Scripts/python.exe tests/corrections/test_corrections_api.py
"""
from __future__ import annotations

import base64
import hashlib
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

from agrolake_service import corrections, governance, reviews, store  # noqa: E402
from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from fixtures import WebFixtureRunner  # noqa: E402

OUT = ROOT / "reports" / "corrections" / "DEV-20261015-015" / datetime.now().strftime(
    "unit-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)


def rid(prefix="corr"):
    return prefix + "-" + datetime.now().strftime("%H%M%S%f") + "-" + str(id(object()))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class CorrectionTests(unittest.TestCase):
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

    def wait_job(self, job_id, want="succeeded", timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.manager.get_job(job_id)["status"]
            if status not in ("queued", "running"):
                assert status == want, f"作业 {job_id} 期望 {want} 实为 {status}"
                return status
            time.sleep(0.03)
        raise AssertionError(f"作业 {job_id} 未结束")

    def success_job(self, content=b"SYNTHETIC-PARENT", filename="parent.tif",
                    source="SYNTHETIC 整改测试", scene_id="CORR"):
        norm = validate_job_payload({
            "filename": filename, "content_base64": b64(content),
            "source": source, "scene_id": scene_id, "request_id": rid("job"),
        })
        job_id = self.manager.submit(norm)["job_id"]
        self.wait_job(job_id, "succeeded")
        return job_id

    def register(self, job_id):
        return governance.register_governance(
            self.cfg, job_id, {"request_id": rid("gov")}
        )["registration"]

    def add_review(self, job_id, decision="questioned", field_id=None, note="合成意见"):
        norm = reviews.normalize_review_payload({
            "field_id": field_id, "decision": decision,
            "reviewer": "合成署名", "note": note, "request_id": rid("rev"),
        })
        return reviews.submit_review(self.cfg.db_path, job_id, norm)["record"]["review_id"]

    def correction_body(self, review_id, request_id=None, operator="合成操作者",
                        reason="合成整改原因", **extra):
        body = {"request_id": request_id or rid(), "review_id": review_id,
                "operator": operator, "reason": reason}
        body.update(extra)
        return body

    def job_row(self, job_id):
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return dict(conn.execute(
                "SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone())
        finally:
            conn.close()

    def table_count(self, table):
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()

    def parent_input_bytes(self, job_id):
        row = self.job_row(job_id)
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            rel = conn.execute(
                "SELECT object_relpath FROM rec_inputs WHERE input_id = ?",
                (row["input_id"],)).fetchone()[0]
        finally:
            conn.close()
        return (Path(self.cfg.store_root) / rel).read_bytes()

    # ---- 1. 元数据整改 -----------------------------------------------------

    def test_metadata_correction_creates_child(self):
        parent = self.success_job(b"PARENT-META-1")
        reg = self.register(parent)
        review_id = self.add_review(parent, "questioned")
        before_row = self.job_row(parent)
        before_bytes = self.parent_input_bytes(parent)
        before_arts = self.manager.get_job(parent)["artifact_checksums"]
        before_links = self.table_count("rec_governance_links")
        before_reviews = self.table_count("rec_reviews")
        calls_before = self.runner.calls

        status, raw = self.request("POST", f"/jobs/{parent}/corrections",
                                   self.correction_body(review_id, changes={"scene_id": "CORRECTED"}))
        self.assertEqual(status, 201, raw)
        body = json.loads(raw)
        self.assertTrue(body["created"])
        self.assertEqual(body["parent_job_id"], parent)
        self.assertEqual(body["root_job_id"], parent)
        self.assertEqual(body["revision_no"], 1)
        self.assertEqual(body["changes"], {"scene_id": "CORRECTED"})
        self.assertIsNotNone(body["child_job_id"])
        child = body["child_job_id"]
        self.wait_job(child, "succeeded")
        self.assertEqual(self.runner.calls, calls_before + 1, "新作业只应推理一次")

        # 父作业原事实、输入文件、产物、意见、治理链接均不变。
        after_row = self.job_row(parent)
        for key in ("source", "scene_id", "source_link", "license_status",
                    "input_sha256", "model_sha256", "finished_at", "status"):
            self.assertEqual(before_row[key], after_row[key], f"父作业 {key} 被改动")
        self.assertEqual(self.parent_input_bytes(parent), before_bytes, "父输入文件被改动")
        self.assertEqual(self.manager.get_job(parent)["artifact_checksums"], before_arts)
        self.assertEqual(self.table_count("rec_governance_links"), before_links)
        self.assertEqual(self.table_count("rec_reviews"), before_reviews)

        # 子作业沿用父来源、套用修订场景，同模型。
        child_row = self.job_row(child)
        self.assertEqual(child_row["scene_id"], "CORRECTED")
        self.assertEqual(child_row["source"], before_row["source"])
        self.assertEqual(child_row["model_sha256"], before_row["model_sha256"])

        # 详情：对照明确，父治理 ID 与子空治理不串。
        status, raw = self.request("GET", f"/corrections/{body['correction_id']}")
        self.assertEqual(status, 200, raw)
        detail = json.loads(raw)
        comp = detail["comparison"]
        self.assertTrue(comp["child_succeeded"])
        self.assertTrue(comp["same_model_sha256"])
        self.assertTrue(comp["same_input_sha256"])  # 未替换输入
        self.assertEqual(comp["changed_fields"], ["scene_id"])
        self.assertFalse(comp["bytes_changed"])
        self.assertEqual(detail["parent"]["governance"]["version_id"], reg["version_id"])
        self.assertIsNone(detail["child"]["governance"])
        self.assertEqual(comp["parent_problem_review"]["review_id"], review_id)

    # ---- 2. 输入替换 -------------------------------------------------------

    def test_input_replacement(self):
        parent = self.success_job(b"PARENT-REPLACE-1")
        self.register(parent)
        review_id = self.add_review(parent, "rejected")
        before_bytes = self.parent_input_bytes(parent)

        new_content = b"REPLACED-DIFFERENT-CONTENT"
        status, raw = self.request("POST", f"/jobs/{parent}/corrections",
                                   self.correction_body(review_id, replacement={
                                       "filename": "new.tif", "content_base64": b64(new_content)}))
        self.assertEqual(status, 201, raw)
        body = json.loads(raw)
        child = body["child_job_id"]
        self.wait_job(child, "succeeded")

        child_row = self.job_row(child)
        self.assertEqual(child_row["input_sha256"],
                         hashlib.sha256(new_content).hexdigest())
        self.assertEqual(child_row["original_filename"], "new.tif")
        # 父输入文件与摘要不变。
        self.assertEqual(self.parent_input_bytes(parent), before_bytes)
        self.assertEqual(self.job_row(parent)["input_sha256"],
                         hashlib.sha256(b"PARENT-REPLACE-1").hexdigest())

        _, raw = self.request("GET", f"/corrections/{body['correction_id']}")
        comp = json.loads(raw)["comparison"]
        self.assertTrue(comp["bytes_changed"])
        self.assertFalse(comp["same_input_sha256"])
        self.assertEqual(comp["changed_fields"], [])  # 仅替换字节

    # ---- 3. 幂等与不重复推理 ----------------------------------------------

    def test_idempotent_replay_no_double_run(self):
        parent = self.success_job(b"PARENT-IDEM-1")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        req = rid("same")
        body = self.correction_body(review_id, request_id=req, changes={"source": "SYNTHETIC 改来源"})

        s1, raw1 = self.request("POST", f"/jobs/{parent}/corrections", body)
        self.assertEqual(s1, 201, raw1)
        first = json.loads(raw1)
        self.wait_job(first["child_job_id"], "succeeded")
        jobs_after_first = self.table_count("rec_jobs")
        calls_after_first = self.runner.calls

        s2, raw2 = self.request("POST", f"/jobs/{parent}/corrections", body)
        self.assertEqual(s2, 200, raw2)
        second = json.loads(raw2)
        self.assertFalse(second["created"])
        self.assertEqual(second["correction_id"], first["correction_id"])
        self.assertEqual(self.table_count("rec_jobs"), jobs_after_first)
        self.assertEqual(self.runner.calls, calls_after_first, "重放不得再次推理")

        # 同号不同内容 -> 409。
        s3, raw3 = self.request("POST", f"/jobs/{parent}/corrections",
                                self.correction_body(review_id, request_id=req,
                                                     changes={"source": "SYNTHETIC 另一个来源"}))
        self.assertEqual(s3, 409, raw3)

    # ---- 4. 拒绝边界 -------------------------------------------------------

    def test_rejections(self):
        # 未知作业 404。
        s, _ = self.request("POST", "/jobs/999999/corrections",
                            self.correction_body(1, changes={"source": "未知来源"}))
        self.assertEqual(s, 404)
        # 未登记治理的成功作业 409。
        lone = self.success_job(b"PARENT-NO-REG")
        lone_review = self.add_review(lone, "questioned")
        s, _ = self.request("POST", f"/jobs/{lone}/corrections",
                            self.correction_body(lone_review, changes={"source": "X 来源"}))
        self.assertEqual(s, 409)

        parent = self.success_job(b"PARENT-REJECT")
        self.register(parent)
        good_review = self.add_review(parent, "questioned")

        # accepted 不能作来源。
        accepted = self.add_review(parent, "accepted")
        s, _ = self.request("POST", f"/jobs/{parent}/corrections",
                            self.correction_body(accepted, changes={"source": "Y 来源"}))
        self.assertEqual(s, 409)

        # 已被后续意见取代 -> 409（good_review 之后又有 accepted）。
        s, _ = self.request("POST", f"/jobs/{parent}/corrections",
                            self.correction_body(good_review, changes={"source": "Z 来源"}))
        self.assertEqual(s, 409)

        # 跨作业 / 不存在意见 404。
        other = self.success_job(b"PARENT-OTHER")
        self.register(other)
        other_review = self.add_review(other, "questioned")
        s, _ = self.request("POST", f"/jobs/{parent}/corrections",
                            self.correction_body(other_review, changes={"source": "W 来源"}))
        self.assertEqual(s, 404)
        s, _ = self.request("POST", f"/jobs/{parent}/corrections",
                            self.correction_body(999999, changes={"source": "V 来源"}))
        self.assertEqual(s, 404)

        # 最新 questioned 意见可用于；构造非法请求体。
        latest = self.add_review(parent, "questioned")
        base = self.correction_body(latest, changes={"source": "U 来源"})
        for bad in (
            {**base, "review_id": True},          # bool 不算整数
            {**base, "review_id": 0},             # 越界
            {**base, "operator": ""},             # 空
            {**base, "operator": "x" * 81},       # 超长
            {**base, "reason": ""},               # 空
            {**base, "unknown": 1},               # 未知字段
            {**base, "changes": {"bogus": "x"}},  # 非法 changes 键
            {**base, "changes": {"source": 1}},   # 类型错误
            {**base, "changes": {"source": "U 来源", "replacement": {}}},  # 无真实变化+空替换
            {**base, "replacement": {"filename": "a.tif",
                                     "content_base64": b64(b"x"), "extra": 1}},  # 三字段
            {**base, "replacement": {"filename": "a.png", "content_base64": b64(b"x")}},  # 非 tif
            {**base, "replacement": {"filename": "a.tif", "content_base64": "!!not base64!!"}},
            {**base, "changes": {"source": "SYNTHETIC 整改测试"}},  # 与父来源相同 -> 无真实变化
        ):
            s, raw = self.request("POST", f"/jobs/{parent}/corrections", bad)
            self.assertEqual(s, 400, (bad, raw))

        # 仅改文件名（字节不变）不算整改 -> 400。
        parent_bytes = self.parent_input_bytes(parent)
        s, _ = self.request("POST", f"/jobs/{parent}/corrections",
                            self.correction_body(latest, replacement={
                                "filename": "renamed.tif", "content_base64": b64(parent_bytes)}))
        self.assertEqual(s, 400)

    # ---- 5. 并发同号 / 跨 parent 冲突 --------------------------------------

    def test_concurrent_same_number_cross_parent(self):
        p1 = self.success_job(b"PARENT-CONC-1")
        p2 = self.success_job(b"PARENT-CONC-2")
        self.register(p1)
        self.register(p2)
        r1 = self.add_review(p1, "questioned")
        r2 = self.add_review(p2, "questioned")
        same = rid("shared")

        def post(args):
            parent, review_id = args
            return self.request("POST", f"/jobs/{parent}/corrections",
                                self.correction_body(review_id, request_id=same,
                                                     changes={"source": "SYNTHETIC 并发"}))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(post, ((p1, r1), (p2, r2))))
        statuses = sorted(s for s, _ in results)
        self.assertIn(409, statuses, results)
        self.assertLessEqual(statuses.count(201) + statuses.count(200), 1)
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            n = conn.execute("SELECT COUNT(*) FROM rec_corrections WHERE request_id = ?",
                             (same,)).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(n, 1)

    # ---- 6. 准备事务中途失败回滚 ------------------------------------------

    def test_prepare_transaction_rollback(self):
        parent = self.success_job(b"PARENT-ROLLBACK")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        before_corr = self.table_count("rec_corrections")
        before_jobs = self.table_count("rec_jobs")

        def boom(*args, **kwargs):
            raise RuntimeError("injected mid-prepare failure")

        with patch.object(corrections, "_build_snapshot", boom):
            with self.assertRaises(RuntimeError):
                corrections.submit_correction(
                    self.manager, parent,
                    self.correction_body(review_id, replacement={
                        "filename": "rollback.tif", "content_base64": b64(b"ROLLBACK-BYTES")}))

        self.assertEqual(self.table_count("rec_corrections"), before_corr, "回滚后不得留下半记录")
        self.assertEqual(self.table_count("rec_jobs"), before_jobs)

    # ---- 7. 丢回执 / 链接失败，同号恢复不重复推理 --------------------------

    def test_link_failure_then_recover(self):
        parent = self.success_job(b"PARENT-LINK-BREAK")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        body = self.correction_body(review_id, changes={"source": "SYNTHETIC 恢复"})
        calls_before = self.runner.calls

        with patch.object(corrections, "_link_child",
                          side_effect=sqlite3.OperationalError("simulated link failure")):
            with self.assertRaises(corrections.ValidationError) as ctx:
                corrections.submit_correction(self.manager, parent, body)
        self.assertEqual(ctx.exception.status,503)

        # 记录保留 prepared，子作业已提交但尚未链接。
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = dict(conn.execute(
                "SELECT * FROM rec_corrections WHERE request_id = ?",
                (body["request_id"],)).fetchone())
            child = conn.execute(
                "SELECT job_id FROM rec_requests WHERE request_id = ?",
                (corrections._derive_child_request_id(body["request_id"]),)).fetchone()
        finally:
            conn.close()
        self.assertEqual(row["submission_state"], "prepared")
        self.assertIsNone(row["child_job_id"])
        self.assertIsNotNone(child, "第一次提交应已创建子作业")
        self.wait_job(child["job_id"], "succeeded")
        self.assertEqual(self.runner.calls, calls_before + 1, "第一次提交应推理一次")

        # 同号重试：只补链接，不再推理。
        outcome = corrections.submit_correction(self.manager, parent, body)
        self.assertFalse(outcome["created"])
        self.assertIsNotNone(outcome["record"]["child_job_id"])
        self.assertEqual(outcome["record"]["submission_state"], "linked")
        self.assertEqual(self.runner.calls, calls_before + 1, "恢复不得重复推理")

    # ---- 8. 提交失败 503 保留 prepared，可同号恢复 ------------------------

    def test_submit_failure_keeps_prepared_and_recovers(self):
        parent = self.success_job(b"PARENT-SUBMIT-BREAK")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        body = self.correction_body(review_id, changes={"source": "SYNTHETIC 503"})

        with patch.object(self.manager, "submit",
                          side_effect=sqlite3.OperationalError("simulated submit failure")):
            with self.assertRaises(corrections.ValidationError) as ctx:
                corrections.submit_correction(self.manager, parent, body)
        self.assertEqual(ctx.exception.status, 503)
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = dict(conn.execute(
                "SELECT * FROM rec_corrections WHERE request_id = ?",
                (body["request_id"],)).fetchone())
        finally:
            conn.close()
        self.assertEqual(row["submission_state"], "prepared")
        self.assertIsNone(row["child_job_id"])
        self.assertIsNotNone(row["error"])

        outcome = corrections.submit_correction(self.manager, parent, body)
        self.assertFalse(outcome["created"])
        self.assertIsNotNone(outcome["record"]["child_job_id"])

    # ---- 9. 重启 GET 不自动提交 prepared；中断子 job 不重放 --------------

    def test_restart_get_no_autosubmit(self):
        parent = self.success_job(b"PARENT-RESTART")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        body = self.correction_body(review_id, changes={"source": "SYNTHETIC 重启"})

        with patch.object(corrections, "_link_child",
                          side_effect=sqlite3.OperationalError("simulated")):
            with self.assertRaises(corrections.ValidationError) as ctx:
                corrections.submit_correction(self.manager, parent, body)
        self.assertEqual(ctx.exception.status,503)
        jobs_before = self.table_count("rec_jobs")
        calls_before = self.runner.calls

        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cid = conn.execute("SELECT correction_id FROM rec_corrections WHERE request_id = ?",
                               (body["request_id"],)).fetchone()[0]
        finally:
            conn.close()

        # 当前进程 GET：不自动提交。
        s, raw = self.request("GET", f"/corrections/{cid}")
        self.assertEqual(s, 200, raw)
        self.assertEqual(json.loads(raw)["correction"]["submission_state"], "prepared")
        self.assertEqual(self.table_count("rec_jobs"), jobs_before)

        # 重启服务/manager 后 GET：依旧不自动提交。
        self.server.shutdown(); self.server.server_close(); self.thread.join(5)
        self.manager.stop()
        cfg2 = ServiceConfig(
            db_path=self.cfg.db_path, store_root=self.cfg.store_root,
            runs_root=self.cfg.runs_root, port=0, synthetic_mode=True, model_sha256="f" * 64)
        runner2 = WebFixtureRunner("success")
        mgr2 = JobManager(cfg2, runner2, auto_start=True)
        srv2 = make_server(cfg2, mgr2)
        t2 = threading.Thread(target=srv2.serve_forever, daemon=True)
        t2.start()
        try:
            port2 = srv2.server_address[1]
            c2 = http.client.HTTPConnection("127.0.0.1", port2, timeout=30)
            try:
                c2.request("GET", f"/corrections/{cid}")
                resp = c2.getresponse()
                detail = json.loads(resp.read())
            finally:
                c2.close()
            self.assertEqual(detail["correction"]["submission_state"], "prepared")
            self.assertIsNone(detail["correction"]["child_job_id"])
            self.assertEqual(runner2.calls, 0, "GET 不得触发任何推理")
            self.assertEqual(self.table_count("rec_jobs"), jobs_before)
        finally:
            srv2.shutdown(); srv2.server_close(); t2.join(5); mgr2.stop()

    def test_interrupted_child_not_replayed(self):
        parent = self.success_job(b"PARENT-INTERRUPT")
        self.register(parent)
        review_id = self.add_review(parent, "questioned")
        s, raw = self.request("POST", f"/jobs/{parent}/corrections",
                              self.correction_body(review_id, changes={"source": "SYNTHETIC 中断"}))
        self.assertEqual(s, 201, raw)
        child = json.loads(raw)["child_job_id"]
        cid = json.loads(raw)["correction_id"]
        self.wait_job(child, "succeeded")

        # 人为把子作业标为中断（模拟重启后遗留）；GET 只读，不重放。
        conn = sqlite3.connect(self.cfg.db_path)
        try:
            conn.execute("UPDATE rec_jobs SET status='interrupted' WHERE job_id = ?", (child,))
            conn.commit()
        finally:
            conn.close()
        jobs_before = self.table_count("rec_jobs")
        runs_before = self.table_count("rec_runs")

        s, raw = self.request("GET", f"/corrections/{cid}")
        self.assertEqual(s, 200, raw)
        detail = json.loads(raw)
        self.assertEqual(detail["child"]["status"], "interrupted")
        self.assertFalse(detail["comparison"]["child_succeeded"])
        self.assertIsNone(detail["comparison"]["class_pixel_comparison"])
        self.assertEqual(self.table_count("rec_jobs"), jobs_before)
        self.assertEqual(self.table_count("rec_runs"), runs_before)

    # ---- 10. 父子计数 / NoData 分母 / 治理 ID / 意见不串 -------------------

    def test_no_cross_connection(self):
        parent = self.success_job(b"PARENT-CROSS")
        reg = self.register(parent)
        parent_review = self.add_review(parent, "questioned")
        s, raw = self.request("POST", f"/jobs/{parent}/corrections",
                              self.correction_body(parent_review, changes={"scene_id": "CROSSED"}))
        self.assertEqual(s, 201, raw)
        body = json.loads(raw)
        child = body["child_job_id"]
        self.wait_job(child, "succeeded")
        child_review = self.add_review(child, "accepted", note="子作业合成意见")

        _, raw = self.request("GET", f"/corrections/{body['correction_id']}")
        detail = json.loads(raw)
        self.assertEqual(detail["parent"]["latest_task_review"]["review_id"], parent_review)
        self.assertEqual(detail["comparison"]["child_latest_task_review"]["review_id"], child_review)
        self.assertEqual(detail["parent"]["governance"]["version_id"], reg["version_id"])
        self.assertIsNone(detail["child"]["governance"])

        # 类计数来自各自作业（fixture 相同），NoData 分母为四类合计。
        p_counts = detail["parent"]["class_counts"]
        c_counts = detail["child"]["class_counts"]
        self.assertEqual(sorted(p_counts), ["0", "1", "2", "255"])
        self.assertEqual(p_counts, c_counts)
        nd = detail["comparison"]["nodata"]["parent"]
        self.assertEqual(nd["denominator"], sum(int(p_counts[k]) for k in ("0", "1", "2", "255")))
        self.assertEqual(nd["pixels"], int(p_counts["255"]))
        self.assertEqual(detail["comparison"]["class_pixel_comparison"]["delta"],
                         {k: 0 for k in ("0", "1", "2", "255")})

    # ---- 11. 50 条容量边界后旧号仍可读/确认 ------------------------------

    def test_capacity_boundary(self):
        self.assertEqual(corrections.MAX_CORRECTIONS_PER_ROOT, 50)
        parent = self.success_job(b"PARENT-CAPACITY")
        self.register(parent)
        first_request_id = None
        first_correction_id = None
        for i in range(50):
            review_id = self.add_review(parent, "questioned", note=f"合成意见 {i}")
            req = rid(f"cap{i}")
            s, raw = self.request("POST", f"/jobs/{parent}/corrections",
                                  self.correction_body(review_id, request_id=req,
                                                       changes={"source": f"SYNTHETIC 容量 {i}"}))
            self.assertEqual(s, 201, (i, raw))
            if i == 0:
                first_request_id = req
                first_correction_id = json.loads(raw)["correction_id"]

        # 第 51 条被拒。
        review_51 = self.add_review(parent, "questioned", note="合成意见 51")
        s, raw = self.request("POST", f"/jobs/{parent}/corrections",
                              self.correction_body(review_51, changes={"source": "SYNTHETIC 超限"}))
        self.assertEqual(s, 409, raw)
        self.assertEqual(self.table_count("rec_corrections"), 50)

        # 列表满 50、不截断。
        s, raw = self.request("GET", f"/jobs/{parent}/corrections")
        self.assertEqual(s, 200, raw)
        listing = json.loads(raw)
        self.assertEqual(listing["total"], 50)
        self.assertEqual(listing["limit"], 50)
        self.assertEqual(len(listing["corrections"]), 50)
        self.assertFalse(listing["truncated"])

        # 旧号仍可读/确认（幂等命中先于容量检查）。
        s, raw = self.request("GET", f"/corrections/{first_correction_id}")
        self.assertEqual(s, 200, raw)
        s, raw = self.request("POST", f"/jobs/{parent}/corrections",
                              self.correction_body(
                                  json.loads(raw)["correction"]["review_id"],
                                  request_id=first_request_id,
                                  changes={"source": "SYNTHETIC 容量 0"}))
        self.assertEqual(s, 200, raw)
        self.assertFalse(json.loads(raw)["created"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
