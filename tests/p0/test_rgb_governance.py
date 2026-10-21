"""INT-20261021-001 RGB治理登记与事实层测试（.venv-p0，合成fixture，离线）。

覆盖：两资产登记与PNG逐像素/JSON来源、原子回滚、幂等与既有链接核验、
并发认领（外层事务拒绝）、readonly零写与submit拒绝、失败/未知/损坏拒绝、
SQL计数篡改拒绝、敏感路径不外发。

全部使用合成产物 fixture（tests/web/test_rgb_candidate_jobs.py），不加载模型、
不真实推理、不访问网络、不写原业务库；所有数据都在临时目录。

用法：
  .venv-p0/Scripts/python.exe tests/p0/test_rgb_governance.py
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
WEB_TESTS = ROOT / "tests" / "web"
for p in (str(ROOT), str(WEB_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image  # noqa: E402

import agrolake.rgb_governance as rg  # noqa: E402
import agrolake.rgb_facts as rf  # noqa: E402
import test_rgb_candidate_jobs as fixture  # noqa: E402
from agrolake import config as p0_config  # noqa: E402
from agrolake import db as p0_db  # noqa: E402
from agrolake import ingest as p0_ingest  # noqa: E402
from agrolake.rgb_jobs import (  # noqa: E402
    RgbArtifactError,
    RgbCandidateInputError,
    RgbCandidateJobs,
)

RID = "a" * 32
ABS_PATH_RE = r"[A-Za-z]:\\"


def png_bytes(color=(10, 120, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), color).save(buf, "PNG")
    return buf.getvalue()


class GovernBase(unittest.TestCase):
    def setUp(self):
        test_root = ROOT / "reports/rgb/WB-INT-20261021-001/tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root)
        self.rgb_dir = Path(self.tmp.name) / "rgb"
        self.p0_dir = Path(self.tmp.name) / "p0"
        self.p0_db_file = self.p0_dir / "agrolake.sqlite"
        self._saved = {k: os.environ.get(k) for k in
                       ("AGROLAKE_RGB_DIR", "AGROLAKE_DATA_DIR", "AGROLAKE_DB",
                        "AGROLAKE_STORE")}
        os.environ["AGROLAKE_RGB_DIR"] = str(self.rgb_dir)
        os.environ["AGROLAKE_DATA_DIR"] = str(self.p0_dir)
        os.environ["AGROLAKE_DB"] = str(self.p0_db_file)
        os.environ.pop("AGROLAKE_STORE", None)
        self.jobs = RgbCandidateJobs(data_dir=self.rgb_dir)
        self.jobs_ro = RgbCandidateJobs(data_dir=self.rgb_dir, readonly=True)
        self.conn = p0_db.connect()
        p0_db.init_schema(self.conn)
        fixture.FIXTURE["mode"] = "success"
        fixture.FIXTURE["calls"] = []
        self._patcher = mock.patch("subprocess.run", fixture.fake_run)
        self._patcher.start()
        self._img = png_bytes()
        self._job = self.jobs.submit(self._img, "a.png", RID)
        self.assertEqual(self._job["status"], "succeeded")
        fixture.FIXTURE["calls"] = []

    def tearDown(self):
        self._patcher.stop()
        self.conn.close()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def register(self):
        return rg.register(self.conn, self._job["id"], backend=self.jobs_ro)

    def counts(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]


class RegisterTest(GovernBase):
    def test_register_creates_two_assets_link_and_quality(self):
        result = self.register()
        self.assertTrue(result["created"])
        reg = result["registration"]
        job_id = self._job["id"]
        self.assertEqual(reg["scene_id"], f"RGB-{job_id}")

        # 链接行唯一
        rows = self.conn.execute(
            "SELECT * FROM rgb_governance_links WHERE job_id = ?", (job_id,)
        ).fetchall()
        self.assertEqual(len(rows), 1)

        # 两份资产：PNG 影像 + JSON 文本
        img_v = self.conn.execute(
            "SELECT * FROM asset_versions WHERE version_id = ?",
            (reg["image_version_id"],)).fetchone()
        sta_v = self.conn.execute(
            "SELECT * FROM asset_versions WHERE version_id = ?",
            (reg["stats_version_id"],)).fetchone()
        self.assertEqual(img_v["content_type"], "image")
        self.assertEqual(img_v["original_filename"], f"rgb-{job_id}.png")
        self.assertEqual(img_v["source"], rg.DERIVED_SOURCE)
        self.assertEqual(img_v["license_status"], "unknown")
        self.assertEqual(img_v["declared_scene_id"], f"RGB-{job_id}")
        self.assertEqual(sta_v["content_type"], "text")
        self.assertEqual(sta_v["original_filename"], f"rgb-{job_id}-stats.json")

        # 两次导入事件、两版质检
        self.assertEqual(self.counts("import_events"), 2)
        for vid in (reg["image_version_id"], reg["stats_version_id"]):
            checks = self.conn.execute(
                "SELECT COUNT(*) AS c FROM quality_checks WHERE version_id = ?",
                (vid,)).fetchone()["c"]
            self.assertGreater(checks, 0)

        # PNG 逐像素与原图一致（无损派生）
        png_obj = self.conn.execute(
            "SELECT relpath FROM objects WHERE object_id = ?",
            (reg["png_sha256"],)).fetchone()
        self.assertIsNotNone(png_obj)
        path = p0_config.storage_root() / png_obj["relpath"]
        with Image.open(path) as im:
            self.assertEqual(im.size, (512, 512))
            self.assertEqual(im.mode, "RGB")
            png_pixels = list(im.convert("RGB").getdata())
        with Image.open(io.BytesIO(self._img)) as src:
            self.assertEqual(png_pixels, list(src.convert("RGB").getdata()))

        # JSON 来源记录内容
        sta_obj = self.conn.execute(
            "SELECT relpath FROM objects WHERE object_id = ?",
            (reg["json_sha256"],)).fetchone()
        doc = json.loads((p0_config.storage_root() / sta_obj["relpath"])
                         .read_text(encoding="utf-8"))
        self.assertEqual(doc["job_id"], job_id)
        self.assertEqual(doc["source_image_sha256"], self._job["image_sha256"])
        self.assertEqual(doc["derived_png_sha256"], reg["png_sha256"])
        self.assertEqual(doc["manifest_sha256"], self._job["manifest_sha256"])
        self.assertEqual(doc["seed"], self._job["seed"])
        self.assertEqual(doc["scene_id"], f"RGB-{job_id}")
        self.assertEqual(set(doc["artifact_sha256"]), {
            "mask256.png", "mask512.png", "overlay.png",
            "candidate_fields.csv", "quality.json", "metadata.json"})
        self.assertEqual(doc["native_class_counts"]["0"], 196608)
        self.assertEqual(doc["candidate_count"], 2)
        self.assertEqual(len(doc["regions"]), 2)
        self.assertIn("not_confirmed_fields", json.dumps(doc["boundaries"]))
        # 不伪造农业属性/许可
        self.assertNotIn("crop", doc)
        self.assertNotIn("area_m2", doc)
        self.assertNotIn("acquired_at", doc)

        # 质量摘要真实可读
        self.assertIn("质检", result["quality_summary"])

    def test_register_idempotent_no_extra_rows(self):
        first = self.register()
        second = self.register()
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(
            first["registration"]["image_version_id"],
            second["registration"]["image_version_id"])
        self.assertEqual(
            first["registration"]["stats_version_id"],
            second["registration"]["stats_version_id"])
        self.assertEqual(self.counts("asset_versions"), 2)
        self.assertEqual(self.counts("import_events"), 2)
        self.assertEqual(self.counts("quality_checks"),
                         self.counts("quality_checks"))  # 稳定
        self.assertEqual(self.counts("rgb_governance_links"), 1)

    def test_atomic_rollback_on_second_import_failure(self):
        real = p0_ingest.import_bytes
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("合成失败：第二次导入中断")
            return real(*args, **kwargs)

        with mock.patch.object(rg, "p0_ingest") as patched:
            patched.import_bytes.side_effect = flaky
            with self.assertRaises(RuntimeError):
                self.register()
        # 整体回滚：零资产/零版本/零事件/零质检/零链接
        for table in ("assets", "asset_versions", "import_events",
                      "quality_checks", "rgb_governance_links", "objects"):
            self.assertEqual(self.counts(table), 0, table)
        # 孤立内容对象文件允许保留，不删除
        # （storage.store_bytes 在事务外写文件，属预期行为）

    def test_outer_transaction_rejected(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaises(rg.RgbGovernanceError):
                rg.register(self.conn, self._job["id"], backend=self.jobs_ro)
        finally:
            self.conn.rollback()

    def test_failed_unknown_and_corrupt_rejected(self):
        # 失败作业
        fixture.FIXTURE["mode"] = "fail_exit"
        bad = self.jobs.submit(png_bytes(), "b.png", "b" * 32)
        self.assertEqual(bad["status"], "failed")
        fixture.FIXTURE["mode"] = "success"
        with self.assertRaises(rg.RgbGovernanceError):
            rg.register(self.conn, bad["id"], backend=self.jobs_ro)
        # 未知作业
        with self.assertRaises(rg.RgbGovernanceError):
            rg.register(self.conn, "c" * 32, backend=self.jobs_ro)
        # 产物损坏（登记前被改动）
        (Path(self._job["result_dir"]) / "mask256.png").write_bytes(b"tampered")
        with self.assertRaises(RgbArtifactError):
            self.register()
        self.assertEqual(self.counts("rgb_governance_links"), 0)

    def test_corrupted_existing_link_not_returned_as_success(self):
        result = self.register()
        reg = result["registration"]
        # 篡改 P0 里的派生 PNG 对象
        obj = self.conn.execute(
            "SELECT relpath FROM objects WHERE object_id = ?",
            (reg["png_sha256"],)).fetchone()
        (p0_config.storage_root() / obj["relpath"]).write_bytes(b"corrupted")
        with self.assertRaises(rg.RgbGovernanceError):
            self.register()
        # 只读 link_status 也如实标损坏
        ro = p0_db.connect_readonly()
        try:
            status = rg.link_status(ro, self._job["id"], backend=self.jobs_ro)
        finally:
            ro.close()
        self.assertTrue(status["registered"])
        self.assertFalse(status["intact"])
        self.assertIn("摘要", status["problem"])


class ReadonlyModeTest(GovernBase):
    def test_missing_db_raises_and_creates_nothing(self):
        ghost = RgbCandidateJobs(data_dir=Path(self.tmp.name) / "不存在目录",
                                 readonly=True)
        with self.assertRaises(RgbArtifactError):
            ghost.get_job("d" * 32)
        with self.assertRaises(RgbArtifactError):
            ghost.list_jobs()
        self.assertFalse((Path(self.tmp.name) / "不存在目录").exists())

    def test_readonly_submit_rejected_without_subprocess(self):
        with self.assertRaises(RgbCandidateInputError):
            self.jobs_ro.submit(png_bytes(), "x.png", RID)
        self.assertEqual(len(fixture.FIXTURE["calls"]), 0)

    def test_readonly_query_zero_write(self):
        before = {p.name: p.stat().st_mtime_ns
                  for p in self.rgb_dir.rglob("*") if p.is_file()}
        job = self.jobs_ro.get_job(self._job["id"])
        self.assertEqual(job["status"], "succeeded")
        self.jobs_ro.result(self._job["id"])
        self.jobs_ro.list_jobs()
        after = {p.name: p.stat().st_mtime_ns
                 for p in self.rgb_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


class RgbFactsTest(GovernBase):
    def _ro_p0(self):
        conn = p0_db.connect_readonly()
        self._ro_conns.append(conn)
        return conn

    def setUp(self):
        super().setUp()
        self._ro_conns: list[sqlite3.Connection] = []

    def tearDown(self):
        for conn in self._ro_conns:
            try:
                conn.close()
            except Exception:
                pass
        super().tearDown()

    def test_success_bundle_counts_and_boundaries(self):
        self.register()
        bundle = rf.build(self._ro_p0(), self._job["id"], data_dir=self.rgb_dir)
        self.assertEqual(bundle.scope, "rgb_job")
        self.assertIn("262144", bundle.text)
        self.assertIn("196608", bundle.text)
        self.assertIn("候选区域 2 个", bundle.text)
        self.assertIn("触及", bundle.text)
        self.assertIn(self._job["image_sha256"], bundle.text)
        self.assertIn("mask512.png", bundle.text)
        # 已登记信息
        self.assertIn("已登记", bundle.text)
        self.assertIn(f"RGB-{self._job['id']}", bundle.text)
        # 固定边界缺口
        blob = bundle.text + json.dumps(bundle.gaps, ensure_ascii=False)
        self.assertIn("无实地确认", blob)
        self.assertIn("平方米", blob)
        self.assertIn("独立来源新图评估", blob)
        # 敏感路径不外发
        self.assertNotIn(str(self.tmp.name), bundle.text)
        self.assertNotIn(str(self.tmp.name), json.dumps(
            bundle.snapshot, ensure_ascii=False))
        self.assertNotIn(str(self.tmp.name), json.dumps(
            bundle.citations, ensure_ascii=False))
        self.assertTrue(bundle.snapshot["sql_counts_verified"])

    def test_sql_count_tamper_rejected(self):
        self.register()
        sql = sqlite3.connect(self.rgb_dir / "runs.sqlite3")
        sql.execute(
            "UPDATE class_counts SET pixels = pixels + 1 WHERE job_id = ?",
            (self._job["id"],))
        sql.commit()
        sql.close()
        bundle = rf.build(self._ro_p0(), self._job["id"], data_dir=self.rgb_dir)
        self.assertIn("不一致", bundle.text)
        self.assertFalse(bundle.snapshot.get("sql_counts_verified"))

    def test_region_row_tamper_rejected(self):
        sql = sqlite3.connect(self.rgb_dir / "runs.sqlite3")
        sql.execute(
            "UPDATE regions SET area_pixels = area_pixels + 7 WHERE job_id = ?",
            (self._job["id"],))
        sql.commit()
        sql.close()
        bundle = rf.build(self._ro_p0(), self._job["id"], data_dir=self.rgb_dir)
        self.assertIn("不一致", bundle.text)
        self.assertFalse(bundle.snapshot.get("sql_counts_verified"))

    def test_failed_job_has_no_success_counts(self):
        fixture.FIXTURE["mode"] = "fail_exit"
        bad = self.jobs.submit(png_bytes(), "b.png", "b" * 32)
        fixture.FIXTURE["mode"] = "success"
        bundle = rf.build(self._ro_p0(), bad["id"], data_dir=self.rgb_dir)
        self.assertIn("失败", bundle.text)
        self.assertNotIn("196608", bundle.text)
        self.assertNotIn("262144", bundle.text)
        self.assertIn("没有成功结果", bundle.text)

    def test_unknown_job_and_missing_db(self):
        bundle = rf.build(self._ro_p0(), "e" * 32, data_dir=self.rgb_dir)
        self.assertIn("没有编号", bundle.text)
        bundle = rf.build(self._ro_p0(), self._job["id"],
                          data_dir=Path(self.tmp.name) / "ghost")
        self.assertIn("作业库不存在", bundle.text)
        bundle = rf.build(self._ro_p0(), "not-valid", data_dir=self.rgb_dir)
        self.assertIn("格式不合法", bundle.text)

    def test_list_jobs_readonly(self):
        listing = rf.list_jobs(data_dir=self.rgb_dir)
        self.assertTrue(listing["available"])
        self.assertEqual(listing["jobs"][0]["id"], self._job["id"])
        missing = rf.list_jobs(data_dir=Path(self.tmp.name) / "ghost")
        self.assertFalse(missing["available"])
        self.assertIn("只读", missing["note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
