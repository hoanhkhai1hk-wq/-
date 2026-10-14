"""AgroLake P0 固定验收测试（标准库 unittest，不额外安装测试框架）。

覆盖：重复导入事件、同名异内容、空/坏图、单位换算与未知单位、JSON 失配、
来源补充前后留痕、隔离后复检、独立进程重读持久化、内容对象路径边界。

使用独立临时存储目录，不污染演示数据库。
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 固定测试写入项目内目录（不再使用系统 temp），并保留历史结果。
BASE_TEST_DIR = PROJECT_ROOT / "reports" / "p0" / "DEV-20261015-002" / "tests"
BASE_TEST_DIR.mkdir(parents=True, exist_ok=True)

from agrolake import config, db, disposition, ingest, quality, reporting, scenes  # noqa: E402
from agrolake import storage, semantics  # noqa: E402

from PIL import Image  # noqa: E402


def make_png(color=(30, 120, 90), size=(64, 48), noise=False) -> bytes:
    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for y in range(size[1]):
            for x in range(size[0]):
                px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agrolake_p0_test_", dir=str(BASE_TEST_DIR)))
        self._old = {k: os.environ.get(k) for k in
                     ("AGROLAKE_DATA_DIR", "AGROLAKE_STORE", "AGROLAKE_DB")}
        os.environ["AGROLAKE_DATA_DIR"] = str(self.tmp)
        os.environ["AGROLAKE_STORE"] = str(self.tmp / "store")
        os.environ["AGROLAKE_DB"] = str(self.tmp / "test.sqlite")
        self.root = config.storage_root()
        self.conn = db.connect()
        db.init_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def imp(self, data: bytes, name: str, **kw):
        return ingest.import_bytes(self.conn, data, name, store_root=self.root, **kw)

    def statuses(self, version_id):
        return {c["rule_id"]: c["status"] for c in quality.latest_checks(self.conn, version_id)}


class RuleCaseTests(BaseCase):
    """10 种规则情况的期望与实际结果。"""

    def test_01_normal_image_pass(self):
        r = self.imp(make_png(noise=True), "ok.png", source="syn", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["image_decode"], "pass")
        self.assertEqual(st["file_not_empty"], "pass")
        self.assertEqual(st["crs_applicability"], "na")  # 普通图像 CRS 不适用
        self.assertEqual(scenes.list_versions(self.conn)[0]["overall_status"], "pass")

    def test_02_empty_file_block(self):
        r = self.imp(b"", "empty.png", source="syn", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["file_not_empty"], "block")
        self.assertEqual(st["image_decode"], "block")

    def test_03_broken_image_block(self):
        r = self.imp(b"not-a-real-png", "broken.png", source="syn", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["image_decode"], "block")

    def test_04_exact_duplicate_warn(self):
        data = make_png()
        r1 = self.imp(data, "dup.png", source="syn", scene_id="S1")
        r2 = self.imp(data, "dup.png", source="other", scene_id="S1")
        self.assertEqual(r2["event_type"], "duplicate_content")
        st = self.statuses(r2["version_id"])
        self.assertEqual(st["exact_duplicate"], "warn")
        # 保留本次导入事件与本次来源声明，未复用第一次声明
        ev = self.conn.execute(
            "SELECT declared_source FROM import_events WHERE version_id = ?",
            (r2["version_id"],),
        ).fetchone()
        self.assertEqual(ev["declared_source"], "other")

    def test_05_same_name_diff_content_no_overwrite(self):
        # 同名、同场景、同来源的新内容才构成同一资产的新版本（身份边界见数据字典）。
        r1 = self.imp(make_png(color=(1, 2, 3)), "same.png", source="a", scene_id="S1")
        r2 = self.imp(make_png(color=(9, 9, 9)), "same.png", source="a", scene_id="S1")
        self.assertEqual(r2["event_type"], "new_version_same_name")
        self.assertNotEqual(r1["sha256"], r2["sha256"])
        # 两个对象文件都在，原始未被覆盖
        for sha in (r1["sha256"], r2["sha256"]):
            self.assertTrue((self.root / storage.object_relpath(sha)).exists())
        vers = self.conn.execute(
            "SELECT COUNT(*) c FROM asset_versions WHERE asset_id = ?", (r1["asset_id"],)
        ).fetchone()["c"]
        self.assertEqual(vers, 2)

    def test_06_missing_source_block(self):
        r = self.imp(make_png(), "nosrc.png", source=None, scene_id="S1")
        self.assertEqual(self.statuses(r["version_id"])["required_source"], "block")

    def test_07_csv_missing_columns_block(self):
        csv_text = "scene_id,field_id\nS1,F1\n"
        r = self.imp(csv_text.encode("utf-8"), "attrs.csv", source="syn", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["csv_required_columns"], "block")

    def test_08_csv_units(self):
        csv_text = (
            "scene_id,field_id,crop,area,unit\n"
            "S1,F1,rice,1,ha\n"       # 1 公顷 => 10000 m2
            "S1,F2,rice,1,亩\n"        # 1 亩 => 666.67 m2
            "S1,F3,rice,5,acre\n"      # 未知单位 => 错误
            "S1,F4,rice,-2,m2\n"       # 负数 => 错误
        )
        r = self.imp(csv_text.encode("utf-8"), "u.csv", source="syn", scene_id="S1")
        rows = self.conn.execute(
            "SELECT field_id, area_m2, issues FROM attribute_records WHERE version_id=?"
            " ORDER BY row_index", (r["version_id"],)).fetchall()
        self.assertAlmostEqual(rows[0]["area_m2"], 10000.0)
        self.assertAlmostEqual(rows[1]["area_m2"], 666.6666666, places=4)
        self.assertIsNone(rows[2]["area_m2"])
        self.assertIn("未定义", rows[2]["issues"])
        self.assertIsNone(rows[3]["area_m2"])
        self.assertIn("负数", rows[3]["issues"])
        self.assertEqual(self.statuses(r["version_id"])["csv_area_units"], "block")

    def test_09_json_scene_mismatch_warn(self):
        payload = json.dumps({"scene_id": "S2", "note": "复核"}, ensure_ascii=False)
        r = self.imp(payload.encode("utf-8"), "note.json", source="syn", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["scene_consistency"], "warn")
        note = self.conn.execute(
            "SELECT issues FROM text_records WHERE version_id=?", (r["version_id"],)
        ).fetchone()
        self.assertIn("失配", note["issues"])

    def test_10_unit_helpers(self):
        self.assertEqual(semantics.normalize_unit("公顷"), "ha")
        self.assertIsNone(semantics.normalize_unit("acre"))
        m2, status, _ = semantics.convert_area("2", "平方米")
        self.assertEqual(status, "ok")
        self.assertEqual(m2, 2.0)


class DispositionTests(BaseCase):
    def test_source_supplement_keeps_history_and_rechecks(self):
        r = self.imp(make_png(), "x.png", source=None, scene_id="S1")
        vid = r["version_id"]
        self.assertEqual(self.statuses(vid)["required_source"], "block")
        disposition.supplement_metadata(
            self.conn, vid, "tester", "补来源", {"source": "补充来源"}, self.root)
        self.assertEqual(self.statuses(vid)["required_source"], "pass")
        hist = self.conn.execute(
            "SELECT old_value, new_value FROM meta_history WHERE version_id=? AND field='source'",
            (vid,)).fetchall()
        self.assertEqual(len(hist), 1)
        self.assertIsNone(hist[0]["old_value"])
        self.assertEqual(hist[0]["new_value"], "补充来源")
        # 复检与被复检的旧检查关联
        linked = self.conn.execute(
            "SELECT COUNT(*) c FROM quality_checks WHERE version_id=? AND recheck_of_check_id"
            " IS NOT NULL", (vid,)).fetchone()["c"]
        self.assertGreater(linked, 0)

    def test_isolate_not_restored_by_recheck(self):
        r = self.imp(make_png(), "y.png", source="syn", scene_id="S1")
        vid = r["version_id"]
        disposition.isolate(self.conn, vid, "tester", "人工隔离")
        disposition.recheck(self.conn, vid, "tester", self.root)
        status = self.conn.execute(
            "SELECT status FROM asset_versions WHERE version_id=?", (vid,)).fetchone()["status"]
        self.assertEqual(status, "isolated")
        # 隔离版本不计入当前有效版本与通过率分母
        self.assertNotIn(vid, [v["version_id"] for v in scenes.current_versions(self.conn)])

    def test_duplicate_source_edit_isolated(self):
        data = make_png()
        r1 = self.imp(data, "d.png", source="A", scene_id="S1")
        r2 = self.imp(data, "d.png", source="B", scene_id="S1")
        disposition.supplement_metadata(
            self.conn, r2["version_id"], "t", "改来源", {"source": "C"}, self.root)
        srcs = {e["version_id"]: e["declared_source"] for e in self.conn.execute(
            "SELECT version_id, declared_source FROM import_events").fetchall()}
        self.assertEqual(srcs[r1["version_id"]], "A")
        # 导入事件保留原始声明，未被其他版本的来源修改影响
        self.assertEqual(srcs[r2["version_id"]], "B")
        cur = self.conn.execute(
            "SELECT source FROM asset_versions WHERE version_id=?", (r2["version_id"],)
        ).fetchone()["source"]
        self.assertEqual(cur, "C")


class BoundaryTests(BaseCase):
    def test_object_path_boundary(self):
        with self.assertRaises(storage.PathBoundaryError):
            storage.ensure_within(self.root, self.root / ".." / "escape.txt")
        with self.assertRaises(storage.PathBoundaryError):
            storage.ensure_within(self.root, Path(tempfile.gettempdir()) / "outside.txt")
        with self.assertRaises(ValueError):
            storage.object_path(self.root, "zz/" + "a" * 62)

    def test_user_filename_not_used_for_path(self):
        r = self.imp(make_png(), "../../evil.png", source="syn", scene_id="S1")
        rel = r["object_relpath"]
        self.assertTrue(rel.startswith("objects/"))
        self.assertNotIn("evil", rel)
        self.assertNotIn("..", rel)
        # 原始文件名仅作为元数据，且已剥离目录分量
        name = self.conn.execute(
            "SELECT original_filename FROM asset_versions WHERE version_id=?",
            (r["version_id"],)).fetchone()["original_filename"]
        self.assertEqual(name, "evil.png")


class PersistenceTests(BaseCase):
    def test_independent_process_rereads(self):
        self.imp(make_png(), "p.png", source="syn", scene_id="S1")
        self.imp("scene_id,field_id,crop,area,unit\nS1,F1,rice,1,ha\n".encode(),
                 "p.csv", source="syn", scene_id="S1")
        self.conn.commit()
        code = (
            "import sys; sys.path.insert(0, r'%s');"
            "from agrolake import db, scenes;"
            "c = db.connect();"
            "print(len(scenes.list_versions(c)), len(scenes.list_scenes(c)))"
            % str(PROJECT_ROOT)
        )
        env = dict(os.environ)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, cwd=str(PROJECT_ROOT))
        self.assertEqual(out.returncode, 0, out.stderr)
        versions, scene_count = out.stdout.strip().split()
        self.assertEqual(versions, "2")
        self.assertEqual(scene_count, "1")

    def test_report_export_rereadable(self):
        self.imp(make_png(), "r.png", source="syn", scene_id="S1")
        out = self.tmp / "rep"
        p = reporting.export_assets_csv(self.conn, out / "assets.csv")
        with open(p, encoding="utf-8-sig") as fh:
            self.assertIn("version_id", fh.readline())
        jp = reporting.export_report_json(self.conn, out / "report.json")
        data = json.loads(jp.read_text(encoding="utf-8"))
        self.assertIn("summary", data)


def make_jpeg(color=(40, 90, 140), size=(160, 160), noise=True) -> bytes:
    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for y in range(size[1]):
            for x in range(size[0]):
                px[x, y] = ((x * 5) % 256, (y * 9) % 256, ((x * y) * 3) % 256)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class RegressionTests(BaseCase):
    """Codex 预审发现的语义问题的回归检查。"""

    def test_current_versions_excludes_older(self):
        r1 = self.imp(make_png(color=(1, 1, 1)), "ev.png", source="a", scene_id="S1")
        r2 = self.imp(make_png(color=(2, 2, 2)), "ev.png", source="a", scene_id="S1")
        cur = [v["version_id"] for v in scenes.current_versions(self.conn)]
        self.assertEqual(cur, [r2["version_id"]])  # 仅最新内容版本
        self.assertNotIn(r1["version_id"], cur)

    def test_cross_filename_duplicate_single_effective(self):
        data = make_png(color=(7, 7, 7))
        r1 = self.imp(data, "one.png", source="a", scene_id="S1")
        r2 = self.imp(data, "another.png", source="b", scene_id="S1")
        self.assertEqual(r2["event_type"], "duplicate_content")
        status = self.conn.execute(
            "SELECT status FROM asset_versions WHERE version_id=?", (r2["version_id"],)
        ).fetchone()["status"]
        self.assertEqual(status, "duplicate")
        cur = [v["version_id"] for v in scenes.current_versions(self.conn)]
        self.assertEqual(cur, [r1["version_id"]])

    def test_set_scene_moves_current_membership(self):
        r = self.imp(make_png(noise=True), "sc.png", source="a", scene_id="SA")
        vid = r["version_id"]
        self.assertIn(vid, [m["version_id"] for m in scenes.scene_members(self.conn, "SA")["roles"]["image"]])
        disposition.set_scene(self.conn, vid, "SB", "tester", "改场景", self.root)
        self.assertNotIn(vid, [m["version_id"] for m in scenes.scene_members(self.conn, "SA")["roles"]["image"]])
        self.assertIn(vid, [m["version_id"] for m in scenes.scene_members(self.conn, "SB")["roles"]["image"]])

    def test_area_overflow_is_error(self):
        m2, status, reason = semantics.convert_area("1e308", "ha")
        self.assertIsNone(m2)
        self.assertEqual(status, "error")
        self.assertIn("非有限", reason)

    def test_truncated_jpeg_blocks(self):
        data = make_jpeg()
        truncated = data[: int(len(data) * 0.6)]
        r = self.imp(truncated, "trunc.jpg", source="a", scene_id="S1")
        self.assertEqual(self.statuses(r["version_id"])["image_decode"], "block")

    def test_csv_scene_mismatch_not_counted(self):
        csv_text = "scene_id,field_id,crop,area,unit\nSA,F1,rice,1,ha\n"
        r = self.imp(csv_text.encode(), "m.csv", source="a", scene_id="SB")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["csv_scene_consistency"], "warn")
        v = [x for x in scenes.list_versions(self.conn) if x["version_id"] == r["version_id"]][0]
        eff, note = scenes.effective_scene(self.conn, v)
        self.assertIsNone(eff)
        self.assertIn("不一致", note)

    def test_json_declaration_fix_recomputes(self):
        payload = json.dumps({"scene_id": "SB", "note": "x"}, ensure_ascii=False)
        r = self.imp(payload.encode(), "j.json", source="a", scene_id="SA")
        vid = r["version_id"]
        self.assertEqual(self.statuses(vid)["scene_consistency"], "warn")
        disposition.set_scene(self.conn, vid, "SB", "tester", "补正声明", self.root)
        self.assertEqual(self.statuses(vid)["scene_consistency"], "pass")
        # 旧结论保留
        hist = self.conn.execute(
            "SELECT COUNT(*) c FROM quality_checks WHERE version_id=? AND rule_id='scene_consistency'",
            (vid,)).fetchone()["c"]
        self.assertGreaterEqual(hist, 2)


class Dev002SemanticsTests(BaseCase):
    """DEV-20261015-002：隔离/复制事件/身份边界/未处置失配/编码/类别。"""

    def _vdict(self, vid):
        return [x for x in scenes.list_versions(self.conn) if x["version_id"] == vid][0]

    def test_isolate_latest_does_not_revive_old(self):
        r1 = self.imp(make_png(color=(1, 1, 1)), "v.png", source="s", scene_id="S1")
        r2 = self.imp(make_png(color=(2, 2, 2)), "v.png", source="s", scene_id="S1")
        self.assertEqual(r1["asset_id"], r2["asset_id"])
        disposition.isolate(self.conn, r2["version_id"], "t", "隔离最新版本")
        cur = [v["version_id"] for v in scenes.current_versions(self.conn)]
        self.assertEqual(cur, [])  # 不自动复活旧历史版本 r1
        self.assertNotIn(r1["version_id"], cur)

    def test_duplicate_upload_keeps_original_current(self):
        data = make_png()
        r1 = self.imp(data, "d.png", source="s", scene_id="S1")
        r2 = self.imp(data, "d.png", source="s", scene_id="S1")  # 同身份再次导入同一内容
        self.assertEqual(r2["event_type"], "duplicate_content")
        cur = [v["version_id"] for v in scenes.current_versions(self.conn)]
        self.assertEqual(cur, [r1["version_id"]])  # 复制事件不改变原件的当前指向
        st = self.conn.execute(
            "SELECT status FROM asset_versions WHERE version_id=?", (r1["version_id"],)
        ).fetchone()["status"]
        self.assertEqual(st, "active")

    def test_manual_mark_latest_duplicate_no_revive(self):
        r1 = self.imp(make_png(color=(3, 3, 3)), "m.png", source="s", scene_id="S1")
        r2 = self.imp(make_png(color=(4, 4, 4)), "m.png", source="s", scene_id="S1")
        disposition.mark_duplicate(self.conn, r2["version_id"], "t", "人工标重复",
                                   r1["version_id"])
        cur = [v["version_id"] for v in scenes.current_versions(self.conn)]
        self.assertEqual(cur, [])  # 人工把当前内容版本标重复，不复活旧版

    def test_asset_identity_separates_scene_and_source(self):
        a1 = self.imp(make_png(color=(1, 1, 1)), "f.png", source="s1", scene_id="A")
        a2 = self.imp(make_png(color=(2, 2, 2)), "f.png", source="s1", scene_id="B")
        a3 = self.imp(make_png(color=(3, 3, 3)), "f.png", source="s2", scene_id="A")
        ids = {a1["asset_id"], a2["asset_id"], a3["asset_id"]}
        self.assertEqual(len(ids), 3)  # 不同场景/来源各自独立，不共用资产
        for a in (a1, a2, a3):
            self.assertEqual(a["event_type"], "new_asset")

    def test_unresolved_json_not_associated(self):
        payload = json.dumps({"scene_id": "B", "note": "x"}, ensure_ascii=False)
        r = self.imp(payload.encode(), "n.json", source="s", scene_id="A")
        v = self._vdict(r["version_id"])
        eff, note = scenes.effective_scene(self.conn, v)
        self.assertIsNone(eff)          # 矛盾未处置：不归属
        self.assertIn("失配", note or "")
        mem_b = scenes.scene_members(self.conn, "B")
        self.assertNotIn(r["version_id"], [m["version_id"] for m in mem_b["roles"]["text"]])
        details = " ".join(str(i.get("detail")) for i in scenes.scene_issues(self.conn))
        self.assertIn("失配", details)  # 问题被呈现

    def test_json_fix_restores_association(self):
        payload = json.dumps({"scene_id": "B", "note": "x"}, ensure_ascii=False)
        r = self.imp(payload.encode(), "n2.json", source="s", scene_id="A")
        vid = r["version_id"]
        disposition.set_scene(self.conn, vid, "B", "t", "补正声明", self.root)
        v = self._vdict(vid)
        eff, note = scenes.effective_scene(self.conn, v)
        self.assertEqual(eff, "B")
        self.assertIsNone(note)
        self.assertEqual(self.statuses(vid)["scene_consistency"], "pass")

    def test_multiscene_csv_not_silently_first(self):
        csv_text = "scene_id,field_id,crop,area,unit\nA,F1,rice,1,ha\nB,F2,rice,1,ha\n"
        r = self.imp(csv_text.encode(), "multi.csv", source="s", scene_id=None)
        v = self._vdict(r["version_id"])
        eff, note = scenes.effective_scene(self.conn, v)
        self.assertIsNone(eff)          # 不偷偷只取第一行场景
        self.assertIn("多", note or "")

    def test_csv_required_value_blank_recorded(self):
        csv_text = "scene_id,field_id,crop,area,unit\nS1,,rice,1,ha\n"
        r = self.imp(csv_text.encode(), "blank.csv", source="s", scene_id="S1")
        self.assertEqual(self.statuses(r["version_id"])["csv_required_values"], "block")

    def test_encoding_error_registered_no_partial_rows(self):
        raw = "scene_id,field_id,crop,area,unit\nS1,F1,稻,1,ha\n".encode("gbk")
        r = self.imp(raw, "gbk.csv", source="s", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["csv_required_columns"], "block")  # 登记为解析问题
        rows = self.conn.execute(
            "SELECT COUNT(*) c FROM attribute_records WHERE version_id=?",
            (r["version_id"],)).fetchone()["c"]
        self.assertEqual(rows, 0)  # 不产生半截入库记录

    def test_text_encoding_error_blocks(self):
        r = self.imp(b"\xff\xfe not utf8 text", "bad.txt", source="s", scene_id="S1")
        st = self.statuses(r["version_id"])
        self.assertEqual(st["text_parsed"], "block")
        rec = self.conn.execute(
            "SELECT payload_json, issues FROM text_records WHERE version_id=?",
            (r["version_id"],)).fetchone()
        self.assertIsNone(rec["payload_json"])
        self.assertIn("编码", rec["issues"])

    def test_category_normalization_and_unknown(self):
        csv_text = (
            "scene_id,field_id,crop,category,area,unit\n"
            "S1,F1,rice,田块,1,ha\n"
            "S1,F2,rice,不存在的类别,1,ha\n"
        )
        r = self.imp(csv_text.encode(), "cat.csv", source="s", scene_id="S1")
        rows = self.conn.execute(
            "SELECT category_raw, category_std, issues FROM attribute_records"
            " WHERE version_id=? ORDER BY row_index", (r["version_id"],)).fetchall()
        self.assertEqual(rows[0]["category_raw"], "田块")
        self.assertEqual(rows[0]["category_std"], "FIELD")
        self.assertIsNone(rows[1]["category_std"])          # 未知保留原值
        self.assertIn("未知", rows[1]["issues"])

    def test_category_vs_crop_not_mixed(self):
        self.assertEqual(semantics.map_category("田块"), "FIELD")
        self.assertIsNone(semantics.map_category("水稻"))    # 作物不属地物类别字典
        std, note = semantics.normalize_category("水稻")
        self.assertIsNone(std)
        self.assertIn("未知", note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
