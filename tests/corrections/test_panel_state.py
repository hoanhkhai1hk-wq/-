"""整改面板提交状态新增回归（DEV-20261015-018）。

真实 HTTP/SQLite 配合合成 runner（WebFixtureRunner），只验证“提交状态/治理前置/
NoData 分母比例”这几项本轮修复的行为；合成 fixture 明确不代表真实模型或农业效果。
独立只读探针 tests/corrections/test_panel_regressions.py 由 Codex 维护，本文件不改动它。

用法：
  .venv-p0/Scripts/python.exe tests/corrections/test_panel_state.py
"""
import json, os, sqlite3, sys, threading, unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent), str(ROOT / "tests/web")]

import test_correction_panel_apptest as helpers  # noqa: E402
from agrolake_service import governance, reviews  # noqa: E402
from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from fixtures import WebFixtureRunner  # noqa: E402

OUT = ROOT / "reports/corrections/DEV-20261015-020" / datetime.now().strftime(
    "state-%Y%m%d-%H%M%S-%f")
OUT.mkdir(parents=True, exist_ok=False)


class PanelStateTests(unittest.TestCase):
    def setUp(self):
        self.out = OUT / self._testMethodName
        self.out.mkdir()
        self.env = patch.dict(os.environ)
        self.env.start()
        self.cfg = ServiceConfig(db_path=self.out / "db.sqlite", store_root=self.out / "store",
                                 runs_root=self.out / "runs", port=0, synthetic_mode=True,
                                 model_sha256="f" * 64)
        self.manager = helpers.CountingManager(self.cfg, WebFixtureRunner("success"),
                                               auto_start=True)
        self.server = make_server(self.cfg, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        helpers.set_env(self.out, f"http://127.0.0.1:{self.server.server_address[1]}")
        self.parent = helpers.make_job(self.manager, b"STATE-PANEL")
        if "ungoverned" not in self._testMethodName:
            governance.register_governance(self.cfg, self.parent, {"request_id": "state-gov"})
        reviews.submit_review(self.cfg.db_path, self.parent, {
            "request_id": "state-opinion", "decision": "questioned",
            "reviewer": "SYNTHETIC reviewer", "note": "软件流程问题非真值"})

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        self.manager.stop()
        self.env.stop()

    def app(self):
        at = helpers.new_app()
        helpers.select_job(at, self.parent)
        self.assertFalse(helpers.exceptions(at))
        return at

    def prepare(self, at, scene):
        at.text_input(key=f"rec_corr_operator_{self.parent}").set_value("SYNTHETIC operator")
        at.text_area(key=f"rec_corr_reason_{self.parent}").set_value("独立行为验证，非田间业务证据")
        at.text_input(key=f"rec_corr_scene_{self.parent}").set_value(scene)

    def create(self, at, scene):
        self.prepare(at, scene)
        at.button(key=f"rec_corr_create_{self.parent}").click().run()
        self.assertFalse(helpers.exceptions(at))
        return self.rows()

    def rows(self):
        conn = sqlite3.connect(self.cfg.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM rec_corrections"
                                                  " ORDER BY correction_id")]
        finally:
            conn.close()

    def test_same_click_reuses_and_aba_makes_three_ids(self):
        at = self.app()
        first = self.create(at, "CASE-A")[0]
        helpers.wait_child(self.manager, first["child_job_id"])
        at.run()
        self.create(at, "CASE-A")                      # 当前内容相同：复用同号/同子作业
        self.assertEqual(len(self.rows()), 1)
        self.create(at, "CASE-B")
        self.create(at, "CASE-A")                      # A→B→A：必须新号/新版本
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r["request_id"] for r in rows}), 3)
        self.assertEqual([r["revision_no"] for r in rows], [1, 2, 3])

    def test_http503_locks_new_intent(self):
        at = self.app()
        with patch.object(self.manager, "submit", side_effect=OSError("SYNTHETIC SUBMIT UNAVAILABLE")):
            rows = self.create(at, "CASE-UNKNOWN")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["submission_state"], "prepared")
        at.run()
        self.prepare(at, "CASE-DIFFERENT")
        at.run()
        buttons = [b for b in at.button if b.key == f"rec_corr_create_{self.parent}"]
        if buttons and not buttons[0].disabled:
            buttons[0].click().run()
        self.assertFalse(helpers.exceptions(at))
        self.assertEqual(len(self.rows()), 1)          # 未换号新建
        self.assertEqual(self.rows()[0]["request_id"], rows[0]["request_id"])

    def test_ungoverned_parent_has_no_create_form(self):
        at = self.app()
        self.assertFalse([b for b in at.button if b.key == f"rec_corr_create_{self.parent}"])
        self.assertTrue(any("登记" in str(e.value) for e in [*at.info, *at.warning, *at.caption]))

    def test_nodata_fraction_and_ratio_are_visible(self):
        at = self.app()
        row = self.create(at, "CASE-DISPLAY")[0]
        helpers.wait_child(self.manager, row["child_job_id"])
        at.run()
        text = " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.info, *at.metric])
        self.assertRegex(text, r"2\s*/\s*12")
        self.assertRegex(text, r"16\.6\d*\s*%")
        self.assertFalse(helpers.exceptions(at))

    def test_intent_includes_replacement_filename(self):
        # 同字节换名（相对父输入仍有真实变化）必须产生新意图，不能复用旧号导致 409。
        from agrolake_web import corrections_panel as panel
        base = panel._intent(1, 3, "op", "why", {"scene_id": "S"}, "a" * 64, "x.tif")
        self.assertEqual(base, panel._intent(1, 3, "op", "why", {"scene_id": "S"}, "a" * 64, "x.tif"))
        self.assertNotEqual(base, panel._intent(1, 3, "op", "why", {"scene_id": "S"},
                                               "a" * 64, "y.tif"))
        self.assertNotEqual(base, panel._intent(1, 3, "op", "why", {"scene_id": "S"},
                                               "b" * 64, "x.tif"))
        self.assertNotEqual(base, panel._intent(1, 3, "op", "why", {"scene_id": "S"}, None, None))

    def test_flash_is_isolated_per_job(self):
        # 提示按作业存储/消费：切到另一作业不得显示本作业的成功提示。
        at = self.app()
        other = helpers.make_job(self.manager, b"OTHER-PARENT")
        at.run()                                        # 刷新历史下拉以包含新作业
        at.session_state["rec_corr_flash"] = {self.parent: "FLASH-FOR-PARENT",
                                              other: "FLASH-FOR-OTHER"}
        helpers.select_job(at, other)
        self.assertFalse(helpers.exceptions(at))
        success = " ".join(str(e.value) for e in at.success)
        self.assertNotIn("FLASH-FOR-PARENT", success)
        self.assertIn("FLASH-FOR-OTHER", success)
        self.assertEqual(at.session_state["rec_corr_flash"].get(self.parent), "FLASH-FOR-PARENT")


if __name__ == "__main__":
    with (OUT / "tests.log").open("x", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(PanelStateTests))
    summary = {"tests": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "successful": result.wasSuccessful(),
               "directory": str(OUT)}
    (OUT / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    print((OUT / "tests.log").read_text(encoding="utf-8"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
