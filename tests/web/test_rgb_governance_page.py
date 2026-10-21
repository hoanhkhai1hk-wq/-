"""INT-20261021-001 页面 AppTest（.venv-p0，隔离后端+合成fixture，全程离线）。

覆盖：
- RGB 页成功结果出现显式“登记到数据目录”按钮；普通浏览/刷新/历史零登记；
- 点击登记真实写入 P0 临时库，重复点击同链接不多写；
- 失败作业不可登记；
- DeepSeek 问答页 RGB 口径选择框、mock 提问、回答按上下文绑定、切换不串回答；
- 全程 mock ds.load_config/ask，无 .env 读取、零外部请求、零真实推理。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_rgb_governance_page.py
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
P0_TESTS = ROOT / "tests" / "p0"
WEB_TESTS = ROOT / "tests" / "web"
for p in (str(ROOT), str(P0_TESTS), str(WEB_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from streamlit.testing.v1 import AppTest  # noqa: E402

import agrolake.deepseek_client as ds  # noqa: E402
import test_rgb_candidate_jobs as fixture  # noqa: E402
import test_rgb_governance as gov  # noqa: E402  (复用 png_bytes)

APP = str(ROOT / "app.py")

CANNED_ANSWER = "RGB任务边界测试回答STUB-ANSWER-7f31。"


def _canned_reply():
    return ds.DeepSeekReply(
        ok=True, status="ok", status_zh="成功", answer=CANNED_ANSWER,
        model="deepseek-chat", attempts=1, latency_ms=42,
        usage={"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        provider_request_id="stub-rgb-1",
    )


class PageBase(unittest.TestCase):
    def setUp(self):
        test_root = ROOT / "reports/rgb/WB-INT-20261021-001/page-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=test_root)
        self.rgb_dir = Path(self.tmp.name) / "rgb"
        self.p0_db_file = Path(self.tmp.name) / "app.sqlite"
        self._saved = {k: os.environ.get(k) for k in
                       ("AGROLAKE_RGB_DIR", "AGROLAKE_DB", "AGROLAKE_DATA_DIR",
                        "AGROLAKE_STORE")}
        os.environ["AGROLAKE_RGB_DIR"] = str(self.rgb_dir)
        os.environ["AGROLAKE_DB"] = str(self.p0_db_file)
        os.environ["AGROLAKE_DATA_DIR"] = str(Path(self.tmp.name) / "p0")
        os.environ.pop("AGROLAKE_STORE", None)
        self._env_patch = mock.patch.object(
            ds, "ENV_FILE", Path(self.tmp.name) / "无.env")
        self._env_patch.start()
        self._config_patch = mock.patch.object(ds, "load_config", return_value=ds.DeepSeekConfig())
        self._config_patch.start()
        self._ask_guard = mock.patch.object(ds, "ask", side_effect=AssertionError("unexpected model call"))
        self._ask_guard.start()
        self._patcher = mock.patch("subprocess.run", fixture.fake_run)
        self._patcher.start()
        fixture.FIXTURE["mode"] = "success"
        fixture.FIXTURE["calls"] = []

    def tearDown(self):
        self._ask_guard.stop()
        self._config_patch.stop()
        self._patcher.stop()
        self._env_patch.stop()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def calls(self) -> int:
        return len(fixture.FIXTURE["calls"])

    def _links(self) -> int:
        conn = sqlite3.connect(self.p0_db_file)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM rgb_governance_links").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def _open_rgb_page(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        at.sidebar.radio[0].set_value("RGB 图片识别")
        at.run()
        at.radio[0].set_value("已审示例 U11")
        at.run()
        return at

    def _run_sample_job(self, at: AppTest) -> str:
        next(b for b in at.button if b.key == "rgb_run").click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        self.assertEqual(self.calls(), 1)
        return at.session_state["rgb_current_job"]


class RegisterButtonTest(PageBase):
    def test_explicit_button_registers_once_and_repeats_are_idempotent(self):
        at = self._open_rgb_page()
        job_id = self._run_sample_job(at)
        self.assertEqual(self._links(), 0)  # 浏览/识别本身不登记
        btn = next(b for b in at.button
                   if b.key == f"rgb_register_cur_{job_id}")
        btn.click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        self.assertEqual(self._links(), 1)
        success = "\n".join(s.value for s in at.success)
        self.assertIn("登记完成", success)
        self.assertIn(f"RGB-{job_id}", success)
        # 重复点击：同一链接，不多写
        next(b for b in at.button
             if b.key == f"rgb_register_cur_{job_id}").click()
        at.run()
        self.assertEqual(self._links(), 1)
        info = "\n".join(i.value for i in at.info)
        self.assertIn("已重新核验通过", info)
        # 普通刷新不新增
        at.run()
        self.assertEqual(self._links(), 1)

    def test_history_success_can_register_with_own_key(self):
        # 历史作业由后端直接创建（mock推理），与页面当前作业不同，
        # 历史渲染走 readonly/scope=hist 分支，按钮 key 含 hist 与完整 job_id。
        from agrolake.rgb_jobs import RgbCandidateJobs

        jobs = RgbCandidateJobs(data_dir=self.rgb_dir)
        job = jobs.submit(gov.png_bytes(), "hist.png", "a" * 32)
        self.assertEqual(job["status"], "succeeded")
        fixture.FIXTURE["calls"] = []
        at = self._open_rgb_page()
        pick = next(sb for sb in at.selectbox if sb.key == "rgb_history_pick")
        labels = [o for o in pick.options if o != "（选择历史作业查看）"]
        self.assertTrue(labels)
        pick.set_value(labels[0])
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        hist_btn = next(b for b in at.button
                        if b.key == f"rgb_register_hist_{job['id']}")
        self.assertEqual(self._links(), 0)
        hist_btn.click()
        at.run()
        self.assertEqual(self._links(), 1)

    def test_failed_job_has_no_register_path(self):
        at = self._open_rgb_page()
        fixture.FIXTURE["mode"] = "fail_exit"
        next(b for b in at.button if b.key == "rgb_run").click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        job_id = at.session_state["rgb_current_job"]
        errors = "\n".join(str(e.value) for e in at.error)
        self.assertIn("识别失败", errors)
        self.assertFalse(any(b.key.startswith("rgb_register_") for b in at.button))
        self.assertEqual(self._links(), 0)


class QaRgbScopeTest(PageBase):
    def _prep_job(self):
        """先用后端直接造一个成功作业（不经过页面，保持测试聚焦问答）。"""
        from agrolake.rgb_jobs import RgbCandidateJobs

        jobs = RgbCandidateJobs(data_dir=self.rgb_dir)
        job = jobs.submit(gov.png_bytes(), "a.png", "a" * 32)
        self.assertEqual(job["status"], "succeeded")
        return job["id"]

    def _open_qa_page(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        at.sidebar.radio[0].set_value("DeepSeek 问答")
        at.run()
        return at

    def test_rgb_scope_lists_jobs_and_stub_answer_binds_to_context(self):
        job_id = self._prep_job()
        at = self._open_qa_page()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        scope = next(r for r in at.radio if r.key == "qa_scope")
        self.assertIn("RGB 识别任务事实", scope.options)
        scope.set_value("rgb_job")
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        rgb_pick = next(sb for sb in at.selectbox if sb.key == "qa_rgb_job")
        labels = [o for o in rgb_pick.options if o != "（选择 RGB 任务）"]
        self.assertTrue(any(job_id[:8] in o for o in labels),
                        f"任务未出现在选择框：{labels}")
        rgb_pick.set_value(labels[0])
        at.run()
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("RGB 任务事实", md)
        cfg = ds.DeepSeekConfig(api_key="sk-" + "stub" * 8)
        with mock.patch.object(ds, "load_config", return_value=cfg), \
                mock.patch.object(ds, "ask", return_value=_canned_reply()) as ask:
            at.text_area[-1].set_value("这个RGB任务怎么样？")
            next(b for b in at.button if b.key == "qa_submit").click()
            at.run()
        ask.assert_called_once()  # 零外部请求：ask 被替换为桩
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn(CANNED_ANSWER, md)

        captions = "\n".join(str(c.value) for c in at.caption)
        self.assertIn(f"RGB任务 {job_id}", captions)  # 绑定说明带任务号
        self.assertIn("当时事实摘要 SHA 前16位", captions)

        # 提交后留在 RGB 口径：回答仍在
        scope = next(r for r in at.radio if r.key == "qa_scope")
        scope.set_value("overview")
        at.run()
        infos = "\n".join(str(i.value) for i in at.info)
        self.assertIn("输入问题后点击「提问」", infos)
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertNotIn(CANNED_ANSWER, md)  # 旧回答不挂在新口径下

        # 切回 RGB 口径+同一任务：绑定回答仍在（同上下文）
        scope = next(r for r in at.radio if r.key == "qa_scope")
        scope.set_value("rgb_job")
        at.run()
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn(CANNED_ANSWER, md)

        # Same job, changed governance facts: hide the old answer under new citations.
        from agrolake import db, rgb_governance
        conn = db.connect()
        try: rgb_governance.register(conn, job_id)
        finally: conn.close()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        self.assertNotIn(CANNED_ANSWER, "\n".join(str(m.value) for m in at.markdown))
        self.assertIn("事实或治理状态已变化", "\n".join(str(i.value) for i in at.info))

    def test_rgb_scope_without_db_shows_placeholder(self):
        at = self._open_qa_page()
        scope = next(r for r in at.radio if r.key == "qa_scope")
        scope.set_value("rgb_job")
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        rgb_pick = next(sb for sb in at.selectbox if sb.key == "qa_rgb_job")
        self.assertIn("（RGB 作业库不可用）", rgb_pick.options)
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("任务编号", md)  # 事实摘要写明无可解释任务

    def test_browsing_qa_page_never_registers(self):
        job_id = self._prep_job()
        at = self._open_qa_page()
        scope = next(r for r in at.radio if r.key == "qa_scope")
        scope.set_value("rgb_job")
        at.run()
        rgb_pick = next(sb for sb in at.selectbox if sb.key == "qa_rgb_job")
        labels = [o for o in rgb_pick.options if o != "（选择 RGB 任务）"]
        rgb_pick.set_value(labels[0])
        at.run()
        at.run()
        self.assertEqual(self._links(), 0)
        self.assertIsNotNone(job_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
