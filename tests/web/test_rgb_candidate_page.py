"""INT-20261020-002 RGB图片识别页面 AppTest（.venv-p0，隔离后端+mock运行器）。

全程离线：后端数据目录指向临时目录（AGROLAKE_RGB_DIR），应用数据库指向
临时副本路径（AGROLAKE_DB），qa 模块 ENV_FILE 指向不存在的隔离路径；
subprocess.run 被替换为合成 fixture，不加载模型、不真实推理。
真实浏览器与真实模型网页流程由 Root 独立验收。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_rgb_candidate_page.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
WEB_TESTS = Path(__file__).resolve().parent
for p in (str(ROOT), str(WEB_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from streamlit.testing.v1 import AppTest  # noqa: E402

import agrolake.deepseek_client as ds  # noqa: E402
import test_rgb_candidate_jobs as fixture  # noqa: E402

APP = str(ROOT / "app.py")


def fake_run(cmd, **kwargs):
    """委托后端测试文件中的合成 fixture，并独立计数。"""
    return fixture.fake_run(cmd, **kwargs)


class PageTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "rgb"
        self._saved = {k: os.environ.get(k) for k in
                       ("AGROLAKE_RGB_DIR", "AGROLAKE_DB")}
        os.environ["AGROLAKE_RGB_DIR"] = str(self.data_dir)
        os.environ["AGROLAKE_DB"] = str(Path(self.tmp.name) / "app.sqlite")
        self._env_patch = mock.patch.object(
            ds, "ENV_FILE", Path(self.tmp.name) / "无.env")
        self._env_patch.start()
        self._patcher = mock.patch("subprocess.run", fake_run)
        self._patcher.start()
        fixture.FIXTURE["mode"] = "success"
        fixture.FIXTURE["calls"] = []

    def tearDown(self):
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

    def _open_page(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=180)
        at.run()
        at.sidebar.radio[0].set_value("RGB 图片识别")
        at.run()
        return at

    def _run_button(self, at: AppTest):
        for btn in at.button:
            if btn.key == "rgb_run":
                return btn
        raise AssertionError("页面上没有“开始识别”按钮")


class RgbCandidatePageTest(PageTestBase):
    def test_initial_render_zero_inference(self):
        at = self._open_page()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        subs = [s.value for s in at.subheader]
        self.assertIn("RGB 图片识别", subs)
        self.assertEqual(self.calls(), 0)
        # 上传控件与示例/上传选择存在；默认上传来源
        self.assertGreaterEqual(len(at.get("file_uploader")), 1)
        radios = [r.value for r in at.radio]
        self.assertIn("上传图片", radios)
        # 页面说明包含零推理与候选模型提示
        blob = "\n".join(str(c.value) for c in at.caption)
        self.assertIn("不会触发新的识别", blob)

    def test_sample_run_once_then_rerun_and_repeat_zero_new(self):
        at = self._open_page()
        at.radio[0].set_value("已审示例 U11")
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        btn = self._run_button(at)
        btn.click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        self.assertEqual(self.calls(), 1)
        # 结果展示出现（图片/表格元素 AppTest 不直接暴露，用固定文案与下载核对）
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("像素统计", md)
        self.assertIn("候选区域", md)
        self.assertIn("结果下载", md)
        dls = at.get("download_button")
        self.assertEqual(
            sum(1 for d in dls if "下载" in str(d.label)), 6)
        job_id_1 = at.session_state["rgb_current_job"]
        # 普通 rerun（刷新）零新增
        at.run()
        self.assertEqual(self.calls(), 1)
        # 相同输入重复点击：返回原任务，零新增
        self._run_button(at).click()
        at.run()
        self.assertEqual(self.calls(), 1)
        self.assertEqual(at.session_state["rgb_current_job"], job_id_1)
        # 历史查看零新增
        pick = next(sb for sb in at.selectbox if sb.key == "rgb_history_pick")
        labels = [o for o in pick.options if o != "（选择历史作业查看）"]
        self.assertTrue(labels)
        pick.set_value(labels[0])
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        self.assertEqual(self.calls(), 1)

    def test_failure_shown_then_no_fake_success_on_retry(self):
        at = self._open_page()
        at.radio[0].set_value("已审示例 U11")
        at.run()
        fixture.FIXTURE["mode"] = "fail_exit"
        self._run_button(at).click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        errors = "\n".join(str(e.value) for e in at.error)
        self.assertIn("识别失败", errors)
        self.assertEqual(self.calls(), 1)
        job_id_1 = at.session_state["rgb_current_job"]
        # 修复 fixture 后同号重试：返回已存失败作业，不自动重跑、不显示成功
        fixture.FIXTURE["mode"] = "success"
        self._run_button(at).click()
        at.run()
        self.assertEqual(self.calls(), 1)
        self.assertEqual(at.session_state["rgb_current_job"], job_id_1)
        errors = "\n".join(str(e.value) for e in at.error)
        self.assertIn("识别失败", errors)
        # 明确“新建一次识别”后才真正再跑一次并成功
        new_btn = next(b for b in at.button if b.key == "rgb_new")
        new_btn.click()
        at.run()
        self._run_button(at).click()
        at.run()
        self.assertEqual(self.calls(), 2)
        self.assertNotEqual(at.session_state["rgb_current_job"], job_id_1)
        self.assertEqual(len(at.error), 0, "新建识别成功后不应残留失败提示")
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("像素统计", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
