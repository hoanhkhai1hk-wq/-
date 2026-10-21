"""T15 问答页面的 AppTest 验收（Streamlit AppTest，全程离线）。

覆盖：
- 按名称进入问答页，页面能渲染不报错；
- 未配置密钥时显示准确状态，本地事实摘要照常可见，提交后显示「未配置」而不是假答案；
- 配置就绪时（把 ask 换成桩）提交能拿到回答、写入 qa_runs 并显示引用来源。

本测试**不发真实请求**：未配置用例靠清空密钥，成功用例把 ask 换成桩。
真实调用记录见 reports/DeepSeek接入验收.md。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_qa_page.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from agrolake import deepseek_client as ds, qa_store  # noqa: E402

APP = str(ROOT / "app.py")
DEMO_DB = ROOT / "data" / "p0" / "agrolake.sqlite"


def _copy_db(tmpdir: str) -> Path:
    target = Path(tmpdir) / "qa_test.sqlite"
    shutil.copy2(DEMO_DB, target)
    return target


class QaPageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_file = _copy_db(self.tmp.name)
        self._saved = {k: os.environ.get(k) for k in
                       ("AGROLAKE_DB", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL")}
        os.environ["AGROLAKE_DB"] = str(self.db_file)
        os.environ.pop("DEEPSEEK_MODEL", None)
        # 防止 .env 里的真 Key 被读到：本测试一律走显式设置
        self._env_patch = mock.patch.object(ds, "ENV_FILE", Path(self.tmp.name) / "无.env")
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def _run(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        at.sidebar.radio[0].set_value("DeepSeek 问答")
        at.run()
        return at

    def test_page_renders_without_key(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        at = self._run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        subs = [s.value for s in at.subheader]
        self.assertIn("DeepSeek 数据问答", subs)
        blob = "\n".join(w.value for w in at.warning)
        self.assertIn("未配置", blob)

    def test_facts_visible_without_key(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        at = self._run()
        blob = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("程序从 SQLite 直接统计", blob)
        self.assertIn("事实摘要", blob)

    def test_submit_without_key_reports_not_configured(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        at = self._run()
        at.text_area[-1].set_value("当前批次待复核资料有哪些？")
        at.button(key="qa_submit").click()
        at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        blob = "\n".join(w.value for w in at.warning)
        self.assertIn("未配置", blob)
        # 失败也要留档，且状态准确
        import sqlite3
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status, answer FROM qa_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "not_configured")
        self.assertEqual(row["answer"], "")

    def test_submit_with_stub_ask_shows_answer(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-" + "stub" * 8
        canned = ds.DeepSeekReply(
            ok=True, status="ok", status_zh="成功",
            answer="当前共 5 条待复核资料，均在场景 S1。",
            model="deepseek-chat", attempts=1, latency_ms=42,
            usage={"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            provider_request_id="stub-1",
        )
        with mock.patch.object(ds, "ask", return_value=canned):
            at = self._run()
            at.text_area[-1].set_value("当前批次待复核资料有哪些？")
            at.button(key="qa_submit").click()
            at.run()
        self.assertFalse(at.exception, [str(e) for e in at.exception])
        success = "\n".join(s.value for s in at.success)
        self.assertIn("成功", success)
        md = "\n".join(str(m.value) for m in at.markdown)
        self.assertIn("当前共 5 条待复核资料", md)

        import sqlite3
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT status, answer, total_tokens FROM qa_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "ok")
        self.assertIn("5 条待复核", row["answer"])
        self.assertEqual(row["total_tokens"], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
