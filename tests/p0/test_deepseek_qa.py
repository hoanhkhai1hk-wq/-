"""T15 DeepSeek 数据问答的固定验收测试（标准库 unittest）。

覆盖：配置优先级、密钥脱敏、本机路径清理、六类失败状态、重试上限、
请求体不含密钥、问答留档不含密钥、事实层只读与真实数字。

不联网：所有 HTTP 用 FakeSession 替代；真实调用记录单独放
``reports/DeepSeek接入验收.md``，本文件只证明逻辑正确。
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from agrolake import db, deepseek_client as ds, qa_facts, qa_store  # noqa: E402

FAKE_KEY = "sk-" + "test" * 8          # 假密钥，故意做成 35 位长度


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """按预设顺序返回响应或抛异常，并记录每次请求。"""

    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "json": json, "timeout": timeout})
        if not self.items:
            raise AssertionError("调用次数超出预期，疑似无上限重试")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def ok_payload(content: str = "当前共 5 条待复核资料。") -> dict:
    return {
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def make_cfg(**over) -> ds.DeepSeekConfig:
    base = dict(api_key=FAKE_KEY, base_url="https://api.deepseek.com",
                model="deepseek-chat", timeout=30.0, max_tokens=900,
                temperature=0.2, max_attempts=2, key_source="测试", env_file="")
    base.update(over)
    return ds.DeepSeekConfig(**base)


# ── 临时库：造一套最小可查的治理数据 ──────────────────────────────────────
def build_temp_db(dir_path: Path) -> Path:
    p = dir_path / "t.sqlite"
    conn = db.connect(p)
    db.init_schema(conn)
    now = "2026-10-18T00:00:00+08:00"
    conn.execute(
        "INSERT INTO objects(object_id, relpath, size_bytes, created_at)"
        " VALUES ('obj1','objects/aa/obj1',100,?)", (now,))
    conn.execute(
        "INSERT INTO assets(asset_id, asset_key, original_filename, created_at)"
        " VALUES (1,'a.png','a.png',?)", (now,))
    conn.execute(
        "INSERT INTO asset_versions(version_id, asset_id, version_no, content_type,"
        " object_id, original_filename, source, declared_scene_id, status,"
        " created_at, updated_at) VALUES"
        " (1,1,1,'image','obj1','a_ok.png','src','S1','active',?,?),"
        " (2,1,2,'image','obj1','a_bad.png','src','S1','active',?,?)",
        (now, now, now, now),
    )
    conn.execute(
        "INSERT INTO import_events(imported_at, asset_id, version_id, original_filename,"
        " content_sha256, object_relpath, size_bytes, event_type)"
        " VALUES (?,1,1,'a_ok.png','obj1','objects/aa/obj1',100,'new_asset')", (now,))
    conn.execute(
        "INSERT INTO quality_checks(version_id, rule_id, rule_version, status, reason, checked_at)"
        " VALUES (1,'image_decode','QR-1.0.0','pass','ok',?),"
        "        (2,'image_decode','QR-1.0.0','block',?,?)",
        (now, r"图像解码失败：cannot identify image file 'D:\\proj\\data\\store\\objects\\ff'", now),
    )
    conn.commit()
    conn.close()
    return p


class ConfigTest(unittest.TestCase):
    def test_env_file_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / ".env"
            f.write_text(
                "# 注释行\n"
                "A=1\n"
                'B="带引号"\n'
                "export C=3\n"
                "D = 4 \n"
                "没有等号的行\n",
                encoding="utf-8",
            )
            got = ds.read_env_file(f)
        self.assertEqual(got["A"], "1")
        self.assertEqual(got["B"], "带引号")
        self.assertEqual(got["C"], "3")
        self.assertEqual(got["D"], "4")
        self.assertNotIn("没有等号的行", got.values())

    def test_missing_env_file_returns_empty(self):
        self.assertEqual(ds.read_env_file(Path("不存在的文件.env")), {})

    def test_priority_env_over_file_over_default(self):
        table = {ds.ENV_KEY: "from-file", ds.ENV_MODEL: "file-model"}
        with mock.patch.dict(os.environ, {ds.ENV_KEY: "from-env"}, clear=False):
            os.environ.pop(ds.ENV_MODEL, None)
            cfg = ds.load_config(table)
        self.assertEqual(cfg.api_key, "from-env")
        self.assertEqual(cfg.model, "file-model")      # 环境变量没给就用文件
        self.assertEqual(cfg.base_url, ds.DEFAULT_BASE_URL)  # 都没有就用默认
        self.assertTrue(cfg.configured)

    def test_no_key_means_not_configured(self):
        cfg = ds.load_config({})
        self.assertFalse(cfg.configured)
        self.assertIn("未配置", cfg.key_hint)

    def test_describe_has_no_key(self):
        text = str(make_cfg().describe())
        self.assertNotIn(FAKE_KEY, text)

    def test_attempt_cap(self):
        cfg = ds.load_config({ds.ENV_MAX_ATTEMPTS: "99"})
        self.assertLessEqual(cfg.max_attempts, ds.HARD_MAX_ATTEMPTS)

    def test_chat_ready_unconfigured(self):
        ok, note = ds.chat_ready(ds.load_config({}))
        self.assertFalse(ok)
        self.assertIn(ds.ENV_KEY, note)


class RedactTest(unittest.TestCase):
    def test_redact_key(self):
        self.assertNotIn(FAKE_KEY, ds.redact(f"失败 {FAKE_KEY} 结束", FAKE_KEY))

    def test_redact_generic_sk(self):
        self.assertNotIn("sk-abcdefghijklmn", ds.redact("token=sk-abcdefghijklmn"))

    def test_sanitize_removes_windows_path(self):
        raw = r"图像解码失败：cannot identify image file 'D:\proj\data\store\objects\ff'"
        out = ds.sanitize_facts_text(raw)
        self.assertNotIn("D:\\proj", out)
        self.assertIn("本机路径已省略", out)

    def test_sanitize_keeps_business_text(self):
        self.assertIn("共 5 条", ds.sanitize_facts_text("共 5 条待复核"))


class AskTest(unittest.TestCase):
    def test_not_configured_never_calls(self):
        session = FakeSession()   # 空队列：一旦被调用就 AssertionError
        reply = ds.ask("问题", "事实", config=ds.load_config({}), session=session)
        self.assertEqual(reply.status, "not_configured")
        self.assertEqual(reply.answer, "")
        self.assertEqual(session.calls, [])

    def test_empty_question_never_calls(self):
        session = FakeSession()
        reply = ds.ask("   ", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "invalid_request")
        self.assertEqual(session.calls, [])

    def test_ok(self):
        session = FakeSession(FakeResponse(200, ok_payload()))
        reply = ds.ask("当前批次待复核资料有哪些？", "事实", config=make_cfg(), session=session)
        self.assertTrue(reply.ok)
        self.assertEqual(reply.status, "ok")
        self.assertIn("待复核", reply.answer)
        self.assertEqual(reply.attempts, 1)
        self.assertEqual(reply.usage["total_tokens"], 150)
        self.assertEqual(reply.provider_request_id, "chatcmpl-test-001")
        self.assertEqual(len(session.calls), 1)

    def test_request_body_has_no_key(self):
        session = FakeSession(FakeResponse(200, ok_payload()))
        ds.ask("问题", "事实", config=make_cfg(), session=session)
        body = session.calls[0]["json"]
        self.assertNotIn(FAKE_KEY, str(body))
        # Key 只能出现在 Authorization 头里，这是协议要求
        self.assertIn(FAKE_KEY, session.calls[0]["headers"]["Authorization"])

    def test_request_sends_model_and_params(self):
        session = FakeSession(FakeResponse(200, ok_payload()))
        cfg = make_cfg(model="deepseek-chat", temperature=0.3, max_tokens=777)
        ds.ask("问题", "事实", config=cfg, session=session)
        body = session.calls[0]["json"]
        self.assertEqual(body["model"], "deepseek-chat")
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["max_tokens"], 777)
        self.assertFalse(body["stream"])
        self.assertEqual(session.calls[0]["url"], "https://api.deepseek.com/chat/completions")

    def test_auth_failed_no_retry(self):
        session = FakeSession(FakeResponse(401, {"error": {
            "message": "Authentication Fails", "type": "authentication_error"}}))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "auth_failed")
        self.assertEqual(reply.attempts, 1)
        self.assertEqual(len(session.calls), 1)     # 认证失败不重试
        self.assertEqual(reply.answer, "")

    def test_insufficient_balance_no_retry(self):
        session = FakeSession(FakeResponse(402, {"error": {
            "message": "Insufficient Balance", "type": "insufficient_balance"}}))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "insufficient_balance")
        self.assertEqual(len(session.calls), 1)

    def test_invalid_params_no_retry(self):
        session = FakeSession(FakeResponse(422, {"error": {
            "message": "bad param", "code": "invalid_parameter"}}))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "invalid_request")
        self.assertEqual(len(session.calls), 1)

    def test_rate_limited_retries_then_gives_up(self):
        resp = FakeResponse(429, {"error": {"message": "Rate Limit Reached"}})
        session = FakeSession(resp, FakeResponse(429, {"error": {"message": "Rate Limit Reached"}}))
        with mock.patch.object(ds.time, "sleep", lambda *_: None):
            reply = ds.ask("问题", "事实", config=make_cfg(max_attempts=2), session=session)
        self.assertEqual(reply.status, "rate_limited")
        self.assertEqual(reply.attempts, 2)
        self.assertEqual(len(session.calls), 2)     # 到上限就停，不会无限重试

    def test_timeout_retries_once_then_reports(self):
        session = FakeSession(requests.Timeout("slow"), requests.Timeout("slow"))
        with mock.patch.object(ds.time, "sleep", lambda *_: None):
            reply = ds.ask("问题", "事实", config=make_cfg(max_attempts=2), session=session)
        self.assertEqual(reply.status, "timeout")
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(reply.retried)

    def test_server_error_then_success(self):
        session = FakeSession(
            FakeResponse(503, {"error": {"message": "Server Overloaded"}}),
            FakeResponse(200, ok_payload("重试成功")),
        )
        with mock.patch.object(ds.time, "sleep", lambda *_: None):
            reply = ds.ask("问题", "事实", config=make_cfg(max_attempts=2), session=session)
        self.assertTrue(reply.ok)
        self.assertEqual(reply.attempts, 2)

    def test_network_error(self):
        session = FakeSession(requests.ConnectionError("down"))
        reply = ds.ask("问题", "事实", config=make_cfg(max_attempts=1), session=session)
        self.assertEqual(reply.status, "network_error")
        self.assertEqual(reply.answer, "")

    def test_empty_content_is_bad_response(self):
        payload = ok_payload()
        payload["choices"][0]["message"]["content"] = "   "
        session = FakeSession(FakeResponse(200, payload))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "bad_response")
        self.assertEqual(reply.answer, "")

    def test_non_json_body_is_bad_response(self):
        session = FakeSession(FakeResponse(200, None, "<html>502</html>"))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertEqual(reply.status, "bad_response")

    def test_error_message_never_leaks_key(self):
        session = FakeSession(FakeResponse(401, {"error": {
            "message": f"bad key {FAKE_KEY}"}}))
        reply = ds.ask("问题", "事实", config=make_cfg(), session=session)
        self.assertNotIn(FAKE_KEY, reply.error_message)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = build_temp_db(Path(self.tmp.name))
        self.conn = db.connect(self.path)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_schema_has_no_secret_column(self):
        qa_store.ensure_schema(self.conn)
        cols = [r["name"].lower() for r in
                self.conn.execute("PRAGMA table_info(qa_runs)").fetchall()]
        for bad in ("api_key", "secret", "password"):
            self.assertNotIn(bad, cols)
        self.assertIn("model", cols)

    def test_save_and_read_back(self):
        qa_store.ensure_schema(self.conn)
        ro = db.connect_readonly(self.path)
        try:
            bundle = qa_facts.build(ro, "batch_review")
        finally:
            ro.close()
        reply = ds.DeepSeekReply(ok=True, status="ok", status_zh="成功",
                                 answer="有 1 条待复核", model="deepseek-chat",
                                 attempts=1, latency_ms=123,
                                 usage={"prompt_tokens": 1, "completion_tokens": 2,
                                        "total_tokens": 3})
        run_id = qa_store.save_run(self.conn, question="问", bundle=bundle,
                                   reply=reply, config=make_cfg())
        self.assertGreater(run_id, 0)
        rows = qa_store.recent_runs(self.conn)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["total_tokens"], 3)
        self.assertIn("snapshot", rows[0])
        self.assertTrue(rows[0]["facts_sha256"])

    def test_stored_row_never_contains_key(self):
        qa_store.ensure_schema(self.conn)
        ro = db.connect_readonly(self.path)
        try:
            bundle = qa_facts.build(ro, "overview")
        finally:
            ro.close()
        reply = ds.DeepSeekReply(ok=False, status="auth_failed", status_zh="认证失败",
                                 error_message=f"bad key {FAKE_KEY}", model="deepseek-chat")
        qa_store.save_run(self.conn, question=f"用 {FAKE_KEY} 问", bundle=bundle,
                          reply=reply, config=make_cfg())
        self.assertEqual(qa_store.scan_for_secret(self.conn, FAKE_KEY), [])


class FactsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = build_temp_db(Path(self.tmp.name))
        self.ro = db.connect_readonly(self.path)

    def tearDown(self):
        self.ro.close()
        self.tmp.cleanup()

    def test_readonly_connection_rejects_writes(self):
        with self.assertRaises(Exception):
            self.ro.execute("INSERT INTO meta(key,value) VALUES ('x','y')")

    def test_all_scopes_build(self):
        for scope in qa_facts.SCOPES:
            b = qa_facts.build(self.ro, scope)
            self.assertEqual(b.scope, scope)
            self.assertTrue(b.text.strip())
            self.assertIn("程序从 SQLite 直接统计", b.text)
            self.assertTrue(b.citations)

    def test_counts_come_from_data(self):
        b = qa_facts.build(self.ro, "batch_review")
        self.assertIn("待复核资料条数：1", b.text)
        self.assertIn("a_bad.png", b.text)
        self.assertEqual(b.snapshot["pending_count"], 1)

    def test_facts_have_no_local_path(self):
        b = qa_facts.build(self.ro, "batch_review")
        self.assertNotIn("D:\\proj", b.text)
        self.assertIn("本机路径已省略", b.text)

    def test_missing_recognition_is_stated_as_gap(self):
        b = qa_facts.build(self.ro, "scene")
        self.assertIn("识别结果：暂无", b.text)
        self.assertTrue(any("识别结果" in g for g in b.gaps))

    def test_unknown_scope_falls_back(self):
        b = qa_facts.build(self.ro, "乱写")
        self.assertEqual(b.scope, "overview")


if __name__ == "__main__":
    unittest.main(verbosity=2)
