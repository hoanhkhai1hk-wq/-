"""问答运行留档（T15）。

只写 ``qa_runs`` 这一张表：问题、事实快照、回答、模型、状态、用量、耗时。
设计上的三条边界：

- 表结构里没有密钥字段，也没有任何能放密钥的列；
  写入前把回答与错误信息再过一次 :func:`deepseek_client.redact`。
- 只 INSERT 这一次问答，不写 asset_versions、quality_checks、dispositions；
  模型给的是建议，落库的只是"这次问了什么、答了什么、成没成"。
- ``facts_sha256`` 记录事实摘要的摘要值，便于日后核对"当时看到的是哪一版事实"。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any

from . import deepseek_client as ds

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at TEXT NOT NULL,
    question TEXT NOT NULL,
    scope TEXT NOT NULL,
    scene_id TEXT,
    facts_text TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    gaps_json TEXT NOT NULL,
    facts_sha256 TEXT NOT NULL,
    answer TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    status_zh TEXT NOT NULL,
    model TEXT,
    response_model TEXT,
    base_url TEXT,
    http_status INTEGER,
    error_code TEXT,
    error_message TEXT,
    attempts INTEGER,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    provider_request_id TEXT,
    retried INTEGER
);
CREATE INDEX IF NOT EXISTS idx_qa_asked ON qa_runs(asked_at);
"""

# 建表语句里不允许出现的字样：出现即说明有人改了表结构想存密钥。
FORBIDDEN_COLUMNS = ("api_key", "key", "token_secret", "secret", "password")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # 对既有库补列，保留历史问答记录。
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(qa_runs)").fetchall()}
    if "response_model" not in cols:
        conn.execute("ALTER TABLE qa_runs ADD COLUMN response_model TEXT")
    conn.commit()


def facts_digest(facts_text: str) -> str:
    return hashlib.sha256((facts_text or "").encode("utf-8")).hexdigest()


def save_run(
    conn: sqlite3.Connection,
    *,
    question: str,
    bundle: Any,
    reply: ds.DeepSeekReply,
    config: ds.DeepSeekConfig | None = None,
) -> int:
    """写入一次问答。返回 ``run_id``。"""
    cfg = config or ds.load_config()
    facts_text = getattr(bundle, "text", "") or ""
    snapshot = getattr(bundle, "snapshot", {}) or {}
    citations = getattr(bundle, "citations", []) or []
    gaps = getattr(bundle, "gaps", []) or []
    scope = getattr(bundle, "scope", "")
    scene_id = (snapshot or {}).get("scene_id")

    cur = conn.execute(
        """
        INSERT INTO qa_runs (
            asked_at, question, scope, scene_id,
            facts_text, snapshot_json, citations_json, gaps_json, facts_sha256,
            answer, status, status_zh, model, response_model, base_url, http_status,
            error_code, error_message, attempts, latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            provider_request_id, retried
        ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?, ?,?,?,?,?)
        """,
        (
            _now(),
            ds.redact(question, cfg.api_key),
            scope,
            scene_id,
            ds.redact(facts_text, cfg.api_key),
            json.dumps(snapshot, ensure_ascii=False),
            json.dumps(citations, ensure_ascii=False),
            json.dumps(gaps, ensure_ascii=False),
            facts_digest(facts_text),
            ds.redact(reply.answer or "", cfg.api_key),
            reply.status,
            reply.status_zh,
            reply.model or cfg.model,
            reply.response_model or "",
            cfg.base_url,
            reply.http_status,
            ds.redact(reply.error_code or "", cfg.api_key),
            ds.redact(reply.error_message or "", cfg.api_key),
            reply.attempts,
            reply.latency_ms,
            reply.usage.get("prompt_tokens"),
            reply.usage.get("completion_tokens"),
            reply.usage.get("total_tokens"),
            reply.provider_request_id,
            1 if reply.retried else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def recent_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM qa_runs ORDER BY run_id DESC LIMIT ?", (int(limit),)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("snapshot_json", "citations_json", "gaps_json"):
            try:
                d[k.replace("_json", "")] = json.loads(d.pop(k) or "null")
            except (ValueError, TypeError):
                d[k.replace("_json", "")] = None
        out.append(d)
    return out


def count_runs(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM qa_runs").fetchone()["c"]


def scan_for_secret(conn: sqlite3.Connection, secret: str) -> list[int]:
    """检查库里有没有混进密钥原文，返回命中的 run_id。

    仅用于测试与自检；正常流程不会命中，因为写入前已脱敏且没有密钥字段。
    """
    if not secret or len(secret) < 8:
        return []
    hits: list[int] = []
    rows = conn.execute("SELECT * FROM qa_runs").fetchall()
    for r in rows:
        for value in r:
            if isinstance(value, str) and secret in value:
                hits.append(r["run_id"])
                break
    return hits
