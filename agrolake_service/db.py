"""独立识别表 SQLite 模式与连接（DEV-20261015-009）。

只新建 ``rec_`` 前缀的识别表；与 P0 共享同一 ``AGROLAKE_DB`` 文件时，P0 既有表与数据
原样保留。不修改、不迁移、不删除任何既有表。

连接规则：连接不跨线程复用；每次请求/每次工作由调用方新建并显式关闭
（显式finally关闭；sqlite连接的with只控制事务，不负责close）。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "AGROLAKE-RECOGNITION-DB-1.0.0"

# 作业状态：queued | running | succeeded | failed | interrupted
SCHEMA = """
CREATE TABLE IF NOT EXISTS rec_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rec_inputs (
    input_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,           -- 内容地址
    object_relpath TEXT NOT NULL,          -- 相对存储根的对象路径
    size_bytes INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    sample_kind TEXT NOT NULL,             -- public_adapter_sample | user_upload | synthetic_test
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rec_requests (
    request_id TEXT PRIMARY KEY,           -- 可选幂等键
    request_fingerprint TEXT NOT NULL,     -- 规范化请求内容摘要
    job_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rec_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id INTEGER NOT NULL REFERENCES rec_inputs(input_id),
    input_sha256 TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    source TEXT NOT NULL,
    source_link TEXT,
    license_status TEXT,
    scene_id TEXT NOT NULL,
    sample_kind TEXT NOT NULL,
    request_id TEXT,
    status TEXT NOT NULL,                  -- queued | running | succeeded | failed | interrupted
    error TEXT,
    run_dir_relpath TEXT,                  -- 相对 runs 根
    cli_output_relpath TEXT,
    log_relpath TEXT,
    model_path TEXT,
    model_sha256 TEXT,
    cli_path TEXT,
    worker_python TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS rec_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES rec_jobs(job_id),
    run_seq INTEGER NOT NULL,
    status TEXT NOT NULL,                  -- succeeded | failed | interrupted
    exit_code INTEGER,
    adapter TEXT,
    versions_json TEXT,
    params_json TEXT,
    input_sha256 TEXT,
    model_sha256 TEXT,
    elapsed_seconds REAL,
    field_count INTEGER,
    output_relpath TEXT,
    run_json_relpath TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS rec_regions (
    region_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    field_id INTEGER NOT NULL,
    pixels INTEGER NOT NULL,
    min_row INTEGER NOT NULL,
    min_col INTEGER NOT NULL,
    max_row INTEGER NOT NULL,
    max_col INTEGER NOT NULL,
    area_unit TEXT NOT NULL DEFAULT 'pixel'  -- 仅像素，不代表平方米
);

CREATE TABLE IF NOT EXISTS rec_class_counts (
    count_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    class_value INTEGER NOT NULL,
    meaning TEXT NOT NULL,
    pixels INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rec_artifacts (
    job_id INTEGER NOT NULL REFERENCES rec_jobs(job_id),
    run_id INTEGER NOT NULL REFERENCES rec_runs(run_id),
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    PRIMARY KEY(job_id, name)
);

CREATE INDEX IF NOT EXISTS idx_rec_jobs_status ON rec_jobs(status);
CREATE INDEX IF NOT EXISTS idx_rec_runs_job ON rec_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_rec_regions_job ON rec_regions(job_id);
CREATE INDEX IF NOT EXISTS idx_rec_counts_job ON rec_class_counts(job_id);

-- 人工复核记录（DEV-20261015-013）。只新增，不改旧表；历史不可覆盖。
CREATE TABLE IF NOT EXISTS rec_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    field_id INTEGER,                      -- NULL 表示对任务整体的复核，非 NULL 为某候选区域
    decision TEXT NOT NULL,                -- accepted | questioned | rejected
    reviewer TEXT NOT NULL,                -- 本机署名，不是已认证身份
    note TEXT NOT NULL,
    request_id TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,     -- 规范化内容指纹，用于同号幂等
    input_sha256 TEXT NOT NULL,            -- 服务端从该作业取得的快照
    model_sha256 TEXT,
    run_seq INTEGER,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_rec_reviews_request ON rec_reviews(request_id);
CREATE INDEX IF NOT EXISTS idx_rec_reviews_job ON rec_reviews(job_id);
CREATE INDEX IF NOT EXISTS idx_rec_reviews_job_field ON rec_reviews(job_id, field_id);

-- 识别输入登记单与 P0 资产的治理链接（DEV-20261015-014）。只新增，不改旧表/既有记录。
-- 每个作业固定一条登记单；job_id 与 request_id 均唯一，用于同号幂等与跨作业冲突。
CREATE TABLE IF NOT EXISTS rec_governance_links (
    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL UNIQUE,
    request_id TEXT NOT NULL UNIQUE,
    asset_id INTEGER NOT NULL,              -- P0 资产编号
    version_id INTEGER NOT NULL,            -- P0 资产版本编号
    manifest_sha256 TEXT NOT NULL,          -- 登记单JSON内容摘要（不是原TIFF摘要）
    input_sha256 TEXT NOT NULL,             -- 原TIFF输入摘要（与登记单摘要分开）
    manifest_relpath TEXT NOT NULL,         -- 服务存储中登记单对象相对路径
    snapshot_json TEXT NOT NULL,            -- 登记时的服务端原事实与质量检查快照
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rec_governance_job ON rec_governance_links(job_id);
CREATE INDEX IF NOT EXISTS idx_rec_governance_version ON rec_governance_links(version_id);
"""

# 候选区域类计数含义（与 CLI run.json 一致）。
CLASS_MEANINGS = {
    0: "background",
    1: "field_interior",
    2: "boundary",
    255: "nodata/unlabelled",
}

CLASS_PIXEL_KEY = {
    0: "background_0",
    1: "field_interior_1",
    2: "boundary_2",
    255: "nodata_255",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """新建连接。不跨线程复用；调用方负责 close。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # 不切换 journal 模式，避免改动共享库既有设置；用忙等待处理并发写。
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO rec_meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM rec_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
