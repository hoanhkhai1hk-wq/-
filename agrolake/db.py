"""SQLite 模式与连接。

记录模型：对象(内容) / 业务资产 / 资产版本 / 独立导入事件 / 质检 / 处置 / 场景关联。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config

SCHEMA_VERSION = "AGROLAKE-P0-DB-1.0.0"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,          -- 内容 SHA256
    relpath TEXT NOT NULL,               -- 相对存储根的对象路径
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_key TEXT NOT NULL,             -- 规范化原始文件名，用于聚合版本
    original_filename TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS asset_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL REFERENCES assets(asset_id),
    version_no INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES objects(object_id),
    original_filename TEXT NOT NULL,
    source TEXT,
    source_link TEXT,
    license_status TEXT,
    acquired_at TEXT,
    declared_scene_id TEXT,
    intended_use TEXT,
    width INTEGER,
    height INTEGER,
    bands INTEGER,
    bitdepth INTEGER,
    status TEXT NOT NULL DEFAULT 'active',   -- active | isolated | duplicate
    duplicate_of_version_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    imported_at TEXT NOT NULL,
    asset_id INTEGER NOT NULL,
    version_id INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    declared_source TEXT,
    declared_scene_id TEXT,
    content_sha256 TEXT NOT NULL,
    object_relpath TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    event_type TEXT NOT NULL,            -- new_asset | new_version_same_name | duplicate_content
    notes TEXT
);

CREATE TABLE IF NOT EXISTS attribute_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    scene_id TEXT,
    field_id TEXT,
    crop TEXT,
    category_raw TEXT,                   -- 地物类别原值（与作物 crop 分开）
    category_std TEXT,                   -- 标准化后的地物类别，未知为空并记 issues
    area_raw TEXT,                       -- 保留原值
    area_unit_raw TEXT,                  -- 保留原值
    area_m2 REAL,                        -- 仅对已定义单位转换
    row_index INTEGER,
    issues TEXT
);

CREATE TABLE IF NOT EXISTS text_records (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    scene_id TEXT,
    source TEXT,
    fmt TEXT,
    payload_json TEXT,
    issues TEXT
);

CREATE TABLE IF NOT EXISTS quality_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    params_json TEXT,
    measured_json TEXT,
    status TEXT NOT NULL,                -- pass | warn | block | na
    reason TEXT,
    checked_at TEXT NOT NULL,
    recheck_of_check_id INTEGER
);

CREATE TABLE IF NOT EXISTS dispositions (
    disp_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    action TEXT NOT NULL,                -- supplement_metadata | isolate | mark_duplicate | recheck | set_scene | set_source
    operator TEXT,
    reason TEXT,
    before_json TEXT,
    after_json TEXT,
    related_check_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    disposition_id INTEGER,
    changed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id TEXT PRIMARY KEY,
    label TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scene_members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id TEXT NOT NULL,
    version_id INTEGER NOT NULL,
    role TEXT NOT NULL,                  -- image | attribute | text
    basis TEXT,
    time_condition TEXT,
    confirm_status TEXT NOT NULL DEFAULT 'auto',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_versions_asset ON asset_versions(asset_id);
CREATE INDEX IF NOT EXISTS idx_events_version ON import_events(version_id);
CREATE INDEX IF NOT EXISTS idx_attrs_version ON attribute_records(version_id);
CREATE INDEX IF NOT EXISTS idx_texts_version ON text_records(version_id);
CREATE INDEX IF NOT EXISTS idx_checks_version ON quality_checks(version_id);
CREATE INDEX IF NOT EXISTS idx_members_scene ON scene_members(scene_id);
"""


def connect(db_file: Path | None = None) -> sqlite3.Connection:
    path = Path(db_file) if db_file is not None else config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用默认的线程检查：连接不跨线程复用，由调用方（应用每次运行）新建并关闭。
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_readonly(db_file: Path | None = None) -> sqlite3.Connection:
    """只读连接。

    问答页的事实查询走这个连接：写操作在 SQLite 层面就被拒绝，
    不依赖"调用方保证不写"。数据库文件不存在时由调用方先建好主连接。
    """
    path = Path(db_file) if db_file is not None else config.db_path()
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """对既有数据库补齐新增列，保留历史数据。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(attribute_records)").fetchall()}
    for name in ("category_raw", "category_std"):
        if name not in cols:
            conn.execute(f"ALTER TABLE attribute_records ADD COLUMN {name} TEXT")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
