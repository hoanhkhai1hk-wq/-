"""问题处置：补充元数据、隔离、标记重复、复检。

- 每次处置写入 dispositions，保留处置前后记录。
- 修改来源/场景写入 meta_history，保留旧版本与当前指向。
- 复检重新执行规则并与旧检查关联；不修改历史结论，也不悄悄恢复隔离状态。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config, quality

METADATA_FIELDS = [
    "source", "source_link", "license_status", "acquired_at",
    "declared_scene_id", "intended_use",
]

VALID_ACTIONS = {
    "supplement_metadata", "isolate", "unisolate", "mark_duplicate",
    "recheck", "set_scene", "set_source",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _get_version(conn: sqlite3.Connection, version_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM asset_versions WHERE version_id = ?", (version_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"版本不存在：{version_id}")
    return row


def _record(conn: sqlite3.Connection, version_id: int, action: str, operator: str | None,
            reason: str | None, before: dict, after: dict,
            related_check_id: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO dispositions(version_id, action, operator, reason, before_json,"
        " after_json, related_check_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            version_id, action, operator, reason,
            json.dumps(before, ensure_ascii=False),
            json.dumps(after, ensure_ascii=False),
            related_check_id, _now(),
        ),
    )
    return cur.lastrowid


def _history(conn: sqlite3.Connection, version_id: int, field: str, old, new,
             reason: str | None, disp_id: int) -> None:
    conn.execute(
        "INSERT INTO meta_history(version_id, field, old_value, new_value, reason,"
        " disposition_id, changed_at) VALUES (?,?,?,?,?,?,?)",
        (version_id, field, None if old is None else str(old),
         None if new is None else str(new), reason, disp_id, _now()),
    )


def _recheck_map(conn: sqlite3.Connection, version_id: int) -> dict[str, int]:
    return {c["rule_id"]: c["check_id"] for c in quality.latest_checks(conn, version_id)}


def recheck(conn: sqlite3.Connection, version_id: int, operator: str | None = None,
            store_root: Path | None = None) -> dict:
    """重新执行质检，保留旧结论，并关联被复检的检查。"""
    ver = _get_version(conn, version_id)
    rm = _recheck_map(conn, version_id)
    quality.run_checks(conn, version_id, store_root, recheck_of_map=rm)
    disp_id = _record(
        conn, version_id, "recheck", operator, "复检",
        {"recheck_of": rm},
        {"status_unchanged": ver["status"]},
    )
    conn.commit()
    return {"disposition_id": disp_id, "status": ver["status"]}


def supplement_metadata(conn: sqlite3.Connection, version_id: int, operator: str | None,
                        reason: str, updates: dict, store_root: Path | None = None) -> dict:
    """补充/修订元数据，逐字段留痕，并重新验证（复检）。"""
    ver = _get_version(conn, version_id)
    before = {f: ver[f] for f in METADATA_FIELDS}
    applied = {k: v for k, v in updates.items() if k in METADATA_FIELDS and v is not None}
    if not applied:
        raise ValueError("没有可应用的元数据字段")
    disp_id = _record(conn, version_id, "supplement_metadata", operator, reason, before,
                      applied)
    for field, new_value in applied.items():
        _history(conn, version_id, field, ver[field], new_value, reason, disp_id)
    sets = ", ".join(f"{f} = ?" for f in applied)
    conn.execute(
        f"UPDATE asset_versions SET {sets}, updated_at = ? WHERE version_id = ?",
        (*applied.values(), _now(), version_id),
    )
    # 场景变更时补挂场景成员
    if "declared_scene_id" in applied and applied["declared_scene_id"]:
        conn.execute(
            "INSERT OR IGNORE INTO scenes(scene_id, label, created_at) VALUES (?,?,?)",
            (applied["declared_scene_id"], None, _now()),
        )
        conn.execute(
            "INSERT INTO scene_members(scene_id, version_id, role, basis, time_condition,"
            " confirm_status, created_at) VALUES (?,?,?,?,?,?,?)",
            (applied["declared_scene_id"], version_id, ver["content_type"],
             "supplemented", None, "manual", _now()),
        )
    conn.commit()
    recheck(conn, version_id, operator, store_root)
    return {"disposition_id": disp_id, "applied": list(applied)}


def set_scene(conn: sqlite3.Connection, version_id: int, scene_id: str,
              operator: str | None, reason: str, store_root: Path | None = None) -> dict:
    return supplement_metadata(
        conn, version_id, operator, reason or "更改场景",
        {"declared_scene_id": scene_id}, store_root,
    )


def set_source(conn: sqlite3.Connection, version_id: int, source: str,
               operator: str | None, reason: str, store_root: Path | None = None) -> dict:
    return supplement_metadata(
        conn, version_id, operator, reason or "更改来源", {"source": source}, store_root,
    )


def isolate(conn: sqlite3.Connection, version_id: int, operator: str | None,
            reason: str) -> dict:
    ver = _get_version(conn, version_id)
    before = {"status": ver["status"]}
    conn.execute(
        "UPDATE asset_versions SET status='isolated', updated_at=? WHERE version_id=?",
        (_now(), version_id),
    )
    disp_id = _record(conn, version_id, "isolate", operator, reason, before,
                      {"status": "isolated"})
    conn.commit()
    return {"disposition_id": disp_id, "status": "isolated"}


def unisolate(conn: sqlite3.Connection, version_id: int, operator: str | None,
              reason: str) -> dict:
    """显式解除隔离。必须由人工显式调用，复检不会自动恢复。"""
    ver = _get_version(conn, version_id)
    before = {"status": ver["status"]}
    conn.execute(
        "UPDATE asset_versions SET status='active', updated_at=? WHERE version_id=?",
        (_now(), version_id),
    )
    disp_id = _record(conn, version_id, "unisolate", operator, reason, before,
                      {"status": "active"})
    conn.commit()
    return {"disposition_id": disp_id, "status": "active"}


def mark_duplicate(conn: sqlite3.Connection, version_id: int, operator: str | None,
                   reason: str, of_version_id: int | None = None) -> dict:
    ver = _get_version(conn, version_id)
    before = {"status": ver["status"], "duplicate_of_version_id": ver["duplicate_of_version_id"]}
    conn.execute(
        "UPDATE asset_versions SET status='duplicate', duplicate_of_version_id=?,"
        " updated_at=? WHERE version_id=?",
        (of_version_id, _now(), version_id),
    )
    disp_id = _record(conn, version_id, "mark_duplicate", operator, reason, before,
                      {"status": "duplicate", "duplicate_of_version_id": of_version_id})
    conn.commit()
    return {"disposition_id": disp_id, "status": "duplicate"}
