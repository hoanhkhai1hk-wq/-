"""场景关联查询与统计口径。

统计只针对当前有效版本：
- 每个资产只取最新版本（当前指向），历史版本不重复累加；
- 每个内容对象只保留唯一有效版本（跨文件名精确重复只认最早一次为有效）；
- 隔离(isolated)与重复(duplicate)版本排除；
- 场景归属按“当前生效”判定：文本用内容中的 scene_id，属性要求行内 scene_id 与声明一致。
scene_members 表保留为审计留痕，不作为当前归属依据。
"""
from __future__ import annotations

import json
import sqlite3

from . import quality


def list_versions(conn: sqlite3.Connection, include_inactive: bool = True) -> list[dict]:
    rows = conn.execute(
        """
        SELECT v.*, a.asset_key, o.relpath, o.size_bytes
        FROM asset_versions v
        JOIN assets a ON a.asset_id = v.asset_id
        JOIN objects o ON o.object_id = v.object_id
        ORDER BY v.version_id DESC
        """
    ).fetchall()
    out = []
    for r in rows:
        if not include_inactive and r["status"] != "active":
            continue
        checks = quality.latest_checks(conn, r["version_id"])
        d = dict(r)
        d["overall_status"] = quality.overall_status(checks)
        d["check_count"] = len(checks)
        out.append(d)
    return out


def _event_type_map(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT version_id, event_type FROM import_events").fetchall()
    out: dict[int, str] = {}
    for r in rows:
        out.setdefault(r["version_id"], r["event_type"])
    return out


def current_content_version_ids(conn: sqlite3.Connection) -> dict[int, int]:
    """每个资产的“当前内容版本”：最新一次定义了内容（非纯复制事件）的版本。

    复制事件(event_type=duplicate_content)只是再一次导入同一内容，不改变已有原件的
    当前内容指向。先选出当前内容版本，再由调用方检查其隔离/重复状态。
    """
    ev = _event_type_map(conn)
    rows = conn.execute(
        "SELECT version_id, asset_id, version_no FROM asset_versions ORDER BY version_no"
    ).fetchall()
    best: dict[int, int] = {}
    for r in rows:
        if ev.get(r["version_id"]) == "duplicate_content":
            continue
        best[r["asset_id"]] = r["version_id"]  # 升序遍历，较新版本覆盖
    return best


def current_versions(conn: sqlite3.Connection) -> list[dict]:
    """当前有效版本。

    先按资产选当前内容版本，再检查该版本状态：仅 active 计入。
    隔离或人工标重复的是当前内容版本时，资产没有当前有效版本，不自动回退到旧历史版本。
    同一内容对象跨版本/跨文件名只保留一处。
    """
    best = current_content_version_ids(conn)
    by_id = {v["version_id"]: v for v in list_versions(conn)}
    seen_obj: dict[str, int] = {}
    effective: list[dict] = []
    for asset_id, vid in sorted(best.items()):
        v = by_id.get(vid)
        if v is None or v["status"] != "active":
            continue
        if v["object_id"] in seen_obj:
            continue
        seen_obj[v["object_id"]] = vid
        effective.append(v)
    return sorted(effective, key=lambda x: x["version_id"])


def effective_scene(conn: sqlite3.Connection, v: dict) -> tuple[str | None, str | None]:
    """返回 (当前生效场景, 说明)。属性/文本按当前声明与内容重新判断。

    只有当前声明与内容一致时才返回场景；矛盾未处置时返回 (None, 说明)，
    既呈现问题又不计入合格关联。
    """
    ctype = v["content_type"]
    declared = v["declared_scene_id"]
    if ctype == "text":
        rows = conn.execute(
            "SELECT payload_json, fmt FROM text_records WHERE version_id = ?",
            (v["version_id"],),
        ).fetchall()
        content_scene = None
        for r in rows:
            if r["fmt"] == "json" and r["payload_json"]:
                try:
                    obj = json.loads(r["payload_json"])
                    if isinstance(obj, dict):
                        ps = obj.get("scene_id") or obj.get("scene")
                        if ps:
                            content_scene = str(ps)
                except Exception:
                    pass
        if content_scene:
            if declared and str(declared) != content_scene:
                return None, f"JSON scene_id 与声明失配（未处置）：声明={declared} 内容={content_scene}"
            return content_scene, None
        return declared, None
    if ctype == "attribute":
        rows = conn.execute(
            "SELECT scene_id FROM attribute_records WHERE version_id = ?",
            (v["version_id"],),
        ).fetchall()
        sids = sorted({r["scene_id"] for r in rows if r["scene_id"]})
        if len(sids) > 1:
            return None, (
                f"CSV 含多个场景 {sids}，需按行逐条关联，未作为单一场景归属统计"
            )
        if len(sids) == 1:
            only = sids[0]
            if declared and declared != only:
                return None, f"CSV 行 scene_id {only} 与声明 {declared} 不一致"
            return only, None
        if declared:
            return declared, None
        return None, "CSV 行缺少场景标识"
    return declared, None


def scene_members(conn: sqlite3.Connection, scene_id: str) -> dict:
    roles: dict[str, list[dict]] = {"image": [], "attribute": [], "text": []}
    issues: list[str] = []
    for v in current_versions(conn):
        eff, note = effective_scene(conn, v)
        if note:
            issues.append(f"版本 #{v['version_id']}：{note}")
        if eff == scene_id and v["content_type"] in roles:
            roles[v["content_type"]].append({
                "version_id": v["version_id"],
                "confirm_status": "current",
                "basis": v["source"],
            })
    missing = [role for role, items in roles.items() if not items]
    return {"scene_id": scene_id, "roles": roles, "missing_roles": missing, "issues": issues}


def list_scenes(conn: sqlite3.Connection) -> list[dict]:
    declared = conn.execute(
        "SELECT DISTINCT declared_scene_id AS s FROM asset_versions"
        " WHERE declared_scene_id IS NOT NULL"
    ).fetchall()
    ids = {r["s"] for r in declared}
    for v in current_versions(conn):
        eff, _ = effective_scene(conn, v)
        if eff:
            ids.add(eff)
    return [scene_members(conn, sid) for sid in sorted(ids)]


def scene_issues(conn: sqlite3.Connection) -> list[dict]:
    issues = []
    for v in current_versions(conn):
        eff, note = effective_scene(conn, v)
        if note:
            issues.append({"scene_id": v["declared_scene_id"], "kind": "scene_mismatch",
                           "detail": note})
    for sc in list_scenes(conn):
        if sc["missing_roles"]:
            issues.append({"scene_id": sc["scene_id"], "kind": "missing_roles",
                           "detail": sc["missing_roles"]})
    return issues


def association_coverage(conn: sqlite3.Connection) -> dict:
    scenes = list_scenes(conn)
    total = len(scenes)
    complete = sum(1 for s in scenes if not s["missing_roles"])
    return {
        "total_scenes": total,
        "complete_scenes": complete,
        "coverage": (complete / total) if total else None,
        "denominator": "场景清单全部 scene_id",
        "note": "按当前有效版本归属判定；失配/历史/隔离/重复版本不计入",
    }


def quality_pass_rate(conn: sqlite3.Connection) -> dict:
    versions = current_versions(conn)
    counted = [v for v in versions if v["overall_status"] in ("pass", "warn", "block")]
    passed = [v for v in counted if v["overall_status"] == "pass"]
    return {
        "counted_versions": len(counted),
        "passed_versions": len(passed),
        "pass_rate": (len(passed) / len(counted)) if counted else None,
        "denominator": "存在可适用规则且状态为 active 的当前有效版本",
        "excluded": "isolated、duplicate、历史版本、全部规则均为 na 的版本",
    }


def summary(conn: sqlite3.Connection) -> dict:
    total_versions = conn.execute("SELECT COUNT(*) c FROM asset_versions").fetchone()["c"]
    total_events = conn.execute("SELECT COUNT(*) c FROM import_events").fetchone()["c"]
    return {
        "total_versions": total_versions,
        "total_import_events": total_events,
        "current_versions": len(current_versions(conn)),
        "quality_pass_rate": quality_pass_rate(conn),
        "association_coverage": association_coverage(conn),
    }


def list_checks(conn: sqlite3.Connection, version_id: int) -> list[dict]:
    return [dict(r) for r in quality.latest_checks(conn, version_id)]
