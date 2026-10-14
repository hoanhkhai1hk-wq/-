"""质量检查规则执行。

结果状态：pass / warn / block / na。
- 确定性规则（空文件、解码失败、单位错误、JSON 解析失败）给出明确阻断/错误。
- 启发式清晰度规则固定预览尺度、阈值与适用范围，仅作提示。
"""
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config, semantics


def _load_quality_rules() -> dict[str, Any]:
    with open(config.QUALITY_RULES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def rules_version() -> str:
    return _load_quality_rules()["rules_version"]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sharpness(path: Path, rules: dict[str, Any]) -> tuple[float, str]:
    """固定预览尺度下的灰度拉普拉斯方差。返回 (方差, 说明)。"""
    from PIL import Image, ImageFilter, ImageStat

    cfg = rules["sharpness"]
    img = Image.open(path).convert(cfg["band"])
    w, h = img.size
    scale = min(1.0, cfg["preview_max_side"] / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    lap = img.filter(
        ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)
    )
    variance = float(ImageStat.Stat(lap).var[0])
    note = (
        f"预览缩放={scale:.3f}，最长边≤{cfg['preview_max_side']}，"
        f"band={cfg['band']}，阈值 warn<{cfg['warn_below']}"
    )
    return variance, note


def _insert(
    conn: sqlite3.Connection,
    version_id: int,
    rule_id: str,
    rule_version: str,
    params: dict | None,
    measured: dict | None,
    status: str,
    reason: str,
    recheck_of: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO quality_checks(version_id, rule_id, rule_version, params_json,"
        " measured_json, status, reason, checked_at, recheck_of_check_id)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (
            version_id,
            rule_id,
            rule_version,
            json.dumps(params, ensure_ascii=False) if params is not None else None,
            json.dumps(measured, ensure_ascii=False) if measured is not None else None,
            status,
            reason,
            _now(),
            recheck_of,
        ),
    )


def run_checks(
    conn: sqlite3.Connection,
    version_id: int,
    store_root: Path | None = None,
    recheck_of_map: dict[str, int] | None = None,
) -> list[sqlite3.Row]:
    """对某版本执行全部适用规则，写入 quality_checks。返回该版本最新一轮检查。"""
    store_root = store_root or config.storage_root()
    rules = _load_quality_rules()
    qver = rules["rules_version"]
    recheck_of_map = recheck_of_map or {}

    ver = conn.execute(
        "SELECT * FROM asset_versions WHERE version_id = ?", (version_id,)
    ).fetchone()
    if ver is None:
        raise ValueError(f"版本不存在：{version_id}")

    obj = conn.execute(
        "SELECT * FROM objects WHERE object_id = ?", (ver["object_id"],)
    ).fetchone()
    obj_path = store_root / obj["relpath"]
    ctype = ver["content_type"]

    def add(rule_id: str, params, measured, status, reason):
        _insert(
            conn, version_id, rule_id, qver, params, measured, status, reason,
            recheck_of_map.get(rule_id),
        )

    # --- 通用规则 ---
    if obj["size_bytes"] == 0:
        add("file_not_empty", {"limit": 0}, {"size": 0}, "block", "空文件，无内容")
    else:
        add("file_not_empty", {"limit": 0}, {"size": obj["size_bytes"]}, "pass", "文件非空")

    max_bytes = rules["limits"]["max_file_bytes"]
    if obj["size_bytes"] > max_bytes:
        add("size_limit", {"max_bytes": max_bytes}, {"size": obj["size_bytes"]},
            "warn", f"文件超过建议上限 {max_bytes} 字节")
    else:
        add("size_limit", {"max_bytes": max_bytes}, {"size": obj["size_bytes"]}, "pass", "大小在上限内")

    if ver["source"]:
        add("required_source", {}, {"source": ver["source"]}, "pass", "来源已登记")
    else:
        add("required_source", {}, {"source": None}, "block", "缺少必填来源")

    if ver["declared_scene_id"]:
        add("required_scene", {}, {"scene_id": ver["declared_scene_id"]}, "pass", "场景已声明")
    else:
        add("required_scene", {}, {"scene_id": None}, "warn", "未声明场景，无法参与场景关联统计")

    dup = conn.execute(
        "SELECT v.version_id FROM asset_versions v WHERE v.object_id = ? AND v.version_id != ?"
        " ORDER BY v.version_id LIMIT 1",
        (ver["object_id"], version_id),
    ).fetchone()
    if dup is not None:
        add("exact_duplicate", {}, {"same_content_version": dup["version_id"]},
            "warn", f"内容与该资产版本 {dup['version_id']} 完全相同（未覆盖）")
    else:
        add("exact_duplicate", {}, None, "pass", "无重复内容")

    # --- 图像规则 ---
    if ctype == "image":
        try:
            from PIL import Image

            with Image.open(obj_path) as im:
                w, h = im.size
                im.load()  # 真正解码像素；截断/损坏内容在此抛错
            add("image_decode", {}, {"width": w, "height": h}, "pass", "图像可解码")
            max_px = rules["limits"]["max_image_pixels"]
            if w * h > max_px:
                add("pixel_limit", {"max_pixels": max_px}, {"pixels": w * h},
                    "warn", "像素总量超过建议上限")
            else:
                add("pixel_limit", {"max_pixels": max_px}, {"pixels": w * h}, "pass", "像素量在上限内")
            try:
                variance, note = _sharpness(obj_path, rules)
                cfg = rules["sharpness"]
                if variance < cfg["warn_below"]:
                    sev = "远低于阈值，疑似严重模糊" if variance < cfg["block_below"] else "低于阈值"
                    add("sharpness", {"preview_max_side": cfg["preview_max_side"]},
                        {"laplacian_variance": round(variance, 3)},
                        "warn", f"{sev}（提示，非阻断）。{note}")
                else:
                    add("sharpness", {"preview_max_side": cfg["preview_max_side"]},
                        {"laplacian_variance": round(variance, 3)}, "pass", note)
            except Exception as exc:  # 清晰度计算失败不影响其他结论
                add("sharpness", None, None, "na", f"清晰度未计算：{exc}")
        except Exception as exc:
            add("image_decode", {}, None, "block", f"图像解码失败：{exc}")
        add("crs_applicability", {}, {"crs": None}, "na", "普通图像无地理参考，CRS 不适用")

    # --- 属性(CSV)规则 ---
    elif ctype == "attribute":
        rows = conn.execute(
            "SELECT * FROM attribute_records WHERE version_id = ?", (version_id,)
        ).fetchall()
        parse_issue = _read_parse_issue(conn, version_id)
        if parse_issue:
            add("csv_required_columns", {}, None, "block", parse_issue)
        else:
            add("csv_required_columns", {}, None, "pass", "CSV 必填列齐全")
        if not rows:
            add("csv_has_rows", {}, {"rows": 0}, "warn", "CSV 无数据行")
        else:
            add("csv_has_rows", {}, {"rows": len(rows)}, "pass", f"{len(rows)} 行数据")
            err = [r["record_id"] for r in rows if r["issues"]]
            if err:
                add("csv_area_units", {}, {"error_rows": err},
                    "block", f"{len(err)} 行面积/单位或类别错误：{err}")
            else:
                add("csv_area_units", {}, {"error_rows": []}, "pass", "面积与单位均可转换")

        value_issue = _read_meta(conn, f"csv_value_issue:{version_id}")
        if value_issue:
            add("csv_required_values", {}, None, "block", f"必填值缺失：{value_issue}")
        else:
            add("csv_required_values", {}, None, "pass", "场景/地块/作物必填值均已填写")

        declared = ver["declared_scene_id"]
        sids = {r["scene_id"] for r in rows if r["scene_id"]}
        if sids and declared and sids != {declared}:
            add("csv_scene_consistency", {}, {"declared": declared, "rows": sorted(sids)},
                "warn", f"CSV 行 scene_id {sorted(sids)} 与声明 {declared} 不一致，不参与合格覆盖统计")
        elif sids and not declared:
            add("csv_scene_consistency", {}, {"rows": sorted(sids)},
                "warn", "CSV 含 scene_id 但版本未声明场景")
        else:
            add("csv_scene_consistency", {}, None, "pass", "CSV 场景标识一致")

    # --- 文本(JSON/TXT)规则 ---
    elif ctype == "text":
        notes = conn.execute(
            "SELECT * FROM text_records WHERE version_id = ?", (version_id,)
        ).fetchall()
        if not notes:
            add("text_parsed", {}, None, "block", "文本记录解析失败")
        else:
            parse_fail = None
            payload_scene = None
            for n in notes:
                if n["fmt"] == "json":
                    if n["payload_json"] is None:
                        parse_fail = n["issues"] or "JSON 解析失败"
                    else:
                        try:
                            obj = json.loads(n["payload_json"])
                            if isinstance(obj, dict):
                                ps = obj.get("scene_id") or obj.get("scene")
                                if ps:
                                    payload_scene = str(ps)
                        except Exception as exc:
                            parse_fail = f"JSON 解析失败：{exc}"
                elif n["payload_json"] is None and n["issues"]:
                    parse_fail = n["issues"]  # 例如编码错误
            if parse_fail:
                add("text_parsed", {}, None, "block", parse_fail)
            else:
                add("text_parsed", {}, {"notes": len(notes)}, "pass", "文本已解析")
            # 场景一致性按当前版本元数据重新判断（复检可因补正而改变结论）
            declared = ver["declared_scene_id"]
            if payload_scene and declared and payload_scene != str(declared):
                add("scene_consistency", {}, {"declared": declared, "content": payload_scene},
                    "warn", f"JSON scene_id 与声明失配：声明={declared} 内容={payload_scene}")
            elif payload_scene and not declared:
                add("scene_consistency", {}, {"content": payload_scene},
                    "warn", "JSON 含 scene_id 但版本未声明场景")
            else:
                add("scene_consistency", {}, None, "pass", "场景标识一致")

    conn.commit()
    return latest_checks(conn, version_id)


def _read_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _read_parse_issue(conn: sqlite3.Connection, version_id: int) -> str | None:
    return _read_meta(conn, f"csv_parse_issue:{version_id}")


def latest_checks(conn: sqlite3.Connection, version_id: int) -> list[sqlite3.Row]:
    """返回每个规则最近一次的检查结果。"""
    return conn.execute(
        """
        SELECT c.* FROM quality_checks c
        JOIN (
            SELECT rule_id, MAX(check_id) AS max_id
            FROM quality_checks WHERE version_id = ? GROUP BY rule_id
        ) m ON c.check_id = m.max_id
        ORDER BY c.rule_id
        """,
        (version_id,),
    ).fetchall()


def overall_status(checks: list[sqlite3.Row]) -> str:
    """版本总体状态：有阻断→block，有警告→warn，全通过→pass，全不适用→na。"""
    statuses = {c["status"] for c in checks}
    if "block" in statuses:
        return "block"
    if "warn" in statuses:
        return "warn"
    if statuses == {"na"} or not statuses:
        return "na"
    return "pass"
