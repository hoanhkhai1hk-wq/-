"""报告与清单导出：CSV / JSON，可重新读取。

报告包含资产/版本、质检、处置、场景关联与统计口径。
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from . import scenes


def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def export_assets_csv(conn: sqlite3.Connection, path: Path) -> Path:
    versions = scenes.list_versions(conn)
    header = [
        "version_id", "asset_id", "version_no", "content_type", "original_filename",
        "source", "declared_scene_id", "status", "overall_status", "object_id",
        "size_bytes", "width", "height", "bands", "bitdepth",
    ]
    rows = [[v.get(h) for h in header] for v in versions]
    return _write_csv(Path(path), header, rows)


def export_checks_csv(conn: sqlite3.Connection, path: Path) -> Path:
    header = [
        "version_id", "rule_id", "rule_version", "status", "reason",
        "measured_json", "checked_at", "recheck_of_check_id",
    ]
    rows = []
    for v in scenes.list_versions(conn):
        for c in scenes.list_checks(conn, v["version_id"]):
            rows.append([c.get(h) for h in header])
    return _write_csv(Path(path), header, rows)


def build_report(conn: sqlite3.Connection) -> dict:
    from . import db as _db

    return {
        "schema_version": _db.get_meta(conn, "schema_version"),
        "summary": scenes.summary(conn),
        "scenes": [
            {
                "scene_id": s["scene_id"],
                "missing_roles": s["missing_roles"],
                "image_versions": [m["version_id"] for m in s["roles"]["image"]],
                "attribute_versions": [m["version_id"] for m in s["roles"]["attribute"]],
                "text_versions": [m["version_id"] for m in s["roles"]["text"]],
            }
            for s in scenes.list_scenes(conn)
        ],
        "scene_issues": scenes.scene_issues(conn),
        "versions": [
            {
                "version_id": v["version_id"],
                "original_filename": v["original_filename"],
                "content_type": v["content_type"],
                "status": v["status"],
                "overall_status": v["overall_status"],
                "source": v["source"],
                "scene_id": v["declared_scene_id"],
            }
            for v in scenes.list_versions(conn)
        ],
    }


def export_report_json(conn: sqlite3.Connection, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_report(conn), fh, ensure_ascii=False, indent=2)
    return path


def export_report_markdown(conn: sqlite3.Connection, path: Path) -> Path:
    rep = build_report(conn)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# AgroLake P0 治理报告", ""]
    s = rep["summary"]
    lines += [
        f"- 资产版本总数：{s['total_versions']}",
        f"- 导入事件总数：{s['total_import_events']}",
        f"- 当前有效版本：{s['current_versions']}",
        f"- 质量通过率：{s['quality_pass_rate']['pass_rate']}"
        f"（分母：{s['quality_pass_rate']['denominator']}；排除：{s['quality_pass_rate']['excluded']}）",
        f"- 关联覆盖率：{s['association_coverage']['coverage']}"
        f"（分母：{s['association_coverage']['denominator']}）",
        "",
        "## 版本清单",
        "",
        "| version | 文件 | 类型 | 场景 | 状态 | 质检 |",
        "|---|---|---|---|---|---|",
    ]
    for v in rep["versions"]:
        lines.append(
            f"| {v['version_id']} | {v['original_filename']} | {v['content_type']} |"
            f" {v['scene_id'] or ''} | {v['status']} | {v['overall_status']} |"
        )
    if rep["scene_issues"]:
        lines += ["", "## 待处理场景问题", ""]
        for it in rep["scene_issues"]:
            lines.append(f"- {it['scene_id']}：{it['kind']} {it['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
