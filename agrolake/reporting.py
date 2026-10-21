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


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _check_distribution(conn: sqlite3.Connection) -> dict:
    """按规则条目统计质检分布（每个版本每条规则取最近一次结果）。"""
    counts = {"pass": 0, "warn": 0, "block": 0, "na": 0}
    for v in scenes.list_versions(conn):
        for c in scenes.list_checks(conn, v["version_id"]):
            if c["status"] in counts:
                counts[c["status"]] += 1
    total = sum(counts.values())
    applicable = total - counts["na"]
    return {
        **counts,
        "total": total,
        "applicable": applicable,
        "pass_rate": (counts["pass"] / applicable) if applicable else None,
    }


def _version_distribution(conn: sqlite3.Connection) -> dict:
    """当前有效版本的总体质检结论分布。"""
    counts = {"pass": 0, "warn": 0, "block": 0, "na": 0}
    for v in scenes.current_versions(conn):
        if v["overall_status"] in counts:
            counts[v["overall_status"]] += 1
    counts["flagged"] = counts["warn"] + counts["block"]
    counts["counted"] = sum(
        counts[k] for k in ("pass", "warn", "block")
    )
    return counts


def build_report(conn: sqlite3.Connection) -> dict:
    from . import db as _db

    return {
        "schema_version": _db.get_meta(conn, "schema_version"),
        "summary": scenes.summary(conn),
        "check_distribution": _check_distribution(conn),
        "version_distribution": _version_distribution(conn),
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
    lines = ["# AgroLake 治理报告", ""]
    s = rep["summary"]
    q = s["quality_pass_rate"]
    cd = rep["check_distribution"]
    vd = rep["version_distribution"]
    lines += [
        f"- 资产版本总数：{s['total_versions']}",
        f"- 导入事件总数：{s['total_import_events']}",
        f"- 当前有效版本：{s['current_versions']}",
        f"- 版本质量通过率：{_pct(q['pass_rate'])}"
        f"（{q['passed_versions']}/{q['counted_versions']} 个当前有效版本通过；"
        f"分母：{q['denominator']}；排除：{q['excluded']}）",
        f"- 当前有效版本质检分布：通过 {vd['pass']} · 警告 {vd['warn']} · "
        f"拦下 {vd['block']}（共 {vd['flagged']} 个版本被标记待处置）",
        f"- 规则条目通过率：{_pct(cd['pass_rate'])}"
        f"（{cd['pass']}/{cd['applicable']} 条适用规则通过；"
        f"全部 {cd['total']} 条中警告 {cd['warn']} · 拦下 {cd['block']} · 不适用 {cd['na']}）",
        f"- 关联覆盖率：{_pct(s['association_coverage']['coverage'])}"
        f"（分母：{s['association_coverage']['denominator']}）",
        "",
        "> 口径说明：版本通过率的分母是当前有效版本数，样本规模较小时波动很大；"
        "当样本集中包含为规则验证故意构造的问题样本（空文件、损坏文件、模糊影像等）时，"
        "该通过率反映的是样本构成，不直接代表真实生产数据的质量水平。"
        "评价治理效果应结合规则条目通过率、问题版本的标记与处置情况综合判断。",
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
