"""数据问答的事实层（T15）。

只读地把治理库里的真实状态整理成三样东西：

- ``text``       给模型看的事实摘要（结构化文本，数字全部由程序算出）
- ``citations``  引用来源（哪个表、哪些版本号、哪一版规则）
- ``snapshot``   结构化快照（存库留档，也用于页面展示）

三条口径：

1. 数字只来自 SQL 聚合或既有统计函数（``scenes.summary`` 等），
   模型只负责解释，不负责计算；快照里已算好的值直接给模型引用。
2. 资料缺失写进 ``gaps``，由回答明确说明缺口，不用常识补齐。
3. 质检 reason 原文带有本机绝对路径，这里统一去掉，只保留业务结论。
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import quality, scenes, semantics

SCOPES: dict[str, str] = {
    "overview": "治理概览",
    "batch_review": "当前批次待复核资料",
    "scene": "场景质量与识别结果",
    "governance_report": "治理报告",
}

_ABS_PATH = re.compile(r"[A-Za-z]:\\[^\s\"'）)，,；;]*")

OVERALL_ZH = {"pass": "通过", "warn": "警告", "block": "阻断", "na": "不适用"}
STATUS_ZH = {"active": "有效", "isolated": "已隔离", "duplicate": "已标记重复"}


@dataclass
class FactBundle:
    """一次问答所用的事实快照。"""

    scope: str
    title: str
    text: str = ""
    citations: list[dict[str, str]] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "title": self.title,
            "text": self.text,
            "citations": self.citations,
            "snapshot": self.snapshot,
            "gaps": self.gaps,
        }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _clean(text: str | None) -> str:
    """去掉本机绝对路径，保留可读的业务结论。"""
    if not text:
        return ""
    return re.sub(r"\s{2,}", " ", _ABS_PATH.sub("<本机路径已省略>", str(text))).strip()


def _pct(value: float | None) -> str:
    return "无法计算（分母为 0）" if value is None else f"{value:.0%}"


# ── 待复核清单 ────────────────────────────────────────────────────────────
def pending_review(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """按当前库状态列出需要人处理的资料，每条都带可追溯的原因。"""
    items: list[dict[str, Any]] = []
    for v in scenes.list_versions(conn, include_inactive=True):
        vid = v["version_id"]
        checks = quality.latest_checks(conn, vid)
        overall = quality.overall_status(checks)
        reasons: list[str] = []
        rules: list[str] = []

        if v["status"] == "isolated":
            reasons.append("版本处于已隔离状态，需确认解除隔离或替换文件")
        if v["status"] == "duplicate":
            reasons.append("版本已标记为重复，需确认保留哪一份")
        for c in checks:
            if c["status"] in ("block", "warn"):
                label = "阻断" if c["status"] == "block" else "警告"
                reasons.append(f"质检{label} · {c['rule_id']}：{_clean(c['reason'])}")
                rules.append(c["rule_id"])

        bad_attr = conn.execute(
            "SELECT COUNT(*) AS c FROM attribute_records"
            " WHERE version_id = ? AND issues IS NOT NULL AND issues <> ''",
            (vid,),
        ).fetchone()["c"]
        if bad_attr:
            reasons.append(f"属性记录有 {bad_attr} 行带问题标记")
        bad_text = conn.execute(
            "SELECT COUNT(*) AS c FROM text_records"
            " WHERE version_id = ? AND issues IS NOT NULL AND issues <> ''",
            (vid,),
        ).fetchone()["c"]
        if bad_text:
            reasons.append(f"文本记录有 {bad_text} 条带问题标记")

        if not reasons:
            continue
        items.append({
            "version_id": vid,
            "filename": v["original_filename"],
            "content_type": v["content_type"],
            "scene_id": v["declared_scene_id"],
            "version_status": v["status"],
            "version_status_zh": STATUS_ZH.get(v["status"], v["status"]),
            "overall_status": overall,
            "overall_status_zh": OVERALL_ZH.get(overall, overall),
            "rule_ids": sorted(set(rules)),
            "reasons": reasons,
        })
    return items


# ── 识别结果 ──────────────────────────────────────────────────────────────
def _recognition_facts(conn: sqlite3.Connection, scene_id: str | None) -> dict[str, Any]:
    """读取识别服务落库的结果；没有记录时明确返回"暂无"。"""
    try:
        if scene_id:
            jobs = conn.execute(
                "SELECT job_id, status, error, created_at FROM rec_jobs"
                " WHERE scene_id = ? ORDER BY job_id DESC",
                (scene_id,),
            ).fetchall()
        else:
            jobs = conn.execute(
                "SELECT job_id, status, error, created_at FROM rec_jobs"
                " ORDER BY job_id DESC"
            ).fetchall()
    except sqlite3.Error:
        return {"available": False, "jobs": [], "runs": [], "classes": [],
                "note": "识别结果表不存在，尚无识别记录"}

    if not jobs:
        return {"available": False, "jobs": [], "runs": [], "classes": [],
                "note": "识别结果库中暂无记录"}

    job_ids = [j["job_id"] for j in jobs]
    marks = ",".join("?" * len(job_ids))
    runs = conn.execute(
        f"SELECT run_id, job_id, status, field_count, elapsed_seconds, finished_at"
        f" FROM rec_runs WHERE job_id IN ({marks}) ORDER BY run_id DESC",
        job_ids,
    ).fetchall()
    classes = conn.execute(
        f"SELECT job_id, class_value, meaning, pixels FROM rec_class_counts"
        f" WHERE job_id IN ({marks}) ORDER BY job_id DESC, pixels DESC",
        job_ids,
    ).fetchall()
    return {
        "available": True,
        "jobs": [dict(r) for r in jobs],
        "runs": [dict(r) for r in runs],
        "classes": [dict(r) for r in classes],
        "note": "",
    }


# ── 各口径的事实组装 ──────────────────────────────────────────────────────
def _overview_block(conn: sqlite3.Connection) -> tuple[list[str], dict, list[dict], list[str]]:
    s = scenes.summary(conn)
    qr = s["quality_pass_rate"]
    ac = s["association_coverage"]
    lines = [
        f"- 资产版本总数：{s['total_versions']}",
        f"- 导入事件总数：{s['total_import_events']}",
        f"- 当前有效版本：{s['current_versions']}",
        f"- 质量通过率：{_pct(qr['pass_rate'])}（通过 {qr['passed_versions']} / 计入 {qr['counted_versions']}）",
        f"  分母口径：{qr['denominator']}",
        f"  排除项：{qr['excluded']}",
        f"- 关联覆盖率：{_pct(ac['coverage'])}"
        f"（{ac['complete_scenes']}/{ac['total_scenes']} 个场景三类资料齐全）",
        f"  分母口径：{ac['denominator']}",
    ]
    snapshot = {
        "total_versions": s["total_versions"],
        "total_import_events": s["total_import_events"],
        "current_versions": s["current_versions"],
        "quality_pass_rate": qr,
        "association_coverage": ac,
    }
    citations = [{
        "label": "治理概览统计",
        "source": "asset_versions / import_events / quality_checks / scene_members",
        "detail": (
            f"统计由程序生成；规则版本 {quality.rules_version()}，"
            f"语义字典 {semantics.semantic_version()}"
        ),
    }]
    gaps = []
    if s["total_versions"] == 0:
        gaps.append("数据库里还没有任何资产版本，所有比例均无法计算")
    return lines, snapshot, citations, gaps


def _pending_block(conn: sqlite3.Connection) -> tuple[list[str], dict, list[dict], list[str]]:
    items = pending_review(conn)
    lines = [f"- 待复核资料条数：{len(items)}"]
    if not items:
        lines.append("- 结论：当前没有命中阻断、警告或处置待确认的版本")
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. 版本 #{it['version_id']} · {it['filename']} "
            f"（{it['content_type']}，场景 {it['scene_id'] or '未填'}，"
            f"状态 {it['version_status_zh']}，总体结论 {it['overall_status_zh']}）"
        )
        for r in it["reasons"]:
            lines.append(f"   - {r}")
    citations = [{
        "label": f"版本 #{it['version_id']} · {it['filename']}",
        "source": f"asset_versions.version_id={it['version_id']}；quality_checks 最新一轮",
        "detail": "；".join(it["reasons"])[:300],
    } for it in items]
    if not citations:
        citations = [{
            "label": "待复核清单",
            "source": "asset_versions + quality_checks + attribute_records + text_records",
            "detail": f"按最新一轮质检与记录问题标记汇总，规则版本 {quality.rules_version()}",
        }]
    snapshot = {"pending_count": len(items), "items": items}
    # "有几条待复核"是结论不是缺口；只有库为空才算资料缺口。
    total = conn.execute("SELECT COUNT(*) AS c FROM asset_versions").fetchone()["c"]
    gaps = ["数据库里还没有任何资产版本，待复核清单为空是因为没有资料"] if total == 0 else []
    return lines, snapshot, citations, gaps


def _scene_block(
    conn: sqlite3.Connection, scene_id: str | None
) -> tuple[list[str], dict, list[dict], list[str], str]:
    all_scenes = scenes.list_scenes(conn)
    if not all_scenes:
        return (
            ["- 场景：数据库里还没有任何场景"],
            {"scene_id": None, "scenes": []},
            [],
            ["数据库里没有任何场景资料，无法回答场景相关问题"],
            "场景质量与识别结果",
        )
    ids = [s["scene_id"] for s in all_scenes]
    pick = scene_id if scene_id in ids else ids[0]
    sc = next(s for s in all_scenes if s["scene_id"] == pick)
    by_id = {v["version_id"]: v for v in scenes.list_versions(conn)}

    lines = [f"- 场景编号：{pick}"]
    roles = sc["roles"]
    role_zh = {"image": "影像", "attribute": "属性", "text": "复核文本"}
    for role, items_r in roles.items():
        if items_r:
            names = []
            for m in items_r:
                v = by_id.get(m["version_id"])
                names.append(
                    f"#{m['version_id']} {v['original_filename'] if v else '（版本已不存在）'}"
                )
            lines.append(f"- 当前{role_zh.get(role, role)}：{len(items_r)} 份（{'、'.join(names)}）")
        else:
            lines.append(f"- 当前{role_zh.get(role, role)}：缺失")

    # 每个成员版本的质检结论
    for role, items_r in roles.items():
        for m in items_r:
            v = by_id.get(m["version_id"])
            if not v:
                continue
            checks = quality.latest_checks(conn, v["version_id"])
            overall = quality.overall_status(checks)
            lines.append(
                f"  · #{v['version_id']} {v['original_filename']}："
                f"质检 {OVERALL_ZH.get(overall, overall)}"
            )
            for c in checks:
                if c["status"] in ("block", "warn"):
                    label = "阻断" if c["status"] == "block" else "警告"
                    lines.append(f"      - {label} · {c['rule_id']}：{_clean(c['reason'])}")

    # 属性摘要只统计"当前有效版本"里的属性文件，与页面口径一致；
    # 已隔离/已标重复/历史版本不计入。
    attr_vids = [m["version_id"] for m in roles.get("attribute", [])]
    attr_rows: list = []
    if attr_vids:
        marks = ",".join("?" * len(attr_vids))
        attr_rows = conn.execute(
            f"SELECT field_id, crop, category_std, area_m2 FROM attribute_records"
            f" WHERE version_id IN ({marks}) ORDER BY row_index",
            attr_vids,
        ).fetchall()
    if attr_rows:
        crops = sorted({r["crop"] for r in attr_rows if r["crop"]})
        cats = sorted({r["category_std"] for r in attr_rows if r["category_std"]})
        areas = [r["area_m2"] for r in attr_rows if r["area_m2"] is not None]
        lines.append(f"- 属性记录 {len(attr_rows)} 行；作物：{'、'.join(crops) or '未填'}")
        lines.append(f"- 地物类别（标准值）：{'、'.join(cats) or '未填'}")
        if areas:
            lines.append(
                f"- 已换算面积合计：{sum(areas):.2f} 平方米"
                "（仅统计单位已定义的记录，不代表田块实测面积）"
            )
        else:
            lines.append("- 已换算面积合计：无（所有面积单位均未定义，未换算）")

    rec = _recognition_facts(conn, pick)
    gaps: list[str] = []
    if rec["available"]:
        lines.append(f"- 识别任务 {len(rec['jobs'])} 个")
        for r in rec["runs"][:5]:
            lines.append(
                f"  · 运行 #{r['run_id']}：{r['status']}，田块数 {r['field_count']}，"
                f"耗时 {r['elapsed_seconds']}s"
            )
        for c in rec["classes"][:8]:
            lines.append(f"  · 类别 {c['class_value']}（{c['meaning']}）：{c['pixels']} 像素")
    else:
        lines.append(f"- 识别结果：暂无（{rec['note']}）")
        gaps.append("该场景尚无识别结果落库，无法解释识别结论")

    if sc["missing_roles"]:
        gaps.append(
            "该场景缺少资料：" + "、".join(role_zh.get(r, r) for r in sc["missing_roles"])
        )
    if sc["issues"]:
        for note in sc["issues"]:
            lines.append(f"- 场景关联问题：{_clean(note)}")

    citations = [{
        "label": f"场景 {pick} 的成员版本",
        "source": "asset_versions / scene_members / quality_checks",
        "detail": "按当前有效版本判定归属；隔离、重复与失配版本不计入",
    }]
    if attr_rows:
        citations.append({
            "label": f"场景 {pick} 的属性记录",
            "source": "attribute_records",
            "detail": f"{len(attr_rows)} 行；面积只统计单位已定义的记录",
        })
    if rec["available"]:
        citations.append({
            "label": f"场景 {pick} 的识别结果",
            "source": "rec_jobs / rec_runs / rec_class_counts",
            "detail": f"{len(rec['jobs'])} 个识别任务",
        })
    snapshot = {
        "scene_id": pick,
        "roles": {k: [m["version_id"] for m in v] for k, v in roles.items()},
        "missing_roles": sc["missing_roles"],
        "attribute_rows": len(attr_rows),
        "recognition": rec,
    }
    return lines, snapshot, citations, gaps, f"场景 {pick} 质量与识别结果"


def build(
    conn: sqlite3.Connection, scope: str, *, scene_id: str | None = None
) -> FactBundle:
    """组装事实摘要。``scope`` 取值见 :data:`SCOPES`。"""
    scope = scope if scope in SCOPES else "overview"
    header = [
        f"【口径】统计时点 {_now()}；规则版本 {quality.rules_version()}；"
        f"语义字典 {semantics.semantic_version()}",
        "【说明】以下数字均由程序从 SQLite 直接统计，不是模型推算的。",
    ]
    citations: list[dict[str, str]] = []
    gaps: list[str] = []
    snapshot: dict[str, Any] = {"scope": scope, "generated_at": _now()}
    body: list[str] = []

    if scope == "overview":
        lines, snap, cits, g = _overview_block(conn)
        body = lines; snapshot.update(snap); citations += cits; gaps += g
        title = SCOPES["overview"]
    elif scope == "batch_review":
        lines, snap, cits, g = _pending_block(conn)
        body = lines; snapshot.update(snap); citations += cits; gaps += g
        title = SCOPES["batch_review"]
    elif scope == "scene":
        lines, snap, cits, g, title = _scene_block(conn, scene_id)
        body = lines; snapshot.update(snap); citations += cits; gaps += g
    else:  # governance_report
        o_lines, o_snap, o_cits, o_g = _overview_block(conn)
        p_lines, p_snap, p_cits, p_g = _pending_block(conn)
        s_lines, s_snap, s_cits, s_g, _ = _scene_block(conn, scene_id)
        body = (
            ["【治理概览】"] + o_lines
            + ["", "【待复核资料】"] + p_lines
            + ["", "【场景明细】"] + s_lines
        )
        snapshot.update({"overview": o_snap, "pending": p_snap, "scene": s_snap})
        citations += o_cits[:1] + p_cits[:3] + s_cits[:2]
        gaps += o_g + p_g + s_g
        title = SCOPES["governance_report"]

    if gaps:
        body = body + ["", "【缺口】"] + [f"- {g}" for g in gaps]

    return FactBundle(
        scope=scope,
        title=title,
        text="\n".join(header + [""] + body),
        citations=citations,
        snapshot=snapshot,
        gaps=gaps,
    )
