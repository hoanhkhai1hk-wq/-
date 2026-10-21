"""RGB候选作业事实层（INT-20261021-001）。

只读地把 RGB 候选作业库（``data/rgb_candidate/runs.sqlite3``）的真实状态
整理成 DeepSeek 问答可用的 :class:`FactBundle`。

口径与边界：

- RGB 侧一律只读访问（``RgbCandidateJobs(readonly=True)``）＋只读 SQLite
  连接；问答绝不触发登记或推理，缺库/未知/失败/损坏都写进 ``gaps``，
  不回退其他任务、不引用旧成功计数。
- SQL 里的 ``class_counts`` / ``regions`` 必须与产物核验结果一致；库行被
  篡改时按"不可信"处理并明确说明，不把篡改值当事实发给模型。
- ``text`` / ``snapshot`` / ``citations`` 不含本机绝对路径，也不含任何
  图片字节。
- 固定写入缺口：无实地确认、像素不是平方米、无独立来源新图评估。
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from .rgb_jobs import ARTIFACT_NAMES, JOB_ID_RE, RgbArtifactError, RgbCandidateJobs

_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|file://|\\\\)[^\s\"'）)，,；;]*", re.I)

_FIXED_GAPS = [
    "无实地确认：候选区域未经实地核实，不是已确认的真实田块。",
    "像素面积不是平方米；没有比例尺或地理参考，不能换算实测面积。",
    "模型为候选版本，未用独立来源新图评估，本事实不构成识别精度结论。",
    "本摘要不包含实测作物、土壤或传感器属性，技术场景编号不代表真实农田身份。",
]

_STATUS_ZH = {"running": "运行中/结果未确认", "succeeded": "成功", "failed": "失败"}


def _clean(text) -> str:
    """去掉本机绝对路径，保留可读结论。"""
    if not text:
        return ""
    return re.sub(r"\s{2,}", " ", _ABS_PATH.sub("<本机路径已省略>", str(text))).strip()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _readonly_sql(db_path: Path) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def list_jobs(*, data_dir=None) -> dict:
    """只读列出 RGB 作业（供问答页选择框使用）。

    返回 ``{'available': bool, 'note': str, 'jobs': [...]}``；
    库不存在/不可读时 ``available=False``，绝不创建任何文件。
    """
    backend = RgbCandidateJobs(data_dir=data_dir, readonly=True)
    try:
        jobs = backend.list_jobs(30)
    except RgbArtifactError:
        return {"available": False,
                "note": "RGB 作业库不存在或不可读（只读访问不会创建新库）",
                "jobs": []}
    except (sqlite3.Error, ValueError, OSError, TypeError) as exc:
        return {"available": False,
                "note": f"RGB 作业库读取失败：{_clean(exc)}", "jobs": []}
    return {
        "available": True,
        "note": "",
        "jobs": [{
            "id": j["id"],
            "status": j["status"],
            "status_zh": _STATUS_ZH.get(j["status"], j["status"]),
            "source_name": Path(str(j["source_name"]).replace("\\", "/")).name,
            "created_at": j["created_at"],
            "finished_at": j["finished_at"],
            "error": _clean(j["error"]),
        } for j in jobs],
    }


def _sql_stats(db_path: Path, job_id: str) -> tuple[dict, list[dict]] | None:
    """只读读取 RGB 库的 SQL 计数与区域行；读不到返回 None。"""
    try:
        conn = _readonly_sql(db_path)
    except sqlite3.Error:
        return None
    try:
        counts = {
            str(r["class_id"]): r["pixels"]
            for r in conn.execute(
                "SELECT class_id, pixels FROM class_counts WHERE job_id = ?",
                (job_id,))
        }
        regions = [dict(r) for r in conn.execute(
            "SELECT field_id, area_pixels, bbox_x0, bbox_y0, bbox_x1, bbox_y1,"
            " centroid_x, centroid_y, edge_touch, small_patch FROM regions"
            " WHERE job_id = ? ORDER BY field_id", (job_id,))]
        return counts, regions
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _registration_info(conn: sqlite3.Connection, job_id: str, backend) -> dict | None:
    """只读查询 P0 登记链接与当前质检/版本状态；表不存在视为未登记。"""
    try:
        row = conn.execute(
            "SELECT * FROM rgb_governance_links WHERE job_id = ?",
            (job_id,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    from . import quality as p0_quality
    from . import rgb_governance

    integrity = rgb_governance.link_status(conn, job_id, backend=backend)
    if not integrity.get("intact"):
        return {"intact": False, "problem": _clean(integrity.get("problem"))}

    info: dict = {
        "intact": integrity.get("intact", False),
        "problem": _clean(integrity.get("problem")),
        "scene_id": row["scene_id"],
        "image_asset_id": row["image_asset_id"],
        "image_version_id": row["image_version_id"],
        "stats_asset_id": row["stats_asset_id"],
        "stats_version_id": row["stats_version_id"],
        "png_sha256": row["png_sha256"],
        "json_sha256": row["json_sha256"],
        "created_at": row["created_at"],
    }
    zh = {"pass": "通过", "warn": "警告", "block": "阻断", "na": "不适用"}
    for key, vid in (("image", row["image_version_id"]),
                     ("stats", row["stats_version_id"])):
        try:
            v = conn.execute(
                "SELECT status FROM asset_versions WHERE version_id = ?",
                (vid,)).fetchone()
            checks = p0_quality.latest_checks(conn, vid)
            info[f"{key}_version_status"] = v["status"] if v else "版本缺失"
            info[f"{key}_quality"] = zh.get(
                p0_quality.overall_status(checks), "未知")
        except sqlite3.Error:
            info[f"{key}_version_status"] = "查询失败"
            info[f"{key}_quality"] = "未知"
    return info


def _bundle(snapshot, lines, citations, gaps):
    from .qa_facts import FactBundle, SCOPES
    text = "\n".join(
        ["【口径】统计时点 " + snapshot["generated_at"] + "（RGB 任务事实，全部只读）",
         "【说明】只有来源、产物与SQL明细全部核对一致，才提供识别统计。"]
        + ([""] + lines if lines else [])
        + ["", "【缺口】"] + [f"- {g}" for g in gaps])
    return FactBundle(scope="rgb_job", title=SCOPES["rgb_job"], text=text,
                      citations=citations, snapshot=snapshot, gaps=gaps)


def build(conn: sqlite3.Connection, job_id, *, data_dir=None):
    """组装单个 RGB 任务的事实摘要（scope=rgb_job）。

    ``conn`` 是 P0 只读连接，仅用于查询登记链接；RGB 侧全部只读。
    失败/未知/损坏时如实写缺口，不引用任何成功计数。
    """
    from . import qa_facts  # 延迟导入，避免与 qa_facts 的委托互相循环

    now = _now()
    valid_id = isinstance(job_id, str) and bool(JOB_ID_RE.match(job_id))
    snapshot: dict = {"scope": "rgb_job", "generated_at": now,
                      "job_id": job_id if valid_id else None}
    lines: list[str] = []
    citations: list[dict[str, str]] = []
    gaps: list[str] = list(_FIXED_GAPS)

    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        snapshot["status"] = "invalid"
        bundle = qa_facts.FactBundle(
            scope="rgb_job", title=qa_facts.SCOPES["rgb_job"],
            text="\n".join([
                "【口径】统计时点 " + now + "（RGB 任务事实，全部只读）",
                "【说明】以下数字均由程序从 SQLite 直接统计，不是模型推算的。"
                "任务编号格式不合法（应为 32 位小写十六进制），"
                "没有可解释的 RGB 任务事实。",
                "", "【缺口】",
            ] + [f"- {g}" for g in gaps]),
            citations=citations, snapshot=snapshot, gaps=gaps)
        return bundle

    backend = RgbCandidateJobs(data_dir=data_dir, readonly=True)
    job = None
    try:
        job = backend.get_job(job_id)
    except RgbArtifactError:
        gaps.append("RGB 作业库不存在或不可读；只读访问不创建新库，"
                    "无法给出该任务的事实。")
    except (sqlite3.Error, ValueError, OSError, TypeError) as exc:
        gaps.append(f"RGB 作业库读取失败：{_clean(exc)}")
    if job is None and not any("作业库" in g for g in gaps):
        gaps.append(f"RGB 作业库中没有编号为 {job_id} 的任务记录。")

    if job is not None:
        snapshot["job_id"] = job["id"]
        snapshot["status"] = job["status"]
        lines.append(f"- 任务编号：{job['id']}")
        filename = Path(str(job['source_name']).replace('\\', '/')).name
        lines.append(f"- 输入文件名：{_clean(filename)}")
        lines.append(f"- 任务状态：{_STATUS_ZH.get(job['status'], job['status'])}")
        lines.append(f"- 创建时间：{job['created_at']}")
        if job["status"] != "succeeded":
            if job["error"]:
                lines.append(f"- 失败/未确认原因：{_clean(job['error'])}")
            gaps.append("该任务没有成功结果（状态 "
                        f"{job['status']}），无可解释的识别统计；"
                        "不引用任何历史成功计数。")
        else:
            try:
                result = backend.result(job_id)
                backend.source_bytes(job_id)
            except (RgbArtifactError, ValueError, OSError, TypeError) as exc:
                snapshot["status"] = "corrupt"
                gaps.append(f"产物核验未通过：{_clean(exc)}；"
                            "该任务按损坏处理，不引用任何成功计数。")
            else:
                metadata = result["metadata"]
                native = {str(k): int(v)
                          for k, v in metadata["native_class_counts"].items()}
                regions = result["regions"]
                sql = _sql_stats(backend.db_path, job_id)
                sql_ok = False
                if sql is None:
                    gaps.append("无法读取 RGB 库中的 SQL 统计行，"
                                "无法交叉核对库行与产物。")
                else:
                    counts, sql_regions = sql
                    norm_regions = [{**r, "edge_touch": bool(r["edge_touch"]),
                                     "small_patch": bool(r["small_patch"])}
                                    for r in sql_regions]
                    if counts != native or norm_regions != regions:
                        gaps.append("RGB 作业库中的 class_counts/regions 行与"
                                    "产物核验结果不一致，已按库行篡改处理；"
                                    "该库当前不可信，不发布成功统计。")
                        snapshot["sql_counts_verified"] = False
                    else:
                        sql_ok = True
                        snapshot["sql_counts_verified"] = True

                if not sql_ok:
                    snapshot.update(status="corrupt", sql_counts_verified=False)
                    return _bundle(snapshot, lines, citations, gaps)

                snapshot.update({
                    "source_image_sha256": job["image_sha256"],
                    "manifest_sha256": job["manifest_sha256"],
                    "seed": job["seed"],
                    "artifact_sha256": {n: job["artifact_sha256"].get(n)
                                        for n in ARTIFACT_NAMES},
                    "native_class_counts": native,
                    "candidate_count": len(regions),
                    "edge_touch_count": sum(1 for r in regions if r["edge_touch"]),
                    "regions": regions,
                })
                lines.append(f"- 输入图片 SHA256：{job['image_sha256']}")
                lines.append(f"- 模型清单 SHA256：{job['manifest_sha256']}")
                lines.append(f"- 固定 seed：{job['seed']}")
                lines.append("- 六份产物 SHA256：")
                for name in ARTIFACT_NAMES:
                    lines.append(f"    · {name}: {job['artifact_sha256'].get(name)}")
                lines.append("- 三类像素统计（512×512 原生网格，合计 262144 已核对）：")
                for cls in (0, 1, 2):
                    label = {0: "田块候选", 1: "田埂", 2: "道路"}[cls]
                    lines.append(f"    · {label}：{native.get(str(cls), 0)} 像素")
                edge = sum(1 for r in regions if r["edge_touch"])
                lines.append(f"- 候选区域 {len(regions)} 个；其中 {edge} 个触及图片"
                             "边缘（边缘触碰意味着区域可能被裁切，需要人工注意）。")
                for r in regions:
                    lines.append(
                        f"    · 区域 {r['field_id']}：{r['area_pixels']} 像素，"
                        f"范围 ({r['bbox_x0']},{r['bbox_y0']})–"
                        f"({r['bbox_x1']},{r['bbox_y1']})，"
                        f"{'触及边缘' if r['edge_touch'] else '未触边'}")
                if sql_ok:
                    lines.append("- 库行核对：runs.sqlite3 的 class_counts/regions"
                                 " 与产物核验结果一致。")
                citations.append({
                    "label": f"RGB 任务 {job_id[:8]}… 的识别记录",
                    "source": "runs / regions / class_counts"
                              "（data/rgb_candidate 专用库，只读）",
                    "detail": "产物 SHA 与 metadata/quality 交叉核验通过后取值；"
                              "候选区域是四连通统计，不是已确认田块。",
                })

        # P0 登记状态（只读）
        reg = _registration_info(conn, job_id, backend)
        snapshot["registration"] = reg
        if reg is None:
            lines.append("- P0 数据目录登记：尚未登记"
                         "（如需登记请在 RGB 页面显式点击登记按钮）。")
        elif not reg["intact"]:
            gaps.append("P0登记核验未通过：" + reg["problem"])
            lines.append("- P0登记记录存在，但核验未通过，不作为完好登记引用。")
        else:
            state_zh = {"active": "有效", "isolated": "已隔离", "duplicate": "已标记重复"}
            lines.append(f"- P0 数据目录登记：已登记；场景 {reg['scene_id']}，"
                         f"派生PNG版本 #{reg['image_version_id']}"
                         f"（质检 {reg['image_quality']}，"
                         f"版本状态 {state_zh.get(reg['image_version_status'], '未知')}），"
                         f"统计JSON版本 #{reg['stats_version_id']}"
                         f"（质检 {reg['stats_quality']}，"
                         f"版本状态 {state_zh.get(reg['stats_version_status'], '未知')}）。")
            citations.append({
                "label": f"RGB 任务 {job_id[:8]}… 的 P0 登记",
                "source": "rgb_governance_links / asset_versions / quality_checks",
                "detail": "登记内容为派生 PNG 与统计 JSON；质检只针对派生文件，"
                          "不代表原始输入已通过 P0 质检。",
            })

    return _bundle(snapshot, lines, citations, gaps)
