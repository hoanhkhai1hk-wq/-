"""RGB 图片识别页面（INT-20261020-002）。

把已审单图 CLI 变成网页操作：上传 512×512 RGB 图片（或用已审示例 U11），
点“开始识别”显式提交一次识别；结果原图/叠加图并排展示，下方像素统计、
候选区域表与六产物下载，历史持久化可回看。

普通首次进入、rerun、刷新与历史切换零新推理；只有“开始识别”与用户明确
“新建一次识别”后再次提交才会运行模型。模型仍为候选，未经独立新图评估。
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from agrolake import db as p0_db
from agrolake import rgb_governance
from agrolake.rgb_jobs import (
    ARTIFACT_NAMES,
    ROOT,
    RgbArtifactError,
    RgbCandidateInputError,
    RgbCandidateJobs,
)

SAMPLE_U11_PATH = ROOT / "邓" / "hy0717_tile_r38_c53.tif"
SAMPLE_U11_SHA256 = "ebb6514312e21cd1fbfd9ea3ef1486b825e39ffffd294dfabb7652b6bb5445bf"
DOWNLOAD_LABELS = {
    "mask256.png": "下载分类图（256像素）",
    "mask512.png": "下载分类图（原图尺寸）",
    "overlay.png": "下载叠加图",
    "candidate_fields.csv": "下载候选区域表",
    "quality.json": "下载质量记录",
    "metadata.json": "下载来源记录",
}
STATUS_ZH = {"running": "运行中/结果未确认", "succeeded": "成功", "failed": "失败"}
CLASS_ZH = {0: "田块", 1: "田埂", 2: "道路"}


def _local_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return str(value)


def _backend() -> RgbCandidateJobs:
    return RgbCandidateJobs()


def _load_sample() -> tuple[bool, bytes | None, str]:
    """加载已审示例 U11；文件缺失或摘要不符时禁用示例。"""
    if not SAMPLE_U11_PATH.is_file():
        return False, None, "示例文件缺失，示例已禁用；请改用上传图片。"
    data = SAMPLE_U11_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != SAMPLE_U11_SHA256:
        return False, None, "示例文件摘要与已审记录不符，示例已禁用；请改用上传图片。"
    return True, data, SAMPLE_U11_PATH.name


def _pick_current_input(backend: RgbCandidateJobs) -> tuple[bytes | None, str]:
    """读取当前输入（上传或示例），并在换图时清除当前请求号与结果视图。"""
    state = st.session_state
    source = st.radio("图片来源", ["上传图片", "已审示例 U11"],
                      horizontal=True, key="rgb_input_source")
    content: bytes | None = None
    name = ""
    if source == "上传图片":
        up = st.file_uploader(
            "选择 512×512 RGB 图片（PNG/JPEG/TIFF，不超过 4MiB）",
            type=["png", "jpg", "jpeg", "tif", "tiff"], key="rgb_uploader")
        if up is not None:
            content = up.getvalue()
            name = up.name
    else:
        ok, data, sample_name = _load_sample()
        if ok:
            content, name = data, sample_name
        else:
            st.warning("已审示例 U11 不可用，已禁用；请改用“上传图片”。")
    digest = hashlib.sha256(content).hexdigest() if content else None
    if digest != state.get("rgb_input_sha"):
        # 换了图片（或清空）：当前结果与请求号一并清除；历史记录保留。
        state.rgb_input_sha = digest
        state.pop("rgb_req_id", None)
        state.pop("rgb_req_sha", None)
        state.pop("rgb_current_job", None)
        state.pop("rgb_history_pick", None)
    return content, name


def _submit_controls(backend: RgbCandidateJobs, content: bytes | None, name: str) -> None:
    state = st.session_state
    col_run, col_new, _ = st.columns([1, 1.6, 2])
    with col_run:
        run_clicked = st.button("开始识别", key="rgb_run", type="primary",
                                disabled=content is None,
                                help="只会识别当前选择的这张图片")
    with col_new:
        if st.button("新建一次识别", key="rgb_new",
                     help="更换请求编号；同一张图片再次点“开始识别”会重新运行一次"):
            state.pop("rgb_req_id", None)
            state.pop("rgb_req_sha", None)
            state.pop("rgb_current_job", None)
            state.pop("rgb_history_pick", None)
            st.info("已准备新建：再次点“开始识别”会以新编号运行一次识别。")
    if not run_clicked:
        return
    state.pop("rgb_current_job", None)
    state.pop("rgb_history_pick", None)
    digest = hashlib.sha256(content).hexdigest()
    if digest != state.get("rgb_req_sha"):
        state.rgb_req_id = uuid.uuid4().hex
        state.rgb_req_sha = digest
    try:
        job = backend.submit(content, name, state.rgb_req_id)
    except RgbCandidateInputError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # 环境/存储等意外错误，如实显示
        st.error(f"识别提交失败：{exc}")
        return
    state.rgb_current_job = job["id"]
    st.rerun()


def _pixel_table(result: dict) -> None:
    native = result["quality"].get("native_class_counts", {})
    total = sum(int(v) for v in native.values()) or 1
    rows = []
    for cls in (0, 1, 2):
        pixels = int(native.get(str(cls), 0))
        rows.append({
            "类别": CLASS_ZH[cls],
            "像素数": pixels,
            "占比": f"{pixels / total:.2%}",
        })
    st.markdown("**像素统计（512×512，面积仅为像素，不是平方米）**")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _regions_table(regions: list[dict]) -> None:
    st.markdown("**候选区域（相互连通的田块像素区域，尚未经实地确认）**")
    if not regions:
        st.info("本次识别没有发现候选区域。")
        return
    rows = [{
        "编号": r["field_id"],
        "像素面积": r["area_pixels"],
        "范围": f"({r['bbox_x0']},{r['bbox_y0']})–({r['bbox_x1']},{r['bbox_y1']})",
        "中心": f"({r['centroid_x']:.1f},{r['centroid_y']:.1f})",
        "触及边缘": "是" if r["edge_touch"] else "否",
        "小斑块": "是" if r["small_patch"] else "否",
    } for r in regions]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _downloads(backend: RgbCandidateJobs, job_id: str, scope: str) -> None:
    st.markdown("**结果下载（六份固定文件，读取时核对摘要）**")
    cols = st.columns(3)
    for i, name in enumerate(ARTIFACT_NAMES):
        with cols[i % 3]:
            try:
                data = backend.artifact(job_id, name)
            except RgbArtifactError as exc:
                st.error(str(exc))
                continue
            st.download_button(DOWNLOAD_LABELS[name], data, file_name=name,
                               key=f"rgb_dl_{scope}_{job_id[:8]}_{name}")


def _render_job(backend: RgbCandidateJobs, job_id: str,
                readonly: bool = False, scope: str = "cur") -> None:
    job = backend.get_job(job_id)
    if job is None:
        st.warning("找不到该作业记录。")
        return
    status = STATUS_ZH.get(job["status"], job["status"])
    st.caption(f"{'历史记录' if readonly else '本次记录'}：{job['source_name']} · "
               f"{_local_time(job['created_at'])} · {status}")
    if job["status"] == "running":
        st.info("该作业状态为运行中或结果未确认（可能是历史遗留记录）；"
                "系统不会自动重跑，也不会把它显示为成功。")
        return
    elif job["status"] == "failed":
        st.error(f"识别失败：{job['error'] or '原因未记录'}")
        return
    try:
        result = backend.result(job_id)
    except RgbArtifactError as exc:
        st.error(f"结果读取失败：{exc}")
        return
    except Exception as exc:
        st.error(f"结果读取失败：{exc}")
        return
    source, overlay = None, None
    try:
        source = backend.source_bytes(job_id)
        overlay = backend.artifact(job_id, "overlay.png")
    except RgbArtifactError as exc:
        st.error(f"图片读取失败：{exc}")
    if source is not None and overlay is not None:
        left, right = st.columns(2)
        with left:
            st.image(source, caption="原图", use_container_width=True)
        with right:
            st.image(overlay, caption="识别叠加图（绿=田块候选，黄=田埂，紫=道路）",
                     use_container_width=True)
    _pixel_table(result)
    _regions_table(result.get("regions") or [])
    _downloads(backend, job_id, scope)
    _registration_section(job_id, scope)
    st.caption("说明：当前为候选模型，尚未用独立新图评估识别精度；"
               "候选区域数量与像素面积来自固定程序统计，面积仅为像素，"
               "不能等同真实田块数量或平方米。")


def _registration_section(job_id: str, scope: str) -> None:
    """数据目录登记（INT-20261021-001）。

    只有显式点击按钮才登记；普通浏览/刷新/历史查看只做只读状态查询。
    已登记展示同样经过核验；损坏/失败作业在上方已提前返回，不会到达这里。
    """
    st.markdown("---")
    st.markdown("**数据目录登记**")
    st.caption(
        "把本次成功结果的两份派生资产（512×512 无损 PNG 影像 + JSON 识别统计）"
        "登记进 P0 数据目录并建立链接；登记只读本作业，不触发识别。"
        "重复点击返回既有链接并重新核验。")
    try:
        ro = p0_db.connect_readonly()
    except Exception as exc:
        st.error(f"P0 只读连接失败，无法读取登记状态：{exc}")
        return
    try:
        status = rgb_governance.link_status(ro, job_id)
    except Exception as exc:
        st.error(f"登记状态读取失败：{exc}")
        return
    finally:
        ro.close()
    if status.get("registered"):
        if status.get("intact"):
            quality_zh = {"pass": "通过", "warn": "警告", "block": "阻断", "na": "不适用"}
            st.success(
                f"已登记：场景 {status['scene_id']} · "
                f"派生PNG版本 #{status['image_version_id']}"
                f"（质检 {quality_zh.get(status.get('image_quality'), '未知')}） · "
                f"统计JSON版本 #{status['stats_version_id']}"
                f"（质检 {quality_zh.get(status.get('stats_quality'), '未知')}） · "
                f"登记时间 {status.get('created_at', '未知')}")
        else:
            st.warning(
                f"已有登记记录，但本次核验未通过：{status.get('problem')}；"
                "在核验恢复前请勿采信该登记为完好。")
    if st.button("登记到数据目录",
                 key=f"rgb_register_{scope}_{job_id}",
                 help="仅显式点击才登记；只登记当前这条成功记录"):
        try:
            reg_conn = p0_db.connect()
        except Exception as exc:
            st.error(f"P0 连接失败：{exc}")
            return
        try:
            result = rgb_governance.register(reg_conn, job_id)
        except Exception as exc:
            st.error(f"登记失败：{exc}")
            return
        finally:
            reg_conn.close()
        reg = result["registration"]
        if result["created"]:
            st.success(
                f"登记完成：场景 {reg['scene_id']} · "
                f"派生PNG版本 #{reg['image_version_id']} · "
                f"统计JSON版本 #{reg['stats_version_id']}")
        else:
            st.info(
                f"该作业此前已登记（本次已重新核验通过）：场景 {reg['scene_id']} · "
                f"派生PNG版本 #{reg['image_version_id']} · "
                f"统计JSON版本 #{reg['stats_version_id']}")
        st.caption("质检摘要：" + result.get("quality_summary", "未知"))
        st.caption("边界：登记的是派生 PNG 与统计 JSON，质检不代表原始输入"
                   "已通过 P0 质检；像素不是平方米，候选区域不是已确认田块。")


def render() -> None:
    st.subheader("RGB 图片识别")
    st.caption("上传 512×512 RGB 农田图片（或用已审示例 U11），点“开始识别”运行候选模型；"
               "结果与历史会保存。普通浏览、刷新和历史查看不会触发新的识别。")
    backend = _backend()
    content, name = _pick_current_input(backend)
    _submit_controls(backend, content, name)

    current = st.session_state.get("rgb_current_job")
    if current:
        st.markdown("---")
        _render_job(backend, current)

    st.markdown("---")
    st.markdown("**识别历史**")
    if st.button("刷新历史", key="rgb_history_refresh"):
        pass  # 点击只是重跑脚本，重新读取历史；不产生新推理。
    jobs = backend.list_jobs(30)
    if not jobs:
        st.info("还没有识别记录。")
        return
    options = {"（选择历史作业查看）": None}
    for j in jobs:
        label = (f"{_local_time(j['created_at'])} · {j['source_name'][:24]} · "
                 f"{STATUS_ZH.get(j['status'], j['status'])} · {j['id'][:8]}…")
        options[label] = j["id"]
    picked = st.selectbox("历史作业", list(options), key="rgb_history_pick")
    picked_id = options.get(picked)
    if picked_id and picked_id != current:
        st.markdown("---")
        _render_job(backend, picked_id, readonly=True, scope="hist")
    elif picked_id:
        st.caption("这条历史记录已在上方显示。")
