"""田块识别页面（DEV-20261015-012）。

零基础面向：先解释 8 波段 TIFF 要求，再让用户选择公开样例或上传文件，
点击“开始识别”才调用本机识别服务；成功后显示真实预览、像素统计与候选区域表，
并提供五种产物下载。失败/中断不显示此前成功结果。

设计约束（见任务 DEV-20261015-012）：
- 只在点击“开始识别/重试提交”时 POST；普通 rerun、切页、刷新、下载都不重复 POST。
- 每次提交持有稳定 request_id；提交超时结果未知时复用同一 key。
- 只对本机已审识别服务发起请求；服务未运行只给启动提示。
- 只显示服务返回的事实，不冒充模型精度或平方米面积。
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid

import streamlit as st

from agrolake_service import config as svc

from .client import ApiConfigError, ApiError, RecognitionClient
from . import review_panel

# 会话状态前缀：避免与 P0 页面（ag_flash 等）冲突。
_NS = "rec"

STATUS_ZH = {
    "queued": "排队中",
    "running": "运行中",
    "succeeded": "已成功",
    "failed": "已失败",
    "interrupted": "已中断",
}
TERMINAL = {"succeeded", "failed", "interrupted"}

# 公开适配样例的真实来源说明（与已审服务/文档一致，不是自动取得授权的证明）。
PUBLIC_SAMPLE_META = {
    "source": "Fields of the World 公开上游适配样例；时相身份未核实",
    "source_link": "https://github.com/fieldsoftheworld/ftw-baselines/tree/45d7df7ea0c357f2c34182e4a7e0c4625dd43533",
    "license_status": "上游公开示例；许可/署名分项见 THIRD_PARTY.md；不是教师无人机数据",
    "scene_id": "FTW-PUBLIC-ADAPTER-DEMO",
    "filename": "inference-img.tif",
}

_NO_SELECT = "（未选择）"


def _public_sample() -> tuple[bool, bytes | None, str]:
    """返回（是否可用, 字节, 说明）；只有文件存在且摘要匹配固定公开摘要才可用。"""
    path = svc.default_public_sample_path()
    try:
        if not path.is_file():
            return False, None, f"未找到公开样例文件：{path}"
        if path.stat().st_size>svc.MAX_CONTENT_BYTES:
            return False,None,'公开样例文件超出允许大小，请检查样例配置。'
        data = path.read_bytes()
    except OSError as exc:
        return False, None, f"读取公开样例失败：{exc}"
    sha = hashlib.sha256(data).hexdigest()
    if sha != svc.PUBLIC_SAMPLE_SHA256:
        return False, None, (
            "公开样例文件摘要与固定公开摘要不符，为避免把私有数据当作公开样例，已禁止使用。"
        )
    return True, data, "已确认本机公开适配样例（文件存在且摘要匹配）。"


def _status_badge(status: str) -> str:
    label = STATUS_ZH.get(status, status)
    if status == "succeeded":
        return f"✅ {label}"
    if status in ("failed", "interrupted"):
        return f"⚠️ {label}"
    return f"⏳ {label}"


def _collect_content(mode: str, public_ok: bool, public_bytes, upload_bytes, upload_name):
    if mode == "public":
        if not public_ok:
            return None, None
        return public_bytes, PUBLIC_SAMPLE_META["filename"]
    if upload_bytes is None:
        return None, None
    return upload_bytes, upload_name


def _intent_key(content_sha: str | None, filename: str | None, source: str,
                source_link: str, license_status: str, scene_id: str) -> str:
    parts = [content_sha or "", filename or "", source or "", source_link or "",
             license_status or "", scene_id or ""]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _build_payload(content: bytes, filename: str, source: str, source_link: str,
                   license_status: str, scene_id: str, request_id: str) -> dict:
    import base64

    payload = {
        "filename": filename,
        "content_base64": base64.b64encode(content).decode("ascii"),
        "source": source,
        "scene_id": scene_id,
        "request_id": request_id,
    }
    if source_link:
        payload["source_link"] = source_link
    if license_status:
        payload["license_status"] = license_status
    return payload


def _do_submit(client: RecognitionClient, payload: dict, intent: str) -> None:
    """执行一次提交，并把结果/错误写入会话状态。不在此处重复执行。"""
    st.session_state[f'{_NS}_selected_job']=None
    st.session_state[f'{_NS}_history']=_NO_SELECT
    st.session_state[f'{_NS}_auto_count']=0
    # 回执未知时不能继续借用上一次已结束任务来另开新请求。
    st.session_state.setdefault(f'{_NS}_confirmed', {}).pop(intent, None)
    try:
        result = client.submit_job(payload)
    except ApiError as exc:
        if exc.category in ('timeout','protocol','badjson','offline'):
            st.session_state[f"{_NS}_pending"] = {"intent": intent}
            st.warning(
                "提交结果尚未确认。可查看任务历史，或点击“重试提交（同一请求号）”；"
                "服务端按请求号幂等，不会重复执行。"
            )
            return
        st.error(f"提交失败（{exc.category}）：{exc.message}")
        return
    st.session_state[f"{_NS}_selected_job"] = int(result["job_id"])
    st.session_state[f"{_NS}_pending"] = None
    st.session_state.setdefault(f'{_NS}_confirmed',{})[intent]=int(result['job_id'])
    st.success(
        f"已提交识别任务 #{result['job_id']}"
        f"（{'新建' if result.get('created') else '复用已有'}）。"
    )


def render(api_url: str | None = None) -> None:
    """渲染“田块识别”页；由 app.py 在第八个 tab 中调用。"""
    st.subheader('田块识别')
    st.caption(
        '选择8波段双时相TIFF或公开演示样例，识别田块内部和边界，查看预测图与候选区域像素表。'
    )

    with st.expander("使用前须知（点击展开）", expanded=False):
        st.markdown(
            "- 本模型需要 **8 波段双时相 GeoTIFF** 栅格；普通 JPG/PNG 或 3 通道 RGB 不适配，"
            "不会被伪装成 8 波段。\n"
            "- 来源与场景为必填声明；许可字段是你填写的声明，系统不会自动验证授权。\n"
            "- 结果中的“候选连通区域”是像素统计，**不等于经验证的真实田块数**；"
            "没有独立真值，不报告准确率；面积只以像素计，未取得经核验的地理参考前不换算平方米。\n"
            '- 结果会保存在任务历史中，可查看来源记录并下载。'
        )

    # ---- 服务地址与在线状态 ------------------------------------------------
    try:
        client = RecognitionClient(api_url)
    except ApiConfigError as exc:
        st.error(f"识别服务地址配置错误：{exc.message}")
        st.caption("请检查环境变量 AGROLAKE_API_URL，默认 http://127.0.0.1:8506。")
        return

    st.button('重新连接识别服务',key=f'{_NS}_reconnect')
    try:
        health = client.health()
    except ApiError as exc:
        st.warning(
            f"识别服务当前不可用：{exc.message}\n\n"
            "请先按 docs/网页识别.md 启动本机识别服务（默认 http://127.0.0.1:8506）："
        )
        st.code(
            "$env:PYTHONDONTWRITEBYTECODE = '1'\n"
            "& '.venv-p0/Scripts/python.exe' 'scripts/service/serve.py'",
            language="powershell",
        )
        st.info("识别服务未运行时，本页无法提交任务；上方其余治理页面仍可正常使用。")
        return

    ready_notes = []
    for key, label in (("worker_python_ready", "识别子进程解释器 .venv-recognition"),
                       ("cli_script_ready", "已审识别 CLI"), ("model_ready", "模型权重文件")):
        if not health.get(key):
            ready_notes.append(label)
    if ready_notes:
        st.warning("识别服务已启动，但以下资源缺失，真实识别可能失败：" + "、".join(ready_notes))
    else:
        st.success('识别服务在线，模型文件已就绪。运行结果请看下方任务状态。')

    # ---- 输入 --------------------------------------------------------------
    st.markdown("### 1. 选择输入")
    mode_label = st.radio(
        "输入方式",
        ["使用本机公开适配样例", "上传自己的 8 波段 TIFF"],
        key=f"{_NS}_input_mode",
        horizontal=True,
    )
    mode = "public" if mode_label.startswith("使用") else "upload"

    public_ok, public_bytes, public_note = _public_sample()
    upload_bytes = None
    upload_name = None

    if mode == "public":
        if public_ok:
            st.caption(public_note)
            st.caption(
                "公开演示样例不是自采农业观测；其来源/许可为上游公开说明，时相身份未独立确认。"
                "许可字段为你确认的声明。"
            )
        else:
            st.error(public_note)
    else:
        up = st.file_uploader(
            "选择 .tif/.tiff 文件（≤40 MiB）", type=["tif", "tiff"], key=f"{_NS}_upload"
        )
        if up is not None:
            if up.size > svc.MAX_CONTENT_BYTES:
                st.error(
                    f'文件超过40 MiB上限，请选择较小的TIFF。'
                )
            else:
                data = up.getvalue()
                upload_bytes = data
                upload_name = up.name
                st.caption(f"已选择：{up.name}（{len(data)} 字节）")

    content, filename = _collect_content(mode, public_ok, public_bytes, upload_bytes, upload_name)
    content_sha = hashlib.sha256(content).hexdigest() if content else None

    # ---- 元数据 ------------------------------------------------------------
    st.markdown("### 2. 填写来源与场景（来源、场景必填）")
    meta_default = PUBLIC_SAMPLE_META if mode == "public" else {
        "source": "", "source_link": "", "license_status": "", "scene_id": ""
    }
    suffix = mode
    source = st.text_input('影像来源（必填）', value=meta_default["source"],
                           key=f"{_NS}_source_{suffix}")
    col_a, col_b = st.columns(2)
    scene_id = col_a.text_input('场景编号（必填）', value=meta_default["scene_id"],
                                key=f"{_NS}_scene_{suffix}")
    source_link = col_b.text_input("来源链接（可留空）", value=meta_default["source_link"],
                                   key=f"{_NS}_link_{suffix}")
    license_status = st.text_input("许可/授权状态（可留空；仅为你的声明）",
                                   value=meta_default["license_status"],
                                   key=f"{_NS}_license_{suffix}")

    intent = _intent_key(content_sha, filename, source.strip(), source_link.strip(),
                         license_status.strip(), scene_id.strip())
    rids = st.session_state.setdefault(f"{_NS}_request_ids", {})

    # ---- 提交 --------------------------------------------------------------
    st.markdown("### 3. 开始识别")
    submit_clicked = st.button("开始识别", key=f"{_NS}_submit", type="primary")
    can_submit = content is not None and bool(source.strip()) and bool(scene_id.strip())

    if submit_clicked:
        if content is None:
            st.error("请先选择公开样例或上传一个 TIFF 文件。")
        elif not content:
            st.error('文件为空，请选择有内容的TIFF。')
        elif not source.strip():
            st.error("来源为必填项，请填写来源。")
        elif not scene_id.strip():
            st.error("场景为必填项，请填写场景编号。")
        else:
            rid = rids.get(intent)
            if rid is None:
                rid = uuid.uuid4().hex
                rids[intent] = rid
            payload = _build_payload(content, filename, source.strip(), source_link.strip(),
                                     license_status.strip(), scene_id.strip(), rid)
            _do_submit(client, payload, intent)

    confirmed=st.session_state.setdefault(f'{_NS}_confirmed',{})
    new_run=st.button('重新识别（新任务）',key=f'{_NS}_new_run',
        disabled=not can_submit or intent not in confirmed)
    st.caption('“开始识别”复用当前请求；已完成、失败或中断后，可用“重新识别”明确新建一次运行。')
    if new_run:
        try:
            prior=client.get_job(confirmed[intent])
            if prior['status'] not in TERMINAL:
                st.info('当前输入仍在排队或运行，请先刷新状态等待完成。')
            else:
                rid=uuid.uuid4().hex
                rids[intent]=rid
                _do_submit(client,_build_payload(content,filename,source.strip(),source_link.strip(),
                    license_status.strip(),scene_id.strip(),rid),intent)
                if st.session_state.get(f'{_NS}_pending'):
                    st.rerun()  # 立即禁用已绘制的“新任务”，保留同号重试入口。
        except ApiError as exc:
            st.error(f'未能确认原任务状态，未新建运行：{exc.message}')

    pending = st.session_state.get(f"{_NS}_pending")
    if pending and pending.get("intent") == intent and content is not None:
        st.info('上次提交结果尚未确认，重试会保持同一请求号。')
        if st.button("重试提交（同一请求号）", key=f"{_NS}_retry"):
            rid = rids.get(intent)
            if rid is None:
                rid = uuid.uuid4().hex
                rids[intent] = rid
            payload = _build_payload(content, filename, source.strip(), source_link.strip(),
                                     license_status.strip(), scene_id.strip(), rid)
            _do_submit(client, payload, intent)
    elif pending and pending.get("intent") != intent:
        st.caption("输入内容已变化，将作为新的识别任务提交（新的请求号）。")

    if content is not None and not can_submit:
        st.caption("填写来源与场景后即可提交。")

    # ---- 历史任务 ----------------------------------------------------------
    st.markdown("### 4. 任务状态与历史")
    try:
        jobs = client.list_jobs()
    except ApiError as exc:
        st.error(f"读取任务列表失败（{exc.category}）：{exc.message}")
        return

    selected = st.session_state.get(f"{_NS}_selected_job")
    if selected is not None and selected not in [j['job_id'] for j in jobs]:
        try:
            jobs.append(client.get_job(selected))  # 最近100条以外的本会话选择也能继续查看
        except ApiError:
            selected=None
            st.session_state[f'{_NS}_selected_job']=None
            st.session_state[f'{_NS}_history']=_NO_SELECT

    def _hist_label(job: dict) -> str:
        return (f"#{job['job_id']} · {STATUS_ZH.get(job['status'], job['status'])} · "
                f"{job.get('original_filename', '')}")

    options = [_NO_SELECT] + [_hist_label(j) for j in jobs]

    def _on_history_change():
        val = st.session_state.get(f"{_NS}_history")
        if not val or val == _NO_SELECT:
            st.session_state[f"{_NS}_selected_job"] = None
            return
        # 选项形如 "#<编号> · 状态 · 文件名"；编号解析失败视为未选择。
        try:
            st.session_state[f"{_NS}_selected_job"] = int(str(val)[1:].split(" ", 1)[0])
        except (ValueError, IndexError):
            st.session_state[f"{_NS}_selected_job"] = None

    # 提交成功后把下拉同步到当前选中的作业。
    if selected is not None:
        match = next((lab for lab in options if lab.startswith(f"#{selected} ")), None)
        if match is not None:
            st.session_state[f"{_NS}_history"] = match
    elif f"{_NS}_history" not in st.session_state:
        st.session_state[f"{_NS}_history"] = _NO_SELECT

    if jobs:
        st.selectbox("选择历史任务查看", options, key=f"{_NS}_history",
                     on_change=_on_history_change)
    else:
        st.caption("暂无历史任务。")

    if selected is None:
        st.info("提交新的识别任务，或在上方历史任务中选择一个查看结果。")
        return

    # ---- 选中任务详情 ------------------------------------------------------
    try:
        job = client.get_job(selected)
    except ApiError as exc:
        st.error(f"读取任务 #{selected} 失败（{exc.category}）：{exc.message}")
        return

    status = job.get("status")
    if status in TERMINAL:
        st.session_state[f"{_NS}_auto_count"] = 0

    top = st.columns([3, 1])
    top[0].markdown(f"**任务 #{job['job_id']}** · {_status_badge(status)}")
    if top[1].button("刷新状态", key=f"{_NS}_refresh"):
        st.rerun()

    if status in ("queued", "running"):
        auto = st.checkbox("自动刷新（每 2 秒，最多 60 次）", key=f"{_NS}_auto")
        cnt = st.session_state.get(f"{_NS}_auto_count", 0)
        if auto and cnt < 60:
            st.session_state[f"{_NS}_auto_count"] = cnt + 1
            time.sleep(2)
            st.rerun()
        elif auto:
            st.caption("自动刷新已达上限，请手动点击“刷新状态”。")
        return

    if status == "failed":
        st.error(f"任务失败：{job.get('error') or '服务未给出具体原因，请查看服务日志。'}")
        st.caption("失败任务不显示此前任何成功结果，请检查输入或服务后重新提交。")
        _render_trace(job)
        return

    if status == "interrupted":
        st.warning(
            f"任务已中断：{job.get('error') or '服务重启或停止，未自动重放。'}"
        )
        st.caption("中断任务不自动重放；如需重新识别请重新提交一个新任务。")
        _render_trace(job)
        return

    if status!='succeeded':
        st.error('任务状态无法识别，请刷新后重试。')
        return

    # status == succeeded
    _render_success(client, job)


def _render_trace(job: dict) -> None:
    with st.expander("可追溯事实（任务编号/来源/场景/模型与输入摘要）"):
        st.write({
            "任务编号": job.get("job_id"),
            "来源": job.get("source"),
            "来源链接": job.get("source_link") or "（未填）",
            "许可声明": job.get("license_status") or "（未填）",
            "场景": job.get("scene_id"),
            "输入类型": job.get("sample_kind"),
            "原始文件名": job.get("original_filename"),
            "输入SHA256": job.get("input_sha256"),
            "模型SHA256": job.get("model_sha256"),
            "请求号": job.get("request_id"),
            "尝试次数": job.get("attempt_count"),
            "创建时间": job.get("created_at"),
            "结束时间": job.get("finished_at"),
        })
        run = job.get("run") or {}
        if run:
            st.caption(
                f"适配器：{run.get('adapter')} · 窗口/参数：{run.get('params')} · "
                f"耗时：{run.get('elapsed_seconds')} 秒"
            )


def _render_success(client: RecognitionClient, job: dict) -> None:
    job_id = job["job_id"]

    checksums=job.get('artifact_checksums') or {}
    if any(not re.fullmatch(r'[0-9a-f]{64}',str((checksums.get(n) or {}).get('sha256','')))
           for n in svc.FIXED_ARTIFACTS):
        st.error('任务产物缺少完整校验记录，暂不展示结果，请刷新或重新识别。')
        return

    st.markdown("### 结果")
    counts = job.get("class_counts") or {}
    c = st.columns(4)
    c[0].metric("背景像素", counts.get("0", "—"))
    c[1].metric("田块内部像素", counts.get("1", "—"))
    c[2].metric("边界像素", counts.get("2", "—"))
    c[3].metric("无效像素", counts.get("255", "—"))

    region_count = job.get("region_count")
    st.caption(
        f"候选连通区域数：{region_count}。这是像素级连通统计，**不等于经验证的真实田块数**；"
        "面积只以像素计，未取得经核验的地理参考前不换算平方米；没有独立真值，不报告准确率。"
    )

    # 产物：只对成功作业下载，并在下载时核对服务保存的摘要（改动会被 409 拦下）。
    checksums = job.get("artifact_checksums") or {}
    artifacts: dict[str, bytes] = {}
    for name in svc.FIXED_ARTIFACTS:
        expected = (checksums.get(name) or {}).get("sha256")
        try:
            artifacts[name] = client.get_artifact(job_id, name, expected_sha256=expected)
        except ApiError as exc:
            st.error(
                f"读取产物 {name} 失败（{exc.category}）：{exc.message}\n\n"
                "为防止展示被改动或过期的结果，已停止显示本任务的产物。"
            )
            artifacts = {}
            break

    if artifacts.get("preview.png"):
        st.image(artifacts["preview.png"],
                 caption="左：输入前三波段拉伸　中：预测类别　右：叠加（非真值；无效像素为灰色）")

    try:
        fields = client.get_fields(job_id)
    except ApiError as exc:
        st.error(f"读取候选区域失败（{exc.category}）：{exc.message}")
        fields = None

    if fields and fields.get("regions"):
        rows = [{
            "区域编号": r["field_id"],
            "像素数": r["pixels"],
            "最小行": r["min_row"],
            "最小列": r["min_col"],
            "最大行": r["max_row"],
            "最大列": r["max_col"],
        } for r in fields["regions"]]
        st.markdown(f"**候选区域表（{fields.get('region_count', len(rows))} 个，按像素边框）**")
        st.dataframe(rows, width="stretch", hide_index=True)

    if artifacts:
        st.markdown("**下载产物（下载时会再次核对摘要）**")
        mime = {
            "mask.tif": "image/tiff",
            "preview.png": "image/png",
            "class_pixels.csv": "text/csv",
            "field_pixels.csv": "text/csv",
            "run.json": "application/json",
        }
        cols = st.columns(len(svc.FIXED_ARTIFACTS))
        for i, name in enumerate(svc.FIXED_ARTIFACTS):
            data = artifacts.get(name)
            if data is None:
                continue
            cols[i].download_button(
                f"下载 {name}", data, file_name=name, mime=mime.get(name),
                key=f"{_NS}_dl_{job_id}_{name}",
            )

    # 人工复核面板：只在五项产物校验齐全（未因 409/摘要不符被清空）时出现，
    # 避免在结果未被验证时给出可复核的“成功”观感。
    artifacts_ok = len(artifacts) == len(svc.FIXED_ARTIFACTS)
    if artifacts_ok:
        regions = (fields or {}).get("regions") or []
        review_panel.render(client, job, regions)
    else:
        st.caption("产物未通过完整校验，暂不提供人工复核记录。")

    _render_trace(job)
