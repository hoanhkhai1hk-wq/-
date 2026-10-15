"""识别输入治理登记与血缘面板（DEV-20261015-014）。

放在第 8 页“田块识别”的成功结果里，只通过 :class:`RecognitionClient` 调用本机识别
服务接口，**不直接写数据库、不加载模型、不触发推理**。

边界与规则：
- 只对“已成功且五项产物校验通过”的作业显示（由调用方控制）。
- 只有用户显式点击“登记输入资料”才 POST；普通刷新/切历史/下载都不写。
- 同一作业固定一个请求号：超时/网络未知时提示“重试提交（同一请求号）”，不悄悄换号。
- 登记的是“识别输入登记单 JSON”（P0 文本资产）；其质检结论只针对该 JSON 文本，
  不代表原 TIFF 已通过 P0 质检或已有推理前质量门禁。
- 清晰展示真实 ID，以及登记时与当前的质检/状态差异；不把历史检查冒充实时通过。
- 切换作业不串状态；读取失败不显示旧的成功观感。
"""

from __future__ import annotations

import json
import uuid

import streamlit as st

from .client import ApiError, RecognitionClient

_NS = "rec_gov"

_BOUNDARY = (
    "登记单是识别输入的服务端事实快照，作为 P0 文本资产；其质检结论只针对该 JSON 文本，"
    "不代表原 TIFF 已通过 P0 质检。登记不触发新推理，候选区域仍以像素计，"
    "不等于真实田块数或平方米，不报告精度。"
)


def _ids() -> dict:
    return st.session_state.setdefault(f"{_NS}_ids", {})


def _do_register(client: RecognitionClient, job_id: int) -> None:
    ids = _ids()
    rid = ids.get(job_id)
    if rid is None:
        rid = uuid.uuid4().hex
        ids[job_id] = rid
    pending = st.session_state.setdefault(f"{_NS}_pending", {})
    try:
        result = client.register_governance(job_id, request_id=rid)
    except ApiError as exc:
        if exc.category in ("timeout", "protocol", "badjson", "offline"):
            pending[job_id] = True
            st.warning(
                "登记结果尚未确认。可点击“重试提交（同一请求号）”；"
                "服务端按请求号幂等，不会重复登记。"
            )
            return
        st.error(f"登记失败（{exc.category}）：{exc.message}")
        return
    pending.pop(job_id, None)
    if result.get("created"):
        st.session_state[f"{_NS}_flash"] = (
            f"已登记识别输入资料：P0 资产 #{result['asset_id']}／版本 #{result['version_id']}。"
        )
    else:
        st.session_state[f"{_NS}_flash"] = "该作业已存在登记，本次未重复登记。"
    st.rerun()


def _render_registered(lineage: dict) -> None:
    reg = lineage["registration"]
    job = lineage["job"]
    st.markdown("**治理登记状态：已登记**")
    left, right = st.columns(2)
    with left:
        st.write({
            "P0 资产编号": reg["asset_id"],
            "P0 版本编号": reg["version_id"],
            "登记链接编号": reg["link_id"],
            "登记请求号": reg["request_id"],
            "登记时间": reg["created_at"],
        })
    with right:
        st.write({
            "登记单摘要 (manifest_sha256)": reg["manifest_sha256"],
            "原 TIFF 摘要 (input_sha256)": reg["input_sha256"],
            "登记单资产类型": (reg.get("current", {}).get("version") or {}).get("content_type"),
            "登记单用途": (reg.get("current", {}).get("version") or {}).get("intended_use"),
        })
    st.caption("两个摘要是不同对象：manifest_sha256 是该 JSON 登记单的摘要，不是原图摘要。")

    snap = reg.get("quality_at_registration") or {}
    cur = reg.get("current") or {}
    cur_quality = cur.get("quality") or {}
    diff = cur.get("diff") or {}
    a, b = st.columns(2)
    with a:
        st.markdown("**登记时质检（针对登记单 JSON 文本）**")
        st.write({"规则版本": snap.get("rules_version"), "总体结论": snap.get("overall")})
        st.dataframe(
            [{"规则": c["rule_id"], "结论": c["status"], "说明": c["reason"]}
             for c in (snap.get("checks") or [])],
            width="stretch", hide_index=True,
        )
    with b:
        st.markdown("**当前状态（登记后可能因 P0 处置而变）**")
        st.write({
            "版本状态": (cur.get("version") or {}).get("status"),
            "当前来源": (cur.get("version") or {}).get("source"),
            "当前场景": (cur.get("version") or {}).get("declared_scene_id"),
            "来源是否变化": diff.get("source_changed"),
            "场景是否变化": diff.get("scene_changed"),
            "当前质检总体": cur_quality.get("overall"),
            "处置记录数": (cur.get("dispositions") or {}).get("total"),
        })
        if diff.get("source_changed") or diff.get("scene_changed"):
            st.info("登记后 P0 来源/场景已变化；登记时快照原样保留，不被覆盖。")
        st.dataframe([{"规则":c['rule_id'],"结论":c['status'],"说明":c['reason']}
                      for c in cur_quality.get('checks',{}).get('items',[])],
                     width='stretch', hide_index=True)

    integrity = lineage.get("integrity") or {}
    st.markdown("**对象完整性（本次查询时的字节观察）**")
    st.write({
        "输入对象完整": integrity.get("input_object_intact"),
        "服务登记单对象完整": integrity.get("manifest_service_object_intact"),
        "P0 登记单对象完整": integrity.get("manifest_p0_object_intact"),
    })
    if any(integrity.get(k) is not True for k in
           ('input_object_intact','manifest_service_object_intact','manifest_p0_object_intact')):
        st.warning('本次对象完整性核验有异常；下面保留历史事实，不能作为当前字节完好的证明。')

    st.markdown("**识别血缘摘要**")
    st.write({
        "任务状态": job.get("status"),
        "输入类型": job.get("sample_kind"),
        "原始文件名": job.get("original_filename"),
        "模型摘要": job.get("model_sha256"),
        "候选区域数（像素计）": job.get("region_count"),
        "人工复核记录数": (lineage.get("reviews") or {}).get("total"),
        "产物数": len(lineage.get("artifacts") or []),
    })
    if lineage.get("reviews", {}).get("truncated"):
        st.warning("人工复核记录超过单次读取上限，展示已截断；原记录仍在服务端保留。")


def _render_unregistered(lineage: dict) -> None:
    st.markdown("**治理登记状态：未登记**")
    st.info(
        "尚未把本作业的识别输入登记为 P0 治理资料。"
        "“未登记”只表示治理资料尚未建立，不代表作业未成功或已完成治理。"
    )
    job = lineage.get("job") or {}
    st.write({
        "任务状态": job.get("status"),
        "输入摘要": job.get("input_sha256"),
        "来源": job.get("source"),
        "场景": job.get("scene_id"),
        "候选区域数（像素计）": job.get("region_count"),
        "人工复核记录数": (lineage.get("reviews") or {}).get("total"),
    })


def _report_complete(lineage: dict) -> bool:
    pages=[lineage.get('reviews') or {}]
    if lineage.get('registered'):
        cur=(lineage.get('registration') or {}).get('current') or {}
        pages += [(cur.get('quality') or {}).get('checks') or {},
                  cur.get('dispositions') or {},cur.get('history') or {}]
    return all(p.get('truncated') is False and p.get('total')==p.get('returned') for p in pages)


def _lineage_report(lineage: dict) -> bytes:
    payload = {"boundary": _BOUNDARY, "complete":_report_complete(lineage), "lineage": lineage}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def render(client: RecognitionClient, job: dict, regions: list) -> None:
    """在成功结果里渲染治理登记与血缘面板。regions 为该作业候选区域列表。"""
    job_id = job["job_id"]
    st.markdown("### 治理登记与血缘追溯")
    st.caption(_BOUNDARY)

    flash = st.session_state.pop(f"{_NS}_flash", None)
    if flash:
        st.success(flash)

    try:
        lineage = client.get_lineage(job_id)
    except ApiError as exc:
        if exc.category in ("offline", "timeout", "badjson", "protocol"):
            st.warning(f"暂时无法读取治理血缘：{exc.message}")
        else:
            st.error(f"读取治理血缘失败（{exc.category}）：{exc.message}")
        st.caption("血缘读取失败时不显示登记内容；请恢复服务后重新刷新。")
        return

    if lineage.get("registered"):
        _render_registered(lineage)
    else:
        _render_unregistered(lineage)

    st.markdown("**登记操作**")
    if not lineage.get("registered"):
        if st.button("登记输入资料", key=f"{_NS}_register_{job_id}", type="primary"):
            _do_register(client, job_id)
    else:
        st.caption("本作业已登记；同一作业固定一条登记单，不重复登记。")

    pending = st.session_state.get(f"{_NS}_pending") or {}
    if pending.get(job_id):
        st.info("上次登记结果尚未确认，重试会保持同一请求号。")
        if st.button("重试提交（同一请求号）", key=f"{_NS}_retry_{job_id}"):
            _do_register(client, job_id)

    if _report_complete(lineage):
        st.download_button(
            "下载治理血缘 JSON（UTF-8）", _lineage_report(lineage),
            file_name=f"lineage-job-{job_id}.json", mime="application/json",
            key=f"{_NS}_dl_{job_id}",
            on_click="ignore",
        )
    else:
        st.warning('血缘明细超过单次读取上限，当前展示已截断；暂不提供完整JSON下载。原记录仍保留。')
