"""田块识别结果的人工复核面板（DEV-20261015-013）。

放在第 8 页“田块识别”的成功结果里，只通过 :class:`RecognitionClient` 调用本机
识别服务接口，**不直接写识别数据库、不加载模型**。

边界与规则：
- 只记录人工填写的声明，不改掩膜/类别数/模型输入输出，不声称真值、精度提升或真实面积；
  复核人只是本机署名，不是已认证身份。
- 只有成功作业、且五项产物校验通过时本面板才交给上层渲染（由调用方控制）。
- 只有用户显式点击“保存”才 POST；普通刷新/切历史/下载都不写。
- 每次“任务、区域、署名、结论或原因”变化都算新意图，使用新的请求号追加一条；
  同一意图重复点击复用同一请求号，服务端按请求号幂等，不会重复写入。
- 回执未知（超时/网络/坏 JSON）时提示“重试提交（同一请求号）”，不悄悄换号。
- 历史不可覆盖；下载的 CSV/JSON 都注明人工声明与像素/无真值边界。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid

import streamlit as st

from .client import (
    MAX_NOTE_LEN,
    MAX_REVIEW_PAGE,
    MAX_REVIEWER_LEN,
    ApiError,
    RecognitionClient,
)

_NS = "rec_review"

# 结论中文显示与顺序（与 agrolake_service.reviews 一致）。
_DECISION_LABELS = [
    ("accepted", "可用（人工认为可用）"),
    ("questioned", "存疑待核实"),
    ("rejected", "不采用"),
]
_LABEL_TO_DECISION = {label: value for value, label in _DECISION_LABELS}

# CSV 防公式注入：以下开头字符会被表格软件当成公式，导出时前置单引号；原值保留在 JSON。
_FORMULA_PREFIX = ("=", "+", "-", "@", "\t", "\r")
_MAX_EXPORT_PAGES = 50  # 每页200条，且服务端每任务最多500条；所有遍历均有界。


def _boundary_note() -> str:
    return (
        "人工复核只是本机填写的署名声明，不是真值、不是精度证据；"
        "不改掩膜/类别数/模型输入输出，不代表真实田块数或平方米面积；"
        "没有独立真值，不报告准确率。"
    )


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    probe = text.lstrip(' \t\r\n\ufeff\u00a0')
    if text[:1] in _FORMULA_PREFIX or probe[:1] in _FORMULA_PREFIX:
        return "'" + text
    return text


def _intent(job_id: int, field_id, decision: str, reviewer: str, note: str) -> str:
    parts = [
        str(job_id),
        "" if field_id is None else str(field_id),
        decision,
        reviewer.strip(),
        note.strip(),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _fetch_all(client: RecognitionClient, job_id: int) -> tuple[list, bool]:
    """返回（记录列表, 是否取全）。分页遍历，超上限时明确未取全。"""
    rows: list = []
    after = None
    until = None
    expected_total = None
    for _ in range(_MAX_EXPORT_PAGES):
        page = client.list_reviews(job_id, after_id=after, limit=MAX_REVIEW_PAGE, until_id=until)
        batch = page.get('reviews') or []
        pagination = page.get("pagination") or {}
        bound = pagination.get('until_id')
        if type(bound) is not int or bound < 0 or (until is not None and until != bound):
            raise ApiError('protocol', '无法确认复核导出的快照边界。')
        until = bound
        if expected_total is None:
            expected_total = page.get('total')
        if page.get('total') != expected_total:
            raise ApiError('protocol', '复核导出的快照计数发生变化。')
        if any(r.get('job_id') != job_id or type(r.get('review_id')) is not int for r in batch):
            raise ApiError('protocol', '复核导出混入了其他任务的记录。')
        cursor = after or 0
        for row in batch:
            if not cursor < row['review_id'] <= until:
                raise ApiError('protocol', '复核导出分页没有向前移动。')
            cursor = row['review_id']
        rows.extend(batch)
        if not pagination.get("has_more"):
            return rows, len(rows) == expected_total
        after = pagination.get('next_after_id')
        if not batch or type(after) is not int or after != cursor:
            raise ApiError('protocol', '复核分页缺少有效后续游标。')
    return rows, False


def _to_csv(rows: list) -> bytes:
    buffer = io.StringIO()
    cutoff = max((r['review_id'] for r in rows), default=0)
    buffer.write(f'# 导出快照截至记录号{cutoff}，共{len(rows)}条。' + _boundary_note() + '\n')
    buffer.write(
        "# CSV 转义：以 = + - @ 或制表/回车开头的文本，导出时前加单引号 ' 防止被表格"
        "软件当公式执行；原值请以 JSON 导出为准。\n"
    )
    writer = csv.writer(buffer)
    writer.writerow([
        "review_id", "job_id", "target", "field_id", "decision", "decision_zh",
        "reviewer", "note", "request_id", "input_sha256", "model_sha256",
        "run_seq", "created_at",
    ])
    for row in rows:
        writer.writerow([
            row.get("review_id"),
            row.get("job_id"),
            "任务整体" if row.get("field_id") is None else "候选区域",
            row.get("field_id"),
            row.get("decision"),
            row.get("decision_zh"),
            _csv_safe(row.get("reviewer")),
            _csv_safe(row.get("note")),
            _csv_safe(row.get("request_id")),
            row.get("input_sha256"),
            row.get("model_sha256"),
            row.get("run_seq"),
            row.get("created_at"),
        ])
    return buffer.getvalue().encode("utf-8")


def _to_json(job_id: int, rows: list, complete: bool) -> bytes:
    payload = {
        "job_id": job_id,
        "boundary": _boundary_note(),
        "csv_escaping": (
            "CSV 中以 = + - @ 等开头的文本会前加单引号；本文件保留原始值。"
        ),
        "complete": complete,
        "count": len(rows),
        "snapshot_until_review_id": max((r['review_id'] for r in rows), default=0),
        "reviews": rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _do_submit(client: RecognitionClient, job_id: int, field_id, decision: str,
               reviewer: str, note: str) -> None:
    """执行一次复核提交，并把结果写入会话状态。相同意图复用同一请求号。"""
    intent = _intent(job_id, field_id, decision, reviewer, note)
    ids = st.session_state.setdefault(f"{_NS}_ids", {})
    rid = ids.get(intent)
    if rid is None:
        rid = uuid.uuid4().hex
        ids[intent] = rid
    pending = st.session_state.setdefault(f"{_NS}_pending", {})
    confirmed = st.session_state.setdefault(f'{_NS}_confirmed', {})

    try:
        if intent in confirmed and not pending.get(intent):
            latest = client.list_reviews(job_id)['latest']
            current = latest['task'] if field_id is None else latest['regions'].get(str(field_id))
            if current is None or current['review_id'] != confirmed[intent]:
                # A→B→A是新一轮人工声明；紧接着的重复点击仍复用本轮请求号。
                rid = uuid.uuid4().hex
                ids[intent] = rid
        result = client.submit_review(
            job_id, decision=decision, reviewer=reviewer, note=note,
            request_id=rid, field_id=field_id,
        )
    except ApiError as exc:
        if exc.category in ("timeout", "protocol", "badjson", "offline"):
            pending[intent] = True
            st.warning(
                "复核结果尚未确认。可点击“重试提交（同一请求号）”；"
                "服务端按请求号幂等，不会重复写入。"
            )
            return
        st.error(f"复核保存失败（{exc.category}）：{exc.message}")
        return

    pending.pop(intent, None)
    confirmed[intent] = result['review_id']
    if result.get("created"):
        st.session_state[f"{_NS}_flash"] = "已保存人工复核意见（新建一条）。"
    else:
        st.session_state[f"{_NS}_flash"] = "已存在同一复核意见，本次未重复写入。"
    st.rerun()


def _render_form(client: RecognitionClient, job_id: int, *, target_label: str,
                 field_id, key_suffix: str) -> None:
    """渲染一个目标（任务整体 或 某个区域）的填写+保存+同号重试。"""
    decision_label = st.selectbox(
        f"{target_label}结论", [label for _, label in _DECISION_LABELS],
        key=f"{_NS}_decision_{key_suffix}",
    )
    reviewer = st.text_input(
        f"{target_label}复核人（署名，1–{MAX_REVIEWER_LEN} 字）",
        key=f"{_NS}_reviewer_{key_suffix}",
        max_chars=MAX_REVIEWER_LEN,
    )
    note = st.text_area(
        f"{target_label}具体原因（1–{MAX_NOTE_LEN} 字，必填）",
        key=f"{_NS}_note_{key_suffix}",
        max_chars=MAX_NOTE_LEN,
    )
    decision = _LABEL_TO_DECISION[decision_label]

    if st.button(f"保存{target_label}复核意见", key=f"{_NS}_save_{key_suffix}"):
        if not reviewer.strip():
            st.error("请填写复核人署名。")
        elif not note.strip():
            st.error("请填写具体原因。")
        else:
            _do_submit(client, job_id, field_id, decision, reviewer, note)

    intent = _intent(job_id, field_id, decision, reviewer, note)
    pending = st.session_state.get(f"{_NS}_pending") or {}
    if pending.get(intent):
        st.info("上次复核结果尚未确认，重试会保持同一请求号。")
        if st.button(f"重试提交（同一请求号）", key=f"{_NS}_retry_{key_suffix}"):
            if reviewer.strip() and note.strip():
                _do_submit(client, job_id, field_id, decision, reviewer, note)


def _render_current(latest: dict, regions: list) -> None:
    task = latest.get("task")
    region_latest = latest.get("regions") or {}
    if task is None and not region_latest:
        st.caption("当前还没有复核意见（显示为“未复核”）。")
    else:
        rows = []
        if task is not None:
            rows.append({
                "目标": "任务整体", "当前结论": task.get("decision_zh"),
                "复核人": task.get("reviewer"), "原因": task.get("note"),
                "记录号": str(task.get("review_id")), "时间": task.get("created_at"),
            })
        for region in regions:
            fid = str(region["field_id"])
            item = region_latest.get(fid)
            rows.append({
                "目标": f"区域 {fid}",
                "当前结论": item.get("decision_zh") if item else "未复核",
                "复核人": item.get("reviewer") if item else "",
                "原因": item.get("note") if item else "",
                "记录号": str(item.get("review_id")) if item else "",
                "时间": item.get("created_at") if item else "",
            })
        st.markdown("**当前意见（按目标取最新一条）**")
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_history(rows: list, complete: bool) -> None:
    with st.expander("复核历史（不可覆盖，按记录号递增）"):
        if not rows:
            st.caption("暂无历史记录。")
            return
        show = [{
            "记录号": str(r.get("review_id")),
            "目标": "任务整体" if r.get("field_id") is None else f"区域 {r.get('field_id')}",
            "结论": r.get("decision_zh"),
            "复核人": r.get("reviewer"),
            "原因": r.get("note"),
            "请求号": r.get("request_id"),
            "输入SHA256": (r.get("input_sha256") or "")[:16] + "…",
            "模型SHA256": (r.get("model_sha256") or "")[:16] + "…",
            "运行序号": "" if r.get("run_seq") is None else str(r.get("run_seq")),
            "时间": r.get("created_at"),
        } for r in rows]
        st.dataframe(show, width="stretch", hide_index=True)
        if not complete:
            st.warning("记录数超过单次导出上限，下方导出未取全，请知悉。")


def render(client: RecognitionClient, job: dict, regions: list) -> None:
    """在成功结果里渲染人工复核面板。regions 为该作业的候选区域列表。"""
    job_id = job["job_id"]
    st.markdown("### 人工复核记录")
    st.caption(
        "记录人工对本次识别结果的意见：可对**任务整体**或**某个候选区域**填写。"
        "人工意见不改模型输出，不代表真值、精度提升或真实面积；复核人只是本机署名。"
    )
    st.caption('本地原型每个任务最多保存500条复核历史；每次导出取开始读取时的完整快照。')

    flash = st.session_state.pop(f"{_NS}_flash", None)
    if flash:
        st.success(flash)

    try:
        page = client.list_reviews(job_id)
    except ApiError as exc:
        if exc.category in ("offline", "timeout", "badjson", "protocol"):
            st.warning(f"暂时无法读取复核记录：{exc.message}")
        else:
            st.error(f"读取复核记录失败（{exc.category}）：{exc.message}")
        return

    latest = page.get("latest") or {}
    total = page.get("total", len(page.get("reviews") or []))
    try:
        all_rows, complete = _fetch_all(client, job_id)
    except ApiError as exc:
        st.warning(f"读取全部复核记录失败（{exc.category}）：{exc.message}")
        all_rows, complete = page.get("reviews") or [], False

    _render_current(latest, regions)

    st.markdown("**填写任务整体意见**")
    _render_form(client, job_id, target_label="任务整体", field_id=None,
                 key_suffix=f"task_{job_id}")

    if regions:
        st.markdown("**填写某个候选区域意见（与任务整体分开）**")
        options = [f"区域 {r['field_id']}（{r['pixels']} 像素）" for r in regions]
        pick = st.selectbox("选择候选区域", options, key=f"{_NS}_region_{job_id}")
        field_id = regions[options.index(pick)]["field_id"]
        _render_form(client, job_id, target_label=f"区域 {field_id}",
                     field_id=field_id, key_suffix=f"region_{job_id}_{field_id}")
    else:
        st.caption("本任务没有候选区域可单独复核；可只记录任务整体意见。")

    _render_history(all_rows, complete)

    st.markdown("**导出复核记录（含人工声明与像素/无真值边界）**")
    if not complete:
        st.warning('未能取全复核历史，暂不提供下载。请恢复服务后重新刷新。')
        return
    st.caption(f"共 {len(all_rows)} 条记录。CSV 中以避免公式执行的转义写入，原值见 JSON。")
    cols = st.columns(2)
    cols[0].download_button(
        "下载复核记录 CSV", _to_csv(all_rows),
        file_name=f"reviews-job-{job_id}.csv", mime="text/csv",
        key=f"{_NS}_dl_csv_{job_id}",
    )
    cols[1].download_button(
        "下载复核记录 JSON", _to_json(job_id, all_rows, complete),
        file_name=f"reviews-job-{job_id}.json", mime="application/json",
        key=f"{_NS}_dl_json_{job_id}",
    )
