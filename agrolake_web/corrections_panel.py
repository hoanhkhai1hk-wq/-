"""整改复跑与前后对照面板（DEV-20261015-016）。

放在第 8 页“田块识别”的成功结果里，只通过 :class:`RecognitionClient` 调用本机识别
服务接口，**不直接写识别数据库、不加载模型、不触发推理**。

边界与规则：
- 只在“已成功且已登记治理血缘、且存在最新 questioned/rejected 人工意见”后由用户
  显式创建一条**不可覆盖**的修订，并另起一个**同模型**新作业（服务端行为）。
- 只有用户显式点击“创建整改并复跑”才 POST；普通刷新/切任务/下载/查看历史都只 GET。
- “已提交”不等于“已识别成功”：新作业真实排队/运行/失败/中断/成功都如实显示，
  成功仍要求五项产物校验通过；失败/中断不显示此前成功图。
- 幂等状态按作业独立：只保留“每个作业的一个当前意图”；当前内容相同重复点击复用同一
  请求号，确认后修改内容（含 A→B→A）一律新号/新版本（不使用历史 hash→UUID 表）。
- 回执未知（离线/超时/协议/坏JSON/HTTP 5xx，503 可已 prepared）必须锁住该作业的新意图，
  保留原请求号与 payload；同父存在 prepared 时禁新建，先“确认原提交（同号）”。
- 创建入口要求父作业已登记治理血缘：未登记（或无法确认）时隐藏创建表单并给出登记下一步，
  历史与来源仍可查看。
- 人工意见不是真值；不报精度、平方米或真实田块数；“自动修复成功”不成立。
  same_total_pixels 只是数量相同，不代表同形状/空间对应；comparable_pixel_space 只在
  相同输入字节时为真，不同输入未核实配准。
"""

from __future__ import annotations

import hashlib
import json
import base64
import uuid

import streamlit as st

from agrolake_service import config as svc

from .client import ApiError, RecognitionClient

_NS = "rec_corr"

_MAX_FILENAME = 255

_STATUS_ZH = {
    "queued": "排队中", "running": "运行中", "succeeded": "已成功",
    "failed": "已失败", "interrupted": "已中断", "prepared": "待确认", "linked": "已链接",
}
_DECISION_ZH = {"accepted": "可用", "questioned": "存疑待核实", "rejected": "不采用"}

_BOUNDARY = (
    "整改只是记录人工问题后的不可覆盖修订，并另起一个同模型新作业：人工意见不是真值，"
    "不证明问题自动解决或精度提升，也不代表原 TIFF 通过 P0 质检；原输入/模型/掩膜/意见/"
    "治理快照一律不改。像素计数只表达事实；“合计相同”不代表形状或空间对应，不同输入未核实"
    "配准，不作逐像素或精度、平方米比较。"
)

_HONEST_NOTE = (
    "本面板不显示旧成功图，也不把“已提交/排队/运行”当作“已识别成功”。"
)


def _current() -> dict:
    """每个作业“当前意图” -> {"intent":..., "rid":...}（不是历史 hash 映射）。"""
    return st.session_state.setdefault(f"{_NS}_current", {})


def _pending() -> dict:
    """每个作业上次未确认的提交（含原请求号与原 payload），确认后释放。"""
    return st.session_state.setdefault(f"{_NS}_pending", {})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _intent(job_id: int, review_id: int, operator: str, reason: str,
            changes: dict, replacement_sha: str | None, replacement_name: str | None) -> str:
    """当前意图指纹：同字节换名（相对父输入仍有真实变化）应视为新意图，不能复用旧号。"""
    parts = [str(job_id), str(review_id), operator.strip(), reason.strip(),
             json.dumps(changes, ensure_ascii=False, sort_keys=True),
             replacement_sha or "", replacement_name or ""]
    return _sha("\x1f".join(parts))


def _set_flash(job_id: int, message: str) -> None:
    """成功/确认提示按作业隔离存储，切换作业时不显示其它作业的提示。"""
    st.session_state.setdefault(f"{_NS}_flash", {})[job_id] = message


def _problem_reviews(client: RecognitionClient, job_id: int) -> list[dict]:
    """返回该作业最新的 questioned/rejected 意见（任务整体与各区域）。"""
    latest = client.list_reviews(job_id).get("latest") or {}
    found: list[dict] = []
    task = latest.get("task")
    if task is not None and task.get("decision") in ("questioned", "rejected"):
        found.append({"target": "任务整体", "field_id": None,
                      "pixels": None, "review": task})
    for key, row in (latest.get("regions") or {}).items():
        if row.get("decision") in ("questioned", "rejected"):
            found.append({"target": f"区域 {key}", "field_id": int(key),
                          "pixels": None, "review": row})
    return found


def _label(item: dict) -> str:
    review = item["review"]
    decision = _DECISION_ZH.get(review.get("decision"), review.get("decision"))
    return f"{item['target']} · {decision} · 记录#{review.get('review_id')}"


def _parent_changes(job: dict, source: str, scene_id: str, source_link: str,
                    license_status: str) -> dict:
    """只收集相对父作业当前值真正规范化变化的字段。"""
    changes: dict = {}
    current = {
        "source": job.get("source"), "scene_id": job.get("scene_id"),
        "source_link": job.get("source_link"), "license_status": job.get("license_status"),
    }
    provided = {"source": source.strip(), "scene_id": scene_id.strip(),
                "source_link": source_link.strip(), "license_status": license_status.strip()}
    for key, value in provided.items():
        # 来源/场景必填；许可/链接可清空（空字符串视为清空 -> None）。
        new_value = value if value else None
        if key in ("source", "scene_id") and new_value is None:
            continue
        if new_value != current.get(key):
            changes[key] = new_value
    return changes


def _is_unknown(exc: ApiError) -> bool:
    """回执未知（离线/超时/协议/坏JSON，或 HTTP 5xx，503 可已 prepared）须锁新意图。"""
    if exc.category in ("timeout", "protocol", "badjson", "offline"):
        return True
    return exc.category == "http" and exc.status is not None and exc.status >= 500


def _do_submit(client: RecognitionClient, job_id: int, *, request_id: str, review_id: int,
               operator: str, reason: str, changes: dict, replacement: dict | None,
               intent: str) -> None:
    pending = _pending()
    try:
        result = client.submit_correction(
            job_id, request_id=request_id, review_id=review_id, operator=operator,
            reason=reason, changes=changes or None, replacement=replacement,
        )
    except ApiError as exc:
        if _is_unknown(exc):
            # 保留原请求号与原 payload（每作业仅一份），锁住该作业的新意图，
            # 直到用户显式“确认原提交（同号）”。
            pending[job_id] = {
                "intent": intent, "rid": request_id, "review_id": review_id,
                "operator": operator, "reason": reason, "changes": changes,
                "replacement": replacement,
            }
            st.warning(
                "整改提交结果尚未确认。可稍后刷新，或点击“确认原提交（同号）”；"
                "服务端按请求号幂等，不会重复创建或重复推理。"
            )
            return
        st.error(f"整改创建失败（{exc.category}）：{exc.message}")
        return
    pending.pop(job_id, None)
    state = result.get("submission_state")
    if state == "linked":
        _set_flash(job_id, (
            f"已创建整改 #{result['correction_id']}（修订 {result['revision_no']}），"
            f"新作业 #{result.get('child_job_id')} 已提交；请在下方查看真实排队/运行/结果。"
        ))
    else:
        _set_flash(job_id, (
            f"已保存整改意图 #{result['correction_id']}（待确认）；子作业尚未确认，"
            "可用“确认原提交（同号）”。"
        ))
    st.rerun()


def _confirm_pending(client: RecognitionClient, job_id: int, listing: dict) -> None:
    """显式确认原提交（同号）：先按只读列表找原 rid，linked 仅确认、prepared 用 resume，
    未找到记录时才明确用原 payload 同号重试。"""
    record = _pending().get(job_id)
    if not record:
        return
    rid = record["rid"]
    match = next((r for r in (listing.get("corrections") or [])
                  if r["request_id"] == rid), None)
    try:
        if match is not None and match["submission_state"] == "linked":
            pass  # 服务端已确认链接，无需再次提交。
        elif match is not None:
            client.resume_correction(match["correction_id"], request_id=rid)
        else:
            client.submit_correction(
                job_id, request_id=rid, review_id=record["review_id"],
                operator=record["operator"], reason=record["reason"],
                changes=record["changes"] or None, replacement=record["replacement"],
            )
    except ApiError as exc:
        if _is_unknown(exc):
            st.warning("原提交确认结果尚未确定，请保留原请求号稍后重试。")
        else:
            st.error(f"确认原提交失败（{exc.category}）：{exc.message}")
        return
    _pending().pop(job_id, None)
    _set_flash(job_id, "原提交已按同一请求号确认。")
    st.rerun()


def _render_origin(client: RecognitionClient, origin: dict) -> None:
    st.markdown("**本作业由整改产生（来源记录）**")
    st.write({
        "整改编号": origin["correction_id"],
        "修订号": origin["revision_no"],
        "父作业": origin["parent_job_id"],
        "根作业": origin["root_job_id"],
        "来源问题意见": origin["review_id"],
        "操作者": origin["operator"],
        "原因": origin["reason"],
        "提交态": _STATUS_ZH.get(origin["submission_state"], origin["submission_state"]),
        "人工异常背景": origin["is_synthetic_case"],
    })
    st.caption("这正是“由哪条整改产生”的来源；原父作业的成功图与事实不被覆盖。")


def _render_create(client: RecognitionClient, job: dict, problems: list[dict], *,
                   governed: bool | None, listing: dict) -> None:
    job_id = job["job_id"]
    st.markdown("**创建不可覆盖的修订并复跑**")

    # 治理前置：从实际 GET 治理血缘判断，不靠猜字段。未登记（或无法确认）一律隐藏创建表单。
    if governed is None:
        st.warning(
            "暂时无法确认本作业的治理登记状态，暂不提供创建整改入口；请恢复服务后刷新。"
        )
        return
    if not governed:
        st.info(
            "本作业尚未完成“治理登记”，暂不能创建整改。请先在上方“治理登记与血缘追溯”"
            "点击“登记输入资料”，登记成功后再回到这里创建整改。历史与来源记录仍可在下方查看。"
        )
        return
    if not problems:
        st.info(
            "本作业还没有最新“存疑待核实/不采用”的人工意见，暂不能创建整改。"
            "请先在上方“人工复核记录”为本任务或某个区域填写结论后再回来。"
        )
        return

    labels = [_label(p) for p in problems]
    pick = st.selectbox("选择作为整改来源的问题意见（当前最新）", labels,
                        key=f"{_NS}_pick_{job_id}")
    chosen = problems[labels.index(pick)]
    review = chosen["review"]
    st.caption(
        f"来源意见：{_DECISION_ZH.get(review.get('decision'), review.get('decision'))} · "
        f"署名 {review.get('reviewer')} · {review.get('note')}"
    )

    operator = st.text_input("操作者署名（必填）", key=f"{_NS}_operator_{job_id}", max_chars=80)
    reason = st.text_area("整改原因（必填）", key=f"{_NS}_reason_{job_id}", max_chars=2000)

    with st.expander("修订来源/场景（默认复用父作业，只提交真正变化）", expanded=False):
        source = st.text_input("影像来源（必填）", value=job.get("source") or "",
                               key=f"{_NS}_source_{job_id}")
        col_a, col_b = st.columns(2)
        scene_id = col_a.text_input("场景编号（必填）", value=job.get("scene_id") or "",
                                    key=f"{_NS}_scene_{job_id}")
        source_link = col_b.text_input("来源链接（可清空）", value=job.get("source_link") or "",
                                       key=f"{_NS}_link_{job_id}")
        license_status = st.text_input("许可/授权状态（可清空）",
                                       value=job.get("license_status") or "",
                                       key=f"{_NS}_license_{job_id}")
    st.caption("留空来源/场景不会创建整改；许可/链接留空表示清空该字段。")

    up = st.file_uploader("可选：上传替换的 8 波段 TIFF（≤40 MiB）",
                          type=["tif", "tiff"], key=f"{_NS}_upload_{job_id}")
    replacement = None
    replacement_sha = None
    oversize = False
    if up is not None:
        if up.size > svc.MAX_CONTENT_BYTES:
            oversize = True
            st.error("替换文件超过 40 MiB 上限，不能提交本次整改；请选择较小的 TIFF。")
        else:
            data = up.getvalue()
            replacement = {"filename": up.name,
                           "content_base64": base64.b64encode(data).decode("ascii")}
            replacement_sha = hashlib.sha256(data).hexdigest()
            st.caption(f"将替换为：{up.name}（{len(data)} 字节）")

    changes = _parent_changes(job, source, scene_id, source_link, license_status)
    if not changes and replacement is None:
        st.caption("当前未检测到任何真实变化；至少需要一处来源/场景变化或替换上传。")

    # 未知回执锁新意图 / 同父存在 prepared 时禁新建：只允许“确认原提交（同号）”。
    prepared = [r for r in (listing.get("corrections") or [])
                if r["submission_state"] == "prepared"]
    locked = bool(prepared) or job_id in _pending()

    if st.button("创建整改并复跑", key=f"{_NS}_create_{job_id}", type="primary",
                 disabled=locked):
        if not operator.strip():
            st.error("请填写操作者署名。")
        elif not reason.strip():
            st.error("请填写整改原因。")
        elif not source.strip():
            st.error("来源为必填项，请填写来源。")
        elif not scene_id.strip():
            st.error("场景为必填项，请填写场景编号。")
        elif oversize:
            st.error("替换文件超过 40 MiB 上限，本次整改不会提交"
                     "（不会忽略替换文件却只提交元数据）。")
        elif not changes and replacement is None:
            st.error("整改必须至少带来一处真实变化（来源/场景/许可/链接），或上传替换文件。")
        else:
            intent = _intent(job_id, review["review_id"], operator, reason, changes,
                             replacement_sha, replacement["filename"] if replacement else None)
            current = _current()
            cur = current.get(job_id)
            if cur is not None and cur.get("intent") == intent:
                rid = cur["rid"]                     # 当前内容相同：复用同号/同子作业
            else:
                rid = uuid.uuid4().hex               # 修改内容（含 A→B→A）：必须新号/新版本
                current[job_id] = {"intent": intent, "rid": rid}
            _do_submit(client, job_id, request_id=rid, review_id=review["review_id"],
                       operator=operator.strip(), reason=reason.strip(), changes=changes,
                       replacement=replacement, intent=intent)

    pending = _pending().get(job_id)
    if locked:
        st.warning(
            "本作业已有待确认的整改（或上次提交结果尚未确认）：请先“确认原提交（同号）”，"
            "不要另建新号；普通刷新/切任务不会自动提交。"
        )
    if pending and not prepared:
        st.caption("上次提交使用了原请求号，但未在本作业记录中找到；可按原 payload 同号重试。")
        if st.button("确认原提交（同号）", key=f"{_NS}_confirm_{job_id}"):
            _confirm_pending(client, job_id, listing)


def _render_prepared(client: RecognitionClient, job_id: int, records: list[dict]) -> None:
    prepared = [r for r in records if r["submission_state"] == "prepared"]
    if not prepared:
        return
    st.markdown("**待确认的整改（记录已保存，子任务尚未确认）**")
    st.caption(
        "整页重载丢失原表单后，可据记录“确认原提交（同号）”；父作业成功登记后"
        "仍可在原面板登记、复核子作业。复用旧子任务不叫重新推理。"
    )
    for rec in prepared:
        cols = st.columns([3, 1])
        cols[0].write({
            "整改编号": rec["correction_id"], "修订号": rec["revision_no"],
            "来源问题意见": rec["review_id"], "操作者": rec["operator"],
            "原因": rec["reason"], "错误": rec["error"] or "（无）",
            "原请求号": rec["request_id"],
        })
        if cols[1].button("确认原提交（同号）", key=f"{_NS}_resume_{rec['correction_id']}"):
            try:
                client.resume_correction(rec["correction_id"], request_id=rec["request_id"])
            except ApiError as exc:
                if _is_unknown(exc):
                    st.warning("确认结果尚未确定，请保留原请求号稍后重试。")
                else:
                    st.error(f"确认整改失败（{exc.category}）：{exc.message}")
                return
            _pending().pop(job_id, None)
            _set_flash(job_id, f"整改 #{rec['correction_id']} 已确认。")
            st.rerun()


def _render_comparison(detail: dict) -> None:
    rec = detail["correction"]
    comp = detail["comparison"]
    problem = detail["snapshot"].get("problem_review") or {}
    parent = detail["parent"]
    child = detail["child"]

    st.markdown("**事实对照（只读快照）**")
    st.write({
        "修订号": rec["revision_no"],
        "原问题意见": (f"{_DECISION_ZH.get(problem.get('decision'), problem.get('decision'))}"
                   f" · {problem.get('reviewer')} · {problem.get('note')}"),
        "操作者/原因": f"{rec['operator']} · {rec['reason']}",
        "人工异常背景": rec["is_synthetic_case"],
        "子任务状态": _STATUS_ZH.get(comp.get("child_state"), comp.get("child_state")),
        "子任务是否成功": comp.get("child_succeeded"),
        "输入字节相同": comp.get("same_input_sha256"),
        "模型相同": comp.get("same_model_sha256"),
        "修订字段": comp.get("changed_fields"),
        "输入字节变化": comp.get("bytes_changed"),
    })
    st.caption(_BOUNDARY)

    left, right = st.columns(2)
    for col, name, summary, reg in ((left, "父作业", parent, rec["parent_job_id"]),
                                    (right, "子作业", child, rec.get("child_job_id"))):
        with col:
            st.markdown(f"**{name}**")
            if summary is None:
                st.caption("尚未创建子作业。")
                continue
            st.write({
                "任务编号": summary["job_id"],
                "状态": _STATUS_ZH.get(summary["status"], summary["status"]),
                "来源": summary["source"],
                "场景": summary["scene_id"],
                "输入SHA256": (summary["input_sha256"] or "")[:16] + "…",
                "模型SHA256": (summary["model_sha256"] or "")[:16] + "…" if summary["model_sha256"] else "—",
                "治理登记": (summary.get("governance") or {}).get("version_id"),
                "证据完整": summary["evidence_complete"],
            })
            latest = summary.get("latest_task_review")
            if latest is None:
                st.caption(f"当前{name}整体意见：未复核")
            else:
                st.caption(
                    f"当前{name}整体意见：{latest.get('decision_zh')} · {latest.get('reviewer')} "
                    f"· {latest.get('note')}"
                )

    nodata = comp.get("nodata") or {}

    def _nd(block) -> str:
        if not isinstance(block, dict):
            return "待完成/缺失"
        return f"{block['pixels']} / {block['denominator']}（{block['ratio'] * 100:.2f}%）"

    st.markdown(
        "**NoData（255 类）像素 / 四类合计分母（占比）**："
        f"父 {_nd(nodata.get('parent'))}；子 {_nd(nodata.get('child'))}。"
        "缺证据时显示“待完成/缺失”，不以 0 冒充；占比只是事实计数，不是准确率。"
    )

    pixel = comp.get("class_pixel_comparison")
    if pixel is None:
        st.info("对照像素计数待完成或记录缺失（子任务未成功或校验记录不齐），不以 0 冒充完成。")
        st.caption(f"记录缺失说明：{comp.get('child_state')}；仅计数与占比，不报精度。")
        return
    rows = []
    for key, meaning in (("0", "背景"), ("1", "田块内部"), ("2", "边界"), ("255", "无效")):
        rows.append({
            "类别": f"{key}（{meaning}）",
            "父像素": pixel["parent"][key],
            "子像素": pixel["child"][key],
            "差值(child-parent)": str(pixel["delta"][key]),
        })
    rows.append({"类别": "合计", "父像素": pixel["parent"]["total"],
                 "子像素": pixel["child"]["total"], "差值(child-parent)": "—"})
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        f"comparable_pixel_space={pixel['comparable_pixel_space']}（仅相同输入字节时为真）；"
        f"same_total_pixels={pixel['same_total_pixels']}（只是数量相同，不代表形状或空间对应）。"
        f"{(comp.get('nodata') or {}).get('note') or ''}"
    )


def _render_records(client: RecognitionClient, job_id: int, listing: dict) -> None:
    records: list[dict] = list(listing.get("corrections") or [])
    origin = listing.get("origin")
    if origin is not None:
        _render_origin(client, origin)
        records.append(origin)
    if not records:
        st.caption("暂无整改历史。")
        return

    _render_prepared(client, job_id, records)

    st.markdown("**整改历史与对照（不可覆盖，按整改编号递增）**")
    show = [{
        "整改编号": r["correction_id"], "修订号": r["revision_no"],
        "来源问题意见": r["review_id"], "父作业": r["parent_job_id"],
        "子作业": r["child_job_id"], "提交态": _STATUS_ZH.get(r["submission_state"], r["submission_state"]),
        "操作者": r["operator"], "原因": r["reason"],
    } for r in sorted(records, key=lambda x: x["correction_id"])]
    st.dataframe(show, width="stretch", hide_index=True)

    ids = [r["correction_id"] for r in sorted(records, key=lambda x: x["correction_id"])]
    pick = st.selectbox("查看某条整改的前后对照", ids, key=f"{_NS}_detail_pick_{job_id}")
    try:
        detail = client.get_correction(int(pick))
    except ApiError as exc:
        st.warning(f"读取整改详情失败（{exc.category}）：{exc.message}")
        return
    _render_comparison(detail)

    # 子任务导航：安全回调只设置现有 rec_selected_job，不直接改动已创建的 rec_history 控件。
    child_id = (detail.get("correction") or {}).get("child_job_id")
    if child_id is not None and child_id != job_id:
        st.caption(f"子作业 #{child_id} 的真实排队/运行/结果可在上方“选择历史任务查看”中选择 "
                   f"#{child_id} 查看。")
        if st.button(f"查看子任务 #{child_id}", key=f"{_NS}_goto_{job_id}_{pick}"):
            st.session_state["rec_selected_job"] = int(child_id)
            st.rerun()

    report = json.dumps(
        {"boundary": _BOUNDARY, "honest_note": _HONEST_NOTE, "detail": detail},
        ensure_ascii=False, indent=2,
    ).encode("utf-8")
    st.download_button(
        "下载整改对照 JSON（UTF-8）", report,
        file_name=f"correction-{pick}.json", mime="application/json",
        key=f"{_NS}_dl_{job_id}_{pick}", on_click="ignore",
    )


def _governance_registered(client: RecognitionClient, job_id: int) -> bool | None:
    """从实际 GET 治理血缘判断父作业是否已登记；读取失败返回 None（不猜字段）。"""
    try:
        lineage = client.get_lineage(job_id)
    except ApiError:
        return None
    return bool(lineage.get("registered"))


def render(client: RecognitionClient, job: dict, regions: list) -> None:
    """在成功结果里渲染整改复跑面板（调用方已确认五项产物校验通过）。"""
    job_id = job["job_id"]
    st.markdown("### 整改复跑与前后对照")
    st.caption(_BOUNDARY)

    flash = st.session_state.setdefault(f"{_NS}_flash", {}).pop(job_id, None)
    if flash:
        st.success(flash)

    try:
        listing = client.list_corrections(job_id)
    except ApiError as exc:
        if exc.category in ("offline", "timeout", "badjson", "protocol"):
            st.warning(f"暂时无法读取整改列表：{exc.message}")
        else:
            st.error(f"读取整改列表失败（{exc.category}）：{exc.message}")
        return

    try:
        problems = _problem_reviews(client, job_id)
    except ApiError as exc:
        st.warning(f"暂时无法读取人工意见，暂不提供创建入口：{exc.message}")
        problems = []

    governed = _governance_registered(client, job_id)
    _render_create(client, job, problems, governed=governed, listing=listing)
    _render_records(client, job_id, listing)


def render_origin_only(client: RecognitionClient, job: dict) -> None:
    """失败/中断任务：只读展示“由哪条整改产生”与对照，不显示旧成功图。"""
    job_id = job["job_id"]
    try:
        listing = client.list_corrections(job_id)
    except ApiError as exc:
        if exc.category in ("offline", "timeout", "badjson", "protocol"):
            st.caption(f"暂时无法读取整改来源：{exc.message}")
        return
    origin = listing.get("origin")
    if origin is None:
        return
    _render_origin(client, origin)
    try:
        detail = client.get_correction(origin["correction_id"])
    except ApiError:
        return
    _render_comparison(detail)
