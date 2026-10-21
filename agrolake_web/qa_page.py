"""DeepSeek 数据问答与报告解释页面（T15）。

页面只做三件事：给事实、给回答、给来源。

设计约束：

- 事实查询走**只读连接**（SQLite ``mode=ro``），页面不提供任何改数据、
  改质量状态或自由 SQL 的入口；模型给的是建议，不会自动落到治理库。
- 只有点击「提问」才会发请求；切页、改选项、普通 rerun 都不触发调用。
- 密钥不显示、不入库、不进日志：页面只显示"配没配、配在哪"，不显示内容。
- 任何失败都显示准确状态，本地事实摘要照常可见。
"""
from __future__ import annotations

import hashlib
import json
import re
import streamlit as st

from agrolake import db, deepseek_client as ds, qa_facts, qa_store, rgb_facts, scenes

_NS = "qa"

SCOPE_ORDER = ["batch_review", "scene", "governance_report", "overview", "rgb_job"]

QUICK_QUESTIONS = {
    "batch_review": "当前批次待复核的资料有哪些？分别是什么原因？",
    "scene": "这个场景的质量和识别结果怎么样？还缺哪些资料？",
    "governance_report": "基于现在的实际指标，给出一份治理报告：整体状态、主要问题、下一步建议。",
    "overview": "现在有多少资料、质量通过率和关联覆盖率分别是多少？",
    "rgb_job": "这个 RGB 识别任务的结果和边界怎么样？有哪些需要人工注意的地方？",
}

RGB_STATUS_ZH = {"running": "运行中", "succeeded": "成功", "failed": "失败"}


def _context_key(scope: str, rgb_job_id: str | None, scene_id) -> str:
    """回答绑定上下文：口径 + RGB任务 + 场景。切换任一项后不会串回答。"""
    rgb = rgb_job_id if scope == "rgb_job" else None
    scene = scene_id if scope in ("scene", "governance_report") else None
    return f"{scope}|{rgb or ''}|{scene or ''}"


def _facts_key(bundle) -> str:
    """Bind answers to actual facts, ignoring only the query generation time."""
    snapshot = {k: v for k, v in bundle.snapshot.items() if k != "generated_at"}
    text = re.sub(r"(?<=统计时点 )\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)?",
                  "<查询时点>", bundle.text)
    payload = [snapshot, text, bundle.citations, bundle.gaps]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")).hexdigest()


def _state(key: str, default=None):
    return st.session_state.get(f"{_NS}_{key}", default)


def _set_state(key: str, value) -> None:
    st.session_state[f"{_NS}_{key}"] = value


def _status_style(status: str) -> str:
    """成功用绿、配置/网络用黄、认证与余额用红。"""
    if status == "ok":
        return "success"
    if status in ("not_configured", "timeout", "network_error", "rate_limited"):
        return "warning"
    return "error"


def _render_config(cfg: ds.DeepSeekConfig) -> None:
    ok, note = ds.chat_ready(cfg)
    if ok:
        st.caption(
            f"模型 {cfg.model} · 地址 {cfg.base_url} · 超时 {cfg.timeout:g}s · "
            f"最多 {cfg.max_attempts} 次尝试 · 密钥 {cfg.key_hint}"
        )
    else:
        st.warning(note)


def _render_facts(bundle: qa_facts.FactBundle, expanded: bool = False) -> None:
    with st.expander(f"事实摘要 · {bundle.title}", expanded=expanded):
        st.markdown(
            f"<div class='ag-card'><pre style='white-space:pre-wrap;margin:0;"
            f"font-size:.85rem;line-height:1.6'>{_esc(bundle.text)}</pre></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"事实摘要 SHA256 前 16 位：{qa_store.facts_digest(bundle.text)[:16]}… "
            "下面所有数字都来自这次查询，不是模型推算的。"
        )


def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("本次没有可引用的来源。")
        return
    st.dataframe(
        [{"引用": c.get("label", ""), "来源表": c.get("source", ""),
          "说明": c.get("detail", "")} for c in citations],
        width="stretch", hide_index=True,
    )


def _render_answer(record: dict) -> None:
    style = _status_style(record["status"])
    model_txt = record["model"] or "—"
    served = record.get("response_model") or ""
    if served and served != record["model"]:
        # 服务端可能把别名解析成具体模型，如实并列显示，不偷偷替换
        model_txt = f"{record['model']}（服务端实际 {served}）"
    head = (
        f"状态：{record['status_zh']}（`{record['status']}`）"
        f" · 模型 {model_txt}"
        f" · 耗时 {record['latency_ms'] or 0} ms"
        f" · 尝试 {record['attempts'] or 0} 次"
    )
    if style == "success":
        st.success(head)
        st.markdown(record["answer"])
    else:
        getattr(st, style)(head)
        detail = record.get("error_message") or ""
        if record.get("error_code"):
            detail = f"{detail}（错误码 {record['error_code']}）"
        if detail:
            st.caption(detail)
        if record["status"] == "not_configured":
            st.caption(
                "本地事实摘要与数据库查询不受影响；配置方式见 docs/DeepSeek问答.md。"
            )
        elif record["status"] in ("auth_failed", "insufficient_balance", "invalid_request"):
            st.caption("这类错误重试没有意义，程序不会自动重试。请按提示修正配置后再提问。")
        st.caption("本次没有可用的模型回答，页面不会用模板话术冒充答案。")

    if record.get("total_tokens"):
        st.caption(
            f"token：输入 {record.get('prompt_tokens') or 0} / "
            f"输出 {record.get('completion_tokens') or 0} / "
            f"合计 {record.get('total_tokens') or 0}"
            f" · 服务端请求号 {record.get('provider_request_id') or '—'}"
        )


def _render_history(conn) -> None:
    st.markdown("**最近问答记录（存在 SQLite，不含密钥）**")
    rows = qa_store.recent_runs(conn, limit=10)
    if not rows:
        st.caption("还没有问答记录。")
        return
    st.dataframe(
        [{
            "时间": r["asked_at"],
            "问题": (r["question"] or "")[:40],
            "口径": qa_facts.SCOPES.get(r["scope"], r["scope"]),
            "状态": r["status_zh"],
            "请求模型": r["model"],
            "实际模型": r.get("response_model") or "—",
            "耗时ms": r["latency_ms"],
            "tokens": r["total_tokens"],
        } for r in rows],
        width="stretch", hide_index=True,
    )
    with st.expander("查看某次记录的完整事实快照与回答"):
        pick = st.selectbox(
            "选择 run_id",
            [r["run_id"] for r in rows],
            format_func=lambda x: f"#{x} · {next(r['question'] for r in rows if r['run_id'] == x)[:30]}",
        )
        rec = next(r for r in rows if r["run_id"] == pick)
        st.caption(
            f"事实摘要 SHA256：{rec['facts_sha256']} · 重试过："
            f"{'是' if rec['retried'] else '否'}"
        )
        st.markdown("**当时的事实摘要**")
        st.code(rec["facts_text"], language="text")
        st.markdown("**当时的回答**")
        st.text(rec["answer"] or "（本次没有回答）")


def _rgb_job_options() -> dict[str, str | None]:
    """只读列出 RGB 任务（明确 id/来源/状态）；缺库或为空时正常显示。"""
    listing = rgb_facts.list_jobs()
    if not listing.get("available"):
        return {"（RGB 作业库不可用）": None}
    if not listing.get("jobs"):
        return {"（暂无 RGB 识别任务）": None}
    opts: dict[str, str | None] = {"（选择 RGB 任务）": None}
    for j in listing["jobs"]:
        label = (f"{j['id'][:8]}… · {j['source_name'][:20]} · "
                 f"{RGB_STATUS_ZH.get(j['status'], j['status'])}")
        opts[label] = j["id"]
    return opts


def render(conn) -> None:
    """入口：由 app.py 的标签页调用，``conn`` 是主可写连接（只用来写 qa_runs）。"""
    st.subheader("DeepSeek 数据问答")
    st.caption(
        "后端先从 SQLite 用只读查询取出事实摘要（场景 / 质量 / 识别结果），"
        "再把摘要与问题一起发给 DeepSeek。模型只做解释，不改数据、不改质量状态、"
        "不执行 SQL；没有配置或调用失败时，本地事实仍然照常显示。"
    )

    qa_store.ensure_schema(conn)
    cfg = ds.load_config()
    _render_config(cfg)

    scene_ids: list[str] = [s["scene_id"] for s in scenes.list_scenes(conn)]

    left, right = st.columns([2, 1])
    scope = left.radio(
        "问题类型（决定取哪一份事实）",
        SCOPE_ORDER,
        format_func=lambda k: qa_facts.SCOPES[k],
        horizontal=True,
        key=f"{_NS}_scope",
    )
    with right:
        scene_pick = st.selectbox(
            "场景（仅“场景质量与识别结果”用）",
            scene_ids or ["（暂无场景）"],
            key=f"{_NS}_scene",
            disabled=not bool(scene_ids) or scope not in ("scene", "governance_report"),
        )
        rgb_opts = _rgb_job_options()
        rgb_pick = st.selectbox(
            "RGB 任务（仅“RGB 识别任务事实”用）",
            list(rgb_opts),
            key=f"{_NS}_rgb_job",
            disabled=scope != "rgb_job",
        )
    rgb_job_id = rgb_opts.get(rgb_pick)

    # 回答按上下文绑定：口径/任务/场景任一变化后，旧回答不挂在新引用下。
    ctx = _context_key(scope, rgb_job_id,
                       scene_pick if scene_ids else None)

    # 事实查询走只读连接：写操作在 SQLite 层就被拒绝
    try:
        ro = db.connect_readonly()
    except Exception as exc:
        st.error(f"只读连接失败，无法取事实摘要：{exc}")
        return
    try:
        bundle = qa_facts.build(
            ro, scope,
            scene_id=scene_pick if scene_ids else None,
            rgb_job_id=rgb_job_id,
        )
    finally:
        ro.close()

    _render_facts(bundle, expanded=True)

    with st.form(f"{_NS}_form"):
        # 草稿按口径分开存：切换问题类型时，预填的是该类型对应的示例问题，
        # 而不会把上一个类型手打的内容带过来。
        question = st.text_area(
            "你的问题",
            value=_state(f"draft_{scope}") or QUICK_QUESTIONS.get(scope, ""),
            height=90,
            help="例如：当前批次待复核资料有哪些？",
        )
        submitted = st.form_submit_button("提问", type="primary", key=f"{_NS}_submit")

    if submitted:
        _set_state(f"draft_{scope}", question.strip())
        with st.spinner("正在调用 DeepSeek…"):
            reply = ds.ask(question, bundle.text, config=cfg)
        try:
            run_id = qa_store.save_run(
                conn, question=question, bundle=bundle, reply=reply, config=cfg
            )
        except Exception as exc:  # 留档失败不影响回答展示
            st.warning(f"回答已返回，但写入问答记录失败：{exc}")
            run_id = None
        answers = _state("answers") or {}
        answers[ctx] = {
            "facts_key": _facts_key(bundle),
            "run_id": run_id,
            "ctx": ctx,
            "scope": scope,
            "rgb_job_id": rgb_job_id,
            "scene_id": scene_pick if scene_ids else None,
            "facts_sha": qa_store.facts_digest(bundle.text)[:16],
            "generated_at": (bundle.snapshot or {}).get("generated_at"),
            "status": reply.status,
            "status_zh": reply.status_zh,
            "answer": reply.answer,
            "model": reply.model,
            "response_model": reply.response_model,
            "latency_ms": reply.latency_ms,
            "attempts": reply.attempts,
            "error_code": reply.error_code,
            "error_message": reply.error_message,
            "prompt_tokens": reply.usage.get("prompt_tokens"),
            "completion_tokens": reply.usage.get("completion_tokens"),
            "total_tokens": reply.usage.get("total_tokens"),
            "provider_request_id": reply.provider_request_id,
        }
        _set_state("answers", answers)

    st.markdown("### 回答")
    result = (_state("answers") or {}).get(ctx)
    if result is not None and result.get("facts_key") != _facts_key(bundle):
        result = None
        st.info("此任务的事实或治理状态已变化，旧回答已隐藏；重新提问将使用当前事实。")
    if result is None:
        st.info("输入问题后点击「提问」。只有点这个按钮才会发出请求。"
                "切换口径、场景或 RGB 任务后，需要重新提问；旧回答不会挂到新引用下。")
    else:
        st.caption(
            f"本回答绑定：口径 {qa_facts.SCOPES.get(result['scope'], result['scope'])}"
            f" · RGB任务 {result['rgb_job_id'] or '—'}"
            f" · 场景 {result['scene_id'] or '—'}"
            f" · 当时事实摘要 SHA 前16位 {result['facts_sha']}…"
            f" · 生成于 {result['generated_at'] or '未知'}"
        )
        if result.get("run_id"):
            st.caption(f"本次问答已写入 qa_runs，run_id = {result['run_id']}")
        _render_answer(result)

    st.markdown("### 引用来源")
    _render_citations(bundle.citations)

    if bundle.gaps:
        st.caption("缺口：" + "；".join(bundle.gaps))

    st.divider()
    _render_history(conn)
