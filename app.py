"""AgroLake 治理底座 · 操作网页（Streamlit）。

以“场景预览 + 问题工作台”为中心，紧凑中文布局。
所有操作实际落库，重启后可读。
治理页面与FTW真实识别服务已连接；第 9 页已接入 DeepSeek 数据问答（T15）。
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import streamlit as st

from agrolake import config, db, disposition, ingest, quality, reporting, scenes, semantics
from agrolake import storage
from agrolake_web import qa_page, recognition_page, rgb_candidate_page, rgb_results_page

st.set_page_config(page_title="AgroLake 治理底座", layout="wide")

ACCENT = "#17685D"
WARN = "#B16D13"
INK = "#1D1D1F"
BG = "#FBFBFD"
SURFACE = "#FFFFFF"
GRAY = "#6E6E73"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; color: {INK};
        font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }}
      .block-container {{ padding-top: 4.5rem; max-width: 1180px; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; font-weight: 700; }}
      .ag-hero-eyebrow {{ color: {ACCENT}; font-size: 0.95rem; font-weight: 600;
        letter-spacing: 0.06em; margin-bottom: 0.4rem; }}
      .ag-hero-title {{ font-size: 2.7rem; line-height: 1.25; font-weight: 700;
        letter-spacing: -0.015em; color: {INK}; margin: 0 0 0.8rem 0; }}
      .ag-hero-sub {{ font-size: 1.12rem; line-height: 1.75; color: {GRAY};
        max-width: 860px; margin-bottom: 1.6rem; }}
      .ag-flow {{ display: flex; align-items: stretch; gap: 0; margin: 0.4rem 0 1.8rem 0; }}
      .ag-flow-step {{ flex: 1; background: {SURFACE}; border: 1px solid #E8E8ED;
        padding: 0.9rem 0.6rem; text-align: center; }}
      .ag-flow-step:first-child {{ border-radius: 14px 0 0 14px; }}
      .ag-flow-step:last-child {{ border-radius: 0 14px 14px 0; }}
      .ag-flow-step + .ag-flow-step {{ border-left: none; }}
      .ag-flow-emoji {{ font-size: 1.5rem; }}
      .ag-flow-label {{ font-size: 0.92rem; font-weight: 600; color: {INK}; margin-top: 0.35rem; }}
      .ag-flow-hint {{ font-size: 0.78rem; color: {GRAY}; margin-top: 0.15rem; }}
      .ag-card {{ background: {SURFACE}; border: 1px solid #E8E8ED;
        border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04); }}
      .ag-kpi {{ font-size: 1.9rem; font-weight: 700; color: {INK};
        font-variant-numeric: tabular-nums; }}
      .ag-kpi-label {{ font-size: 0.88rem; color: {GRAY}; margin-top: 0.2rem; }}
      .ag-warn {{ color: {WARN}; font-weight: 600; }}
      .ag-sub {{ color: {GRAY}; font-size: 0.85rem; }}
      .ag-section-title {{ font-size: 1.35rem; font-weight: 700; margin: 2.2rem 0 0.9rem 0; }}
      div[data-testid="stMetricValue"] {{ color: {ACCENT}; }}
      section[data-testid="stSidebar"] {{ background: {SURFACE}; }}

      /* Floating surfaces only; keep Streamlit's native navigation and collapse. */
      section[data-testid="stSidebar"][aria-expanded="true"] {{
        margin: 18px;
        height: calc(100% - 36px) !important;
        border-radius: 24px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.10), 0 2px 8px rgba(0, 0, 0, 0.04);
      }}
      section[data-testid="stSidebar"][aria-expanded="true"]
      [data-testid="stSidebarContent"] {{ border-radius: inherit; }}
      .ag-flow {{ border-radius: 14px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.03); }}

      /* Stretch only the four KPI columns, including Streamlit's HTML wrappers. */
      [data-testid="stHorizontalBlock"]:has(.ag-kpi) {{ align-items: stretch; }}
      [data-testid="stColumn"]:has(.ag-kpi) > [data-testid="stVerticalBlock"] {{
        height: 100%;
      }}
      [data-testid="stColumn"]:has(.ag-kpi)
      [data-testid="stElementContainer"]:has(.ag-kpi) {{
        flex: 1 0 auto; display: flex; flex-direction: column;
      }}
      [data-testid="stColumn"]:has(.ag-kpi) [data-testid="stMarkdown"],
      [data-testid="stColumn"]:has(.ag-kpi) [data-testid="stMarkdownContainer"] {{
        flex: 1; display: flex; flex-direction: column;
      }}
      .ag-card:has(.ag-kpi) {{ flex: 1;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.03); }}
      [data-testid="stColumn"]:has(.ag-kpi) [data-testid="stButton"] button {{
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
      }}
      [data-testid="stColumn"]:has(.ag-kpi) [data-testid="stButton"] button:hover {{
        background-color: #EFF8F3; border-color: {ACCENT}; color: {ACCENT};
      }}
      @media (max-width: 768px) {{
        section[data-testid="stSidebar"][aria-expanded="true"] {{
          margin: 8px;
          height: calc(100% - 16px) !important;
        }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_conn():
    """每次脚本运行建立独立连接，避免跨线程复用裸连接。"""
    conn = db.connect()
    db.init_schema(conn)
    return conn


def _store_root() -> Path:
    root = config.storage_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _overall_badge(status: str) -> str:
    mapping = {"pass": "通过", "warn": "警告", "block": "阻断", "na": "不适用"}
    label = mapping.get(status, status)
    if status in ("block", "warn"):
        return f"<span class='ag-warn'>{label}</span>"
    return label


def _meta_txt(value) -> str:
    """元数据缺失时显示“未知”，不冒充真实测量值。"""
    return "未知" if value in (None, "") else str(value)


def _image_preview_ok(conn, v: dict) -> tuple[bool, str]:
    """依据最新质检结论判断影像能否安全预览，提示只用中文分类说明。

    空文件、截断或不可解码内容在导入时已由质检规则判为阻断，这里直接沿用
    该结论并转成面向用户的中文提示，避免在页面再次触发原始解码异常或
    展示含内部对象路径的 reason 原文（原文仍保留在数据库质检记录中）。
    """
    for c in quality.latest_checks(conn, v["version_id"]):
        if c["rule_id"] == "file_not_empty" and c["status"] == "block":
            return False, "文件为空"
        if c["rule_id"] == "image_decode" and c["status"] == "block":
            return False, "文件损坏或不是支持的图片格式，无法解码"
        if c["rule_id"] == "image_decode" and c["status"] != "block":
            return True, ""
    if v.get("width") is None or v.get("height") is None:
        return False, "图像尺寸缺失，无法解码"
    return True, ""


def _no_preview_notice(filename: str, reason: str) -> None:
    st.warning(
        f"影像无法预览：{filename}；{reason}。"
        "请在“问题处置”隔离或替换文件。"
    )


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flash_next_run(message: str) -> None:
    """成功写入后把提示放入会话 flash 并显式 rerun。

    先渲染后修改的页内成功提示不会刷新概览/目录/场景；显式 rerun 让本轮
    脚本以最新数据库状态重新渲染，成功提示在下一次 run 显示并被消费，
    避免提示消失、无限 rerun 或同一按钮重复写入。
    """
    st.session_state["ag_flash"] = message
    st.rerun()


def consume_flash() -> None:
    """显示并消费上一次成功操作的提示；之后的普通 rerun 不再重复出现。"""
    msg = st.session_state.pop("ag_flash", None)
    if msg:
        st.success(msg)


def page_overview(conn):
    st.subheader("治理概览")
    s = scenes.summary(conn)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("资产版本", s["total_versions"])
    c2.metric("导入事件", s["total_import_events"])
    c3.metric("当前有效版本", s["current_versions"])
    qr = s["quality_pass_rate"]["pass_rate"]
    c4.metric("质量通过率", "—" if qr is None else f"{qr:.0%}")
    ac = s["association_coverage"]["coverage"]
    st.caption(
        f"关联覆盖率：{'—' if ac is None else f'{ac:.0%}'}    "
        f"（分母：{s['association_coverage']['denominator']}）"
    )
    st.caption(
        "质量通过率分母：存在可适用规则的 active 版本；"
        "已排除 isolated、duplicate 以及全部规则为 na 的版本。"
        f"规则版本 {config_rule_version()}，语义字典 {semantics.semantic_version()}。"
    )
    if s["total_versions"] == 0:
        st.info("暂无数据。请到“资料导入”页上传 JPG/PNG 影像、CSV 属性或 JSON/TXT 记录。")


def config_rule_version() -> str:
    from agrolake import quality

    return quality.rules_version()


def page_import(conn):
    st.subheader("资料导入")
    st.caption(
        "支持 JPG/PNG 影像、CSV 属性、JSON/TXT 记录；本页不支持 TIFF/GeoTIFF，"
        "请到第 8 页“田块识别”使用专门的 8 波段识别模型。"
        "CSV/JSON/TXT 需为 UTF-8 编码；编码不符会登记问题并阻断，不做替换损坏。"
    )
    up = st.file_uploader("选择文件", type=["jpg", "jpeg", "png", "csv", "json", "txt"])
    with st.form("import_form"):
        col1, col2 = st.columns(2)
        source = col1.text_input("来源 source（必填）", "")
        scene_id = col2.text_input("场景 scene_id", "")
        intended_use = col1.text_input("预定用途", "")
        source_link = col2.text_input("来源链接", "")
        license_status = col1.text_input("授权/许可状态", "")
        acquired_at = col2.text_input("采集时间 (ISO8601)", "")
        submitted = st.form_submit_button("导入并质检")
    if submitted:
        if up is None:
            st.error("请先选择文件。")
        elif not source.strip():
            st.error("来源为必填项。请填写来源后重试；未填来源会被质检判为阻断。")
        else:
            try:
                res = ingest.import_bytes(
                    conn, up.getvalue(), up.name,
                    store_root=_store_root(), source=source.strip() or None,
                    scene_id=scene_id.strip() or None,
                    intended_use=intended_use.strip() or None,
                    source_link=source_link.strip() or None,
                    license_status=license_status.strip() or None,
                    acquired_at=acquired_at.strip() or None,
                )
                evt = {
                    "new_asset": "新建资产",
                    "new_version_same_name": "同名新内容（新版本）",
                    "duplicate_content": "精确重复（保留本次导入事件）",
                }.get(res["event_type"], res["event_type"])
                _flash_next_run(
                    f"导入完成：版本 {res['version_id']}，事件类型 {evt}，"
                    f"SHA256 {res['sha256'][:16]}…"
                )
            except ingest.UnsupportedFormat as exc:
                st.error(str(exc))
            except storage.PathBoundaryError as exc:
                st.error(f"路径边界错误：{exc}")
            except Exception as exc:  # 明确失败信息
                st.error(f"导入失败：{exc}")


def page_catalog(conn):
    st.subheader("数据目录与预览")
    versions = scenes.list_versions(conn)
    if not versions:
        st.info("暂无数据。请到“资料导入”页上传。")
        return
    header = ["version_id", "original_filename", "content_type", "declared_scene_id",
              "source", "status", "overall_status", "size_bytes"]
    st.dataframe(
        [{k: v.get(k) for k in header} for v in versions],
        width="stretch", hide_index=True,
    )
    opts = {f"#{v['version_id']} · {v['original_filename']}": v for v in versions}
    pick = st.selectbox("查看版本", list(opts))
    v = opts[pick]
    st.markdown(f"**版本 #{v['version_id']}** · 状态 {v['status']} · "
                f"质检 {_overall_badge(v['overall_status'])}", unsafe_allow_html=True)
    st.caption(f"对象：{v['relpath']} · {v['size_bytes']} 字节 · SHA256 {str(v['object_id'])[:24]}…")
    if v["content_type"] == "image":
        obj = _store_root() / v["relpath"]
        ok, reason = _image_preview_ok(conn, v)
        if ok:
            try:
                st.image(str(obj), caption=v["original_filename"])
            except Exception:
                _no_preview_notice(v["original_filename"], "无法预览，请在问题处置复检")
        else:
            _no_preview_notice(v["original_filename"], reason)
        st.caption(
            f"尺寸 {_meta_txt(v['width'])}×{_meta_txt(v['height'])} · "
            f"{_meta_txt(v['bands'])} 波段 · {_meta_txt(v['bitdepth'])} 位"
        )
    elif v["content_type"] == "attribute":
        rows = conn.execute(
            "SELECT scene_id, field_id, crop, category_raw, category_std,"
            " area_raw, area_unit_raw, area_m2, issues"
            " FROM attribute_records WHERE version_id = ? ORDER BY row_index",
            (v["version_id"],),
        ).fetchall()
        st.dataframe([dict(r) for r in rows], width="stretch", hide_index=True)
    else:
        rows = conn.execute(
            "SELECT scene_id, source, fmt, payload_json, issues FROM text_records"
            " WHERE version_id = ?", (v["version_id"],),
        ).fetchall()
        st.dataframe([dict(r) for r in rows], width="stretch", hide_index=True)


def page_disposition(conn):
    st.subheader("问题处置")
    versions = scenes.list_versions(conn, include_inactive=True)
    if not versions:
        st.info("暂无数据。请先导入资料。")
        return
    opts = {f"#{v['version_id']} · {v['original_filename']} · {v['overall_status']}": v
            for v in versions}
    v = opts[st.selectbox("选择版本", list(opts))]
    st.markdown("**当前规则结论**")
    st.dataframe(scenes.list_checks(conn, v["version_id"]),
                 width="stretch", hide_index=True)
    st.markdown("**处置操作**")
    operator = st.text_input("操作者", "operator")
    tab1, tab2, tab3 = st.tabs(["补充元数据", "隔离/解除", "标记重复 / 复检"])
    with tab1:
        col1, col2 = st.columns(2)
        n_source = col1.text_input("来源 source", v["source"] or "")
        n_scene = col2.text_input("场景 scene_id", v["declared_scene_id"] or "")
        reason = col1.text_input("原因", "补充来源/场景")
        if st.button("保存并复检"):
            updates = {}
            if n_source.strip():
                updates["source"] = n_source.strip()
            if n_scene.strip():
                updates["declared_scene_id"] = n_scene.strip()
            try:
                disposition.supplement_metadata(
                    conn, v["version_id"], operator, reason, updates, _store_root()
                )
                _flash_next_run("已补充元数据并复检，旧结论与旧值已留档。")
            except Exception as exc:
                st.error(f"处置失败：{exc}")
    with tab2:
        c1, c2 = st.columns(2)
        if c1.button("隔离该版本"):
            disposition.isolate(conn, v["version_id"], operator, "人工隔离")
            _flash_next_run("已隔离。复检不会自动恢复；需显式解除。")
        if c2.button("解除隔离"):
            disposition.unisolate(conn, v["version_id"], operator, "人工解除隔离")
            _flash_next_run("已解除隔离。")
    with tab3:
        if st.button("标记为重复"):
            disposition.mark_duplicate(
                conn, v["version_id"], operator, "人工标记重复",
                v["duplicate_of_version_id"],
            )
            _flash_next_run("已标记重复。")
        if st.button("仅复检（不改状态）"):
            disposition.recheck(conn, v["version_id"], operator, _store_root())
            _flash_next_run("已复检：写入新一轮结论并关联旧检查。")


def _versions_by_id(conn):
    return {v["version_id"]: v for v in scenes.list_versions(conn)}


def page_scenes(conn):
    st.subheader("场景关联")
    st.caption("只显示当前有效版本；隔离、标重复、历史与未处置失配均不计入。")
    scs = scenes.list_scenes(conn)
    if not scs:
        st.info("暂无场景。到“资料导入”页填写场景编号并上传资料，即会建立场景。")
        return
    cov = scenes.association_coverage(conn)
    cov_txt = "—" if cov["coverage"] is None else f"{cov['coverage']:.0%}"
    st.caption(
        f"关联覆盖率：{cov_txt}"
        f"（{cov['complete_scenes']}/{cov['total_scenes']} 个场景三类齐全）"
    )
    pick = st.selectbox("选择场景", [s["scene_id"] for s in scs])
    sc = scenes.scene_members(conn, pick)
    by_id = _versions_by_id(conn)
    store = _store_root()

    st.markdown("### 当前影像")
    if sc["roles"]["image"]:
        for m in sc["roles"]["image"]:
            v = by_id.get(m["version_id"])
            if v is None:
                continue
            ok, reason = _image_preview_ok(conn, v)
            if ok:
                try:
                    st.image(str(store / v["relpath"]),
                             caption=f"版本 #{v['version_id']} · {v['original_filename']}")
                except Exception:
                    _no_preview_notice(
                        f"版本 #{v['version_id']} · {v['original_filename']}",
                        "无法预览，请在问题处置复检",
                    )
            else:
                _no_preview_notice(
                    f"版本 #{v['version_id']} · {v['original_filename']}", reason
                )
            st.caption(
                f"来源：{v['source'] or '未填'} · 尺寸 {_meta_txt(v['width'])}×{_meta_txt(v['height'])} · "
                f"{_meta_txt(v['bands'])} 波段 · {_meta_txt(v['bitdepth'])} 位"
            )
    else:
        st.markdown("<span class='ag-warn'>该场景暂无当前影像</span>", unsafe_allow_html=True)

    st.markdown("### 关联属性（原值 / 换算值）")
    attr_rows = []
    for m in sc["roles"]["attribute"]:
        v = by_id.get(m["version_id"])
        if v is None:
            continue
        rows = conn.execute(
            "SELECT row_index, field_id, crop, category_raw, category_std,"
            " area_raw, area_unit_raw, area_m2, issues FROM attribute_records"
            " WHERE version_id = ? ORDER BY row_index",
            (v["version_id"],),
        ).fetchall()
        for r in rows:
            d = dict(r)
            d["版本"] = f"#{v['version_id']}"
            attr_rows.append(d)
    if attr_rows:
        show = [{
            "版本": d["版本"], "行": d["row_index"], "地块": d["field_id"],
            "作物(crop)": d["crop"], "地物类别(原值)": d["category_raw"],
            "地物类别(标准值)": d["category_std"], "面积原值": d["area_raw"],
            "原单位": d["area_unit_raw"], "换算面积(m²)": d["area_m2"],
            "问题": d["issues"],
        } for d in attr_rows]
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.markdown("<span class='ag-warn'>该场景暂无当前属性资料</span>", unsafe_allow_html=True)

    st.markdown("### 文本记录")
    text_rows = []
    for m in sc["roles"]["text"]:
        v = by_id.get(m["version_id"])
        if v is None:
            continue
        rows = conn.execute(
            "SELECT fmt, payload_json, issues FROM text_records WHERE version_id = ?",
            (v["version_id"],),
        ).fetchall()
        for r in rows:
            d = dict(r)
            d["版本"] = f"#{v['version_id']}"
            text_rows.append(d)
    if text_rows:
        st.dataframe(
            [{"版本": d["版本"], "格式": d["fmt"], "内容": d["payload_json"],
              "问题": d["issues"]} for d in text_rows],
            width="stretch", hide_index=True,
        )
    else:
        st.markdown("<span class='ag-warn'>该场景暂无当前文本记录</span>", unsafe_allow_html=True)

    issues = scenes.scene_issues(conn)
    if issues:
        st.markdown("**待处理场景问题（不计入合格关联）**")
        st.dataframe(issues, width="stretch", hide_index=True)
    else:
        st.success("场景关联未发现缺失或失配。")


def page_category(conn):
    st.subheader("类别字典与标准化")
    st.caption(
        f"字典版本 {semantics.category_dict_version()}。"
        "地物类别（田块/道路/沟渠）与作物类别（如水稻）是两套口径，分开管理，不互相替代。"
    )
    rules = semantics.load_semantic_rules()
    dict_rows = [{"标准值": std, "可识别的原值": "、".join(aliases)}
                 for std, aliases in rules["category_aliases"].items()]
    st.markdown("**地物类别字典**")
    st.dataframe(dict_rows, width="stretch", hide_index=True)
    st.markdown("**标准化入口（示例）**")
    raw = st.text_input("输入原始地物类别值，查看标准化结果", "")
    if raw.strip():
        std, note = semantics.normalize_category(raw)
        if std:
            st.success(
                f"原值 {raw!r} → 标准值 {std}（字典版本 {semantics.category_dict_version()}）"
            )
        else:
            st.warning(f"{note}（标准值为空，原值保留并记录为问题）")
    st.caption(
        "提示：上传 CSV 时可额外提供“地物类别”列（category/地物类别），"
        "系统会保留原值并写入对应标准值；未知值留空并在问题中列出。"
    )


def page_reports(conn):
    st.subheader("报告导出")
    st.caption("导出 CSV 清单与 JSON/Markdown 报告，可重新读取。")
    out = config.reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    # 本会话已生成的文件快照：普通 rerun（含点击其它按钮）后下载按钮仍在。
    store = st.session_state.setdefault("generated_reports", {})

    def generate(key: str, label: str, filename: str, exporter) -> None:
        if st.button(label, key=f"gen_{key}"):
            p = exporter(conn, out / filename)
            store[key] = {
                "filename": filename,
                "bytes": Path(p).read_bytes(),
                "generated_at": _now_stamp(),
            }

    c1, c2, c3 = st.columns(3)
    with c1:
        generate("assets.csv", "生成资产清单 CSV", "assets.csv", reporting.export_assets_csv)
    with c2:
        generate("checks.csv", "生成质检明细 CSV", "checks.csv", reporting.export_checks_csv)
    with c3:
        generate("report.json", "生成报告 JSON", "report.json", reporting.export_report_json)
    generate("report.md", "生成报告 Markdown", "report.md", reporting.export_report_markdown)

    st.markdown("**已生成下载（本会话保留）**")
    if not store:
        st.info("尚未生成。点击上方按钮生成文件；生成后下载按钮在普通 rerun 后仍保留。")
        return
    for key, item in store.items():
        col_a, col_b = st.columns([3, 1])
        col_a.caption(
            f"{item['filename']} · {len(item['bytes'])} 字节 · 生成于 {item['generated_at']}"
        )
        col_b.download_button(
            f"下载 {item['filename']}", item["bytes"], item["filename"], key=f"dl_{key}"
        )
    st.caption(
        "下载内容是“生成时”的快照；数据变化后请重新生成，"
        "不会自动把不同批次当作最新报告。当前统计口径见“治理概览”。"
    )


NAV_PAGES = ["首页", "资料导入", "数据目录", "问题处置", "场景关联",
             "类别规范", "田块识别", "RGB 图片识别", "RGB 识别实验",
             "DeepSeek 问答", "报告导出", "治理总览"]


def _go(page_name: str) -> None:
    """导航跳转回调（在脚本重跑前执行，可安全改 widget 状态）。"""
    st.session_state["nav_page"] = page_name


def _latest_recognition(conn):
    """取最近一次成功的识别运行（字段名与服务库表 rec_runs 对齐）。"""
    try:
        row = conn.execute(
            "SELECT r.field_count, r.output_relpath, r.finished_at, j.sample_kind "
            "FROM rec_runs r JOIN rec_jobs j ON j.job_id = r.job_id "
            "WHERE r.status = 'succeeded' ORDER BY r.run_id DESC LIMIT 1"
        ).fetchone()
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM rec_runs WHERE status = 'succeeded'"
        ).fetchone()["c"]
    except Exception:
        return None, 0
    return row, total


def page_home(conn):
    """首页：评委第一屏。主张、动线、真实 KPI、主视觉、处置闭环、AI 直达。"""
    s = scenes.summary(conn)
    rec_row, rec_total = _latest_recognition(conn)
    field_count = rec_row["field_count"] if rec_row else None
    n_checks = conn.execute("SELECT COUNT(*) AS c FROM quality_checks").fetchone()["c"]
    n_disp = conn.execute("SELECT COUNT(*) AS c FROM dispositions").fetchone()["c"]

    st.markdown(
        "<div class='ag-hero-eyebrow'>第三届“兴智杯” · 云智专项赛</div>"
        "<div class='ag-hero-title'>数据先治理，AI 才可信。</div>"
        "<div class='ag-hero-sub'>AgroLake 面向数字农服：登记影像与台账资料，"
        "检查质量、追溯来源，查看农田识别结果并保存复核意见。"
        "当前为本地原型，治理样例包含合成资料，识别示范使用公开影像。</div>",
        unsafe_allow_html=True,
    )

    flow = [
        ("📥", "数据接入", "影像与台账入库"),
        ("🧹", "质量治理", "规则质检与处置"),
        ("🛰️", "AI 识别", "FTW 田块圈定"),
        ("👁️", "人工复核", "意见留痕"),
        ("📄", "报告导出", "一键生成报告"),
    ]
    cells = "".join(
        f"<div class='ag-flow-step'><div class='ag-flow-emoji'>{e}</div>"
        f"<div class='ag-flow-label'>{t}</div><div class='ag-flow-hint'>{h}</div></div>"
        for e, t, h in flow
    )
    st.markdown(f"<div class='ag-flow'>{cells}</div>", unsafe_allow_html=True)

    kpis = [
        ("治理资产", f"{s['total_versions']}", "份版本 · 支持来源追溯", "数据目录"),
        ("质检执行", f"{n_checks}", "条规则检查记录", "治理总览"),
        ("问题处置", f"{n_disp}", "次操作 · 前后状态留痕", "问题处置"),
        ("候选连通区域", f"{field_count if field_count is not None else '—'}",
         f"个区域 · {rec_total} 次已保存运行", "田块识别"),
    ]
    cols = st.columns(4)
    for col, (label, value, hint, target) in zip(cols, kpis):
        with col:
            st.markdown(
                f"<div class='ag-card'><div class='ag-kpi'>{value}</div>"
                f"<div class='ag-kpi-label'>{label}</div>"
                f"<div class='ag-sub'>{hint}</div></div>",
                unsafe_allow_html=True,
            )
            st.button("查看明细 →", key=f"home_go_{target}",
                      on_click=_go, args=(target,), use_container_width=True)

    st.markdown("<div class='ag-section-title'>AI 识别实景</div>", unsafe_allow_html=True)
    if rec_row:
        preview = (Path(__file__).parent / "data" / "service" / "runs"
                   / rec_row["output_relpath"] / "preview.png")
        c_img, c_txt = st.columns([3, 2])
        with c_img:
            if preview.exists():
                st.image(str(preview), use_container_width=True)
            else:
                st.info("识别预览图暂缺，请到“田块识别”页运行一次识别。")
        with c_txt:
            sample_label = {"public_adapter_sample": "公开适配样例",
                            "user_upload": "上传影像",
                            "synthetic_test": "合成软件测试"}.get(
                                rec_row["sample_kind"], "来源类别待核的影像")
            st.markdown(
                "**FTW v1（8 波段）本机推理结果**\n\n"
                f"{sample_label}的最近一次成功运行得到 **{field_count} 个候选连通区域**。"
                "输入、掩膜和区域像素表保存在识别记录中。\n\n"
                "候选区域未经人工确认；同图复跑不等于新增独立场景，像素也不能直接换算为实测面积。"
            )
            st.button("去田块识别页亲手跑一次 →", key="home_go_rec2",
                      on_click=_go, args=("田块识别",))
    else:
        st.info("尚无成功识别记录，请到“田块识别”页运行一次。")

    st.markdown(
        "<div class='ag-card'>🖼️ <b>RGB 图片识别</b>：上传 512×512 RGB 农田图片，"
        "运行候选识别模型并保存结果与历史，可下载掩膜、叠加图与候选区域表。</div>",
        unsafe_allow_html=True,
    )
    st.button("打开 RGB 图片识别 →", key="home_go_rgb",
              on_click=_go, args=("RGB 图片识别",))

    st.markdown("<div class='ag-section-title'>问题处置：查出问题只是开始</div>",
                unsafe_allow_html=True)
    st.caption("以下为治理库中的真实处置记录：每个问题版本都有处置前、处置后状态，"
               "处置可撤销、全程留痕——这正是“治理”与“只报警”的区别。")
    rows = conn.execute(
        "SELECT version_id, action, reason, before_json, after_json, created_at "
        "FROM dispositions ORDER BY disp_id DESC LIMIT 5"
    ).fetchall()
    import json as _json
    table = []
    action_name = {"isolate": "隔离", "unisolate": "解除隔离",
                   "mark_duplicate": "标记重复", "unmark_duplicate": "取消重复标记"}
    status_name = {"active": "有效", "isolated": "已隔离", "duplicate": "重复"}
    for r in rows:
        before = status_name.get(_json.loads(r["before_json"]).get("status"),
                                 _json.loads(r["before_json"]).get("status"))
        after = status_name.get(_json.loads(r["after_json"]).get("status"),
                                _json.loads(r["after_json"]).get("status"))
        table.append({
            "版本": f"版本 {r['version_id']}",
            "处置": action_name.get(r["action"], r["action"]),
            "原因": r["reason"],
            "状态变化": f"{before} → {after}",
            "时间": str(r["created_at"])[:10],
        })
    if table:
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("<div class='ag-section-title'>直接向 AI 提问</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ag-card'>🤖 <b>DeepSeek 问答</b>已接入治理事实库：它用只读查询取"
               "数据再作答，页面附事实与来源，模型解释仍需复核。试着问它："
        "“当前批次待复核的资料有哪些？”</div>",
        unsafe_allow_html=True,
    )
    st.button("打开 DeepSeek 问答 →", key="home_go_qa",
              on_click=_go, args=("DeepSeek 问答",))

    st.markdown("<div class='ag-section-title'>演示动线（建议顺序）</div>",
                unsafe_allow_html=True)
    st.markdown(
        "1. **资料导入** — 一份影像如何入库并被质检拦下或放行\n"
        "2. **数据目录** — 每份资产的版本、来源与血缘\n"
        "3. **问题处置** — 问题版本如何隔离、订正、复检\n"
        "4. **田块识别** — AI 现场推理 + 人工复核订正\n"
        "5. **RGB 识别实验** — 普通农田图片识别效果查看\n"
        "6. **DeepSeek 问答** — 向 AI 追问治理现状\n"
        "7. **报告导出** — 生成带统计口径的治理报告"
    )




def main():
    with st.sidebar:
        st.markdown("### 🌱 AgroLake 农智数湖")
        st.caption("面向数字农服的多模态数据治理与田块识别平台")
        page = st.radio("导航", NAV_PAGES, key="nav_page", label_visibility="collapsed")
        st.divider()
        st.caption("第三届“兴智杯”全国人工智能创新应用大赛\n"
                   "赛题：人工智能数据湖与数据治理创新方案赛（云智专项赛）\n"
                   "当前为方案验证阶段：治理样例含合成资料，FTW使用公开影像，RGB实验使用用户图片。")
    consume_flash()
    try:
        conn = get_conn()
    except Exception as exc:
        st.error(f"数据库初始化失败：{exc}")
        return
    try:
        if page == "首页":
            page_home(conn)
        elif page == "资料导入":
            page_import(conn)
        elif page == "数据目录":
            page_catalog(conn)
        elif page == "问题处置":
            page_disposition(conn)
        elif page == "场景关联":
            page_scenes(conn)
        elif page == "类别规范":
            page_category(conn)
        elif page == "田块识别":
            recognition_page.render()
        elif page == "RGB 图片识别":
            rgb_candidate_page.render()
        elif page == "RGB 识别实验":
            rgb_results_page.render()
        elif page == "DeepSeek 问答":
            qa_page.render(conn)
        elif page == "报告导出":
            page_reports(conn)
        elif page == "治理总览":
            page_overview(conn)
    finally:
        # 每次脚本运行独立连接，结束即关闭，避免跨线程复用与泄漏。
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
