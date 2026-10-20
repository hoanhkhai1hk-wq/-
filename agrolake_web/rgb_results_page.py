"""Read-only view of audited local RGB experiment artifacts; no model execution."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "rgb" / "current-results.json"


def _artifact(record: dict) -> bytes:
    relative = record.get("path")
    expected = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError("结果记录缺少文件或摘要")
    path = (ROOT / relative).resolve()
    allowed = [ROOT / "outputs", ROOT / "reports" / "rgb"]
    if not any(path.is_relative_to(folder.resolve()) for folder in allowed):
        raise ValueError("结果文件不在项目结果目录中")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("结果文件已发生变化")
    return content


def render() -> None:
    st.subheader("RGB 识别实验")
    st.caption("查看你提供的农田图片、已保存的模型输出和预测像素统计。")
    st.button("刷新结果", key="rgb_results_refresh")
    try:
        document = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
        if document.get("schema") != "agrolake_rgb_results_view_v1":
            raise ValueError("结果格式尚未准备好")
        cases = document["cases"]
        if not cases:
            raise ValueError("还没有可展示的案例")
    except FileNotFoundError:
        st.info("RGB结果正在整理，完成后会在这里显示，无需重复上传或补标注。")
        return
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("结果记录正在更新，请稍后点“刷新结果”。")
        return

    training = document.get("training", {})
    message = training.get("message", "当前结果尚未定型，请对照原图查看。")
    if training.get("state") == "reviewed_not_adopted":
        st.warning(message)
    elif training.get("state") == "reviewed":
        st.info(message)
    else:
        st.write(message)
    progress = training.get("progress", [])
    if progress and training.get("state") in {"reviewed", "reviewed_not_adopted"}:
        totals = {int(row.get("total", 128)) for row in progress}
        detail = f"每组{next(iter(totals))}步" if len(totals) == 1 else "各组步数见记录"
        st.caption(f"本轮训练已完成：{len(progress)}组，{detail}。")
    elif progress:
        st.caption("训练进度（最近一次记录）")
        for column, row in zip(st.columns(len(progress)), progress):
            step = int(row.get("step", 0))
            total = max(int(row.get("total", 128)), 1)
            with column:
                st.progress(min(max(step / total, 0), 1), text=f"{row['label']}：{step}/{total}步")
    st.caption("记录更新时间：" + str(document.get("updated_at", "未知")))

    options = {case["case_id"]: case for case in cases}
    default = list(options).index("R03") if "R03" in options else 0
    selectors = st.columns([1, 2])
    with selectors[0]:
        selected = st.selectbox("选择图片案例", list(options), index=default, key="rgb_results_case")
    case = options[selected]
    panels = case.get("panels", [])
    if not panels:
        st.info("该案例的对照图尚未导出。")
        return
    panel = panels[0]
    if len(panels) > 1:
        with selectors[1]:
            chosen = st.selectbox("查看哪次训练的对照", range(len(panels)),
                                  format_func=lambda i: panels[i]["label"],
                                  key="rgb_results_panel_" + selected)
        panel = panels[chosen]
    if case.get("participated_in_training"):
        st.caption("这张图片参与过训练；较高的参考重合度不能代表新图识别效果。")
        if case.get("training_role") == "derived_human_feedback_fit":
            st.caption("本轮用到了你确认的局部类别反馈；完整边界未审核，不确定区域继续忽略。")
    else:
        st.caption("这张图片仅作局部参考；只有部分区域得到确认，完整边界尚未审核。")
    st.caption("黄色是预测田埂，灰色是预测道路；田块保持原图颜色。")
    try:
        content = _artifact(panel)
        st.image(content, caption=panel.get("label", "原图与预测对照"), use_container_width=True)
        st.download_button("下载当前对照图", content, file_name=selected + "-对照.png",
                           mime="image/png", key="rgb_results_image_download")
        statistics = document.get("statistics")
        if statistics:
            csv_bytes = _artifact(statistics)
            rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
            chosen_rows = [row for row in rows if selected in row.get("案例", "").split("+")]
            if chosen_rows:
                st.markdown("**这张图片的预测像素统计**")
                st.dataframe(chosen_rows, use_container_width=True, hide_index=True)
            st.download_button("下载全部预测像素统计", csv_bytes,
                               file_name="RGB预测像素统计.csv", mime="text/csv",
                               key="rgb_results_csv_download")
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("结果文件正在更新或已变化，请稍后刷新；原始图片和标注不会因此改变。")
    counts = document.get("selection_counts", {})
    st.caption(f"训练范围：{counts.get('experiment_train', '未知')}张；"
               f"暂缓：{counts.get('deferred_alias', '未知')}张；"
               f"排除：{counts.get('excluded_user_decision', '未知')}张。")
    feedback = document.get("feedback_training", {})
    if feedback:
        st.caption(f"另有{feedback.get('feedback_images', '未知')}张局部反馈学习副本；"
                   f"这些对照图对应的微调涉及{feedback.get('current_unique_images', '未知')}张图片。"
                   "原暂缓清单保留，反馈学习结果只表示这些已见图片的拟合。")
    st.caption("面积单位仅为像素，不是平方米或真实田块数量。本页不是独立精度评估，模型仍需改进。")
