"""用 Streamlit AppTest 实际驱动 app.py：多次 run、数据刷新、处置、导出，
并验证“报告下载按钮在普通 rerun 后仍保留”。

- 每次运行写入项目内新目录 `reports/p0/ui-<唯一时间>/`，保留历史目录，不覆盖旧证据。
- 只使用项目内目录，不写系统 temp，也不触碰演示数据库。
用法：
  .venv-p0/Scripts/python.exe scripts/p0/ui_apptest.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 唯一时间目录：保留旧文件，不重写历史证据。
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
UI_DIR = ROOT / "reports" / "p0" / f"ui-{STAMP}"
STORE = UI_DIR / "store"
DBF = UI_DIR / "ui.sqlite"
EXPORTS = UI_DIR / "exports"
os.environ["AGROLAKE_DATA_DIR"] = str(UI_DIR)
os.environ["AGROLAKE_STORE"] = str(STORE)
os.environ["AGROLAKE_DB"] = str(DBF)
os.environ["AGROLAKE_REPORTS"] = str(EXPORTS)

from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from agrolake import db, ingest, reporting, scenes  # noqa: E402

RESULT: dict = {"ui_dir": str(UI_DIR), "steps": [], "exceptions": []}


def log(step: str, detail) -> None:
    RESULT["steps"].append({"step": step, "detail": detail})
    print(f"[ui_apptest] {step}: {detail}")


def png_bytes(color=(30, 120, 90), size=(64, 48), name="u") -> bytes:
    img = Image.new("RGB", size, color)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = ((x * 7 + hash(name)) % 256, (y * 11) % 256, ((x + y) * 13) % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def seed(name: str, data: bytes, **kw):
    conn = db.connect()
    db.init_schema(conn)
    res = ingest.import_bytes(conn, data, name, store_root=STORE, **kw)
    conn.close()
    return res


def _collect_exceptions(at: AppTest) -> list[str]:
    excs = [str(e.value) for e in at.exception]
    RESULT["exceptions"].extend(excs)
    return excs


def run_app(label: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    excs = _collect_exceptions(at)
    log(f"run {label}", {"exceptions": excs, "n_tabs": len(at.tabs)})
    if excs:
        raise AssertionError(f"{label} 出现异常：{excs}")
    return at


def click(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click().run()
    excs = _collect_exceptions(at)
    if excs:
        raise AssertionError(f"点击 {key} 出现异常：{excs}")
    return at


def download_labels(at: AppTest) -> list[str]:
    """AppTest 以 download_button 元素暴露下载按钮。

    下载按钮原型（DownloadButton）不含 key 字段，故按可见标签识别，
    标签形如“下载 assets.csv”。
    """
    labels: list[str] = []
    for d in at.get("download_button"):
        proto = getattr(d, "proto", None)
        labels.append(getattr(proto, "label", None) or getattr(d, "label", None))
    return labels


def set_selectbox(at: AppTest, label: str, value: str) -> bool:
    for sb in at.selectbox:
        if sb.label == label:
            sb.set_value(value)
            return True
    return False


def main() -> int:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS.mkdir(parents=True, exist_ok=True)

    # 场景 S1 三类齐全
    seed("s1.png", png_bytes((10, 20, 30), name="a"), source="SYN-合成", scene_id="S1")
    seed("s1.csv",
         "scene_id,field_id,crop,category,area,unit\nS1,F1,rice,田块,1,ha\n".encode(),
         source="SYN-合成", scene_id="S1")
    seed("s1.json", json.dumps({"scene_id": "S1", "note": "复核"}, ensure_ascii=False).encode(),
         source="SYN-合成", scene_id="S1")

    run_app("1")
    c1 = db.connect()
    db.init_schema(c1)
    versions_1 = len(scenes.list_versions(c1))
    c1.close()
    log("after run 1", {"versions": versions_1})

    # 数据刷新：S1 再导入一张，另建 S2 三类；并加入坏图/空图用于预览降级验证。
    seed("s1b.png", png_bytes((200, 30, 30), name="b"), source="SYN-合成", scene_id="S1")
    seed("s1bad.png", b"\x89PNG\r\n\x1a\n" + b"not-a-real-png-body",
         source="SYN-合成", scene_id="S1")
    seed("s1empty.png", b"", source="SYN-合成", scene_id="S1")
    seed("s2.png", png_bytes((20, 30, 200), name="c"), source="SYN-合成", scene_id="S2")
    seed("s2.csv",
         "scene_id,field_id,crop,category,area,unit\nS2,F2,maize,田块,2,ha\n".encode(),
         source="SYN-合成", scene_id="S2")
    seed("s2.json", json.dumps({"scene_id": "S2", "note": "复核2"}, ensure_ascii=False).encode(),
         source="SYN-合成", scene_id="S2")

    at2 = run_app("2")
    c2 = db.connect()
    db.init_schema(c2)
    versions_2 = len(scenes.list_versions(c2))
    members = scenes.scene_members(c2, "S1")
    c2.close()
    log("after run 2 (refresh)", {"versions": versions_2})
    if versions_2 <= versions_1:
        raise AssertionError("第二次 run 未看到新导入数据（无刷新）")
    if not members["roles"]["image"] or not members["roles"]["attribute"] \
            or not members["roles"]["text"]:
        raise AssertionError(f"场景 S1 三类未齐全：{members['missing_roles']}")

    # --- 坏图/空图预览降级验证（DEV-003 核心回归） ---
    warnings = [w.value for w in at2.warning]
    captions = [c.value for c in at2.caption]
    markdowns = [m.value for m in at2.markdown]
    for bad in ("s1bad.png", "s1empty.png"):
        if not any(bad in w and "隔离或替换文件" in w for w in warnings):
            raise AssertionError(f"坏图 {bad} 未显示友好提示与操作引导：{warnings}")
    if not any("64×48" in c for c in captions):
        raise AssertionError("好图尺寸未正常显示")
    if not any("未知×未知" in c for c in captions):
        raise AssertionError("坏图未知元数据未显示为“未知”")
    imgs = at2.get("imgs")
    if len(imgs) < 1:
        raise AssertionError("好图应仍触发 st.image 预览，但未渲染任何影像元素")
    all_text = "\n".join(warnings + captions + markdowns)
    for bad_token in ("None×None", "None 波段", "None 位"):
        if bad_token in all_text:
            raise AssertionError(f"页面仍出现原始 None 元数据提示：{bad_token}")
    log("bad-image preview downgrade", {
        "friendly_warnings": [w for w in warnings if "隔离或替换文件" in w][:4],
        "has_unknown_meta": any("未知×未知" in c for c in captions),
        "rendered_image_elements": len(imgs),
    })

    # 处置：隔离 S1 的最新影像，再 run，应无异常且关联仍只含当前有效版本
    from agrolake import disposition
    c3 = db.connect()
    db.init_schema(c3)
    latest_img = max(v["version_id"] for v in scenes.list_versions(c3)
                     if v["original_filename"] == "s1b.png")
    disposition.isolate(c3, latest_img, "apptest", "隔离最新影像")
    c3.close()
    run_app("3 (after isolate)")
    c4 = db.connect()
    db.init_schema(c4)
    cur_ids = [v["version_id"] for v in scenes.current_versions(c4)]
    c4.close()
    log("after isolate", {"isolated": latest_img, "current_versions": cur_ids})
    if latest_img in cur_ids:
        raise AssertionError("隔离后的版本仍出现在当前有效版本中")

    # --- 下载按钮跨 rerun 保留验证（DEV-003 核心回归） ---
    at = run_app("reports-initial")
    before = download_labels(at)
    if before:
        raise AssertionError(f"未点击生成即出现下载按钮：{before}")

    at = click(at, "gen_assets.csv")
    after_csv = download_labels(at)
    log("after gen csv", {"download_labels": after_csv})
    if "下载 assets.csv" not in after_csv:
        raise AssertionError(f"生成资产清单 CSV 后缺少下载按钮：{after_csv}")

    at = click(at, "gen_report.json")
    after_json = download_labels(at)
    log("after gen json", {"download_labels": after_json})
    if "下载 assets.csv" not in after_json or "下载 report.json" not in after_json:
        raise AssertionError(f"生成报告 JSON 后已有下载按钮未保留：{after_json}")

    # 另一交互：切换场景页下拉，触发普通 rerun；下载按钮应仍在
    if not set_selectbox(at, "选择场景", "S2"):
        raise AssertionError("未找到“选择场景”下拉，无法执行另一交互")
    at.run()
    excs = _collect_exceptions(at)
    if excs:
        raise AssertionError(f"切换场景出现异常：{excs}")
    after_other = download_labels(at)
    log("after other interaction", {"download_labels": after_other})
    if "下载 assets.csv" not in after_other or "下载 report.json" not in after_other:
        raise AssertionError(f"其它交互后下载按钮未保留：{after_other}")

    # 导出可重读（直接调用导出接口）
    c5 = db.connect()
    db.init_schema(c5)
    csv_p = reporting.export_assets_csv(c5, EXPORTS / "assets.csv")
    json_p = reporting.export_report_json(c5, EXPORTS / "report.json")
    c5.close()
    with open(csv_p, encoding="utf-8-sig") as fh:
        head = fh.readline().strip()
    rep = json.loads(Path(json_p).read_text(encoding="utf-8"))
    log("export reread", {"csv_header": head, "report_keys": sorted(rep)[:4]})
    if "version_id" not in head or "summary" not in rep:
        raise AssertionError("导出文件不可重读")

    # --- DEV-20261015-005：真实按钮处置后页面即时刷新 / 一次点击一次写入 / flash 消费 ---
    def db_counts(vid: int):
        c = db.connect()
        db.init_schema(c)
        cur = len(scenes.current_versions(c))
        disps = {
            r[0]: r[1] for r in c.execute(
                "SELECT action, COUNT(*) FROM dispositions WHERE version_id=?"
                " GROUP BY action", (vid,))
        }
        src = c.execute(
            "SELECT source FROM asset_versions WHERE version_id=?", (vid,)
        ).fetchone()[0]
        c.close()
        return cur, disps, src

    def overview_current_metric(at: AppTest):
        for m in at.metric:
            if m.label == "当前有效版本":
                return m.value
        return None

    def select_version(at: AppTest, vid: int) -> None:
        sb = next((s for s in at.selectbox if s.label == "选择版本"), None)
        if sb is None:
            raise AssertionError("未找到处置页“选择版本”下拉")
        opt = next((o for o in sb.options if o.startswith(f"#{vid} ")), None)
        if opt is None:
            raise AssertionError(f"下拉中未找到版本 #{vid}：{list(sb.options)}")
        sb.set_value(opt)
        at.run()
        _collect_exceptions(at)

    def click_label(at: AppTest, label: str) -> AppTest:
        btn = next((b for b in at.button if b.label == label), None)
        if btn is None:
            raise AssertionError(f"未找到处置按钮：{label}")
        btn.click()
        at.run()
        excs = _collect_exceptions(at)
        if excs:
            raise AssertionError(f"点击 {label} 出现异常：{excs}")
        return at

    # DEV-005 处置流程在已有下载的同一 AppTest 会话内继续，验证处置多次 rerun 后下载保留。
    at5 = at
    c = db.connect()
    db.init_schema(c)
    target = next(v for v in scenes.list_versions(c)
                  if v["original_filename"] == "s1.png" and v["status"] == "active")
    c.close()
    vid = target["version_id"]

    select_version(at5, vid)
    cur0, disp0, _ = db_counts(vid)

    # 隔离：点击后概览指标立刻与数据库一致，一次点击只写一次隔离处置
    click_label(at5, "隔离该版本")
    cur1, disp1, _ = db_counts(vid)
    log("005 isolate click", {"before": cur0, "after": cur1, "dispositions": disp1})
    if cur1 != cur0 - 1:
        raise AssertionError(f"隔离后当前有效版本数未减少：{cur0} -> {cur1}")
    if disp1.get("isolate", 0) != disp0.get("isolate", 0) + 1:
        raise AssertionError(f"隔离一次点击应只写一次处置：{disp0} -> {disp1}")
    mv1 = overview_current_metric(at5)
    if str(mv1) != str(cur1):
        raise AssertionError(f"隔离后概览“当前有效版本”={mv1}，数据库={cur1}，未即时刷新")
    if not any("已隔离" in s.value for s in at5.success):
        raise AssertionError("隔离成功后未显示 flash 提示")

    # 普通 rerun：不重复写、提示已消费、指标仍与库一致
    at5.run()
    _collect_exceptions(at5)
    cur1b, disp1b, _ = db_counts(vid)
    log("005 plain rerun after isolate", {"current": cur1b, "dispositions": disp1b})
    if disp1b.get("isolate", 0) != disp1.get("isolate", 0):
        raise AssertionError(f"普通 rerun 重复写入隔离处置：{disp1} -> {disp1b}")
    if str(overview_current_metric(at5)) != str(cur1b):
        raise AssertionError("普通 rerun 后概览指标与数据库不一致")
    if any("已隔离" in s.value for s in at5.success):
        raise AssertionError("flash 提示未在下一次 rerun 中被消费")

    # 解除隔离：同一 AppTest 会话真实按钮，指标立即恢复
    select_version(at5, vid)
    click_label(at5, "解除隔离")
    cur2, disp2, _ = db_counts(vid)
    log("005 unisolate click", {"current": cur2, "dispositions": disp2})
    if cur2 != cur0:
        raise AssertionError(f"解除隔离后当前有效版本数未恢复：{cur2} != {cur0}")
    if disp2.get("unisolate", 0) != disp1b.get("unisolate", 0) + 1:
        raise AssertionError(f"解除一次点击应只写一次处置：{disp1b} -> {disp2}")
    if str(overview_current_metric(at5)) != str(cur2):
        raise AssertionError(
            f"解除后概览指标={overview_current_metric(at5)}，数据库={cur2}")

    # 补充元数据成功：真实按钮一次写一次、来源实际更新、flash 显示
    select_version(at5, vid)
    ti = {t.label: t for t in at5.text_input}
    if "来源 source" not in ti or "原因" not in ti:
        raise AssertionError(f"未找到补充元数据输入框：{sorted(ti)}")
    ti["来源 source"].set_value("SYN-合成-补充")
    ti["原因"].set_value("DEV-005 按钮验证")
    _, disp_b_supp, _ = db_counts(vid)
    click_label(at5, "保存并复检")
    _, disp_a_supp, src_a = db_counts(vid)
    log("005 supplement click", {"before": disp_b_supp, "after": disp_a_supp,
                                 "source": src_a})
    if disp_a_supp.get("supplement_metadata", 0) != \
            disp_b_supp.get("supplement_metadata", 0) + 1:
        raise AssertionError(
            f"补充元数据一次点击应只写一次：{disp_b_supp} -> {disp_a_supp}")
    if src_a != "SYN-合成-补充":
        raise AssertionError(f"来源未实际更新：{src_a}")
    if not any("已补充元数据" in s.value for s in at5.success):
        raise AssertionError("补充元数据成功后未显示 flash 提示")

    # 仅复检（不改状态）：一次点击只写一次复检处置
    select_version(at5, vid)
    _, disp_b_rc, _ = db_counts(vid)
    click_label(at5, "仅复检（不改状态）")
    _, disp_a_rc, _ = db_counts(vid)
    log("005 recheck click", {"before": disp_b_rc, "after": disp_a_rc})
    if disp_a_rc.get("recheck", 0) != disp_b_rc.get("recheck", 0) + 1:
        raise AssertionError(f"复检一次点击应只写一次：{disp_b_rc} -> {disp_a_rc}")
    if not any("已复检" in s.value for s in at5.success):
        raise AssertionError("复检成功后未显示 flash 提示")

    # 处置流程多次真实 rerun 后，本会话已有下载仍保留
    dl5 = download_labels(at5)
    log("005 downloads after dispositions", {"download_labels": dl5})
    if "下载 assets.csv" not in dl5 or "下载 report.json" not in dl5:
        raise AssertionError(f"处置多次 rerun 后下载按钮未保留：{dl5}")

    RESULT["ok"] = True
    out = UI_DIR / "ui_apptest_result.json"
    out.write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AppTest 通过，结果写入 {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # 留下失败证据
        RESULT["ok"] = False
        RESULT["error"] = str(exc)
        UI_DIR.mkdir(parents=True, exist_ok=True)
        (UI_DIR / "ui_apptest_result.json").write_text(
            json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"AppTest 失败：{exc}")
        raise SystemExit(1)
