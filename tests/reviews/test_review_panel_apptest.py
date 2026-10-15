"""人工复核网页面板 AppTest（DEV-20261015-013）。

用合成 fixture 识别服务驱动 app.py 第 8 页，覆盖：无记录显示未复核、保存可见、
回执丢失后同号重试、普通 rerun/下载不重复 POST、切换任务不串旧意见、
API 冲突/不可用时不假报成功。合成 fixture 明确不代表真实模型。

用法：
  .venv-p0/Scripts/python.exe tests/reviews/test_review_panel_apptest.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from agrolake_web.client import ApiError, RecognitionClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from fixtures import WebFixtureRunner  # noqa: E402

BASE = ROOT / "reports" / "reviews" / "DEV-20261015-013" / datetime.now().strftime(
    "apptest-%Y%m%d-%H%M%S-%f")
BASE.mkdir(parents=True, exist_ok=False)
RESULT: dict = {"directory": str(BASE), "steps": [], "exceptions": []}


def log(step: str, detail) -> None:
    RESULT["steps"].append({"step": step, "detail": detail})
    print(f"[review_apptest] {step}: {detail}")


class CountingManager(JobManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submit_calls = 0

    def submit(self, normalized):
        self.submit_calls += 1
        return super().submit(normalized)


def set_env(directory: Path, url: str) -> None:
    os.environ["AGROLAKE_DATA_DIR"] = str(directory / "p0")
    os.environ["AGROLAKE_STORE"] = str(directory / "p0" / "store")
    os.environ["AGROLAKE_DB"] = str(directory / "p0" / "agrolake.sqlite")
    os.environ["AGROLAKE_REPORTS"] = str(directory / "p0" / "reports")
    os.environ["AGROLAKE_API_URL"] = url


def new_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    return at


def exceptions(at: AppTest) -> list[str]:
    excs = [str(e.value) for e in at.exception]
    RESULT["exceptions"].extend(excs)
    return excs


def select_job(at: AppTest, job_id: int) -> None:
    sb = None
    for candidate in at.selectbox:
        if getattr(candidate, "key", None) == "rec_history":
            sb = candidate
            break
    if sb is None:
        raise AssertionError("未找到历史任务下拉")
    label = next((o for o in sb.options if str(o).startswith(f"#{job_id} ")), None)
    if label is None:
        raise AssertionError(f"下拉未包含作业 #{job_id}：{sb.options}")
    sb.set_value(label)
    at.run()


def make_job(manager: JobManager, tag: bytes) -> int:
    norm = validate_job_payload({
        "filename": f"{tag.decode()}.tif",
        "content_base64": base64.b64encode(tag).decode(),
        "source": "SYNTHETIC 复核页面测试", "scene_id": "REVIEW-PAGE",
        "request_id": "page-" + tag.decode() + "-" + str(time.time()),
    })
    job_id = manager.submit(norm)["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if manager.get_job(job_id)["status"] not in ("queued", "running"):
            break
        time.sleep(0.03)
    assert manager.get_job(job_id)["status"] == "succeeded"
    return job_id


def review_rows(manager: JobManager, job_id: int) -> int:
    import sqlite3
    conn = sqlite3.connect(manager.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM rec_reviews WHERE job_id=?",
                            (job_id,)).fetchone()[0]
    finally:
        conn.close()


def fill_task_form(at: AppTest, job_id: int, *, reviewer: str, note: str,
                   decision_label: str | None = None) -> None:
    if decision_label is not None:
        at.selectbox(key=f"rec_review_decision_task_{job_id}").set_value(decision_label)
    at.text_input(key=f"rec_review_reviewer_task_{job_id}").set_value(reviewer)
    at.text_area(key=f"rec_review_note_task_{job_id}").set_value(note)


def scenario() -> None:
    d = BASE / "fixture"
    d.mkdir(parents=True, exist_ok=True)
    cfg = ServiceConfig(db_path=d / "service.sqlite", store_root=d / "store",
                        runs_root=d / "runs", port=0, synthetic_mode=True,
                        model_sha256="f" * 64)
    manager = CountingManager(cfg, WebFixtureRunner("success"))
    server = make_server(cfg, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manager.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        set_env(d, url)
        job_a = make_job(manager, b"A")
        job_b = make_job(manager, b"B")

        at = new_app()
        if exceptions(at):
            raise AssertionError("初始渲染出现异常")
        select_job(at, job_a)
        if exceptions(at):
            raise AssertionError("选择作业 A 出现异常")
        if at.button(key=f"rec_review_save_task_{job_a}") is None:
            raise AssertionError("复核面板未渲染保存按钮")
        if not any("未复核" in c.value for c in at.caption):
            raise AssertionError("无记录时未显示“未复核”")
        log("no_records", {"job_a": job_a, "job_b": job_b})

        # 保存一条任务整体意见
        fill_task_form(at, job_a, reviewer="流程测试署名", note="合成测试：整体存疑")
        at.button(key=f"rec_review_save_task_{job_a}").click().run()
        if exceptions(at):
            raise AssertionError("保存后出现异常")
        if not any("已保存人工复核意见" in s.value for s in at.success):
            raise AssertionError("保存后未显示成功提示")
        if review_rows(manager, job_a) != 1:
            raise AssertionError("保存未落库或落库多条")
        log("save", {"review_rows": review_rows(manager, job_a)})

        # 普通 rerun/下载按钮不重复 POST
        manager.submit_calls = 0
        for _ in range(3):
            at.run()
            if exceptions(at):
                raise AssertionError("普通 rerun 出现异常")
        csv_btns = [b for b in at.get("download_button")]
        if not any(getattr(getattr(b, "proto", None), "label", None) == "下载复核记录 CSV"
                   for b in csv_btns):
            raise AssertionError("未提供复核记录 CSV 下载")
        if review_rows(manager, job_a) != 1:
            raise AssertionError("普通 rerun/下载后复核记录数变化")
        log("no_duplicate_on_rerun", {"review_rows": review_rows(manager, job_a)})

        # 回执丢失 -> 同号重试
        original = RecognitionClient.submit_review
        seen: list = []

        def lose_first(client, job_id, **kw):
            seen.append(kw["request_id"])
            result = original(client, job_id, **kw)
            if len(seen) == 1:
                raise ApiError("timeout", "SYNTHETIC reply lost after acceptance")
            return result

        before = review_rows(manager, job_a)
        with patch.object(RecognitionClient, "submit_review", lose_first):
            fill_task_form(at, job_a, reviewer="流程测试署名2", note="合成测试：回执丢失")
            at.button(key=f"rec_review_save_task_{job_a}").click().run()
            if exceptions(at):
                raise AssertionError("回执丢失后出现异常")
            if not any("尚未确认" in w.value for w in at.warning):
                raise AssertionError("回执丢失未给出确认提示")
            if review_rows(manager, job_a) != before + 1:
                raise AssertionError("服务端应已接受但回执丢失时未落库")
            retry = at.button(key=f"rec_review_retry_task_{job_a}")
            if retry is None:
                raise AssertionError("回执丢失后未提供同号重试按钮")
            retry.click().run()
            if exceptions(at):
                raise AssertionError("重试后出现异常")
        if len(seen) != 2 or seen[0] != seen[1]:
            raise AssertionError(f"重试未复用同一请求号：{seen}")
        if review_rows(manager, job_a) != before + 1:
            raise AssertionError("重试重复写入")
        log("lost_reply_retry", {"before": before, "after": review_rows(manager, job_a),
                                 "request_ids": seen})

        # 切换任务不串旧意见
        select_job(at, job_b)
        if exceptions(at):
            raise AssertionError("选择作业 B 出现异常")
        if review_rows(manager, job_b) != 0:
            raise AssertionError("作业 B 被写入了他人的复核记录")
        if not any("未复核" in c.value for c in at.caption):
            raise AssertionError("作业 B 未显示未复核（可能串了 A 的意见）")
        if not any(f"#{job_b}" in m.value for m in at.markdown):
            raise AssertionError("未切换到作业 B")
        log("switch_no_bleed", {"job_b": job_b})

        # 409 冲突不假报成功
        def conflict(client, job_id, **kw):
            raise ApiError("conflict", "SYNTHETIC 409 conflict")

        with patch.object(RecognitionClient, "submit_review", conflict):
            fill_task_form(at, job_b, reviewer="冲突测试", note="合成测试：409")
            at.button(key=f"rec_review_save_task_{job_b}").click().run()
            if exceptions(at):
                raise AssertionError("409 后出现异常")
            if not any("复核保存失败" in e.value for e in at.error):
                raise AssertionError("409 未给出错误提示")
            if any("已保存人工复核意见" in s.value for s in at.success):
                raise AssertionError("409 假报成功")
        if review_rows(manager, job_b) != 0:
            raise AssertionError("409 后不应写入记录")
        log("conflict_no_false_success", {"job_b": job_b})

        RESULT["ok"] = True
    finally:
        manager.stop()
        server.shutdown()
        server.server_close()
        thread.join(5)


def main() -> int:
    try:
        scenario()
        RESULT["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        RESULT["ok"] = False
        RESULT["error"] = f"{type(exc).__name__}: {exc}"
        import traceback
        RESULT["traceback"] = traceback.format_exc()
    (BASE / "result.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps({"ok": RESULT["ok"], "steps": [s["step"] for s in RESULT["steps"]],
                      "error": RESULT.get("error")}, ensure_ascii=False))
    return 0 if RESULT["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
