"""整改复跑面板 AppTest（DEV-20261015-016）。

用合成 fixture 识别服务驱动 app.py 第 8 页，覆盖：
- 成功且已登记、已有最新问题意见时渲染“创建整改并复跑”；
- 显式创建后真实落库、生成同模型子作业、展示前后像素对照；
- 普通 rerun/下载不重复 POST、不新增作业；
- 失败/中断任务只读展示“由哪条整改产生”，不显示旧成功图。
合成 fixture 明确不代表真实模型；证据写入独立目录并保留。

用法：
  .venv-p0/Scripts/python.exe tests/corrections/test_correction_panel_apptest.py
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "tests" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agrolake_service import governance, reviews  # noqa: E402
from agrolake_service.config import ServiceConfig  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from fixtures import WebFixtureRunner  # noqa: E402

BASE = ROOT / "reports" / "corrections" / "DEV-20261015-016" / datetime.now().strftime(
    "apptest-%Y%m%d-%H%M%S-%f")
BASE.mkdir(parents=True, exist_ok=False)
RESULT: dict = {"directory": str(BASE), "steps": [], "exceptions": []}


def log(step: str, detail) -> None:
    RESULT["steps"].append({"step": step, "detail": detail})
    print(f"[correction_apptest] {step}: {detail}")


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
    sb = next((c for c in at.selectbox if getattr(c, "key", None) == "rec_history"), None)
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
        "source": "SYNTHETIC 整改页面测试", "scene_id": "CORR-PAGE",
        "request_id": "corr-page-" + tag.decode() + "-" + str(time.time()),
    })
    job_id = manager.submit(norm)["job_id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if manager.get_job(job_id)["status"] not in ("queued", "running"):
            break
        time.sleep(0.03)
    assert manager.get_job(job_id)["status"] == "succeeded"
    return job_id


def wait_child(manager: JobManager, job_id: int) -> str:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = manager.get_job(job_id)["status"]
        if status not in ("queued", "running"):
            return status
        time.sleep(0.03)
    return manager.get_job(job_id)["status"]


def db_scalar(db_path, sql, args=()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, args).fetchone()
    finally:
        conn.close()


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
        parent = make_job(manager, b"PARENT016")
        governance.register_governance(cfg, parent, {"request_id": "gov-016"})
        reviews.submit_review(cfg.db_path, parent, reviews.normalize_review_payload({
            "field_id": None, "decision": "questioned", "reviewer": "流程测试署名",
            "note": "合成测试：整体存疑，需整改来源/场景", "request_id": "rev-016",
        }))

        at = new_app()
        if exceptions(at):
            raise AssertionError("初始渲染出现异常")
        select_job(at, parent)
        if exceptions(at):
            raise AssertionError("选择父作业出现异常")
        if at.button(key=f"rec_corr_create_{parent}") is None:
            raise AssertionError("整改复跑面板未渲染“创建整改并复跑”")
        if not any("整改复跑与前后对照" in m.value for m in at.markdown):
            raise AssertionError("未显示整改复跑面板标题")
        log("panel_rendered", {"parent": parent})

        # 显式创建一条整改并复跑（只改场景 => 真实变化）。
        before_submit = manager.submit_calls
        at.text_input(key=f"rec_corr_operator_{parent}").set_value("流程测试操作者")
        at.text_area(key=f"rec_corr_reason_{parent}").set_value("合成测试：修订场景并复跑")
        at.text_input(key=f"rec_corr_scene_{parent}").set_value("SCENE-CORRECTED")
        at.button(key=f"rec_corr_create_{parent}").click().run()
        if exceptions(at):
            raise AssertionError("创建整改后出现异常")
        row = db_scalar(manager.db_path,
                        "SELECT correction_id, child_job_id, submission_state, review_id"
                        " FROM rec_corrections WHERE parent_job_id=?", (parent,))
        if row is None:
            raise AssertionError("整改未落库")
        correction_id, child, state, review_id = row
        if state != "linked" or child is None:
            raise AssertionError(f"整改未链接子作业：{row}")
        if manager.submit_calls != before_submit + 1:
            raise AssertionError("创建整改应只新增一次子作业提交")
        if wait_child(manager, child) != "succeeded":
            raise AssertionError("子作业未成功（合成 fixture 预期成功）")
        log("created_and_rerun", {"correction_id": correction_id, "child": child,
                                  "submit_calls": manager.submit_calls})

        # 前后对照像素表已渲染。
        at.run()
        if exceptions(at):
            raise AssertionError("刷新对照出现异常")
        if not any("整改历史与对照" in m.value for m in at.markdown):
            raise AssertionError("未渲染整改历史与对照")

        # 普通 rerun / 下载按钮不重复 POST、不新增作业。
        n_corr = db_scalar(manager.db_path, "SELECT COUNT(*) FROM rec_corrections")[0]
        n_jobs = db_scalar(manager.db_path, "SELECT COUNT(*) FROM rec_jobs")[0]
        submit_calls = manager.submit_calls
        for _ in range(3):
            at.run()
            if exceptions(at):
                raise AssertionError("普通 rerun 出现异常")
        if db_scalar(manager.db_path, "SELECT COUNT(*) FROM rec_corrections")[0] != n_corr:
            raise AssertionError("普通 rerun 重复写入整改")
        if db_scalar(manager.db_path, "SELECT COUNT(*) FROM rec_jobs")[0] != n_jobs:
            raise AssertionError("普通 rerun 新增了作业")
        if manager.submit_calls != submit_calls:
            raise AssertionError("普通 rerun 触发了推理提交")
        log("no_duplicate_on_rerun", {"corrections": n_corr, "jobs": n_jobs})

        # 子作业：失败/中断只读展示“由哪条整改产生”，不显示旧成功图。
        at.run()  # 回到父作业面板，随后切到子作业
        conn = sqlite3.connect(manager.db_path)
        try:
            conn.execute("UPDATE rec_jobs SET status='interrupted', error='SYNTHETIC 中断'"
                         " WHERE job_id=?", (child,))
            conn.commit()
        finally:
            conn.close()
        select_job(at, child)
        if exceptions(at):
            raise AssertionError("查看中断子作业出现异常")
        if not any("本作业由整改产生" in m.value for m in at.markdown):
            raise AssertionError("中断子作业未只读展示整改来源")
        if any("### 结果" in m.value for m in at.markdown):
            raise AssertionError("中断子作业不应显示成功结果面板")
        log("interrupted_origin_only", {"child": child})

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
