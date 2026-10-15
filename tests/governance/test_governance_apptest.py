"""治理登记网页面板 AppTest（DEV-20261015-014）。

用合成 fixture 识别服务驱动 app.py 第 8 页，覆盖：无登记显示未登记、显式登记可见、
回执丢失后同号重试、普通 rerun/下载不重复登记、切换任务不串、API 409 不假报成功。
合成 fixture 明确不代表真实模型或真实 TIFF。

用法：
  .venv-p0/Scripts/python.exe tests/governance/test_governance_apptest.py
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

BASE = ROOT / "reports" / "governance" / "DEV-20261015-014" / datetime.now().strftime(
    "apptest-%Y%m%d-%H%M%S-%f")
BASE.mkdir(parents=True, exist_ok=False)
RESULT: dict = {"directory": str(BASE), "steps": [], "exceptions": []}


def log(step: str, detail) -> None:
    RESULT["steps"].append({"step": step, "detail": detail})
    print(f"[gov_apptest] {step}: {detail}")


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
        "source": "SYNTHETIC 治理页面测试", "scene_id": "GOV-PAGE",
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


def link_rows(manager: JobManager, job_id: int | None = None) -> int:
    conn = sqlite3.connect(manager.db_path)
    try:
        if job_id is None:
            return conn.execute("SELECT COUNT(*) FROM rec_governance_links").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM rec_governance_links WHERE job_id=?",
                            (job_id,)).fetchone()[0]
    finally:
        conn.close()


def scenario() -> None:
    d = BASE / "fixture"
    d.mkdir(parents=True, exist_ok=True)
    cfg = ServiceConfig(db_path=d / "service.sqlite", store_root=d / "store",
                        runs_root=d / "runs", port=0, synthetic_mode=True,
                        model_sha256="f" * 64)
    manager = JobManager(cfg, WebFixtureRunner("success"))
    server = make_server(cfg, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manager.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        set_env(d, url)
        job_a = make_job(manager, b"GA")
        job_b = make_job(manager, b"GB")

        at = new_app()
        if exceptions(at):
            raise AssertionError("初始渲染出现异常")
        select_job(at, job_a)
        if exceptions(at):
            raise AssertionError("选择作业 A 出现异常")
        if at.button(key=f"rec_gov_register_{job_a}") is None:
            raise AssertionError("治理面板未渲染登记按钮")
        if not any("未登记" in str(i.value) for i in at.info):
            raise AssertionError("无登记时未显示“未登记”")
        if link_rows(manager) != 0:
            raise AssertionError("仅查看不应写入登记")
        log("no_registration", {"job_a": job_a, "job_b": job_b})

        # 显式登记
        at.button(key=f"rec_gov_register_{job_a}").click().run()
        if exceptions(at):
            raise AssertionError("登记后出现异常")
        if not any("已登记识别输入资料" in s.value for s in at.success):
            raise AssertionError("登记后未显示成功提示")
        if link_rows(manager, job_a) != 1:
            raise AssertionError("登记未落库或落库多条")
        log("register", {"links": link_rows(manager, job_a)})

        # 普通 rerun/下载不重复登记
        for _ in range(3):
            at.run()
            if exceptions(at):
                raise AssertionError("普通 rerun 出现异常")
        dl = [b for b in at.get("download_button")]
        if not any("下载治理血缘 JSON" in str(getattr(getattr(b, "proto", None), "label", ""))
                   for b in dl):
            raise AssertionError("未提供治理血缘 JSON 下载")
        if link_rows(manager, job_a) != 1:
            raise AssertionError("普通 rerun/下载后登记数变化")
        log("no_duplicate_on_rerun", {"links": link_rows(manager, job_a)})

        # 回执丢失 -> 同号重试
        original = RecognitionClient.register_governance
        seen: list = []

        def lose_first(client, job_id, *, request_id, **kw):
            seen.append(request_id)
            result = original(client, job_id, request_id=request_id, **kw)
            if len(seen) == 1:
                raise ApiError("timeout", "SYNTHETIC reply lost after acceptance")
            return result

        before = link_rows(manager, job_b)
        with patch.object(RecognitionClient, "register_governance", lose_first):
            select_job(at, job_b)
            at.button(key=f"rec_gov_register_{job_b}").click().run()
            if exceptions(at):
                raise AssertionError("回执丢失后出现异常")
            if not any("尚未确认" in w.value for w in at.warning):
                raise AssertionError("回执丢失未给出确认提示")
            if link_rows(manager, job_b) != before + 1:
                raise AssertionError("服务端应已接受但回执丢失时未落库")
            retry = at.button(key=f"rec_gov_retry_{job_b}")
            if retry is None:
                raise AssertionError("回执丢失后未提供同号重试按钮")
            retry.click().run()
            if exceptions(at):
                raise AssertionError("重试后出现异常")
        if len(seen) != 2 or seen[0] != seen[1]:
            raise AssertionError(f"重试未复用同一请求号：{seen}")
        if link_rows(manager, job_b) != before + 1:
            raise AssertionError("重试重复登记")
        log("lost_reply_retry", {"before": before, "after": link_rows(manager, job_b),
                                 "request_ids": seen})

        # 切换任务不串状态
        select_job(at, job_a)
        if exceptions(at):
            raise AssertionError("切回作业 A 出现异常")
        if not any("已登记" in str(m.value) for m in at.markdown):
            raise AssertionError("作业 A 应显示已登记")
        log("switch_no_bleed", {"job_a": job_a})

        # 409 冲突不假报成功
        job_c = make_job(manager, b"GC")
        at.run()  # 刷新历史列表，使新作业进入下拉

        def conflict(client, job_id, *, request_id, **kw):
            raise ApiError("conflict", "SYNTHETIC 409 conflict")

        with patch.object(RecognitionClient, "register_governance", conflict):
            select_job(at, job_c)
            at.button(key=f"rec_gov_register_{job_c}").click().run()
            if exceptions(at):
                raise AssertionError("409 后出现异常")
            if not any("登记失败" in e.value for e in at.error):
                raise AssertionError("409 未给出错误提示")
            if any("已登记识别输入资料" in s.value for s in at.success):
                raise AssertionError("409 假报成功")
        if link_rows(manager, job_c) != 0:
            raise AssertionError("409 后不应写入登记")
        log("conflict_no_false_success", {"job_c": job_c})

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

    (BASE / "result.json").write_text(
        json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gov_apptest] ok={RESULT['ok']} -> {BASE / 'result.json'}")
    if not RESULT["ok"]:
        print(RESULT.get("traceback", RESULT.get("error")))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
