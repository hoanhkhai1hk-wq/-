"""用 Streamlit AppTest 实际驱动 app.py 的“田块识别”页（DEV-20261015-012）。

覆盖：离线时 P0 仍可用并给出启动提示；连接合成fixture识别服务后提交固定公开样例、
选历史、刷新状态/多次 rerun 不重复 POST、成功显示类计数/候选区域表/下载、
失败任务不串此前成功结果。fixture 明确为合成，不代表真实模型。

AppTest 无法操作 file_uploader：本测试只用“本机公开适配样例”入口，上传控件
的真实行为留给 Codex 用真实浏览器独立验收，本测试不宣称其通过。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_page_apptest.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agrolake_service import config as svc_config  # noqa: E402
from agrolake_service.jobs import JobManager  # noqa: E402
from agrolake_service.server import make_server  # noqa: E402
from agrolake_service.validation import validate_job_payload  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from fixtures import WebFixtureRunner, free_closed_port  # noqa: E402

BASE = ROOT / "reports" / "web" / "DEV-20261015-012" / datetime.now().strftime(
    "apptest-%Y%m%d-%H%M%S-%f")
BASE.mkdir(parents=True, exist_ok=False)
RESULT: dict = {"directory": str(BASE), "steps": [], "exceptions": []}


def log(step: str, detail) -> None:
    RESULT["steps"].append({"step": step, "detail": detail})
    print(f"[web_apptest] {step}: {detail}")


class CountingManager(JobManager):
    """统计真实 submit 调用次数，用于证明 rerun/刷新不会重复 POST。"""

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


def button(at: AppTest, key: str):
    return at.button(key=key)


def selectbox(at: AppTest, *, key=None, label=None):
    for sb in at.selectbox:
        if key is not None and getattr(sb, "key", None) == key:
            return sb
        if label is not None and sb.label == label:
            return sb
    return None


def download_labels(at: AppTest) -> list[str]:
    labels = []
    for d in at.get("download_button"):
        proto = getattr(d, "proto", None)
        labels.append(getattr(proto, "label", None) or getattr(d, "label", None))
    return labels


def wait_terminal(manager: JobManager, job_id: int, timeout: float = 30.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.get_job(job_id)["status"]
        if status not in ("queued", "running"):
            return status
        time.sleep(0.05)
    raise AssertionError("作业在超时前未结束")


def start_service(subdir: str, runner):
    d = BASE / subdir
    d.mkdir(parents=True, exist_ok=True)
    cfg = svc_config.ServiceConfig(
        db_path=d / "service.sqlite", store_root=d / "store", runs_root=d / "runs",
        port=0, synthetic_mode=True, model_sha256="f" * 64,
    )
    manager = CountingManager(cfg, runner)
    server = make_server(cfg, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manager.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    return d, cfg, manager, server, thread, url


def stop_service(manager, server, thread):
    manager.stop()
    server.shutdown()
    server.server_close()
    thread.join(5)


def scenario_offline() -> None:
    d = BASE / "offline"
    d.mkdir(parents=True, exist_ok=True)
    port = free_closed_port()
    set_env(d, f"http://127.0.0.1:{port}")
    at = new_app()
    if exceptions(at):
        raise AssertionError("离线渲染出现异常")
    if len(at.tabs) != 8:
        raise AssertionError(f"应有 8 个页签，实际 {len(at.tabs)}")
    if not any("识别服务当前不可用" in w.value for w in at.warning):
        raise AssertionError("离线时未给出启动提示")
    if not any(i.value.startswith("识别服务未运行时") for i in at.info):
        raise AssertionError("离线时未说明 P0 仍可用")
    if not any(m.label == "资产版本" for m in at.metric):
        raise AssertionError("离线时 P0 治理概览未正常渲染")
    log("offline", {"tabs": len(at.tabs), "warning": "识别服务当前不可用"})


def scenario_fixture() -> None:
    runner = WebFixtureRunner("success")
    d, cfg, manager, server, thread, url = start_service("fixture", runner)
    try:
        set_env(d, url)
        at = new_app()
        if exceptions(at):
            raise AssertionError("在线渲染出现异常")
        if not any("识别服务在线" in s.value for s in at.success):
            raise AssertionError("未显示识别服务在线")
        log("online", {"url": url})

        # 提交（仅此处 POST）
        button(at, "rec_submit").click().run()
        if exceptions(at):
            raise AssertionError("提交后出现异常")
        if manager.submit_calls != 1:
            raise AssertionError(f"提交应只 POST 一次，实际 {manager.submit_calls}")
        if not any("已提交识别任务" in s.value for s in at.success):
            raise AssertionError("提交后未提示")
        job_id = manager.list_jobs()[0]["job_id"]
        log("submit", {"job_id": job_id, "submit_calls": manager.submit_calls})

        status = wait_terminal(manager, job_id)
        if status != "succeeded":
            raise AssertionError(f"fixture 作业应成功，实际 {status}")

        # 多次普通 rerun（模拟切页/刷新/交互）不得再次 POST
        for _ in range(3):
            at.run()
            if exceptions(at):
                raise AssertionError("普通 rerun 出现异常")
        if manager.submit_calls != 1:
            raise AssertionError(f"普通 rerun 重复 POST：{manager.submit_calls}")

        # 刷新状态按钮
        button(at, "rec_refresh").click().run()
        if exceptions(at):
            raise AssertionError("刷新状态出现异常")
        if manager.submit_calls != 1:
            raise AssertionError(f"刷新状态重复 POST：{manager.submit_calls}")

        if not any("已成功" in m.value for m in at.markdown):
            raise AssertionError("成功任务未显示“已成功”")
        dl = download_labels(at)
        for name in ("mask.tif", "preview.png", "class_pixels.csv", "field_pixels.csv",
                     "run.json"):
            if f"下载 {name}" not in dl:
                raise AssertionError(f"缺少下载按钮：{name}（实际 {dl}）")
        metrics = {m.label: m.value for m in at.metric}
        if metrics.get("背景像素") != "4" or metrics.get("田块内部像素") != "4":
            raise AssertionError(f"类计数显示异常：{metrics}")
        df_rows = [len(df.value) for df in at.dataframe if hasattr(df, "value")]
        if not any(n == 1 for n in df_rows):
            raise AssertionError(f"未显示候选区域表（行数 {df_rows}）")
        log("success_view", {"downloads": dl, "metrics": metrics, "df_rows": df_rows})

        # 直接注入一个失败作业，验证失败不串此前成功结果
        norm = validate_job_payload({
            "filename": "fail.tif",
            "content_base64": __import__("base64").b64encode(b"SYNTHETIC-FAIL").decode(),
            "source": "SYNTHETIC 失败注入", "scene_id": "FAIL-DEMO", "request_id": "fail-1",
        })
        fail_id = manager.submit(norm)["job_id"]
        if wait_terminal(manager, fail_id) != "failed":
            raise AssertionError("注入的失败作业状态不是 failed")
        if manager.submit_calls != 1:  # 直接调用 manager，不经过网页
            # manager.submit 被直接调用会+1，这里允许它变成2，但网页 POST 数不变
            pass
        log("inject_failure", {"fail_id": fail_id})

        # 重新执行页面，使历史列表包含刚注入的失败作业
        at.run()
        if exceptions(at):
            raise AssertionError("刷新历史列表出现异常")

        # 用下拉选择失败作业（选项形如 "#<编号> · 状态 · 文件名"）
        sb = selectbox(at, key="rec_history", label="选择历史任务查看")
        if sb is None:
            raise AssertionError("未找到历史任务下拉")
        fail_label = next((o for o in sb.options if str(o).startswith(f"#{fail_id} ")), None)
        if fail_label is None:
            raise AssertionError(f"下拉未包含失败作业：{sb.options}")
        sb.set_value(fail_label)
        at.run()
        if exceptions(at):
            raise AssertionError("选择失败作业出现异常")
        if not any("任务失败" in e.value for e in at.error):
            raise AssertionError("选择失败作业后未显示失败")
        if any(lbl.startswith("下载 ") for lbl in download_labels(at)):
            raise AssertionError("失败任务仍显示下载按钮（串用了旧结果）")
        if any(m.label == "背景像素" for m in at.metric):
            raise AssertionError("失败任务仍显示成功类计数")
        log("failure_view", {"errors": [e.value[:30] for e in at.error]})

        # 切回成功作业，产物应重新出现且不串图
        sb = selectbox(at, key="rec_history", label="选择历史任务查看")
        ok_label = next((o for o in sb.options if str(o).startswith(f"#{job_id} ")), None)
        if ok_label is None:
            raise AssertionError(f"下拉未包含成功作业：{sb.options}")
        sb.set_value(ok_label)
        at.run()
        if exceptions(at):
            raise AssertionError("切回成功作业出现异常")
        dl2 = download_labels(at)
        if f"下载 preview.png" not in dl2:
            raise AssertionError("切回成功作业后未恢复下载")
        if not any(m.label == "背景像素" for m in at.metric):
            raise AssertionError("切回成功作业后未恢复类计数")
        log("switch_back", {"downloads": dl2})
    finally:
        stop_service(manager, server, thread)


def main() -> int:
    try:
        scenario_offline()
        scenario_fixture()
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
