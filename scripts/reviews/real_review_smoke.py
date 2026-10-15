"""真实识别服务 + 人工复核接口验收（DEV-20261015-013）。

用已审 serve.py + .venv-recognition + 真实 FTW v1 权重跑一次真实识别，再通过
HTTP 复核接口保存“流程测试署名”的人工存疑意见，并对一个**真实存在的候选区域**
追加两次意见，核对历史、快照、CSV/JSON 导出与 SQLite 一致，重启后仍可读且无额外推理。

明确边界：
- 用固定公开适配样例；时相身份未独立核实，不是真值。
- 人工意见是自动流程写入的**流程测试署名**，不是真实专家标注。
- 候选区域不是真实田块数；无真值，不报告准确率或平方米。
- 每次写入 reports/reviews/DEV-20261015-013/ 新独立目录并保留；结束前停止自启进程树。

用法：
  .venv-p0/Scripts/python.exe scripts/reviews/real_review_smoke.py
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrolake_service.config import EXPECTED_MODEL_SHA256, PUBLIC_SAMPLE_SHA256  # noqa: E402
from agrolake_service.processes import ProcessTree  # noqa: E402
from agrolake_web.client import ApiError, RecognitionClient  # noqa: E402
from agrolake_web import review_panel  # noqa: E402


class ServiceProcess:
    """独立启动已审识别后端；从子目录启动以覆盖“非项目 cwd”。"""

    def __init__(self, directory: Path, sequence: int):
        self.logpath = directory / f"service-{sequence}.log"
        self.log = self.logpath.open("x", encoding="utf-8")
        command = [
            str(ROOT / ".venv-p0" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "service" / "serve.py"),
            "--port", "0", "--db", str(directory / "service.sqlite"),
            "--store", str(directory / "store"), "--runs", str(directory / "runs"),
        ]
        self.tree = ProcessTree(
            command, cwd=str(directory),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"),
            stdout=self.log,
        )
        self.port = None
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                text = self.logpath.read_text(encoding="utf-8")
                found = re.search(r"http://127\.0\.0\.1:(\d+)/", text)
                if found:
                    self.port = int(found.group(1))
                    break
                if self.tree.process.poll() is not None:
                    raise RuntimeError("服务启动失败：" + text[-1500:])
                time.sleep(0.05)
            if self.port is None:
                raise RuntimeError("服务启动超时")
        except BaseException:
            self.stop()
            raise

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        try:
            self.tree.terminate()
        finally:
            self.tree.close()
            self.log.close()


def http_get(port: int, path: str, timeout: float = 30.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type", ""), resp.read()
    finally:
        conn.close()


def wait_job(port: int, job_id: int, timeout: float = 240.0):
    deadline = time.monotonic() + timeout
    states = []
    while time.monotonic() < deadline:
        status, _c, data = http_get(port, f"/jobs/{job_id}")
        assert status == 200, status
        job = json.loads(data.decode("utf-8"))
        if not states or states[-1] != job["status"]:
            states.append(job["status"])
        if job["status"] not in ("queued", "running"):
            return job, states
        time.sleep(0.1)
    raise AssertionError(f"作业未按期结束：{states}")


def main() -> int:
    directory = ROOT / "reports" / "reviews" / "DEV-20261015-013" / datetime.now().strftime(
        "real-%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)

    checks: dict = {}
    evidence = {
        "directory": str(directory),
        "checks": checks,
        "synthetic_tests_are_separate": True,
        "note": ("真实公开适配样例经真实模型；人工意见为流程测试署名，非真实专家标注；"
                 "候选区域非真实田块数；无真值，不报告准确率或平方米"),
    }

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    service = None
    restart = None
    try:
        sample = ROOT / "samples" / "ftw_smoke" / "inference-img.tif"
        content = sample.read_bytes()
        check("input_sha256_pinned",
              hashlib.sha256(content).hexdigest() == PUBLIC_SAMPLE_SHA256)

        service = ServiceProcess(directory, 1)
        client = RecognitionClient(service.url, timeout=30)
        check("health_ok", client.health().get("status") == "ok")

        # 真实识别（仅一次推理）
        import base64
        job_id = client.submit_job({
            "filename": "inference-img.tif",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "source": "Fields of the World 公开上游适配样例；时相身份未核实",
            "scene_id": "FTW-PUBLIC-ADAPTER-DEMO",
            "request_id": "review-real-1",
        })["job_id"]
        job, states = wait_job(service.port, job_id)
        evidence["job_id"] = job_id
        evidence["observed_states"] = states
        check("real_job_succeeded", job["status"] == "succeeded")
        check("model_identity", job["model_sha256"] == EXPECTED_MODEL_SHA256)
        check("has_regions", job["region_count"] >= 1)

        fields = client.get_fields(job_id)
        region_field = fields["regions"][0]["field_id"]
        evidence["region_field_id"] = region_field

        # 任务整体：一条“流程测试署名”的人工存疑意见
        task_note = "流程测试署名的人工存疑意见，非真实专家标注"
        rec_task = client.submit_review(
            job_id, decision="questioned", reviewer="流程测试署名",
            note=task_note, request_id="rev-task-1",
        )
        check("task_review_created", rec_task.get("created") is True)
        check("task_review_target", rec_task.get("target") == "task")
        check("task_snapshot_input",
              rec_task.get("input_sha256") == job["input_sha256"])
        check("task_snapshot_model",
              rec_task.get("model_sha256") == job["model_sha256"])

        # 对同一真实区域追加两次意见：accepted -> rejected（历史保留，最新为后者）
        first = client.submit_review(
            job_id, decision="accepted", reviewer="流程测试署名",
            note="流程测试：区域初判可用，非真值", request_id="rev-region-1",
            field_id=region_field,
        )
        second = client.submit_review(
            job_id, decision="rejected", reviewer="流程测试署名",
            note="=流程测试：公式转义前缀；复核后不采用，非真值",
            request_id="rev-region-2", field_id=region_field,
        )
        check("region_two_records",
              first["review_id"] != second["review_id"] and second["created"] is True)

        listing = client.list_reviews(job_id)
        check("history_total_3", listing["total"] == 3)
        latest = listing["latest"]
        check("latest_task_questioned", latest["task"]["decision"] == "questioned")
        check("latest_region_rejected",
              latest["regions"][str(region_field)]["decision"] == "rejected")
        check("latest_region_is_second",
              latest["regions"][str(region_field)]["review_id"] == second["review_id"])

        # 独立 SQLite 查询核对
        conn = sqlite3.connect(directory / "service.sqlite")
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM rec_reviews WHERE job_id=? ORDER BY review_id", (job_id,)
            ).fetchall()
            check("sqlite_three_rows", len(rows) == 3)
            check("sqlite_snapshot_consistent",
                  all(r["input_sha256"] == job["input_sha256"]
                      and r["model_sha256"] == job["model_sha256"] for r in rows))
            check("sqlite_task_field_null",
                  rows[0]["field_id"] is None and rows[0]["decision"] == "questioned")
            check("sqlite_region_two_rows",
                  sum(1 for r in rows if r["field_id"] == region_field) == 2)
            evidence["sqlite_review_ids"] = [r["review_id"] for r in rows]
        finally:
            conn.close()

        # 导出（网页使用的同一函数）：CSV/JSON 与 SQLite 一致
        all_rows, complete = review_panel._fetch_all(client, job_id)
        check("export_complete", complete and len(all_rows) == 3)
        csv_text = review_panel._to_csv(all_rows).decode("utf-8")
        json_text = json.loads(review_panel._to_json(job_id, all_rows, complete).decode("utf-8"))
        check("csv_boundary_noted",
              csv_text.splitlines()[0].startswith("# ") and "不是真值" in csv_text.splitlines()[0])
        check("csv_escapes_formula", "'=流程测试：公式转义前缀" in csv_text)
        check("json_keeps_raw_formula",
              any(r["note"].startswith("=流程测试") for r in json_text["reviews"]))
        check("json_boundary", "不是真值" in json_text.get("boundary", ""))
        check("json_count_matches_sqlite", json_text["count"] == 3)

        # 重启后仍可读，且没有额外推理
        runs_before = _count_runs(directory / "service.sqlite")
        jobs_before = len(http_get(service.port, "/jobs")[2])
        service.stop()
        service = None
        restart = ServiceProcess(directory, 2)
        client2 = RecognitionClient(restart.url, timeout=30)
        after = client2.list_reviews(job_id)
        check("restart_history_readable", after["total"] == 3)
        check("restart_latest_same",
              after["latest"]["regions"][str(region_field)]["review_id"] == second["review_id"])
        runs_after = _count_runs(directory / "service.sqlite")
        jobs_after = len(http_get(restart.port, "/jobs")[2])
        check("no_extra_inference", runs_before == runs_after)
        check("no_extra_jobs", jobs_before == jobs_after)

        check("api_process_does_not_import_torch", "torch" not in sys.modules)
        evidence["successful"] = True
    except BaseException as exc:  # noqa: BLE001
        evidence["successful"] = False
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["traceback"] = traceback.format_exc()
    finally:
        for proc in (service, restart):
            if proc is not None:
                proc.stop()
        (directory / "verification.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "directory": str(directory),
        "successful": evidence["successful"],
        "checks": checks,
        "error": evidence.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["successful"] else 1


def _count_runs(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM rec_runs").fetchone()[0]
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
