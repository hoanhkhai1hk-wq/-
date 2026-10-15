"""真实HTTP服务 + Streamlit/AppTest 网页工作流验收（DEV-20261015-012）。

启动一个独立端口的真实识别后端（已审 serve.py + .venv-recognition + 真实权重），
再用 Streamlit AppTest 实际驱动 app.py 的“田块识别”页，提交固定公开适配样例，
核对网页展示、下载产物与 SQLite 记录一致。

明确边界：
- 使用固定公开样例 inference-img.tif；其时相/波段身份未独立核实，不是真值。
- 候选连通区域不是真实田块数；无真值，不报告准确率或平方米。
- 每次写入 reports/web/DEV-20261015-012/ 新的独立目录并保留，包含服务日志与全部下载。
- 结束前停止本脚本启动的进程树；不安装、不下载、不删除他人资料。

用法：
  .venv-p0/Scripts/python.exe scripts/web/real_web_smoke.py
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrolake_service.config import (  # noqa: E402
    EXPECTED_MODEL_SHA256, FIXED_ARTIFACTS, PUBLIC_SAMPLE_SHA256,
)
from agrolake_service.processes import ProcessTree  # noqa: E402


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
        data = resp.read()
        ctype = resp.getheader("Content-Type", "")
        return resp.status, ctype, data
    finally:
        conn.close()


def get_job(port: int, job_id: int) -> dict:
    status, _ctype, data = http_get(port, f"/jobs/{job_id}")
    if status != 200:
        raise RuntimeError(f"读取作业失败：{status} {data[:200]!r}")
    return json.loads(data.decode("utf-8"))


def list_job_ids(port: int) -> list[int]:
    status, _ctype, data = http_get(port, "/jobs")
    if status != 200:
        raise RuntimeError(f"读取作业列表失败：{status}")
    return [j["job_id"] for j in json.loads(data.decode("utf-8"))["jobs"]]


def wait_job(port: int, job_id: int, timeout: float = 240.0):
    deadline = time.monotonic() + timeout
    states = []
    while time.monotonic() < deadline:
        job = get_job(port, job_id)
        if not states or states[-1] != job["status"]:
            states.append(job["status"])
        if job["status"] not in ("queued", "running"):
            return job, states
        time.sleep(0.1)
    raise AssertionError(f"作业状态未按期结束：{states}")


def main() -> int:
    directory = ROOT / "reports" / "web" / "DEV-20261015-012" / datetime.now().strftime(
        "real-%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)

    checks: dict = {}
    evidence = {
        "directory": str(directory),
        "checks": checks,
        "synthetic_tests_are_separate": True,
        "note": "真实公开适配样例经真实模型；时相身份未独立核实；候选区域非真实田块数",
    }

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    service = None
    try:
        sample = ROOT / "samples" / "ftw_smoke" / "inference-img.tif"
        content = sample.read_bytes()
        check("input_sha256_pinned",
              hashlib.sha256(content).hexdigest() == PUBLIC_SAMPLE_SHA256)

        service = ServiceProcess(directory, 1)
        health_status, _c, health_raw = http_get(service.port, "/health")
        check("health_200", health_status == 200)
        health = json.loads(health_raw.decode("utf-8"))
        check("health_files_present",
              all(health[k] for k in ("worker_python_ready", "cli_script_ready", "model_ready")))

        # 网页进程使用独立 P0 目录，绝不写生产演示库。
        p0 = directory / "p0"
        os.environ["AGROLAKE_DATA_DIR"] = str(p0)
        os.environ["AGROLAKE_STORE"] = str(p0 / "store")
        os.environ["AGROLAKE_DB"] = str(p0 / "agrolake.sqlite")
        os.environ["AGROLAKE_REPORTS"] = str(p0 / "reports")
        os.environ["AGROLAKE_API_URL"] = service.url

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at.run()
        check("page_no_exception", not [str(e.value) for e in at.exception])
        check("page_online", any("识别服务在线" in s.value for s in at.success))
        check("page_has_8_tabs", len(at.tabs) == 8)

        # 通过网页提交固定公开样例（仅此一次 POST）
        at.button(key="rec_submit").click().run()
        check("submit_no_exception", not [str(e.value) for e in at.exception])
        check("submit_prompt", any("已提交识别任务" in s.value for s in at.success))
        job_ids = list_job_ids(service.port)
        check("exactly_one_job", len(job_ids) == 1)
        job_id = job_ids[0]
        evidence["job_id"] = job_id

        job, states = wait_job(service.port, job_id)
        evidence["observed_states"] = states
        check("real_job_succeeded", job["status"] == "succeeded")
        check("sample_identity", job["sample_kind"] == "public_adapter_sample")
        check("model_identity", job["model_sha256"] == EXPECTED_MODEL_SHA256)

        # 刷新网页并核对展示
        at.button(key="rec_refresh").click().run()
        check("refresh_no_exception", not [str(e.value) for e in at.exception])
        metrics = {m.label: str(m.value) for m in at.metric}
        check("page_class_counts",
              metrics.get("背景像素") == "105910"
              and metrics.get("田块内部像素") == "74580"
              and metrics.get("边界像素") == "38572"
              and metrics.get("无效像素") == "0")
        check("page_region_count",
              any("候选连通区域数：722" in c.value for c in at.caption))
        dl_labels = []
        for d in at.get("download_button"):
            proto = getattr(d, "proto", None)
            dl_labels.append(getattr(proto, "label", None) or getattr(d, "label", None))
        check("page_five_downloads",
              all(f"下载 {n}" in dl_labels for n in FIXED_ARTIFACTS))
        evidence["page_metrics"] = metrics
        evidence["page_download_labels"] = dl_labels

        # 独立下载并对产物、SQLite 逐项核对
        downloads = directory / "downloads"
        downloads.mkdir()
        checksums = {}
        for name in FIXED_ARTIFACTS:
            status, _ctype, data = http_get(service.port, f"/jobs/{job_id}/artifacts/{name}")
            check(f"artifact_http200_{name}", status == 200)
            (downloads / name).write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            checksums[name] = sha
            check(f"artifact_sha_{name}", sha == job["artifact_checksums"][name]["sha256"])
        evidence["download_sha256"] = checksums

        run = json.loads((downloads / "run.json").read_text(encoding="utf-8"))
        check("strict_634_tensors",
              run["checkpoint"]["load_report"]["strict"] is True
              and run["checkpoint"]["load_report"]["model_tensors_loaded"] == 634)
        check("full_window_coverage",
              run["inference"]["windows"] == 6
              and run["inference"]["coverage_ratio_valid"] == 1.0)
        check("run_hashes",
              run["input"]["sha256"] == PUBLIC_SAMPLE_SHA256
              and run["checkpoint"]["sha256"] == EXPECTED_MODEL_SHA256)

        from PIL import Image
        with Image.open(downloads / "mask.tif") as mask:
            mask.load()
            counts = Counter(mask.getdata())
            check("mask_shape", (mask.height, mask.width) == (379, 578))
            check("mask_class_counts", counts == {0: 105910, 1: 74580, 2: 38572})
        with Image.open(downloads / "preview.png") as preview:
            preview.load()
            check("preview_decodes", preview.width == 1754 and preview.height == 397)
        with (downloads / "field_pixels.csv").open(encoding="utf-8", newline="") as fh:
            fields = list(csv.DictReader(fh))
        check("candidate_regions_pixel_only",
              len(fields) == 722 and sum(int(r["pixels"]) for r in fields) == 74580)

        conn = sqlite3.connect(directory / "service.sqlite")
        try:
            conn.row_factory = sqlite3.Row
            rec = conn.execute(
                "SELECT j.source, j.scene_id, r.model_sha256, r.field_count"
                " FROM rec_jobs j JOIN rec_runs r USING(job_id) WHERE job_id=?",
                (job_id,),
            ).fetchone()
            check("sqlite_provenance",
                  rec["model_sha256"] == EXPECTED_MODEL_SHA256 and rec["field_count"] == 722)
            check("sqlite_regions",
                  conn.execute("SELECT count(*), sum(pixels) FROM rec_regions WHERE job_id=?",
                               (job_id,)).fetchone()[:] == (722, 74580))
            check("sqlite_class_counts",
                  dict(conn.execute("SELECT class_value, pixels FROM rec_class_counts"
                                    " WHERE job_id=? ORDER BY class_value", (job_id,)).fetchall())
                  == {0: 105910, 1: 74580, 2: 38572, 255: 0})
        finally:
            conn.close()

        check("api_process_does_not_import_torch", "torch" not in sys.modules)

        # 产物被改动后，网页刷新必须察觉 409（不使用会话缓存掩盖）
        conn = sqlite3.connect(directory / "service.sqlite")
        try:
            cli_rel = conn.execute("SELECT cli_output_relpath FROM rec_jobs WHERE job_id=?",
                                   (job_id,)).fetchone()[0]
        finally:
            conn.close()
        tampered = directory / "runs" / cli_rel / "mask.tif"
        tampered.write_bytes(b"TAMPER EVIDENCE - not the original artifact")
        at.button(key="rec_refresh").click().run()
        check("tamper_no_exception", not [str(e.value) for e in at.exception])
        check("page_detects_tamper",
              any("409" in e.value or "产物" in e.value for e in at.error))
        tamper_dl = []
        for d in at.get("download_button"):
            proto = getattr(d, "proto", None)
            tamper_dl.append(getattr(proto, "label", None) or getattr(d, "label", None))
        check("tamper_hides_downloads", not any(lbl and lbl.startswith("下载 ") for lbl in tamper_dl))
        evidence["tamper_error"] = [e.value[:160] for e in at.error]

        evidence["successful"] = True
    except BaseException as exc:  # noqa: BLE001
        evidence["successful"] = False
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["traceback"] = traceback.format_exc()
    finally:
        if service is not None:
            service.stop()
        (directory / "verification.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "directory": str(directory),
        "successful": evidence["successful"],
        "checks": checks,
        "error": evidence.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
