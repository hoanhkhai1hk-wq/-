"""识别输入登记治理与血缘真实HTTP验收（DEV-20261015-014）。

启动一个独立端口的真实识别后端（已审 serve.py + .venv-recognition + 真实权重），
经 HTTP 提交固定公开适配样例，真实推理一次后：
- POST /jobs/<id>/governance 登记识别输入资料为 P0 文本资产；
- GET /jobs/<id>/lineage 核对 P0 对象 JSON、SQLite ID/质检、原输入与登记单摘要、
  人工复核记录与产物摘要；
- 重启服务后重新读取，确认登记保留且不产生多余推理。

明确边界：
- 固定公开样例 inference-img.tif 的时相/波段身份未独立核实，不是真值。
- 候选连通区域不是真实田块数；无真值，不报告准确率或平方米。
- 登记单的质检结论只针对该 JSON 文本，不代表原 TIFF 已通过 P0 质检。
- 每次写入 reports/governance/DEV-20261015-014/ 新目录并保留；结束前停止本脚本进程树。

用法：
  .venv-p0/Scripts/python.exe scripts/governance/real_smoke.py
"""
from __future__ import annotations

import base64
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

from agrolake_service.config import (  # noqa: E402
    EXPECTED_MODEL_SHA256, PUBLIC_SAMPLE_SHA256,
)
from agrolake_service.processes import ProcessTree  # noqa: E402


class ServiceProcess:
    """独立启动已审识别后端；从子目录启动以覆盖“非项目 cwd”。"""

    def __init__(self, directory: Path, sequence: int, extra_env: dict):
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
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                     **extra_env),
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


def http_json(port, method, path, body=None, timeout=60.0):
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, data, headers)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw
    finally:
        conn.close()


def get_job(port, job_id):
    status, raw = http_json(port, "GET", f"/jobs/{job_id}")
    if status != 200:
        raise RuntimeError(f"读取作业失败：{status} {raw[:200]!r}")
    return json.loads(raw.decode("utf-8"))


def wait_job(port, job_id, timeout=300.0):
    deadline = time.monotonic() + timeout
    states = []
    while time.monotonic() < deadline:
        job = get_job(port, job_id)
        if not states or states[-1] != job["status"]:
            states.append(job["status"])
        if job["status"] not in ("queued", "running"):
            return job, states
        time.sleep(0.2)
    raise AssertionError(f"作业状态未按期结束：{states}")


def main() -> int:
    directory = ROOT / "reports" / "governance" / "DEV-20261015-014" / datetime.now().strftime(
        "real-%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    p0 = directory / "p0"
    p0.mkdir()

    checks: dict = {}
    evidence = {"directory": str(directory), "checks": checks,
                "note": "真实公开样例经真实模型；候选区域非真实田块数；登记单质检仅针对JSON文本"}

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    service = None
    try:
        sample = ROOT / "samples" / "ftw_smoke" / "inference-img.tif"
        content = sample.read_bytes()
        sample_sha = hashlib.sha256(content).hexdigest()
        check("input_sha256_pinned", sample_sha == PUBLIC_SAMPLE_SHA256)

        # P0 存储根指向本次独立目录（服务进程继承该环境变量）。
        extra_env = {
            "AGROLAKE_DATA_DIR": str(p0),
            "AGROLAKE_STORE": str(p0 / "store"),
        }
        service = ServiceProcess(directory, 1, extra_env)
        status, raw = http_json(service.port, "GET", "/health")
        check("health_200", status == 200)
        health = json.loads(raw.decode("utf-8"))
        check("health_files_present",
              all(health[k] for k in ("worker_python_ready", "cli_script_ready", "model_ready")))

        # 真实推理一次
        submit = {
            "filename": "inference-img.tif",
            "content_base64": base64.b64encode(content).decode("ascii"),
            "source": "Fields of the World 公开上游适配样例；时相身份未核实",
            "scene_id": "FTW-PUBLIC-ADAPTER-DEMO",
            "source_link": "https://github.com/fieldsoftheworld/ftw-baselines",
            "license_status": "上游公开示例；许可分项见 THIRD_PARTY.md",
            "request_id": "gov-real-" + datetime.now().strftime("%H%M%S%f"),
        }
        status, raw = http_json(service.port, "POST", "/jobs", submit, timeout=120)
        check("submit_accepted", status in (200, 202))
        job_id = json.loads(raw.decode("utf-8"))["job_id"]
        evidence["job_id"] = job_id
        job, states = wait_job(service.port, job_id)
        evidence["observed_states"] = states
        check("real_job_succeeded", job["status"] == "succeeded")
        check("sample_identity", job["sample_kind"] == "public_adapter_sample")
        check("model_identity", job["model_sha256"] == EXPECTED_MODEL_SHA256)
        check("region_count_722", job["region_count"] == 722)

        # 未登记时读取血缘：registered=False，且不写登记。
        status, raw = http_json(service.port, "GET", f"/jobs/{job_id}/lineage")
        check("lineage_pre_200", status == 200)
        pre = json.loads(raw.decode("utf-8"))
        check("pre_unregistered", pre["registered"] is False)
        check("pre_no_registration", pre["registration"] is None)

        # 显式登记
        status, raw = http_json(service.port, "POST", f"/jobs/{job_id}/governance",
                                {"request_id": "greg-" + datetime.now().strftime("%H%M%S%f")})
        check("governance_created", status == 201)
        reg = json.loads(raw.decode("utf-8"))
        evidence["registration"] = reg
        check("manifest_sha_is_hex", bool(re.fullmatch(r"[0-9a-f]{64}", reg["manifest_sha256"])))
        check("input_sha_equals_sample", reg["input_sha256"] == PUBLIC_SAMPLE_SHA256)
        check("manifest_differs_from_input", reg["manifest_sha256"] != reg["input_sha256"])

        # 同号重试复用
        status, raw = http_json(service.port, "POST", f"/jobs/{job_id}/governance",
                                {"request_id": reg["request_id"]})
        check("governance_idempotent_200", status == 200)
        check("governance_not_created", json.loads(raw.decode("utf-8"))["created"] is False)

        # 人工复核一条，验证血缘包含复核记录
        status, raw = http_json(service.port, "POST", f"/jobs/{job_id}/reviews", {
            "decision": "questioned", "reviewer": "流程测试署名",
            "note": "合成流程意见，不代表真实专家意见",
            "request_id": "grev-" + datetime.now().strftime("%H%M%S%f"),
        })
        check("review_created", status == 201)

        # GET 血缘
        status, raw = http_json(service.port, "GET", f"/jobs/{job_id}/lineage")
        check("lineage_200", status == 200)
        lin = json.loads(raw.decode("utf-8"))
        evidence["lineage"] = lin
        check("registered_true", lin["registered"] is True)
        check("lineage_asset_version",
              lin["registration"]["asset_id"] == reg["asset_id"]
              and lin["registration"]["version_id"] == reg["version_id"])
        check("lineage_manifest_sha", lin["registration"]["manifest_sha256"] == reg["manifest_sha256"])
        check("lineage_input_sha", lin["registration"]["input_sha256"] == PUBLIC_SAMPLE_SHA256)
        check("integrity_input", lin["integrity"]["input_object_intact"] is True)
        check("integrity_manifest_p0", lin["integrity"]["manifest_p0_object_intact"] is True)
        check("registration_quality_present",
              isinstance(lin["registration"]["quality_at_registration"], dict))
        check("current_version_present", isinstance(lin["registration"]["current"]["version"], dict))
        check("reviews_single", lin["reviews"]["total"] == 1)
        check("artifacts_five", len(lin["artifacts"]) == 5)

        # P0 对象 JSON 可重读且与摘要一致
        manifest_obj = p0 / "store" / "objects" / reg["manifest_sha256"][:2] / reg["manifest_sha256"]
        check("p0_object_exists", manifest_obj.is_file())
        manifest_bytes = manifest_obj.read_bytes()
        check("p0_object_sha", hashlib.sha256(manifest_bytes).hexdigest() == reg["manifest_sha256"])
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        check("manifest_kind", manifest["kind"] == "agrolake.recognition.input_manifest")
        check("manifest_job", manifest["job_id"] == job_id)
        check("manifest_input_sha", manifest["input_sha256"] == PUBLIC_SAMPLE_SHA256)
        check("manifest_no_local_path", "\\" not in manifest_bytes.decode("utf-8"))

        # SQLite：登记链接 + P0 资产/版本/导入事件/质检
        conn = sqlite3.connect(directory / "service.sqlite")
        try:
            conn.row_factory = sqlite3.Row
            link = conn.execute("SELECT * FROM rec_governance_links WHERE job_id=?",
                                (job_id,)).fetchone()
            check("sqlite_link_exists", link is not None)
            check("sqlite_link_sha", link["manifest_sha256"] == reg["manifest_sha256"])
            ver = conn.execute("SELECT * FROM asset_versions WHERE version_id=?",
                               (reg["version_id"],)).fetchone()
            check("p0_version_exists", ver is not None)
            check("p0_version_object", ver["object_id"] == reg["manifest_sha256"])
            check("p0_version_source", ver["source"] == manifest["source"])
            check("p0_import_event",
                  conn.execute("SELECT COUNT(*) FROM import_events WHERE version_id=?",
                               (reg["version_id"],)).fetchone()[0] == 1)
            check("p0_quality_checks",
                  conn.execute("SELECT COUNT(*) FROM quality_checks WHERE version_id=?",
                               (reg["version_id"],)).fetchone()[0] >= 1)
            check("no_tiff_in_p0", conn.execute(
                "SELECT COUNT(*) FROM objects WHERE object_id=?", (PUBLIC_SAMPLE_SHA256,)
            ).fetchone()[0] == 0)
        finally:
            conn.close()

        # 重启：登记保留，无多余推理
        service.stop()
        service = ServiceProcess(directory, 2, extra_env)
        status, raw = http_json(service.port, "GET", f"/jobs/{job_id}/lineage")
        check("restart_lineage_200", status == 200)
        lin2 = json.loads(raw.decode("utf-8"))
        check("restart_registered", lin2["registered"] is True)
        check("restart_same_version", lin2["registration"]["version_id"] == reg["version_id"])
        check("restart_integrity", lin2["integrity"]["manifest_p0_object_intact"] is True)
        status, raw = http_json(service.port, "GET", "/jobs")
        jobs_after = json.loads(raw.decode("utf-8"))["jobs"]
        check("no_extra_inference", len(jobs_after) == 1 and jobs_after[0]["job_id"] == job_id)

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
