"""RGB候选图片识别作业后端（INT-20261020-002）。

把已审单图CLI（scripts/rgb/predict_candidate.py，INF-20261020-002）变成可由
网页显式调用的同步作业接口，并把每次识别持久化到专用 SQLite
``<data_dir>/runs.sqlite3``，与原 FTW/P0 业务库完全分开。

- 输入：非空、≤4MiB、PIL 预读恰为 512×512 / RGB、格式 PNG/JPEG/TIFF；
  原始字节原样保存（不重编码），保证图像 SHA 与上传一致。
- 运行：仅在 ``submit`` 里同步 ``subprocess.run`` 一次已审 CLI
  （shell=False、cwd=ROOT、timeout=120、capture_output=True，固定线程环境变量）；
  网页进程本身不导入 torch / recognition 模型模块。
- 校验：子进程 exit 0 不等于成功；六固定产物存在、SHA 一致、
  metadata/quality/CSV/掩膜相互一致才记为成功；空/损坏/不一致标失败。
- 幂等：同一 request_id（32位小写十六进制）+ 相同原图 SHA + 相同清单 SHA +
  相同 seed 返回已存作业（无论成功/失败/running 都不重跑）；同号不同内容拒绝。
- 候选区域只是 class0 四邻接连通块，像素面积不是平方米，也不是已核实真实地块。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# 与已审 CLI（recognition/rgb_candidate.py）的默认值一致；此处不复用该模块，
# 避免网页进程间接引入 torch。
DEFAULT_MANIFEST = ROOT / "outputs" / "RGB概率融合候选模型-20261020.json"
DEFAULT_SEED = 20261017
CLI_SCRIPT = ROOT / "scripts" / "rgb" / "predict_candidate.py"
INTERPRETER = ROOT / ".venv-recognition" / "Scripts" / "python.exe"
RUN_TIMEOUT_SECONDS = 120
MAX_INPUT_BYTES = 4 * 1024 * 1024
ALLOWED_FORMATS = ("PNG", "JPEG", "TIFF")
ARTIFACT_NAMES = (
    "mask256.png", "mask512.png", "overlay.png",
    "candidate_fields.csv", "quality.json", "metadata.json",
)
HASHED_ARTIFACTS = ("mask256.png", "mask512.png", "overlay.png", "candidate_fields.csv")
CSV_HEADER = ["field_id", "area_pixels", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
              "centroid_x", "centroid_y", "edge_touch", "small_patch"]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE", "PYTHONHASHSEED": "0",
}
REQUEST_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")
JOB_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  job_id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  source_name TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  seed INTEGER NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  input_format TEXT,
  input_path TEXT,
  result_dir TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  artifact_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS regions (
  job_id TEXT NOT NULL,
  field_id INTEGER NOT NULL,
  area_pixels INTEGER NOT NULL,
  bbox_x0 INTEGER NOT NULL,
  bbox_y0 INTEGER NOT NULL,
  bbox_x1 INTEGER NOT NULL,
  bbox_y1 INTEGER NOT NULL,
  centroid_x REAL NOT NULL,
  centroid_y REAL NOT NULL,
  edge_touch INTEGER NOT NULL,
  small_patch INTEGER NOT NULL,
  PRIMARY KEY (job_id, field_id)
);
CREATE TABLE IF NOT EXISTS class_counts (
  job_id TEXT NOT NULL,
  class_id INTEGER NOT NULL,
  pixels INTEGER NOT NULL,
  PRIMARY KEY (job_id, class_id)
);
"""


class RgbCandidateInputError(ValueError):
    """输入被拒绝：不产生成功记录，也不触发推理。"""


class RgbArtifactError(ValueError):
    """产物缺失、被改动或请求不合法，明确报错而不是展示旧值。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json_bytes(data: bytes):
    return json.loads(data.decode("utf-8-sig"))


def validate_image_bytes(content: bytes) -> str:
    """PIL 预读校验：恰为 512×512 / RGB，格式 PNG/JPEG/TIFF。返回检测格式名。"""
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise RgbCandidateInputError("文件内容为空，请选择有效的图片文件。")
    if len(content) > MAX_INPUT_BYTES:
        raise RgbCandidateInputError(
            f"文件超过大小上限（{MAX_INPUT_BYTES} 字节），当前 {len(content)} 字节。")
    try:
        probe = Image.open(io.BytesIO(content))
        fmt, size, mode = probe.format, probe.size, probe.mode
    except Exception as exc:
        raise RgbCandidateInputError(f"无法解码图片：{exc}") from exc
    if fmt not in ALLOWED_FORMATS:
        raise RgbCandidateInputError(
            f"格式 {fmt} 不受支持，仅允许 PNG/JPEG/TIFF。")
    if size != (512, 512):
        raise RgbCandidateInputError(
            f"图片尺寸为 {size[0]}×{size[1]}，当前只支持 512×512 的 RGB 切片。")
    if mode != "RGB":
        raise RgbCandidateInputError(
            f"图片模式为 {mode}，只接受 RGB 8 位图像。")
    try:
        probe.load()
    except Exception as exc:
        raise RgbCandidateInputError(f"无法解码图片：{exc}") from exc
    return fmt


def _parse_regions_csv(data: bytes) -> list[dict]:
    """解析候选 CSV 为字典行；表头或数值不合法即报错（由调用方标失败）。"""
    import csv as _csv

    reader = _csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    if not rows or rows[0] != CSV_HEADER:
        raise ValueError("candidate_fields.csv 表头与固定格式不符")
    regions = []
    for line in rows[1:]:
        if len(line) != len(CSV_HEADER):
            raise ValueError("candidate_fields.csv 列数与固定格式不符")
        regions.append({
            "field_id": int(line[0]),
            "area_pixels": int(line[1]),
            "bbox_x0": int(line[2]), "bbox_y0": int(line[3]),
            "bbox_x1": int(line[4]), "bbox_y1": int(line[5]),
            "centroid_x": float(line[6]), "centroid_y": float(line[7]),
            "edge_touch": line[8] == "true",
            "small_patch": line[9] == "true",
        })
    return regions


def verify_outputs(result_dir: Path, image_sha256: str, manifest_path: Path,
                   seed: int) -> dict:
    """自包含纯校验：六产物齐全且相互一致才返回解析结果，否则抛 ValueError。

    不依赖 torch / recognition 模块；检查项：
    - 六固定文件存在；mask/overlay/CSV 的 SHA 与 metadata 记录一致；
    - metadata 图片 SHA、清单 SHA、seed、三模型 SHA 与当前清单选组相符；
    - mask256(256×256,L) 与 mask512(512×512,L) 呈最近邻 repeat2 关系，
      类别只含 0/1/2，重算计数与 metadata/quality 一致；
    - CSV 行数=候选数、面积和=native 田块像素数、边缘/小斑块计数一致。
    """
    missing = [n for n in ARTIFACT_NAMES if not (result_dir / n).is_file()]
    if missing:
        raise ValueError("产物缺失：" + "、".join(missing))
    metadata = _read_json_bytes((result_dir / "metadata.json").read_bytes())
    quality = _read_json_bytes((result_dir / "quality.json").read_bytes())

    for name in HASHED_ARTIFACTS:
        recorded = metadata.get("output_files_sha256", {}).get(name)
        actual = _sha256_bytes((result_dir / name).read_bytes())
        if not isinstance(recorded, str) or recorded != actual:
            raise ValueError(f"{name} 的 SHA 与 metadata 记录不一致")

    if metadata.get("image_sha256") != image_sha256:
        raise ValueError("metadata 图片 SHA 与本作业输入不一致")
    manifest_bytes = manifest_path.read_bytes()
    if metadata.get("manifest_sha256") != _sha256_bytes(manifest_bytes):
        raise ValueError("metadata 清单 SHA 与当前候选模型清单不一致")
    if metadata.get("seed") != seed:
        raise ValueError("metadata seed 与固定 seed 不一致")
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    group = next((m for m in manifest.get("models", []) if m.get("seed") == seed), None)
    if group is None:
        raise ValueError(f"清单中没有 seed {seed} 的模型组")
    for role in ("baseline_W5", "retention_003", "feedback_005"):
        if metadata.get("weights_sha256", {}).get(role) != group[role]["sha256"]:
            raise ValueError(f"metadata 中 {role} 权重 SHA 与当前清单选组不符")

    with Image.open(result_dir / "mask256.png") as m256:
        if m256.size != (256, 256) or m256.mode != "L":
            raise ValueError("mask256.png 尺寸/模式不符合 256×256 灰度要求")
        grid = list(m256.getdata())
    with Image.open(result_dir / "mask512.png") as m512:
        if m512.size != (512, 512) or m512.mode != "L":
            raise ValueError("mask512.png 尺寸/模式不符合 512×512 灰度要求")
        native = list(m512.getdata())
    with Image.open(result_dir / "overlay.png") as ov:
        if ov.size != (512, 512):
            raise ValueError("overlay.png 尺寸不符合 512×512")
    if any(v not in (0, 1, 2) for v in grid):
        raise ValueError("mask256.png 出现 0/1/2 之外的类别值")
    # repeat2：512 掩膜每 2×2 块必须与 256 掩膜一致
    for r in range(256):
        base = r * 256
        up = r * 1024  # native 第 2r 行起点（每行 512 像素）
        row256 = grid[base:base + 256]
        for c in range(256):
            v = row256[c]
            if native[up + c * 2] != v or native[up + c * 2 + 1] != v \
                    or native[up + 512 + c * 2] != v or native[up + 512 + c * 2 + 1] != v:
                raise ValueError("mask512.png 不是 mask256.png 的最近邻 repeat2")

    grid_counts = {c: grid.count(c) for c in (0, 1, 2)}
    native_counts = {c: native.count(c) for c in (0, 1, 2)}
    if sum(grid_counts.values()) != 65536 or sum(native_counts.values()) != 262144:
        raise ValueError("掩膜像素总数不等于 256×256 / 512×512")
    if {str(k): v for k, v in grid_counts.items()} != {str(int(k)): v for k, v in
            metadata.get("grid_class_counts", {}).items()}:
        raise ValueError("重算的 256 网格类别计数与 metadata 不一致")
    if {str(k): v for k, v in native_counts.items()} != {str(int(k)): v for k, v in
            metadata.get("native_class_counts", {}).items()}:
        raise ValueError("重算的 native 类别计数与 metadata 不一致")
    if quality.get("grid_class_counts") != metadata.get("grid_class_counts") \
            or quality.get("native_class_counts") != metadata.get("native_class_counts"):
        raise ValueError("quality.json 与 metadata 的类别计数不一致")

    regions = _parse_regions_csv((result_dir / "candidate_fields.csv").read_bytes())
    count = metadata.get("candidate_count")
    if not isinstance(count, int) or count != len(regions):
        raise ValueError("CSV 候选行数与 metadata 的 candidate_count 不一致")
    if quality.get("candidate_count") != count:
        raise ValueError("quality.json 的候选数与 metadata 不一致")
    if sum(r["area_pixels"] for r in regions) != native_counts[0]:
        raise ValueError("CSV 像素面积之和不等于 native 田块(class0)像素数")
    if sum(1 for r in regions if r["edge_touch"]) != quality.get("edge_touch_count"):
        raise ValueError("CSV 边缘触碰计数与 quality.json 不一致")
    if sum(1 for r in regions if r["small_patch"]) != quality.get("small_patch_count"):
        raise ValueError("CSV 小斑块计数与 quality.json 不一致")
    for r in regions:
        if not (0 <= r["bbox_x0"] < r["bbox_x1"] <= 512
                and 0 <= r["bbox_y0"] < r["bbox_y1"] <= 512):
            raise ValueError("候选区域范围超出 512×512")
    return {"metadata": metadata, "quality": quality, "regions": regions}


class RgbCandidateJobs:
    """RGB候选识别作业：同步提交、专用库持久化、产物核摘要读取。"""

    def __init__(self, project_root: Path = ROOT, data_dir: Path | str | None = None,
                 *, readonly: bool = False):
        self.project_root = Path(project_root).resolve()
        if data_dir is None:
            env = os.environ.get("AGROLAKE_RGB_DIR")
            data_dir = Path(env) if env else self.project_root / "data" / "rgb_candidate"
        self.data_dir = Path(data_dir).resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.db_path = self.data_dir / "runs.sqlite3"
        self.readonly = bool(readonly)

    # ---------- 存储基础 ----------

    def _connect(self) -> sqlite3.Connection:
        if self.readonly:
            # 只读访问（INT-20261021-001）：mode=ro + query_only，不建目录、不建库、
            # 不执行任何 DDL；库不存在时明确报错，绝不悄悄新建。
            if not self.db_path.is_file():
                raise RgbArtifactError("RGB作业库不存在，只读模式不会创建新库。")
            uri = self.db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            return conn
        self.data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def _job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id or ""):
            raise RgbArtifactError("作业编号格式不合法")
        return self.jobs_dir / job_id

    # ---------- 查询 ----------

    def _row_to_job(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["job_id"],
            "status": row["status"],
            "source_name": row["source_name"],
            "image_sha256": row["image_sha256"],
            "manifest_sha256": row["manifest_sha256"],
            "seed": row["seed"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": row["error"],
            "input_format": row["input_format"],
            "source_path": row["input_path"],
            "result_dir": row["result_dir"],
            "artifact_sha256": json.loads(row["artifact_sha256"])
            if row["artifact_sha256"] else {},
        }

    def get_job(self, job_id: str) -> dict | None:
        if not JOB_ID_RE.match(job_id or ""):
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row else None
        finally:
            conn.close()

    def list_jobs(self, limit: int = 30) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, job_id DESC LIMIT ?",
                (max(1, int(limit)),)).fetchall()
            return [self._row_to_job(r) for r in rows]
        finally:
            conn.close()

    # ---------- 提交 ----------

    def submit(self, content: bytes, filename: str, request_id: str) -> dict:
        """同步提交一次识别：校验输入→持久化→运行CLI→核验产物→落库。"""
        if self.readonly:
            raise RgbCandidateInputError(
                "只读模式不允许提交识别；如需运行识别请使用默认（可写）连接。")
        rid = request_id if isinstance(request_id, str) else ""
        if not REQUEST_ID_RE.match(rid):
            raise RgbCandidateInputError(
                "请求编号必须是 32 位小写十六进制；请新建一次识别，不要手工构造编号。")
        fmt = validate_image_bytes(content)
        content = bytes(content)
        image_sha = _sha256_bytes(content)
        source_name = str(filename or "未命名")
        manifest_bytes = DEFAULT_MANIFEST.read_bytes()
        manifest_sha = _sha256_bytes(manifest_bytes)

        conn = self._connect()
        try:
            # Claim the request before filesystem writes, including concurrent
            # submissions from separate browser sessions.
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM runs WHERE request_id = ?", (rid,)).fetchone()
            if existing is not None:
                if (existing["image_sha256"] == image_sha
                        and existing["manifest_sha256"] == manifest_sha
                        and existing["seed"] == DEFAULT_SEED):
                    return self._row_to_job(existing)
                raise RgbCandidateInputError(
                    "该请求的图片或模型版本已变化；请用“新建一次识别”更换编号。")
            job_id = uuid.uuid4().hex
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True)
            input_path = job_dir / f"input.{fmt.lower()}"
            input_path.write_bytes(content)  # 原字节原样保存，保持输入 SHA
            (job_dir / "model-manifest.json").write_bytes(manifest_bytes)
            result_dir = job_dir / "result"
            conn.execute(
                "INSERT INTO runs (job_id, request_id, source_name, image_sha256,"
                " manifest_sha256, seed, status, error, input_format, input_path,"
                " result_dir, created_at, started_at, finished_at, artifact_sha256)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                (job_id, rid, source_name, image_sha, manifest_sha, DEFAULT_SEED,
                 "running", None, fmt, str(input_path), str(result_dir),
                 _utc_now(), _utc_now()))
            conn.commit()
        finally:
            conn.close()

        self._run_job(job_id)
        job = self.get_job(job_id)
        assert job is not None
        return job

    # ---------- 运行与落库 ----------

    def _run_job(self, job_id: str) -> None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        input_path, result_dir = Path(row["input_path"]), Path(row["result_dir"])
        manifest_path = self._job_dir(job_id) / "model-manifest.json"
        try:
            error = self._environment_error()
            if error is None:
                env = dict(os.environ)
                env.update(THREAD_ENV)
                env["PYTHONUTF8"] = "1"
                cmd = [str(INTERPRETER), "-B", str(CLI_SCRIPT),
                       "--image", str(input_path),
                       "--output-dir", str(result_dir),
                       "--manifest", str(manifest_path)]
                try:
                    proc = subprocess.run(
                        cmd, shell=False, cwd=str(ROOT),
                        timeout=RUN_TIMEOUT_SECONDS, capture_output=True, env=env)
                except subprocess.TimeoutExpired:
                    error = f"识别超时（{RUN_TIMEOUT_SECONDS} 秒），已保留记录，不自动重试。"
                else:
                    if proc.returncode != 0:
                        tail = (proc.stderr or proc.stdout or b"")[-800:].decode(
                            "utf-8", errors="replace").strip()
                        error = f"识别程序退出码 {proc.returncode}：{tail or '无输出'}"
            if error is None:
                try:
                    if _sha256_bytes(manifest_path.read_bytes()) != row["manifest_sha256"]:
                        raise ValueError("本次任务的模型清单快照已变化")
                    parsed = verify_outputs(
                        result_dir, row["image_sha256"], manifest_path,
                        DEFAULT_SEED)
                except Exception as exc:
                    error = f"产物校验未通过：{exc}"
            if error is not None:
                self._finish_failed(job_id, error)
                return
            self._finish_success(job_id, result_dir, parsed)
        except Exception as exc:  # 意外异常也如实标失败，不留 running 假象
            self._finish_failed(job_id, f"识别过程发生异常：{exc}")

    def _environment_error(self) -> str | None:
        if not INTERPRETER.is_file():
            return f"未找到识别解释器：{INTERPRETER}。不自动安装，请联系维护者。"
        if not CLI_SCRIPT.is_file():
            return f"未找到识别脚本：{CLI_SCRIPT}。"
        if not DEFAULT_MANIFEST.is_file():
            return f"未找到候选模型清单：{DEFAULT_MANIFEST}。"
        return None

    def _finish_failed(self, job_id: str, error: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status='failed', error=?, finished_at=?"
                " WHERE job_id=? AND status='running'",
                (error, _utc_now(), job_id))
            conn.commit()
        finally:
            conn.close()

    def _finish_success(self, job_id: str, result_dir: Path, parsed: dict) -> None:
        """成功状态、候选区域与摘要在一个事务内写入。"""
        shas = {name: _sha256_bytes((result_dir / name).read_bytes())
                for name in ARTIFACT_NAMES}
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status='succeeded', error=NULL, finished_at=?,"
                " artifact_sha256=? WHERE job_id=? AND status='running'",
                (_utc_now(), json.dumps(shas), job_id))
            conn.execute("DELETE FROM regions WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM class_counts WHERE job_id=?", (job_id,))
            for class_id in (0, 1, 2):
                conn.execute(
                    "INSERT INTO class_counts (job_id,class_id,pixels) VALUES (?,?,?)",
                    (job_id, class_id, parsed["metadata"]["native_class_counts"][str(class_id)]))
            for r in parsed["regions"]:
                conn.execute(
                    "INSERT INTO regions (job_id, field_id, area_pixels, bbox_x0,"
                    " bbox_y0, bbox_x1, bbox_y1, centroid_x, centroid_y,"
                    " edge_touch, small_patch) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, r["field_id"], r["area_pixels"], r["bbox_x0"],
                     r["bbox_y0"], r["bbox_x1"], r["bbox_y1"], r["centroid_x"],
                     r["centroid_y"], int(r["edge_touch"]), int(r["small_patch"])))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------- 结果与产物 ----------

    def result(self, job_id: str) -> dict | None:
        """返回作业结果；实际重读产物并核对摘要，篡改时明确报错。"""
        job = self.get_job(job_id)
        if job is None:
            return None
        out = {k: job[k] for k in ("id", "status", "source_name",
                                   "image_sha256", "created_at", "error")}
        if job["status"] != "succeeded":
            out.update({"metadata": None, "quality": None, "regions": []})
            return out
        result_dir = self._job_result_dir(job)
        # Check the persisted snapshot, including JSON files whose whitespace
        # may change without affecting the cross-file content checks below.
        for name in ARTIFACT_NAMES:
            self.artifact(job_id, name)
        manifest_path = self._job_dir(job_id) / "model-manifest.json"
        if _sha256_bytes(manifest_path.read_bytes()) != job["manifest_sha256"]:
            raise RgbArtifactError("本次任务的模型清单快照已变化")
        parsed = verify_outputs(result_dir, job["image_sha256"],
                                manifest_path, job["seed"])
        out["metadata"] = parsed["metadata"]
        out["quality"] = parsed["quality"]
        out["regions"] = parsed["regions"]
        return out

    def _job_result_dir(self, job: dict) -> Path:
        job_dir = self._job_dir(job["id"])
        expected = (job_dir / "result").resolve()
        stored = Path(job.get("result_dir") or "")
        if stored.resolve() != expected or not expected.is_relative_to(
                self.jobs_dir.resolve()):
            raise RgbArtifactError("作业结果目录不合法")
        return expected

    def artifact(self, job_id: str, name: str) -> bytes:
        """读取六固定产物之一；核对存库 SHA，防篡改；限定当前作业目录。"""
        if name not in ARTIFACT_NAMES:
            raise RgbArtifactError(
                "只能下载六份固定结果文件（mask256/mask512/overlay/CSV/quality/metadata）。")
        job = self.get_job(job_id)
        if job is None:
            raise RgbArtifactError("作业不存在")
        if job["status"] != "succeeded":
            raise RgbArtifactError("该作业没有成功结果，无可下载产物。")
        result_dir = self._job_result_dir(job)
        path = result_dir / name
        if not path.is_file():
            raise RgbArtifactError(f"产物 {name} 缺失")
        data = path.read_bytes()
        recorded = job["artifact_sha256"].get(name)
        if not isinstance(recorded, str) or _sha256_bytes(data) != recorded:
            raise RgbArtifactError(f"产物 {name} 已被改动，与保存时的摘要不一致。")
        return data

    def source_bytes(self, job_id: str) -> bytes:
        """返回作业的原始输入字节（核对图像 SHA）。"""
        job = self.get_job(job_id)
        if job is None:
            raise RgbArtifactError("作业不存在")
        job_dir = self._job_dir(job_id)
        source = Path(job["source_path"])
        if not (source.resolve().is_relative_to(job_dir.resolve()) and source.is_file()):
            raise RgbArtifactError("原始输入文件缺失或位置不合法")
        data = source.read_bytes()
        if _sha256_bytes(data) != job["image_sha256"]:
            raise RgbArtifactError("原始输入文件已被改动，与保存时的摘要不一致。")
        return data
