"""RGB候选成功作业 → P0 数据目录登记（INT-20261021-001）。

把**已成功且六产物逐项核验通过**的 RGB 候选识别作业登记为两份 P0 资产，
并写入独立链接表，实现 RGB 作业库与 P0 治理目录的血缘贯通。

诚实边界（务必遵守）：

- 登记的两份资产是**派生内容**：512×512 无损 PNG 影像（与原图逐像素一致）
  和 JSON 识别统计记录。原始输入（可能是 TIFF/JPEG）不复制、不冒称
  "已通过 P0 质检"；质检结论只针对 PNG 与 JSON 本身。
- ``scene_id`` 固定为 ``RGB-<job_id>``，仅是本次作业的技术分组，不代表
  真实农田身份；不伪造 crop/平方米/许可/拍摄日期等农业属性。
  ``source='rgb_candidate_derived'``、``license_status='unknown'``。
- 登记不触发推理、不改 RGB 作业库；RGB 侧访问一律只读
  （:class:`RgbCandidateJobs` ``readonly=True``）。
- 原子性：表初始化在事务开始前完成；两次 ``ingest.import_bytes`` 与链接
  行在同一事务提交（延迟提交代理包住 P0 importer，本模块不导入
  agrolake_service 包），任何异常整体回滚；``BEGIN IMMEDIATE`` 认领保证
  重复/并发同作业不产生多余资产/事件/质检。连接上已有未提交外层事务时
  预先拒绝。
- 幂等：同作业重复登记先核对来源/产物/P0 对象摘要，全部完好才返回既有
  链接；任何损坏都明确报错，不显示成功。失败遗留的孤立内容对象文件保留，
  不删除。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from . import config as p0_config
from . import db as p0_db
from . import ingest as p0_ingest
from . import quality as p0_quality
from .rgb_jobs import (
    ARTIFACT_NAMES,
    JOB_ID_RE,
    RgbArtifactError,
    RgbCandidateJobs,
)

LINKS_TABLE = "rgb_governance_links"
DERIVED_SOURCE = "rgb_candidate_derived"
SCENE_PREFIX = "RGB-"
STATS_KIND = "agrolake.rgb_candidate.stats"
STATS_SCHEMA = "AGROLAKE-RGB-STATS-1.0.0"

_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")

_LINKS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {LINKS_TABLE} (
  link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL UNIQUE,
  scene_id TEXT NOT NULL,
  image_asset_id INTEGER NOT NULL,
  image_version_id INTEGER NOT NULL,
  stats_asset_id INTEGER NOT NULL,
  stats_version_id INTEGER NOT NULL,
  source_image_sha256 TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  png_sha256 TEXT NOT NULL,
  json_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

_BOUNDARIES = {
    "pixel_area_is_not_square_meters":
        "像素面积不是平方米；没有比例尺或地理参考不能换算实测面积。",
    "candidates_not_confirmed_fields":
        "候选区域是程序统计的四连通块，未经实地确认，不是已核实真实田块。",
    "no_independent_evaluation":
        "模型为候选版本，尚未用独立来源新图评估；本记录不构成识别精度结论。",
    "original_input_not_p0_checked":
        "原始输入未复制进 P0，其质检结论只针对派生 PNG 与 JSON 本身。",
}


class RgbGovernanceError(ValueError):
    """登记请求不合法或核验未通过；明确报错而不是显示旧成功。"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _readonly_backend(backend=None):
    if backend is None:
        return RgbCandidateJobs(readonly=True)
    if not backend.readonly:
        return RgbCandidateJobs(project_root=backend.project_root,
                                data_dir=backend.data_dir, readonly=True)
    return backend


def _looks_like_local_path(value: str) -> bool:
    if "\\" in value:
        return True
    if _WIN_ABS_RE.match(value):
        return True
    if value.lower().startswith("file://"):
        return True
    return False


def _require_no_paths(obj, where: str) -> None:
    """登记 JSON 不得携带本机绝对路径（防泄漏）。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _require_no_paths(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _require_no_paths(value, f"{where}[{index}]")
    elif isinstance(obj, str):
        if _looks_like_local_path(obj):
            raise RgbGovernanceError(f"登记记录字段 {where} 疑似包含本地路径，已拒绝")


class _DeferredCommit:
    """把底层连接的 commit 变成 no-op，使 P0 importer 的多次写入归入同一事务。

    与 agrolake_service.governance._DeferredCommit 同构；按任务要求独立实现，
    不导入整个服务包。
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def commit(self) -> None:  # 延迟提交：由调用方统一 commit/rollback
        return None

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _derived_png_bytes(source: bytes) -> bytes:
    """从原始输入解码并生成 512×512 RGB 无损 PNG；逐像素比对后返回字节。"""
    try:
        with Image.open(io.BytesIO(source)) as im:
            im.load()
            if im.size != (512, 512):
                raise RgbGovernanceError(
                    f"原始输入尺寸为 {im.size[0]}×{im.size[1]}，不是 512×512")
            rgb = im.convert("RGB")
            pixels = list(rgb.getdata())
            buf = io.BytesIO()
            rgb.save(buf, "PNG")
    except RgbGovernanceError:
        raise
    except Exception as exc:
        raise RgbGovernanceError(f"原始输入无法解码：{exc}") from exc
    with Image.open(io.BytesIO(buf.getvalue())) as check:
        if check.size != (512, 512) or check.mode != "RGB" \
                or list(check.getdata()) != pixels:
            raise RgbGovernanceError("派生 PNG 与原图逐像素比对不一致，拒绝登记")
    return buf.getvalue()


def _build_stats_bytes(job: dict, parsed: dict, png_sha: str, scene_id: str,
                       registered_at: str) -> bytes:
    """构建 JSON 识别统计记录（仅含安全事实字段，不含本机路径与图片字节）。"""
    native = parsed["metadata"].get("native_class_counts", {})
    stats = {
        "kind": STATS_KIND,
        "schema": STATS_SCHEMA,
        "job_id": job["id"],
        "source_image_sha256": job["image_sha256"],
        "source_name": Path(str(job["source_name"]).replace("\\", "/")).name,
        "source_format": job["input_format"],
        "derived_png_sha256": png_sha,
        "manifest_sha256": job["manifest_sha256"],
        "seed": job["seed"],
        "artifact_sha256": {name: job["artifact_sha256"].get(name)
                            for name in ARTIFACT_NAMES},
        "native_class_counts": native,
        "class_labels": {"0": "田块候选", "1": "田埂", "2": "道路"},
        "area_unit": "pixel",
        "grid_class_counts": parsed["metadata"].get("grid_class_counts", {}),
        "native_total_pixels": 262144,
        "candidate_count": len(parsed["regions"]),
        "regions": parsed["regions"],
        "scene_id": scene_id,
        "scene_id_meaning": "技术分组编号（RGB-作业号），不代表真实农田身份",
        "boundaries": _BOUNDARIES,
        "registered_at": registered_at,
    }
    _require_no_paths(stats, "stats")
    return json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _load_succeeded_job(backend: RgbCandidateJobs, job_id: str) -> tuple[dict, dict]:
    """经只读后端加载成功作业并全量核验产物；任何不符都报错。"""
    job = backend.get_job(job_id)
    if job is None:
        raise RgbGovernanceError(f"未知作业 {job_id}")
    if job["status"] != "succeeded":
        raise RgbGovernanceError(
            f"作业状态为 {job['status']}，只有成功作业可登记")
    parsed = backend.result(job_id)  # 内部重读并核对六产物/清单快照摘要
    backend.source_bytes(job_id)     # 核对原始输入摘要
    with closing(backend._connect()) as sql:
        counts = {str(r["class_id"]): r["pixels"] for r in sql.execute(
            "SELECT class_id,pixels FROM class_counts WHERE job_id=?", (job_id,))}
        regions = [dict(r) for r in sql.execute(
            "SELECT field_id,area_pixels,bbox_x0,bbox_y0,bbox_x1,bbox_y1,"
            "centroid_x,centroid_y,edge_touch,small_patch FROM regions "
            "WHERE job_id=? ORDER BY field_id", (job_id,))]
    if counts != parsed["metadata"]["native_class_counts"] or regions != parsed["regions"]:
        raise RgbGovernanceError("RGB统计库行与保存产物不一致，拒绝登记")
    return job, parsed


def _verify_registration(conn: sqlite3.Connection, backend: RgbCandidateJobs,
                         row: sqlite3.Row, *, store_root: Path) -> dict:
    """对既有链接做全量核验：RGB 侧 + P0 对象 + 版本行，损坏即报错。"""
    job_id = row["job_id"]
    job, parsed = _load_succeeded_job(backend, job_id)
    if job["image_sha256"] != row["source_image_sha256"] \
            or job["manifest_sha256"] != row["manifest_sha256"]:
        raise RgbGovernanceError("作业输入/清单摘要与登记记录不一致")
    if row["scene_id"] != SCENE_PREFIX + job_id:
        raise RgbGovernanceError("登记场景与任务编号不一致")
    store_root = Path(store_root).resolve()
    contents = {}
    for col in ("png_sha256", "json_sha256"):
        sha = row[col]
        obj = conn.execute(
            "SELECT relpath, size_bytes FROM objects WHERE object_id = ?", (sha,)
        ).fetchone()
        if obj is None:
            raise RgbGovernanceError(f"P0 对象记录缺失（{col}）")
        path = (store_root / obj["relpath"]).resolve()
        if not path.is_relative_to(store_root):
            raise RgbGovernanceError("P0对象路径越界，拒绝读取")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RgbGovernanceError("P0 对象文件缺失或不可读") from exc
        if _sha256_bytes(data) != sha or len(data) != obj["size_bytes"]:
            raise RgbGovernanceError("P0 对象文件与登记摘要不一致，登记已损坏")
        contents[col] = data
    for role, ctype, object_key in (("image", "image", "png_sha256"),
                                    ("stats", "text", "json_sha256")):
        col = role + "_version_id"
        v = conn.execute(
            "SELECT * FROM asset_versions WHERE version_id = ?",
            (row[col],)).fetchone()
        if v is None:
            raise RgbGovernanceError(f"登记引用的 P0 版本 #{row[col]} 不存在")
        asset = conn.execute("SELECT asset_id FROM assets WHERE asset_id=?",
                             (row[role + "_asset_id"],)).fetchone()
        if (asset is None or v["asset_id"] != row[role + "_asset_id"]
                or v["object_id"] != row[object_key]
                or v["content_type"] != ctype
                or v["declared_scene_id"] != row["scene_id"]
                or v["source"] != DERIVED_SOURCE):
            raise RgbGovernanceError("P0版本、资产、对象或场景与登记链接不一致")
    try:
        with Image.open(io.BytesIO(contents["png_sha256"])) as png, \
                Image.open(io.BytesIO(backend.source_bytes(job_id))) as source:
            if (png.mode != source.mode or png.size != source.size
                    or png.tobytes() != source.tobytes()):
                raise RgbGovernanceError("登记PNG与原图像素不一致")
        expected = json.loads(_build_stats_bytes(job, parsed, row["png_sha256"],
                              row["scene_id"], row["created_at"]))
        if json.loads(contents["json_sha256"]) != expected:
            raise RgbGovernanceError("登记JSON与任务来源、模型或统计不一致")
    except (OSError, ValueError) as exc:
        raise RgbGovernanceError("登记派生内容核验未通过") from exc
    return {
        "job_id": job_id,
        "scene_id": row["scene_id"],
        "image_asset_id": row["image_asset_id"],
        "image_version_id": row["image_version_id"],
        "stats_asset_id": row["stats_asset_id"],
        "stats_version_id": row["stats_version_id"],
        "source_image_sha256": row["source_image_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "png_sha256": row["png_sha256"],
        "json_sha256": row["json_sha256"],
        "created_at": row["created_at"],
    }


def _quality_overall(conn: sqlite3.Connection, version_id: int) -> str:
    checks = p0_quality.latest_checks(conn, version_id)
    return p0_quality.overall_status(checks)


def _quality_summary(conn: sqlite3.Connection, image_vid: int, stats_vid: int) -> str:
    zh = {"pass": "通过", "warn": "警告", "block": "阻断", "na": "不适用"}
    img = zh.get(_quality_overall(conn, image_vid), "未知")
    sta = zh.get(_quality_overall(conn, stats_vid), "未知")
    return (f"派生PNG质检 {img} / 统计JSON质检 {sta}"
            f"（规则版本 {p0_quality.rules_version()}；只针对派生文件本身，"
            "不代表原始输入已通过 P0 质检）")


def register(conn: sqlite3.Connection, job_id: str, *,
             backend: RgbCandidateJobs | None = None,
             store_root: Path | str | None = None) -> dict:
    """把成功 RGB 作业的两份派生资产登记进 P0 并建立链接。

    返回 ``{'created': bool, 'registration': {...}, 'quality_summary': str}``。
    重复登记核对全部摘要后返回既有链接（created=False）；任何损坏都报错。
    """
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise RgbGovernanceError("作业编号必须是 32 位小写十六进制")
    if conn.in_transaction:
        raise RgbGovernanceError(
            "连接上已有未提交的外层事务；为避免嵌套提交，拒绝本次登记。")
    backend = _readonly_backend(backend)
    store_root = Path(store_root) if store_root else p0_config.storage_root()

    # 模式初始化（含本模块链接表）必须在事务开始前完成（与治理登记单同口径）。
    p0_db.init_schema(conn)
    conn.executescript(_LINKS_SCHEMA)
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            f"SELECT * FROM {LINKS_TABLE} WHERE job_id = ?", (job_id,)).fetchone()
        if row is not None:
            # 幂等返回前仍须核验来源/产物/P0 对象；损坏不显示成功。
            registration = _verify_registration(
                conn, backend, row, store_root=store_root)
            conn.rollback()  # 只读路径，未写任何内容
            summary = _quality_summary(
                conn, row["image_version_id"], row["stats_version_id"])
            return {"created": False, "registration": registration,
                    "quality_summary": summary}

        job, parsed = _load_succeeded_job(backend, job_id)
        scene_id = SCENE_PREFIX + job_id
        png_bytes = _derived_png_bytes(backend.source_bytes(job_id))
        png_sha = _sha256_bytes(png_bytes)
        registered_at = _now()
        stats_bytes = _build_stats_bytes(job, parsed, png_sha, scene_id, registered_at)
        json_sha = _sha256_bytes(stats_bytes)

        proxy = _DeferredCommit(conn)
        img = p0_ingest.import_bytes(
            proxy, png_bytes, f"rgb-{job_id}.png",
            store_root=store_root,
            source=DERIVED_SOURCE, scene_id=scene_id,
            intended_use="rgb_derived_image", license_status="unknown",
        )
        sta = p0_ingest.import_bytes(
            proxy, stats_bytes, f"rgb-{job_id}-stats.json",
            store_root=store_root,
            source=DERIVED_SOURCE, scene_id=scene_id,
            intended_use="rgb_recognition_stats", license_status="unknown",
        )
        image_vid = int(img["version_id"])
        stats_vid = int(sta["version_id"])
        conn.execute(
            f"INSERT INTO {LINKS_TABLE}(job_id, scene_id, image_asset_id,"
            " image_version_id, stats_asset_id, stats_version_id,"
            " source_image_sha256, manifest_sha256, png_sha256, json_sha256,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, scene_id, int(img["asset_id"]), image_vid,
             int(sta["asset_id"]), stats_vid, job["image_sha256"],
             job["manifest_sha256"], png_sha, json_sha, registered_at),
        )
        inserted = conn.execute(
            f"SELECT * FROM {LINKS_TABLE} WHERE job_id=?", (job_id,)).fetchone()
        _verify_registration(conn, backend, inserted, store_root=store_root)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    row = conn.execute(
        f"SELECT * FROM {LINKS_TABLE} WHERE job_id = ?", (job_id,)).fetchone()
    registration = {
        "job_id": job_id,
        "scene_id": row["scene_id"],
        "image_asset_id": row["image_asset_id"],
        "image_version_id": row["image_version_id"],
        "stats_asset_id": row["stats_asset_id"],
        "stats_version_id": row["stats_version_id"],
        "source_image_sha256": row["source_image_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "png_sha256": row["png_sha256"],
        "json_sha256": row["json_sha256"],
        "created_at": row["created_at"],
    }
    return {
        "created": True,
        "registration": registration,
        "quality_summary": _quality_summary(
            conn, row["image_version_id"], row["stats_version_id"]),
    }


def link_status(conn: sqlite3.Connection, job_id: str, *,
                backend: RgbCandidateJobs | None = None,
                store_root: Path | str | None = None) -> dict:
    """只读查询登记状态并核验；供页面展示，绝不写入。

    返回 ``{'registered': bool, 'intact': bool|None, ...}``；
    ``registered=True, intact=False`` 时附带中文 ``problem``。
    """
    job_id = job_id if isinstance(job_id, str) else ""
    if not JOB_ID_RE.match(job_id):
        return {"registered": False, "intact": None,
                "problem": "作业编号格式不合法"}
    store_root = Path(store_root) if store_root else p0_config.storage_root()
    try:
        row = conn.execute(
            f"SELECT * FROM {LINKS_TABLE} WHERE job_id = ?", (job_id,)).fetchone()
    except sqlite3.Error:
        return {"registered": False, "intact": None}
    if row is None:
        return {"registered": False, "intact": None}
    base = {
        "registered": True,
        "scene_id": row["scene_id"],
        "image_asset_id": row["image_asset_id"],
        "image_version_id": row["image_version_id"],
        "stats_asset_id": row["stats_asset_id"],
        "stats_version_id": row["stats_version_id"],
        "created_at": row["created_at"],
    }
    try:
        image_q = _quality_overall(conn, row["image_version_id"])
        stats_q = _quality_overall(conn, row["stats_version_id"])
    except sqlite3.Error:
        image_q = stats_q = "未知"
    base["image_quality"] = image_q
    base["stats_quality"] = stats_q
    try:
        _verify_registration(conn, _readonly_backend(backend),
                             row, store_root=store_root)
    except (RgbGovernanceError, RgbArtifactError, ValueError, OSError) as exc:
        base["intact"] = False
        base["problem"] = str(exc)
        return base
    base["intact"] = True
    return base
