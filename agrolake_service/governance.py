"""识别输入登记单治理与血缘贯通（DEV-20261015-014）。

目标：给本机识别服务里**已成功**的作业增加“治理登记与追溯”。

诚实边界（务必遵守）：
- 只把**已成功作业**生成的“识别输入登记单 JSON”登记为 P0 文本资产，得到真实的
  asset_id / version_id / quality_checks / import_event。
- 登记单的质检结论只针对这份 JSON 文本，**不等同于原 TIFF 已通过 P0 质检**，
  也不是推理前质量门禁。原始 TIFF 仍只存在于识别服务存储，不复制进 P0。
- 登记不触发任何新推理，不改原图/掩膜/类别计数；不声称闭环、精度提升或平方米。
- 登记单只记录来自作业的服务端事实；客户端不得指定输入/模型摘要等快照。
- 失败/中断/未知作业分别 409/404；每作业固定一条登记单链接。
- 本轮不做元数据修正/新版本复跑。

事务与原子性：
- 建表/模式初始化在事务开始前完成。
- P0 的 ``ingest.import_bytes`` 内部会自行 ``commit``。这里用“延迟提交代理”包住同一
  连接，把它内部的 commit 变成 no-op，使**登记单 P0 记录与 rec 链接同事务提交**；
  中途任何异常都整体 rollback。不修改原 P0 函数。

对象存储：
- 登记单字节先用 ``agrolake_service.store.store_bytes`` 安全发布并**回读核验**，
  再交给 P0 importer 复用同一份字节（P0 存入自身内容寻址存储，摘要相同）。
- 暂存/孤立文件一律保留，不删除、不覆盖。
- 不把损坏输入的事实当作验证成功：登记前核验输入对象字节大小/SHA 与 run.json 产物摘要。
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from agrolake import config as p0_config
from agrolake import db as p0_db
from agrolake import ingest as p0_ingest
from agrolake import quality as p0_quality

from . import db, store, reviews as review_records
from .validation import ValidationError

MANIFEST_KIND = "agrolake.recognition.input_manifest"
MANIFEST_SCHEMA = "AGROLAKE-RECOGNITION-MANIFEST-1.0.0"
MANIFEST_SOURCE = "agrolake_service.governance"

MAX_REQUEST_ID_LEN = 128
MAX_GOVERNANCE_BODY_BYTES = 4 * 1024

# 血缘查询的容量边界：返回总数与截断说明，绝不静默冒充完整。
MAX_LINEAGE_REVIEWS = 500
MAX_LINEAGE_CHECKS = 500
MAX_LINEAGE_DISPOSITIONS = 500
MAX_LINEAGE_HISTORY = 500

_SQLITE_MAX_INT = 9223372036854775807
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % MAX_REQUEST_ID_LEN)
_ALLOWED_KEYS = frozenset({"request_id"})

# 本地绝对路径检测：Windows 盘符路径、UNC、file://。http(s) 外链不受影响。
_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")

_MANIFEST_NOTE = (
    "本登记单为识别输入（公开/上传 TIFF）的服务端事实快照，作为 P0 JSON/TXT 文本资产登记；"
    "其质检结论只针对这份 JSON 文本，不等同于原 TIFF 已通过 P0 质检或存在推理前质量门禁。"
    "原 TIFF 仍只保存在识别服务存储，不复制进 P0。登记不触发新推理，不改掩膜/类别计数。"
)


def _now() -> str:
    return db.now()


def _looks_like_local_path(value: str) -> bool:
    if "\\" in value:
        return True
    if _WIN_ABS_RE.match(value):
        return True
    if value.lower().startswith("file://"):
        return True
    return False


def normalize_request(payload) -> str:
    """校验登记请求体：只允许 {request_id}，沿用复核的 request_id 约定。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise ValidationError(f"不接受字段 {unknown}：本接口只接受 request_id")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise ValidationError(
            f"request_id 必填，且须匹配 ^[A-Za-z0-9._:-]{{1,{MAX_REQUEST_ID_LEN}}}$"
        )
    return request_id


def _require_no_paths(obj, where: str) -> None:
    """登记单字段不得携带本地绝对路径（防泄漏）。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _require_no_paths(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _require_no_paths(value, f"{where}[{index}]")
    elif isinstance(obj, str):
        if _looks_like_local_path(obj):
            raise ValidationError(f"登记单字段 {where} 疑似包含本地路径，已拒绝")


def build_manifest(job: sqlite3.Row, run: sqlite3.Row) -> dict:
    """从服务端作业事实构建登记单字典（不含本地绝对路径）。"""
    if run is None or run["status"] != "succeeded":
        raise ValidationError(f"作业 {job['job_id']} 缺少成功运行记录，不能登记", status=409)
    return {
        "kind": MANIFEST_KIND,
        "schema": MANIFEST_SCHEMA,
        "created_by": MANIFEST_SOURCE,
        "job_id": int(job["job_id"]),
        "input_id": int(job["input_id"]),
        "input_sha256": job["input_sha256"],
        "original_filename": job["original_filename"],
        "source": job["source"],
        "source_link": job["source_link"],
        "license_status": job["license_status"],
        "scene_id": job["scene_id"],
        "sample_kind": job["sample_kind"],
        "model_sha256": job["model_sha256"],
        "run": {
            "run_seq": int(run["run_seq"]),
            "status": run["status"],
            "exit_code": run["exit_code"],
            "adapter": run["adapter"],
            "finished_at": run["finished_at"],
        },
        "note": _MANIFEST_NOTE,
    }


def _stable_manifest_bytes(manifest: dict) -> bytes:
    """稳定 JSON 序列化（排序键，固定缩进），保证同事实产生同字节。"""
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


class _DeferredCommit:
    """把底层连接的 commit 变成 no-op，使多次写入归入同一外层事务。

    只用于 P0 importer：它内部会反复 commit；这里延迟到调用方统一提交/回滚。
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def commit(self) -> None:  # 延迟提交
        return None

    def __getattr__(self, name):
        return getattr(self._conn, name)

    # P0 当前路径未使用 with-conn，但保持语义安全：不在此处隐式提交/回滚。
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _row_to_dict(row) -> dict | None:
    return dict(row) if row is not None else None


def _verify_object(root: Path, relpath: str, expected_sha: str,
                   expected_size: int | None) -> tuple[bool, str]:
    """核验对象文件存在、大小与 SHA 一致；返回 (是否完好, 说明)。"""
    try:
        path = store.ensure_within(root, Path(root) / relpath)
    except store.PathBoundaryError:
        return False, "对象路径越界，拒绝访问"
    if not path.is_file():
        return False, "对象文件缺失"
    try:
        size = path.stat().st_size
        actual = store.sha256_file(path)
    except OSError:
        return False, "对象读取失败，请检查本机存储"
    if expected_size is not None and size != expected_size:
        return False, f"对象大小不符：期望 {expected_size} 实际 {size}"
    if actual != expected_sha:
        return False, "对象摘要不符"
    return True, "对象字节与摘要一致（检查时）"


def _load_succeeded_job(conn: sqlite3.Connection, job_id: int):
    job = conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if job is None:
        raise ValidationError(f"未知作业 {job_id}", status=404)
    if job["status"] != "succeeded":
        raise ValidationError(
            f"作业 {job_id} 当前状态为 {job['status']}，只有成功作业可登记输入资料",
            status=409,
        )
    run = conn.execute(
        "SELECT * FROM rec_runs WHERE job_id = ? ORDER BY run_seq DESC LIMIT 1", (job_id,)
    ).fetchone()
    if run is None or run["status"] != "succeeded":
        raise ValidationError(f"作业 {job_id} 缺少成功运行记录，不能登记", status=409)
    return job, run


def _verify_job_facts(config, conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """核验输入对象与 run.json 产物摘要；不符即 409，绝不把损坏当成验证成功。"""
    inp = conn.execute(
        "SELECT object_relpath, size_bytes FROM rec_inputs WHERE input_id = ?", (job["input_id"],)
    ).fetchone()
    if inp is None:
        raise ValidationError("输入记录缺失，不能登记", status=409)
    ok, note = _verify_object(
        Path(config.store_root), inp["object_relpath"], job["input_sha256"], inp["size_bytes"]
    )
    if not ok:
        raise ValidationError(f"输入对象未通过字节核验（{note}），拒绝登记", status=409)

    cli_rel = job["cli_output_relpath"]
    if not cli_rel:
        raise ValidationError("作业缺少产物目录记录，不能登记", status=409)
    art = conn.execute(
        "SELECT sha256, size_bytes FROM rec_artifacts WHERE job_id = ? AND name = 'run.json'",
        (job["job_id"],),
    ).fetchone()
    if art is None:
        raise ValidationError("缺少 run.json 产物校验记录，不能登记", status=409)
    run_json_rel = (Path(cli_rel) / "run.json").as_posix()
    ok, note = _verify_object(Path(config.runs_root), run_json_rel, art["sha256"], art["size_bytes"])
    if not ok:
        raise ValidationError(f"run.json 产物未通过字节核验（{note}），拒绝登记", status=409)


def _get_link(conn: sqlite3.Connection, job_id: int):
    return conn.execute("SELECT * FROM rec_governance_links WHERE job_id = ?", (job_id,)).fetchone()


def _link_to_dict(row: sqlite3.Row) -> dict:
    return {
        "link_id": row["link_id"],
        "job_id": row["job_id"],
        "request_id": row["request_id"],
        "asset_id": row["asset_id"],
        "version_id": row["version_id"],
        "manifest_sha256": row["manifest_sha256"],
        "input_sha256": row["input_sha256"],
        "manifest_relpath": row["manifest_relpath"],
        "created_at": row["created_at"],
    }


def _checks_snapshot(conn: sqlite3.Connection, version_id: int) -> dict:
    checks = p0_quality.latest_checks(conn, version_id)
    return {
        "rules_version": p0_quality.rules_version(),
        "overall": p0_quality.overall_status(checks),
        "checks": [
            {
                "check_id": c["check_id"],
                "rule_id": c["rule_id"], "rule_version": c["rule_version"],
                "status": c["status"], "reason": c["reason"],
                "measured": json.loads(c["measured_json"]) if c["measured_json"] else None,
            }
            for c in checks
        ],
    }


def register_governance(config, job_id: int, payload) -> dict:
    """为成功作业登记识别输入登记单 JSON 为 P0 文本资产并建立治理链接。

    返回 {'created': bool, 'registration': {...}}。
    - 同号同作业复用既有登记（created=False）。
    - 同作业另号 / 跨作业同号一律 409，不会多建登记单/版本/事件。
    - 登记单 P0 记录与 rec 链接在同一事务提交；中途异常整体回滚。
    """
    request_id = normalize_request(payload)
    db_path = Path(config.db_path)

    conn = db.connect(db_path)
    try:
        # 模式初始化（含 P0 表）必须在事务开始前完成。
        p0_db.init_schema(conn)
        db.init_schema(conn)

        conn.execute("BEGIN IMMEDIATE")
        try:
            job, run = _load_succeeded_job(conn, job_id)

            existing = _get_link(conn, job_id)
            if existing is not None:
                if existing["request_id"] != request_id:
                    raise ValidationError(
                        "该作业已登记识别输入资料，不能以不同请求号重复登记", status=409
                    )
                conn.commit()
                return {"created": False, "registration": _link_to_dict(existing)}

            cross = conn.execute(
                "SELECT job_id FROM rec_governance_links WHERE request_id = ?", (request_id,)
            ).fetchone()
            if cross is not None and cross["job_id"] != job_id:
                raise ValidationError("该请求号已用于其它作业的登记，拒绝复用", status=409)

            _verify_job_facts(config, conn, job)

            manifest = build_manifest(job, run)
            _require_no_paths(manifest, "manifest")
            manifest_bytes = _stable_manifest_bytes(manifest)
            manifest_sha = store.sha256_bytes(manifest_bytes)

            # 1) 服务存储安全发布并回读核验登记单对象。
            sha, obj_path, _created = store.store_bytes(Path(config.store_root), manifest_bytes)
            if sha != manifest_sha:
                raise ValidationError("登记单对象摘要异常，拒绝登记", status=500)
            relpath = store.relative_to_root(Path(config.store_root), obj_path)
            ok, note = _verify_object(
                Path(config.store_root), relpath, manifest_sha, len(manifest_bytes)
            )
            if not ok:
                raise ValidationError(f"登记单对象发布后未通过核验（{note}）", status=500)

            # 2) 延迟提交代理 + P0 importer：登记单 P0 记录纳入同一事务。
            proxy = _DeferredCommit(conn)
            filename = f"recognition-input-manifest-job-{job_id}.json"
            p0_store_root = Path(p0_config.storage_root())
            # 先在真正的P0根安全发布，使旧importer只复用，不走replace写入。
            store.store_bytes(p0_store_root, manifest_bytes)
            imported = p0_ingest.import_bytes(
                proxy,
                manifest_bytes,
                filename,
                store_root=p0_store_root,
                source=job["source"],
                scene_id=job["scene_id"],
                intended_use="recognition_input_manifest",
                source_link=job["source_link"],
                license_status=job["license_status"],
            )
            version_id = int(imported["version_id"])
            asset_id = int(imported["asset_id"])

            # 3) 核验 P0 登记单对象字节。
            p0_obj = conn.execute(
                "SELECT relpath, size_bytes FROM objects WHERE object_id = ?", (manifest_sha,)
            ).fetchone()
            if p0_obj is None:
                raise ValidationError("P0 登记单对象记录缺失，登记回滚", status=500)
            ok, note = _verify_object(
                p0_store_root, p0_obj["relpath"], manifest_sha, p0_obj["size_bytes"]
            )
            if not ok:
                raise ValidationError(f"P0 登记单对象未通过核验（{note}），登记回滚", status=409)

            quality_snapshot = _checks_snapshot(conn, version_id)
            snapshot = {"manifest": manifest, "quality_at_registration": quality_snapshot}
            created_at = _now()
            # 4) 同事务写入治理链接（真实 commit 在此发生）。
            conn.execute(
                "INSERT INTO rec_governance_links(job_id, request_id, asset_id, version_id,"
                " manifest_sha256, input_sha256, manifest_relpath, snapshot_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id, request_id, asset_id, version_id, manifest_sha, job["input_sha256"],
                    relpath, json.dumps(snapshot, ensure_ascii=False, sort_keys=True), created_at,
                ),
            )
            conn.commit()
            link = _get_link(conn, job_id)
            return {"created": True, "registration": _link_to_dict(link)}
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()


def _bounded(conn: sqlite3.Connection, sql: str, params, limit: int) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM (" + sql + ")", params).fetchone()[0]
    items = [dict(r) for r in conn.execute(sql + " LIMIT ?", (*params, limit))]
    return {"total": total, "returned": len(items), "truncated": total > limit,
            "limit": limit, "items": items}


def get_lineage(config, job_id: int) -> dict:
    """统一查询：登记单资产/版本/质量/处置、识别输入/模型/运行、候选区域数量、
    人工意见与产物摘要。单一 SQLite 读快照。
    """
    if type(job_id) is not int or not 0 < job_id <= _SQLITE_MAX_INT:
        raise ValidationError("作业编号非法", status=404)

    conn = db.connect(Path(config.db_path))
    try:
        conn.execute("BEGIN")
        try:
            job = conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if job is None:
                raise ValidationError(f"未知作业 {job_id}", status=404)
            run = conn.execute(
                "SELECT * FROM rec_runs WHERE job_id = ? ORDER BY run_seq DESC LIMIT 1", (job_id,)
            ).fetchone()
            region_count = conn.execute(
                "SELECT COUNT(*) AS n FROM rec_regions WHERE job_id = ?", (job_id,)
            ).fetchone()["n"]
            artifacts = [
                {"name": r["name"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]}
                for r in conn.execute(
                    "SELECT name, sha256, size_bytes FROM rec_artifacts WHERE job_id = ? ORDER BY name",
                    (job_id,),
                )
            ]
            reviews = _bounded(
                conn,
                "SELECT * FROM rec_reviews WHERE job_id = ? ORDER BY review_id",
                (job_id,), MAX_LINEAGE_REVIEWS,
            )

            reviews["items"] = [review_records.record_to_dict(r) for r in reviews["items"]]
            link = _get_link(conn, job_id)
            registration = None
            inp = conn.execute("SELECT object_relpath,size_bytes FROM rec_inputs WHERE input_id=?",
                               (job["input_id"],)).fetchone()
            i_ok, i_note = (False, "输入记录缺失") if inp is None else _verify_object(
                Path(config.store_root), inp["object_relpath"], job["input_sha256"], inp["size_bytes"])
            integrity: dict = {"input_object_intact": i_ok, "input_object_note": i_note}
            if link is not None:
                snapshot = json.loads(link["snapshot_json"])
                manifest = snapshot.get("manifest") or {}
                version = conn.execute(
                    "SELECT * FROM asset_versions WHERE version_id = ?", (link["version_id"],)
                ).fetchone()
                asset = conn.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (link["asset_id"],)
                ).fetchone()
                versions_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM asset_versions WHERE asset_id = ?", (link["asset_id"],)
                ).fetchone()["n"]
                events_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM import_events WHERE version_id = ?", (link["version_id"],)
                ).fetchone()["n"]
                latest_sql = ("SELECT * FROM quality_checks WHERE check_id IN "
                              "(SELECT MAX(check_id) FROM quality_checks WHERE version_id=? GROUP BY rule_id)")
                checks_bounded = _bounded(conn, latest_sql + " ORDER BY rule_id",
                                          (link["version_id"],), MAX_LINEAGE_CHECKS)
                for check in checks_bounded["items"]:
                    raw = check.pop("measured_json")
                    check["measured"] = json.loads(raw) if raw else None
                statuses = {r[0] for r in conn.execute("SELECT DISTINCT status FROM (" + latest_sql + ")",
                                                      (link["version_id"],))}
                overall = next((s for s in ("block", "warn", "pass") if s in statuses), "na")
                dispositions = _bounded(
                    conn,
                    "SELECT * FROM dispositions WHERE version_id = ? ORDER BY disp_id",
                    (link["version_id"],), MAX_LINEAGE_DISPOSITIONS,
                )
                history = _bounded(
                    conn,
                    "SELECT * FROM meta_history WHERE version_id = ? ORDER BY history_id",
                    (link["version_id"],), MAX_LINEAGE_HISTORY,
                )
                cur_source = version["source"] if version is not None else None
                cur_scene = version["declared_scene_id"] if version is not None else None
                cur_status = version["status"] if version is not None else None
                current = {
                    "asset": _row_to_dict(asset),
                    "version": {
                        "version_id": version["version_id"],
                        "asset_id": version["asset_id"],
                        "version_no": version["version_no"],
                        "content_type": version["content_type"],
                        "object_id": version["object_id"],
                        "status": version["status"],
                        "source": version["source"],
                        "source_link": version["source_link"],
                        "license_status": version["license_status"],
                        "declared_scene_id": version["declared_scene_id"],
                        "intended_use": version["intended_use"],
                        "updated_at": version["updated_at"],
                    } if version is not None else None,
                    "version_count_for_asset": versions_count,
                    "import_event_count_for_version": events_count,
                    "quality": {
                        "overall": overall,
                        "checks": checks_bounded,
                    },
                    "dispositions": dispositions,
                    "history": history,
                    "diff": {
                        "source_changed": (manifest.get("source") != cur_source),
                        "scene_changed": (manifest.get("scene_id") != cur_scene),
                        "version_status": cur_status,
                        "note": "差异来自登记后的 P0 处置；登记时快照原样保留，不被覆盖。",
                    },
                }
                registration = dict(_link_to_dict(link))
                registration["quality_at_registration"] = snapshot.get("quality_at_registration")
                registration["manifest_at_registration"] = manifest
                registration["current"] = current

                s_ok, s_note = _verify_object(
                    Path(config.store_root), link["manifest_relpath"], link["manifest_sha256"], None
                )
                inp = conn.execute(
                    "SELECT object_relpath, size_bytes FROM rec_inputs WHERE input_id = ?",
                    (job["input_id"],),
                ).fetchone()
                i_ok, i_note = (False, "输入记录缺失") if inp is None else _verify_object(
                    Path(config.store_root), inp["object_relpath"], job["input_sha256"], inp["size_bytes"]
                )
                p0_ok, p0_note = False, "缺少P0对象记录"
                p0_obj = conn.execute(
                    "SELECT relpath, size_bytes FROM objects WHERE object_id = ?", (link["manifest_sha256"],)
                ).fetchone()
                if p0_obj is not None:
                    p0_ok, p0_note = _verify_object(
                        Path(p0_config.storage_root()), p0_obj["relpath"],
                        link["manifest_sha256"], p0_obj["size_bytes"],
                    )
                integrity = {
                    "input_object_intact": i_ok, "input_object_note": i_note,
                    "manifest_service_object_intact": s_ok, "manifest_service_object_note": s_note,
                    "manifest_p0_object_intact": p0_ok, "manifest_p0_object_note": p0_note,
                    "note": (
                        "对象完整性只代表本次查询时的字节观察；质量检查是登记/最近一次复检的"
                        "历史快照，不是每次实时的通过声明。"
                    ),
                }

            return {
                "job_id": job_id,
                "job": {
                    "input_id": job["input_id"],
                    "status": job["status"],
                    "input_sha256": job["input_sha256"],
                    "original_filename": job["original_filename"],
                    "source": job["source"],
                    "source_link": job["source_link"],
                    "license_status": job["license_status"],
                    "scene_id": job["scene_id"],
                    "sample_kind": job["sample_kind"],
                    "model_sha256": job["model_sha256"],
                    "finished_at": job["finished_at"],
                    "region_count": region_count,
                    "fields_endpoint": f"/jobs/{job_id}/fields",
                },
                "run": None if run is None else {
                    "run_id": run["run_id"], "input_sha256": run["input_sha256"],
                    "model_sha256": run["model_sha256"],
                    "run_seq": run["run_seq"], "status": run["status"],
                    "exit_code": run["exit_code"], "adapter": run["adapter"],
                    "elapsed_seconds": run["elapsed_seconds"], "field_count": run["field_count"],
                    "versions": json.loads(run["versions_json"]) if run["versions_json"] else None,
                    "params": json.loads(run["params_json"]) if run["params_json"] else None,
                },
                "artifacts": artifacts,
                "reviews": reviews,
                "registered": link is not None,
                "registration": registration,
                "integrity": integrity,
                "area_unit": "pixel",
                "note": (
                    "registered=false 表示尚未登记治理资料，不代表作业未成功或已完成治理；"
                    "候选连通区域以像素计，不等于真实田块数或平方米；不报告精度。"
                ),
            }
        finally:
            conn.close()
    except BaseException:
        raise
