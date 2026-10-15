"""整改版本与显式复跑后端（DEV-20261015-015）。

在“人工复核”与“治理血缘”之上，增加一条**不可覆盖**的整改链：

    已成功且已登记治理血缘的父作业
        + 该作业一条 questioned/rejected 且仍为最新的人工意见
        → 记录一条 revision_no 递增的整改
        → 用**同模型**生成一个新作业（走原 JobManager.submit / 原 worker）
        → 前后事实对照。

诚实边界（务必遵守）：
- 人工意见只是本机署名声明，不是真值；整改不证明问题自动解决或精度提升。
- 原输入 / 模型 / 作业 / 掩膜 / 意见 / 治理快照一律不改；只在独立 ``rec_corrections``
  里追加一条新记录，并另起一个新作业。
- 修订版本不等于“同一 P0 资产内自动新增版本”：新作业成功后如需治理登记，由既有登记
  接口另建自己的登记单（可有不同 asset_id）。这里只记录事实与关联。
- 只改元数据（来源/场景/许可）或替换 TIFF，二者可同时；必须有至少一个真实规范化字段
  或输入字节摘要变化，仅改文件名不算整改。

事务与恢复（不修改原 JobManager/jobs.py）：
- 采用可恢复两阶段：先用 ``BEGIN IMMEDIATE`` 校验并持久化 ``prepared`` 修订意图并提交，
  再调用原 ``manager.submit``，最后回写 ``child_job_id``。
- **两次提交之间不是原子**：真实表达边界。若提交成功但回写链接失败，记录保持 prepared，
  同一 ``request_id`` 重试只会补齐链接（原 manager 幂等复用同一 job，不重新推理）。
- 子作业请求号从本记录 ``request_id`` 用 SHA256 固定 namespace 派生，同号重试始终相同；
  不使用随机新号重试。显式不同 ``request_id`` 代表新版本，须重新通过当前校验。
- ``GET`` / 启动 / 普通查询不自动提交、不自动重跑。
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from . import db, store, validation
from . import reviews as review_records
from .config import (
    ALLOWED_TIF_SUFFIXES,
    MAX_FILENAME_LEN,
    MAX_LICENSE_LEN,
    MAX_REQUEST_ID_LEN,
    MAX_SCENE_ID_LEN,
    MAX_SOURCE_LEN,
    MAX_SOURCE_LINK_LEN,
    MAX_CONTENT_BYTES,
    FIXED_ARTIFACTS,
    SAMPLE_KIND_SYNTHETIC,
)
from .jobs import IdempotencyConflict
from .validation import ValidationError, decode_base64_strict

# 每条修订链根的容量边界；幂等命中先于容量/当前意见检查。
MAX_CORRECTIONS_PER_ROOT = 50
# 列表固定每页 50；因每 root ≤ 50，每 parent 也 ≤ 50。
LIST_LIMIT = 50

MAX_OPERATOR_LEN = 80
MAX_REASON_LEN = 2000

_SQLITE_MAX_INT = 9223372036854775807
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % MAX_REQUEST_ID_LEN)
_ALLOWED_KEYS = frozenset(
    {"request_id", "review_id", "operator", "reason", "changes", "replacement"}
)
_CHANGE_KEYS = ("source", "scene_id", "source_link", "license_status")
_CHANGE_MAX_LEN = {
    "source": MAX_SOURCE_LEN,
    "scene_id": MAX_SCENE_ID_LEN,
    "source_link": MAX_SOURCE_LINK_LEN,
    "license_status": MAX_LICENSE_LEN,
}
_REPLACEMENT_KEYS = frozenset({"filename", "content_base64"})
_PROBLEM_DECISIONS = ("questioned", "rejected")
_CLASS_KEYS = ("0", "1", "2", "255")

_CHILD_NAMESPACE = "agrolake.correction.child:"

_NODATA_NOTE = "NoData 占比分母为四类（0/1/2/255）像素合计，是像素占比，不是准确率。"

_CORRECTION_NOTE = (
    "整改只是记录：人工意见不是真值，不证明问题自动解决或精度提升，也不代表原 TIFF 通过 P0 质检。"
    "原输入/模型/作业/掩膜/意见/治理快照不改；新作业为独立 revision，不等于在同一 P0 资产内新增版本。"
)

_COMPARISON_NOTE = (
    "对照只表达事实与像素计数变化，不证明精度提升；人工意见不是真值；"
    "两作业均成功才比较像素计数；合计不同表示输入形状可能不同，不能按同像素位置直接比较。"
)


# ---- 请求规范化 --------------------------------------------------------


def _derive_child_request_id(request_id: str) -> str:
    digest = hashlib.sha256((_CHILD_NAMESPACE + request_id).encode("utf-8")).hexdigest()
    return "corr-" + digest


def _normalize_changes(raw) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError("changes 必须是 JSON 对象")
    unknown = sorted(set(raw) - set(_CHANGE_KEYS))
    if unknown:
        raise ValidationError(
            f"changes 不接受字段 {unknown}：只允许 source/scene_id/source_link/license_status"
        )
    out: dict = {}
    for key in _CHANGE_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            if key in ("source", "scene_id"):
                raise ValidationError(f"changes.{key} 不能为空")
            out[key] = None
            continue
        if not isinstance(value, str):
            raise ValidationError(f"changes.{key} 必须是字符串或 null")
        text = value.strip()
        if key in ("source", "scene_id") and not text:
            raise ValidationError(f"changes.{key} 不能为空")
        if len(text) > _CHANGE_MAX_LEN[key]:
            raise ValidationError(
                f"changes.{key} 超过长度上限 {_CHANGE_MAX_LEN[key]}"
            )
        out[key] = text or None
    return out


def _normalize_replacement(raw, config) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidationError("replacement 必须是 JSON 对象")
    if set(raw) != _REPLACEMENT_KEYS:
        raise ValidationError(
            "replacement 必须且只能包含 filename 与 content_base64 两个字段"
        )
    filename = raw.get("filename")
    if not isinstance(filename, str):
        raise ValidationError("replacement.filename 必须是字符串")
    name = store.safe_name(filename)
    if not name.lower().endswith(ALLOWED_TIF_SUFFIXES):
        raise ValidationError("替换文件仅接受 .tif / .tiff")
    if len(name) > MAX_FILENAME_LEN:
        raise ValidationError(f"replacement.filename 超过长度上限 {MAX_FILENAME_LEN}")
    content_base64 = raw.get("content_base64")
    if not isinstance(content_base64, str):
        raise ValidationError("replacement.content_base64 必须是字符串")
    data = decode_base64_strict(content_base64, max_bytes=config.max_content_bytes)
    return {"filename": name, "content": data, "content_sha256": store.sha256_bytes(data)}


def normalize_correction_payload(payload, config) -> dict:
    """校验并规范化整改请求体；此处只做纯校验+替换内容解码，不访问数据库。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise ValidationError(
            f"不接受字段 {unknown}：本接口只接受 "
            "request_id/review_id/operator/reason/changes/replacement"
        )

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
        raise ValidationError(
            f"request_id 必填，且须匹配 ^[A-Za-z0-9._:-]{{1,{MAX_REQUEST_ID_LEN}}}$"
        )

    review_id = payload.get("review_id")
    if type(review_id) is not int or not 0 < review_id <= _SQLITE_MAX_INT:
        raise ValidationError("review_id 必须是 1 至 9223372036854775807 之间的整数")

    operator = payload.get("operator")
    if not isinstance(operator, str):
        raise ValidationError("operator 必须是字符串")
    operator = operator.strip()
    if not operator:
        raise ValidationError("operator 不能为空")
    if len(operator) > MAX_OPERATOR_LEN:
        raise ValidationError(f"operator 超过长度上限 {MAX_OPERATOR_LEN} 字符")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ValidationError("reason 必须是字符串")
    reason = reason.strip()
    if not reason:
        raise ValidationError("reason 不能为空")
    if len(reason) > MAX_REASON_LEN:
        raise ValidationError(f"reason 超过长度上限 {MAX_REASON_LEN} 字符")

    changes = _normalize_changes(payload.get("changes"))
    replacement = _normalize_replacement(payload.get("replacement"), config)
    if not changes and replacement is None:
        raise ValidationError("至少需要 changes 或 replacement 之一才能形成真实整改")

    return {
        "request_id": request_id,
        "review_id": review_id,
        "operator": operator,
        "reason": reason,
        "changes": changes,
        "replacement": replacement,
        "derived_job_request_id": _derive_child_request_id(request_id),
    }


# ---- 事实计算 ----------------------------------------------------------


def _parent_facts(conn: sqlite3.Connection, config, parent_job_id: int, normalized: dict, *, check_environment=True):
    """读取父作业事实、校验可整改性，并计算修订元数据与真实变化。"""
    parent = conn.execute(
        "SELECT * FROM rec_jobs WHERE job_id = ?", (parent_job_id,)
    ).fetchone()
    if parent is None:
        raise ValidationError(f"未知作业 {parent_job_id}", status=404)
    if parent["status"] != "succeeded":
        raise ValidationError(
            f"作业 {parent_job_id} 当前状态为 {parent['status']}，只有成功作业可创建整改",
            status=409,
        )
    link = conn.execute(
        "SELECT asset_id, version_id, manifest_sha256 FROM rec_governance_links WHERE job_id = ?",
        (parent_job_id,),
    ).fetchone()
    if link is None:
        raise ValidationError(
            "父作业尚未登记治理血缘（输入登记单），不能创建整改", status=409
        )
    if check_environment and (
        parent["model_sha256"]
        and config.model_sha256
        and parent["model_sha256"] != config.model_sha256
    ):
        raise ValidationError(
            "父作业模型摘要与当前服务模型摘要不一致，拒绝整改（须同模型）", status=409
        )

    metadata = {
        "source": parent["source"],
        "scene_id": parent["scene_id"],
        "source_link": parent["source_link"],
        "license_status": parent["license_status"],
    }
    for key, value in normalized["changes"].items():
        metadata[key] = value

    replacement = normalized["replacement"]
    if replacement is not None:
        filename = replacement["filename"]
        content_sha256 = replacement["content_sha256"]
        bytes_changed = content_sha256 != parent["input_sha256"]
    else:
        filename = store.safe_name(parent["original_filename"])
        content_sha256 = parent["input_sha256"]
        bytes_changed = False

    changed_fields = [k for k in _CHANGE_KEYS if metadata[k] != parent[k]]
    if not changed_fields and not bytes_changed:
        raise ValidationError(
            "整改必须带来至少一个规范化字段或输入字节摘要变化（仅改文件名不算整改）"
        )

    return parent, link, metadata, filename, content_sha256, changed_fields, bytes_changed


def _fingerprint(parent_job_id, normalized, metadata, filename, content_sha256) -> str:
    """规范内容指纹：覆盖父作业/意见/规范内容/SHA/操作者/原因，用于同号幂等比较。"""
    blob = json.dumps(
        {
            "parent_job_id": parent_job_id,
            "review_id": normalized["review_id"],
            "operator": normalized["operator"],
            "reason": normalized["reason"],
            "changes": normalized["changes"],
            "metadata": metadata,
            "filename": filename,
            "content_sha256": content_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _root_job_id(conn: sqlite3.Connection, job_id: int) -> int:
    row = conn.execute(
        "SELECT root_job_id FROM rec_corrections WHERE child_job_id = ?", (job_id,)
    ).fetchone()
    return row["root_job_id"] if row is not None else job_id


def _load_problem_review(conn: sqlite3.Connection, job_id: int, review_id: int):
    row = conn.execute(
        "SELECT * FROM rec_reviews WHERE review_id = ?", (review_id,)
    ).fetchone()
    if row is None or row["job_id"] != job_id:
        raise ValidationError(f"作业 {job_id} 中不存在复核意见 {review_id}", status=404)
    if row["decision"] not in _PROBLEM_DECISIONS:
        raise ValidationError(
            "整改来源必须是 questioned 或 rejected 的人工意见，accepted 不能作为整改来源",
            status=409,
        )
    return row


def _ensure_latest_review(conn: sqlite3.Connection, review) -> None:
    """该意见在创建时仍须是同一目标（整体或区域）的最新意见。"""
    if review["field_id"] is None:
        row = conn.execute(
            "SELECT MAX(review_id) AS m FROM rec_reviews WHERE job_id = ? AND field_id IS NULL",
            (review["job_id"],),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(review_id) AS m FROM rec_reviews WHERE job_id = ? AND field_id = ?",
            (review["job_id"], review["field_id"]),
        ).fetchone()
    latest = row["m"] if row is not None else None
    if latest is None or review["review_id"] != latest:
        raise ValidationError(
            "该人工意见已被后续意见取代，不能再作为整改来源", status=409
        )


def _resolve_content(conn: sqlite3.Connection, config, parent, normalized: dict) -> bytes:
    """取得子作业输入字节：替换内容，或按服务存储边界读取并核验原输入。"""
    replacement = normalized["replacement"]
    if replacement is not None:
        return replacement["content"]
    inp = conn.execute(
        "SELECT object_relpath, size_bytes FROM rec_inputs WHERE input_id = ?",
        (parent["input_id"],),
    ).fetchone()
    if inp is None:
        raise ValidationError("父作业输入记录缺失，不能整改", status=409)
    try:
        path = store.ensure_within(
            Path(config.store_root), Path(config.store_root) / inp["object_relpath"]
        )
    except store.PathBoundaryError:
        raise ValidationError("父作业输入对象路径越界，拒绝读取", status=409)
    if not path.is_file():
        raise ValidationError("父作业输入对象缺失，不能整改", status=409)
    try:
        if path.stat().st_size != inp["size_bytes"] or path.stat().st_size > config.max_content_bytes:
            raise ValidationError("父作业输入对象大小异常或超过上限，拒绝读取",409)
        with path.open('rb') as stream:
            data = stream.read(config.max_content_bytes+1)
    except OSError:
        raise ValidationError("父作业输入对象读取失败，不能整改", status=409)
    if len(data) != inp["size_bytes"] or store.sha256_bytes(data) != parent["input_sha256"]:
        raise ValidationError(
            "父作业输入对象未通过字节核验（大小/摘要不符），拒绝整改", status=409
        )
    return data


def _build_snapshot(parent, link, review, metadata, filename, content_sha256,
                    size_bytes, normalized, changed_fields, bytes_changed) -> dict:
    return {
        "schema": "AGROLAKE-CORRECTION-1.0.0",
        "parent_facts": {
            "job_id": parent["job_id"],
            "status": parent["status"],
            "input_id": parent["input_id"],
            "input_sha256": parent["input_sha256"],
            "original_filename": parent["original_filename"],
            "model_sha256": parent["model_sha256"],
            "source": parent["source"],
            "scene_id": parent["scene_id"],
            "source_link": parent["source_link"],
            "license_status": parent["license_status"],
            "sample_kind": parent["sample_kind"],
        },
        "parent_registration": None if link is None else {
            "asset_id": link["asset_id"], "version_id": link["version_id"],
            "manifest_sha256": link["manifest_sha256"],
        },
        "problem_review": review_records.record_to_dict(review),
        "revised": {
            "replaced": normalized["replacement"] is not None,
            "input_sha256": content_sha256,
            "filename": filename,
            "size_bytes": size_bytes,
            "source": metadata["source"],
            "scene_id": metadata["scene_id"],
            "source_link": metadata["source_link"],
            "license_status": metadata["license_status"],
            "changed_fields": changed_fields,
            "bytes_changed": bytes_changed,
        },
        "changes": normalized["changes"],
        "note": (
            "修订事实快照：父作业/父治理登记不清除；只记录本次整改意图与服务端事实。"
            "人工意见不是真值。"
        ),
    }


def _record_to_dict(row) -> dict:
    return {
        "correction_id": row["correction_id"],
        "root_job_id": row["root_job_id"],
        "parent_job_id": row["parent_job_id"],
        "review_id": row["review_id"],
        "revision_no": row["revision_no"],
        "child_job_id": row["child_job_id"],
        "request_id": row["request_id"],
        "derived_job_request_id": row["derived_job_request_id"],
        "operator": row["operator"],
        "reason": row["reason"],
        "changes": json.loads(row["changes_json"]),
        "is_synthetic_case": bool(row["is_synthetic_case"]),
        "submission_state": row["submission_state"],
        "prepared_at": row["prepared_at"],
        "submitted_at": row["submitted_at"],
        "error": row["error"],
    }


# ---- 子作业提交与链接 --------------------------------------------------


def _read_stored_object(root, sha256: str, max_bytes=MAX_CONTENT_BYTES) -> bytes:
    path = store.object_path(Path(root), sha256)
    if not path.is_file():
        raise ValidationError("整改内容对象缺失，无法提交子作业", status=409)
    if path.stat().st_size > max_bytes:
        raise ValidationError("整改内容对象超过大小上限，拒绝读取",409)
    with path.open('rb') as stream:
        data = stream.read(max_bytes+1)
    if len(data)>max_bytes or store.sha256_bytes(data) != sha256:
        raise ValidationError("整改内容对象摘要不符，拒绝提交", status=409)
    return data


def _mark_error(config, correction_id: int, message: str) -> None:
    """尽力记录上次提交错误，保持 prepared；失败不掩盖原始异常。"""
    try:
        conn = db.connect(config.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE rec_corrections SET error=?, submission_state='prepared'"
                " WHERE correction_id = ? AND child_job_id IS NULL",
                (str(message)[:2000], correction_id),
            )
            conn.commit()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        pass


def _link_child(db_path, correction_id: int, child_job_id: int) -> int:
    """回写 child_job_id 与链接状态（独立于原 JobManager 的一次提交）。"""
    conn = db.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        record=conn.execute('SELECT * FROM rec_corrections WHERE correction_id=?',(correction_id,)).fetchone()
        child=conn.execute('SELECT * FROM rec_jobs WHERE job_id=?',(child_job_id,)).fetchone()
        if record is None or child is None:
            raise ValidationError('整改或子作业记录缺失，拒绝建立关联',409)
        snapshot=json.loads(record['snapshot_json']);revised=snapshot['revised']
        expected={'input_sha256':revised['input_sha256'],'original_filename':revised['filename'],
                  'source':revised['source'],'scene_id':revised['scene_id'],
                  'source_link':revised['source_link'],'license_status':revised['license_status'],
                  'model_sha256':snapshot['parent_facts']['model_sha256'],
                  'request_id':record['derived_job_request_id']}
        if (type(child_job_id) is not int or child_job_id<=record['parent_job_id']
                or any(child[k]!=value for k,value in expected.items())
                or record['child_job_id'] not in (None,child_job_id)):
            raise ValidationError('子作业与已保存的整改意图不一致，拒绝关联',409)
        cur = conn.execute(
            "UPDATE rec_corrections SET child_job_id=?, submission_state='linked',"
            " submitted_at=?, error=NULL WHERE correction_id = ? AND child_job_id IS NULL",
            (child_job_id, db.now(), correction_id),
        )
        conn.commit()
        return cur.rowcount
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _submit_and_link(manager, config, record, *, created: bool) -> dict:
    """第二阶段：用修订快照重建 payload、提交子作业、回写链接。

    提交失败保留 prepared（可同号重试）；回写链接失败同样保留 prepared，
    重试时原 manager 幂等复用同一 job，不重新推理。
    """
    snapshot = json.loads(record["snapshot_json"])
    revised = snapshot["revised"]
    try:
        conn=db.connect(config.db_path)
        try:
            existing=conn.execute('SELECT job_id FROM rec_requests WHERE request_id=?',
                                  (record['derived_job_request_id'],)).fetchone()
        finally:conn.close()
        if existing is not None:
            result={'job_id':existing['job_id']}
        else:
            if config.model_sha256 != snapshot['parent_facts']['model_sha256']:
                raise ValidationError('当前服务模型与整改快照不一致，保留原意图；请恢复同模型后重试',409)
            content=_read_stored_object(config.store_root,revised['input_sha256'],config.max_content_bytes)
            payload={k:revised[k] for k in ('filename','source','scene_id','source_link','license_status')}
            payload.update(content_base64=base64.b64encode(content).decode('ascii'),
                           request_id=record['derived_job_request_id'])
            normalized_child=validation.validate_job_payload(payload,max_content_bytes=config.max_content_bytes)
            result=manager.submit(normalized_child)
        _link_child(config.db_path,record['correction_id'],result['job_id'])
    except IdempotencyConflict:
        _mark_error(config, record['correction_id'], '子作业请求号冲突，保留整改意图')
        raise ValidationError('子作业请求号冲突，拒绝复用',409) from None
    except ValidationError as exc:
        _mark_error(config, record["correction_id"], exc.message)
        raise ValidationError(
            f"子作业提交失败：{exc.message}；整改记录已保留为 prepared，可用同一请求号重试",
            status=409 if exc.status==409 else 503,
        )
    except (OSError, sqlite3.Error, ValueError):
        _mark_error(config, record["correction_id"], '提交或关联未能确认；请同号重试，已有资料保留')
        raise ValidationError(
            "子作业提交失败，整改记录已保留为 prepared，可用同一请求号重试", status=503
        ) from None

    conn = db.connect(config.db_path)
    try:
        row = conn.execute(
            "SELECT * FROM rec_corrections WHERE correction_id = ?",
            (record["correction_id"],),
        ).fetchone()
    finally:
        conn.close()
    return {"created": created, "record": _record_to_dict(row)}


# ---- 公开接口 ----------------------------------------------------------


def submit_correction(manager, parent_job_id, payload) -> dict:
    """创建一条整改：持久化 prepared，再提交子作业并链接。

    返回 {'created': bool, 'record': {...}}。同号同内容复用（created=False）；
    同号不同内容 409；未知/非成功/未登记父作业分别 404/409/409。
    """
    if type(parent_job_id) is not int or not 0 < parent_job_id <= _SQLITE_MAX_INT:
        raise ValidationError("作业编号非法", status=404)
    config = manager.config
    normalized = normalize_correction_payload(payload, config)

    conn = db.connect(config.db_path)
    try:
        db.init_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                "SELECT * FROM rec_corrections WHERE request_id = ?",
                (normalized["request_id"],),
            ).fetchone()
            (parent, link, metadata, filename, content_sha256,
             changed_fields, bytes_changed) = _parent_facts(
                conn, config, parent_job_id, normalized, check_environment=existing is None
            )
            fingerprint = _fingerprint(
                parent_job_id, normalized, metadata, filename, content_sha256
            )

            if existing is not None:
                # 幂等命中先于容量/当前意见检查。
                if existing["content_fingerprint"] != fingerprint:
                    raise ValidationError(
                        "同一请求号已存在但内容不同，拒绝复用", status=409
                    )
                conn.commit()
                record = existing
                created = False
                needs_submit = not (
                    existing["child_job_id"] is not None
                    and existing["submission_state"] == "linked"
                )
            else:
                root_job_id = _root_job_id(conn, parent_job_id)
                max_rev = conn.execute(
                    "SELECT COALESCE(MAX(revision_no), 0) FROM rec_corrections"
                    " WHERE root_job_id = ?",
                    (root_job_id,),
                ).fetchone()[0]
                revision_no = max_rev + 1
                if revision_no > MAX_CORRECTIONS_PER_ROOT:
                    raise ValidationError(
                        f"整改链根作业 {root_job_id} 已达到 {MAX_CORRECTIONS_PER_ROOT}"
                        " 条修订上限，已有记录保留，可继续查看。",
                        status=409,
                    )
                review = _load_problem_review(conn, parent_job_id, normalized["review_id"])
                _ensure_latest_review(conn, review)

                content = _resolve_content(conn, config, parent, normalized)
                store.store_bytes(Path(config.store_root), content)

                snapshot = _build_snapshot(
                    parent, link, review, metadata, filename, content_sha256,
                    len(content), normalized, changed_fields, bytes_changed,
                )
                is_synthetic = 1 if parent["sample_kind"] == SAMPLE_KIND_SYNTHETIC else 0
                origin=conn.execute('SELECT is_synthetic_case FROM rec_corrections WHERE child_job_id=?',
                                    (parent_job_id,)).fetchone()
                is_synthetic=int(bool(is_synthetic or (origin and origin['is_synthetic_case'])))
                prepared_at = db.now()
                cur = conn.execute(
                    "INSERT INTO rec_corrections(root_job_id, parent_job_id, review_id,"
                    " revision_no, child_job_id, request_id, content_fingerprint,"
                    " derived_job_request_id, operator, reason, snapshot_json, changes_json,"
                    " is_synthetic_case, prepared_at, submission_state, error)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        root_job_id, parent_job_id, review["review_id"], revision_no, None,
                        normalized["request_id"], fingerprint,
                        normalized["derived_job_request_id"], normalized["operator"],
                        normalized["reason"],
                        json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                        json.dumps(normalized["changes"], ensure_ascii=False, sort_keys=True),
                        is_synthetic, prepared_at, "prepared", None,
                    ),
                )
                correction_id = cur.lastrowid
                conn.commit()
                record = conn.execute(
                    "SELECT * FROM rec_corrections WHERE correction_id = ?",
                    (correction_id,),
                ).fetchone()
                created = True
                needs_submit = True
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()

    if not needs_submit:
        return {"created": False, "record": _record_to_dict(record)}
    return _submit_and_link(manager, config, record, created=created)


def list_corrections(manager, parent_job_id) -> dict:
    """按父作业列出整改记录（按 correction_id 递增，全部有界）。"""
    if type(parent_job_id) is not int or not 0 < parent_job_id <= _SQLITE_MAX_INT:
        raise ValidationError("作业编号非法", status=404)
    config = manager.config
    conn = db.connect(config.db_path)
    try:
        conn.execute("BEGIN")
        job = conn.execute(
            "SELECT job_id FROM rec_jobs WHERE job_id = ?", (parent_job_id,)
        ).fetchone()
        if job is None:
            raise ValidationError(f"未知作业 {parent_job_id}", status=404)
        total = conn.execute(
            "SELECT COUNT(*) FROM rec_corrections WHERE parent_job_id = ?",
            (parent_job_id,),
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM rec_corrections WHERE parent_job_id = ?"
            " ORDER BY correction_id ASC LIMIT ?",
            (parent_job_id, LIST_LIMIT + 1),
        ).fetchall()
        truncated = len(rows) > LIST_LIMIT
        corrections = [_record_to_dict(r) for r in rows[:LIST_LIMIT]]
        origin=conn.execute('SELECT * FROM rec_corrections WHERE child_job_id=?',(parent_job_id,)).fetchone()
        return {
            "job_id": parent_job_id,
            "total": total,
            "limit": LIST_LIMIT,
            "truncated": truncated,
            "corrections": corrections,
            "origin":None if origin is None else _record_to_dict(origin),
            "note": (
                "只读列出改正本作业的整改记录；每 root 最多 "
                f"{MAX_CORRECTIONS_PER_ROOT} 条，故每 parent 也 ≤ {LIST_LIMIT}。"
            ),
        }
    finally:
        conn.close()


def _job_summary(conn: sqlite3.Connection, job_id: int) -> dict | None:
    job = conn.execute("SELECT * FROM rec_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if job is None:
        return None
    run = conn.execute(
        "SELECT * FROM rec_runs WHERE job_id = ? ORDER BY run_seq DESC LIMIT 1", (job_id,)
    ).fetchone()
    count_rows = conn.execute(
        "SELECT class_value, pixels FROM rec_class_counts WHERE job_id = ?", (job_id,)
    ).fetchall()
    class_counts = {str(r["class_value"]): r["pixels"] for r in count_rows} or None
    region_count = conn.execute(
        "SELECT COUNT(*) FROM rec_regions WHERE job_id = ?", (job_id,)
    ).fetchone()[0]
    link = conn.execute(
        "SELECT asset_id, version_id, manifest_sha256 FROM rec_governance_links WHERE job_id = ?", (job_id,)
    ).fetchone()
    latest = conn.execute(
        "SELECT * FROM rec_reviews WHERE job_id = ? AND field_id IS NULL"
        " ORDER BY review_id DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    artifacts=[dict(r) for r in conn.execute('SELECT name,sha256,size_bytes FROM rec_artifacts WHERE job_id=? ORDER BY name',
                                           (job_id,))]
    evidence_complete=bool(job['status']=='succeeded' and run is not None and run['status']=='succeeded'
        and run['input_sha256']==job['input_sha256'] and run['model_sha256']==job['model_sha256']
        and len(artifacts)==len(FIXED_ARTIFACTS) and {a['name'] for a in artifacts}==set(FIXED_ARTIFACTS)
        and all(re.fullmatch(r'[0-9a-f]{64}',a['sha256']) and type(a['size_bytes']) is int and a['size_bytes']>=0 for a in artifacts)
        and class_counts is not None and set(class_counts)==set(_CLASS_KEYS)
        and all(type(v) is int and v>=0 for v in class_counts.values()) and sum(class_counts.values())>0)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "input_sha256": job["input_sha256"],
        "model_sha256": job["model_sha256"],
        "original_filename": job["original_filename"],
        "source": job["source"],
        "scene_id": job["scene_id"],
        "source_link": job["source_link"],
        "license_status": job["license_status"],
        "sample_kind": job["sample_kind"],
        "attempt_count": job["attempt_count"],
        "run_seq": run["run_seq"] if run is not None else None,
        "run_status": run["status"] if run is not None else None,
        "class_counts": class_counts,
        "evidence_complete":evidence_complete,
        "artifacts":artifacts,
        "region_count": region_count,
        "governance": None if link is None else {
            "asset_id": link["asset_id"], "version_id": link["version_id"],
            "manifest_sha256":link['manifest_sha256'],
        },
        "latest_task_review": (
            None if latest is None else review_records.record_to_dict(latest)
        ),
    }


def _nodata(summary: dict | None) -> dict | None:
    if not summary or not summary.get('evidence_complete'):
        return None
    counts = summary.get("class_counts")
    if not counts:
        return None
    total = sum(int(counts.get(k, 0)) for k in _CLASS_KEYS)
    if total <= 0:
        return None
    pixels = int(counts.get("255", 0))
    return {"pixels": pixels, "denominator": total, "ratio": round(pixels / total, 6)}


def _build_comparison(parent: dict, child: dict | None, snapshot: dict) -> dict:
    revised = snapshot.get("revised") or {}
    out = {
        "revised_input_sha256": revised.get("input_sha256"),
        "revised_filename": revised.get("filename"),
        "changed_fields": revised.get("changed_fields"),
        "bytes_changed": revised.get("bytes_changed"),
        "parent_problem_review": snapshot.get("problem_review"),
        "parent_registration": snapshot.get("parent_registration"),
    }
    if child is None:
        out.update({
            "child_state": "not_submitted",
            "child_succeeded": False,
            "same_input_sha256": None,
            "same_model_sha256": None,
            "input_sha256": {"parent": parent["input_sha256"], "child": None},
            "model_sha256": {"parent": parent["model_sha256"], "child": None},
            "class_pixel_comparison": None,
            "nodata": {"parent": _nodata(parent), "child": None, "note": _NODATA_NOTE},
            "child_latest_task_review": None,
            "note": "子作业尚未提交或链接；不填 0 冒充完成。" + _COMPARISON_NOTE,
        })
        return out

    child_succeeded = child["status"] == "succeeded"
    out.update({
        "child_state": child["status"],
        "child_succeeded": child_succeeded,
        "same_input_sha256": child["input_sha256"] == parent["input_sha256"],
        "same_model_sha256": child["model_sha256"] == parent["model_sha256"],
        "input_sha256": {"parent": parent["input_sha256"], "child": child["input_sha256"]},
        "model_sha256": {"parent": parent["model_sha256"], "child": child["model_sha256"]},
        "child_latest_task_review": child["latest_task_review"],
        "nodata": {"parent": _nodata(parent), "child": _nodata(child), "note": _NODATA_NOTE},
    })
    if child_succeeded and parent['evidence_complete'] and child['evidence_complete']:
        p = {k: int(parent["class_counts"].get(k, 0)) for k in _CLASS_KEYS}
        c = {k: int(child["class_counts"].get(k, 0)) for k in _CLASS_KEYS}
        p_total = sum(p.values())
        c_total = sum(c.values())
        out["class_pixel_comparison"] = {
            "comparable_pixel_space": parent['input_sha256']==child['input_sha256'],
            "same_total_pixels":p_total==c_total,
            "parent": {**p, "total": p_total},
            "child": {**c, "total": c_total},
            "delta": {k: c[k] - p[k] for k in _CLASS_KEYS},
            "note": "delta=child-parent，仅像素计数差值；合计相同不能证明形状或空间对应。不同输入未核实配准，不作逐像素或精度比较。",
        }
    else:
        out["class_pixel_comparison"] = None
    out["note"] = _COMPARISON_NOTE
    return out


def get_correction(manager, correction_id) -> dict:
    """只读返回整改记录、父/子事实摘要与对照（同一 SQLite 读快照）。"""
    if type(correction_id) is not int or not 0 < correction_id <= _SQLITE_MAX_INT:
        raise ValidationError("整改编号非法", status=404)
    config = manager.config
    conn = db.connect(config.db_path)
    try:
        conn.execute("BEGIN")
        rec = conn.execute(
            "SELECT * FROM rec_corrections WHERE correction_id = ?", (correction_id,)
        ).fetchone()
        if rec is None:
            raise ValidationError(f"未知整改记录 {correction_id}", status=404)
        parent = _job_summary(conn, rec["parent_job_id"])
        child = (
            _job_summary(conn, rec["child_job_id"])
            if rec["child_job_id"] is not None else None
        )
        snapshot = json.loads(rec["snapshot_json"])
        comparison = _build_comparison(parent, child, snapshot)
        return {
            "correction": _record_to_dict(rec),
            "snapshot":snapshot,
            "parent": parent,
            "child": child,
            "comparison": comparison,
            "note": (
                _CORRECTION_NOTE
                + " 对照为只读快照；不返回其他任务意见或本地绝对路径。"
            ),
        }
    finally:
        conn.close()


def resume_correction(manager, correction_id, payload) -> dict:
    """显式确认已持久化的同一整改意图；重载后无需重传输入或生成新号。"""
    if type(correction_id) is not int or not 0<correction_id<=_SQLITE_MAX_INT:
        raise ValidationError('整改编号非法',404)
    if (not isinstance(payload,dict) or set(payload)!={'request_id'}
            or not isinstance(payload['request_id'],str) or not _REQUEST_ID_RE.fullmatch(payload['request_id'])):
        raise ValidationError('恢复请求仅接受原request_id')
    conn=db.connect(manager.config.db_path)
    try:record=conn.execute('SELECT * FROM rec_corrections WHERE correction_id=?',(correction_id,)).fetchone()
    finally:conn.close()
    if record is None:raise ValidationError('未知整改记录',404)
    if record['request_id']!=payload['request_id']:raise ValidationError('请求号与原整改记录不一致',409)
    if record['child_job_id'] is not None and record['submission_state']=='linked':
        return {'created':False,'record':_record_to_dict(record)}
    return _submit_and_link(manager,manager.config,record,created=False)
