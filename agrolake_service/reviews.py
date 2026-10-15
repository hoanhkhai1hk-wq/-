"""识别结果的人工复核记录（DEV-20261015-013）。

只做三件事：校验请求、把“人工意见”以**追加**方式写入独立 ``rec_reviews`` 表、
按作业分页读取当前意见与历史。规则边界：

- 人工意见不改掩膜/类别数/模型输入输出，也不声称真值、精度提升或真实面积。
- 只有成功作业可写；未知作业/区域 404，非成功作业 409。
- 同一 ``request_id`` 只对应一条记录：内容/任务/区域一致则复用，不同则 409。
- 每次更正都是新记录，历史永不覆盖；删除/改写不提供 API。
- 快照（input_sha256 / model_sha256 / run_seq）与 created_at 一律由服务端从该作业取得，
  客户端不得指定。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from . import db
from .validation import ValidationError

# 固定结论枚举；中文仅用于展示。
DECISIONS = ("accepted", "questioned", "rejected")
DECISION_ZH = {
    "accepted": "可用",
    "questioned": "存疑待核实",
    "rejected": "不采用",
}

MAX_REVIEWER_LEN = 80
MAX_NOTE_LEN = 2000
MAX_REQUEST_ID_LEN = 128
DEFAULT_PAGE_LIMIT = 200
MAX_PAGE_LIMIT = 200
# 本地原型的明确容量边界；总历史及当前意见一起返回时仍低于客户端8MiB上限。
MAX_REVIEWS_PER_JOB = 500
MAX_REVIEW_BODY_BYTES = 16 * 1024

_SQLITE_MAX_INT = 9223372036854775807
_REVIEW_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % MAX_REQUEST_ID_LEN)
_ALLOWED_KEYS = frozenset({"field_id", "decision", "reviewer", "note", "request_id"})


def normalize_review_payload(payload) -> dict:
    """校验并规范化复核请求体，返回 {field_id, decision, reviewer, note, request_id}。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")

    unknown = sorted(set(payload) - _ALLOWED_KEYS)
    if unknown:
        raise ValidationError(
            f"不接受字段 {unknown}：本接口只接受 field_id/decision/reviewer/note/request_id"
        )

    # field_id：缺省或 null 表示任务整体；否则必须是正整数（bool 不算整数）。
    raw_field = payload.get("field_id", None)
    if raw_field is None:
        field_id = None
    elif type(raw_field) is not int:
        raise ValidationError("field_id 必须是正整数或 null")
    elif not 0 < raw_field <= _SQLITE_MAX_INT:
        raise ValidationError("field_id 必须是 1 至 9223372036854775807 之间的正整数")
    else:
        field_id = raw_field

    decision = payload.get("decision")
    if not isinstance(decision, str) or decision not in DECISIONS:
        raise ValidationError(
            "decision 必须是 accepted / questioned / rejected 之一"
        )

    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str):
        raise ValidationError("reviewer 必须是字符串")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValidationError("reviewer 不能为空")
    if len(reviewer) > MAX_REVIEWER_LEN:
        raise ValidationError(f"reviewer 超过长度上限 {MAX_REVIEWER_LEN} 字符")

    note = payload.get("note")
    if not isinstance(note, str):
        raise ValidationError("note 必须是字符串")
    note = note.strip()
    if not note:
        raise ValidationError("note 不能为空")
    if len(note) > MAX_NOTE_LEN:
        raise ValidationError(f"note 超过长度上限 {MAX_NOTE_LEN} 字符")

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not _REVIEW_REQUEST_ID_RE.fullmatch(request_id):
        raise ValidationError(
            f"request_id 必填，且须匹配 ^[A-Za-z0-9._:-]{{1,{MAX_REQUEST_ID_LEN}}}$"
        )

    return {
        "field_id": field_id,
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "request_id": request_id,
    }


def fingerprint(job_id: int, normalized: dict) -> str:
    """规范化内容指纹：覆盖任务、区域、结论、署名与原因；用于同号幂等比较。"""
    blob = json.dumps(
        {
            "job_id": job_id,
            "field_id": normalized["field_id"],
            "decision": normalized["decision"],
            "reviewer": normalized["reviewer"],
            "note": normalized["note"],
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def record_to_dict(row: sqlite3.Row) -> dict:
    """把一行复核记录转成面向接口/网页的字典（含中文结论与目标说明）。"""
    field_id = row["field_id"]
    decision = row["decision"]
    return {
        "review_id": row["review_id"],
        "job_id": row["job_id"],
        "field_id": field_id,
        "target": "region" if field_id is not None else "task",
        "decision": decision,
        "decision_zh": DECISION_ZH.get(decision, decision),
        "reviewer": row["reviewer"],
        "note": row["note"],
        "request_id": row["request_id"],
        "input_sha256": row["input_sha256"],
        "model_sha256": row["model_sha256"],
        "run_seq": row["run_seq"],
        "created_at": row["created_at"],
    }


def _record_from_request(conn: sqlite3.Connection, request_id: str):
    return conn.execute(
        "SELECT * FROM rec_reviews WHERE request_id = ?", (request_id,)
    ).fetchone()


def submit_review(db_path, job_id: int, payload) -> dict:
    """在**一个事务内**校验并追加一条复核记录。

    返回 {'created': bool, 'record': {...}}。同号同内容复用既有记录（created=False），
    同号不同内容/任务/区域抛 status=409 的 ValidationError。并发同号只落一行。
    """
    normalized = normalize_review_payload(payload)
    fp = fingerprint(job_id, normalized)
    request_id = normalized["request_id"]

    conn = db.connect(db_path)
    try:
        # IMMEDIATE：先取写锁再读，保证“幂等检查 + 追加”整体串行、不产生重复行。
        conn.execute("BEGIN IMMEDIATE")
        try:
            job = conn.execute(
                "SELECT status, input_sha256, model_sha256 FROM rec_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if job is None:
                raise ValidationError(f"未知作业 {job_id}", status=404)
            if job["status"] != "succeeded":
                raise ValidationError(
                    f"作业 {job_id} 当前状态为 {job['status']}，只有成功作业可记录人工复核",
                    status=409,
                )

            field_id = normalized["field_id"]
            if field_id is not None:
                region = conn.execute(
                    "SELECT 1 FROM rec_regions WHERE job_id = ? AND field_id = ? LIMIT 1",
                    (job_id, field_id),
                ).fetchone()
                if region is None:
                    raise ValidationError(
                        f"作业 {job_id} 中不存在候选区域 {field_id}", status=404
                    )

            existing = _record_from_request(conn, request_id)
            if existing is not None:
                if existing["content_fingerprint"] != fp:
                    raise ValidationError(
                        "同一请求号已存在但内容、任务或区域不同，拒绝复用", status=409
                    )
                conn.commit()
                return {"created": False, "record": record_to_dict(existing)}

            count = conn.execute(
                'SELECT COUNT(*) FROM rec_reviews WHERE job_id = ?', (job_id,)
            ).fetchone()[0]
            if count >= MAX_REVIEWS_PER_JOB:
                raise ValidationError(
                    f'此任务已达到{MAX_REVIEWS_PER_JOB}条复核历史上限，已有记录已保留，可继续查看和导出。',
                    status=409,
                )

            run = conn.execute(
                "SELECT run_seq FROM rec_runs WHERE job_id = ? ORDER BY run_seq DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            run_seq = run["run_seq"] if run is not None else None
            created_at = db.now()
            cur = conn.execute(
                "INSERT INTO rec_reviews(job_id, field_id, decision, reviewer, note,"
                " request_id, content_fingerprint, input_sha256, model_sha256, run_seq,"
                " created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, field_id, normalized["decision"], normalized["reviewer"],
                    normalized["note"], request_id, fp, job["input_sha256"],
                    job["model_sha256"], run_seq, created_at,
                ),
            )
            review_id = cur.lastrowid
            row = conn.execute(
                "SELECT * FROM rec_reviews WHERE review_id = ?", (review_id,)
            ).fetchone()
            conn.commit()
            return {"created": True, "record": record_to_dict(row)}
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.close()


def _coerce_bound(raw, name: str, *, default: int, minimum: int, maximum: int) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValidationError(f"{name} 非法")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if (not text.isascii() or not text.isdigit() or len(text) > 19):
            raise ValidationError(f"{name} 非法")
        value = int(text)
    else:
        raise ValidationError(f"{name} 非法")
    if not minimum <= value <= maximum:
        raise ValidationError(f"{name} 超出允许范围 {minimum}..{maximum}")
    return value


def list_reviews(db_path, job_id: int, *, after_id=None, limit=None, until_id=None) -> dict:
    """按作业分页读取复核记录，并给出当前意见（latest）。

    - ``reviews``：按 review_id 升序的**一页**记录（分页遍历可拿到全部）。
    - ``latest``：跨全部分页计算的目标最新意见：``task`` 为任务整体，``regions`` 为
      按 field_id 的最新意见；均只代表“人工当前意见”，不是真值。
    - ``pagination``：明确返回 has_more / next_after_id，避免“看似完整实则被截断”。
    """
    after = _coerce_bound(after_id, "after_id", default=0, minimum=0,
                          maximum=_SQLITE_MAX_INT)
    page = _coerce_bound(limit, "limit", default=DEFAULT_PAGE_LIMIT, minimum=1,
                         maximum=MAX_PAGE_LIMIT)

    conn = db.connect(db_path)
    try:
        conn.execute('BEGIN')  # 计数、分页、当前意见均来自同一读快照。
        job = conn.execute(
            "SELECT status FROM rec_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if job is None:
            raise ValidationError(f"未知作业 {job_id}", status=404)

        maximum = conn.execute(
            'SELECT COALESCE(MAX(review_id), 0) FROM rec_reviews WHERE job_id = ?', (job_id,)
        ).fetchone()[0]
        until = _coerce_bound(until_id, 'until_id', default=maximum, minimum=0,
                              maximum=_SQLITE_MAX_INT)
        # 人工编辑/旧版本异常大库不无界加载；原数据保留，明确失败。
        stored_total = conn.execute('SELECT COUNT(*) FROM rec_reviews WHERE job_id=?',
                                    (job_id,)).fetchone()[0]
        if stored_total > MAX_REVIEWS_PER_JOB:
            raise ValidationError('复核历史超出本版本容量，原记录保留，请先由维护者检查。', status=409)

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM rec_reviews WHERE job_id = ? AND review_id <= ?", (job_id, until)
        ).fetchone()["n"]

        sql = "SELECT * FROM rec_reviews WHERE job_id = ? AND review_id <= ?"
        params: list = [job_id, until]
        if after:
            sql += " AND review_id > ?"
            params.append(after)
        sql += " ORDER BY review_id ASC LIMIT ?"
        params.append(page + 1)
        rows = conn.execute(sql, params).fetchall()
        has_more = len(rows) > page
        rows = rows[:page]
        reviews = [record_to_dict(r) for r in rows]

        latest_task = None
        latest_regions: dict = {}
        for r in conn.execute(
            "SELECT * FROM rec_reviews WHERE job_id = ? AND review_id <= ? ORDER BY review_id ASC", (job_id, until)
        ):
            item = record_to_dict(r)
            if r["field_id"] is None:
                latest_task = item
            else:
                latest_regions[str(r["field_id"])] = item

        next_after_id = rows[-1]["review_id"] if (rows and has_more) else None
        return {
            "job_id": job_id,
            "reviews": reviews,
            "latest": {
                "structure": (
                    "task = 任务整体最新意见；regions = 按候选区域 field_id 的最新意见；"
                    "均为该目标 review_id 最大的一条，仅代表人工当前意见，不是真值。"
                ),
                "task": latest_task,
                "regions": latest_regions,
            },
            "pagination": {
                "after_id": after,
                "until_id": until,
                "limit": page,
                "returned": len(reviews),
                "has_more": has_more,
                "next_after_id": next_after_id,
            },
            "total": total,
            "capacity": MAX_REVIEWS_PER_JOB,
            "note": (
                "人工复核只是本机填写的署名声明；不改掩膜/类别数/模型输入输出，"
                "不代表真值、精度提升或真实面积。历史按 review_id 递增追加，不可覆盖。"
            ),
        }
    finally:
        conn.close()
