"""本机识别服务的纯标准库 HTTP 客户端（DEV-20261015-012）。

只做三件事：拼本机地址、发有界请求、把错误翻译成面向零基础用户的中文分类。
不导入 torch，不导入 agrolake_service 的运行模块，不写数据库，不新增依赖。

安全与边界：
- 服务地址只允许本机 http（127.0.0.1 / localhost / ::1），拒绝远程主机、
  用户名/密码、路径、查询串与片段；端口必须合法。
- 每个请求都有有限超时；响应正文按上限读取，超限立即失败，避免内存被拖垮。
- HTTP/网络/JSON 错误不吞掉，转成带类别的中文错误；下载产物会核对服务记录的
  摘要，发现被改动或过期时明确报错，绝不静默交付。
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
from urllib.parse import urlsplit

DEFAULT_API_URL = "http://127.0.0.1:8506"
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}

# 单次请求默认超时（秒）：本机调用，短超时即可；提交大图的超时由调用方放宽。
DEFAULT_TIMEOUT = 10.0
SUBMIT_TIMEOUT = 60.0

# JSON 响应（作业列表/详情）与二进制产物的读取上限（字节），两者都有界。
DEFAULT_MAX_JSON_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

# 产物名与识别服务保持一致（成功作业的五个固定产物）。
FIXED_ARTIFACTS = (
    "mask.tif",
    "preview.png",
    "class_pixels.csv",
    "field_pixels.csv",
    "run.json",
)

# 人工复核结论枚举与字段上限（与 agrolake_service.reviews 保持一致）。
REVIEW_DECISIONS = ("accepted", "questioned", "rejected")
MAX_REVIEWER_LEN = 80
MAX_NOTE_LEN = 2000
MAX_REVIEW_PAGE = 200

# 整改复跑（与 agrolake_service.corrections 保持一致）。
CORRECTION_STATES = ("prepared", "linked")
_JOB_STATES = ("queued", "running", "succeeded", "failed", "interrupted")
_CLASS_KEYS = ("0", "1", "2", "255")
_OPERATOR_MAX_LEN = 80
_REASON_MAX_LEN = 2000
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
# 整改来源意见只允许问题类结论；变更白名单与后端 corrections._CHANGE_KEYS 一致。
_PROBLEM_DECISIONS = ("questioned", "rejected")
_CHANGE_FIELDS = ("source", "scene_id", "source_link", "license_status")
_SYNTHETIC_KIND = "synthetic_test"


class ApiError(Exception):
    """一次识别服务调用失败，带明确类别与中文说明。

    category 取值：
      config    地址/配置错误（本地即能判断，不发请求）
      offline   连接不上（服务可能未启动或地址错误）
      timeout   请求超时，结果未知（可能是网络慢，也可能是服务仍在处理）
      http      服务返回 4xx/5xx 错误状态
      conflict  服务返回 409（例如产物已变化，拒绝交付过期结果）
      badjson   响应不是合法 JSON
      protocol  响应缺少预期字段（接口约定不符）
      toolarge  响应正文超过本地读取上限
      checksum  下载产物摘要与作业记录不符（可能被改动）
      unexpected 其它未预期错误
    """

    def __init__(self, category: str, message: str, *, status: int | None = None,
                 detail: str | None = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.status = status
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return self.message


class ApiConfigError(ApiError):
    """本地配置错误（如地址非法），不发任何网络请求。"""

    def __init__(self, message: str):
        super().__init__("config", message)


def resolve_base_url(raw: str | None = None) -> str:
    """校验并规范化服务地址，只接受本机 http。

    默认取环境变量 ``AGROLAKE_API_URL``，否则 ``http://127.0.0.1:8506``。
    """
    value = raw if raw is not None else os.environ.get("AGROLAKE_API_URL")
    value = (value or DEFAULT_API_URL).strip()
    if not value:
        raise ApiConfigError("识别服务地址为空")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ApiConfigError(f"识别服务地址或端口非法：{exc}") from None

    if parsed.scheme != 'http':
        raise ApiConfigError('识别服务地址须为本机 http:// 地址')
    host = (parsed.hostname or "").lower()
    if host not in LOCAL_HOSTNAMES:
        raise ApiConfigError("只允许连接本机识别服务（127.0.0.1 / localhost / ::1）")
    if parsed.username is not None or parsed.password is not None:
        raise ApiConfigError("识别服务地址不允许包含用户名或密码")
    if parsed.path not in ("", "/"):
        raise ApiConfigError("识别服务地址不允许包含路径")
    if parsed.query or parsed.fragment:
        raise ApiConfigError("识别服务地址不允许包含查询串或片段")
    if port is not None and not (0 < port <= 65535):
        raise ApiConfigError("识别服务端口须在 1 至 65535 之间")

    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return f"{parsed.scheme}://{netloc}"


class RecognitionClient:
    """调用本机识别服务的客户端。每次调用新建连接并关闭，避免跨线程复用。"""

    def __init__(self, base_url: str | None = None, *, timeout: float = DEFAULT_TIMEOUT,
                 max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
                 max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES):
        self.base_url = resolve_base_url(base_url)
        parsed = urlsplit(self.base_url)
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.timeout = float(timeout)
        self.max_json_bytes = int(max_json_bytes)
        self.max_artifact_bytes = int(max_artifact_bytes)
        if not math.isfinite(self.timeout) or self.timeout<=0:
            raise ApiConfigError('请求超时须为大于零的有限秒数')
        if not 0<self.max_json_bytes<=DEFAULT_MAX_JSON_BYTES or not 0<self.max_artifact_bytes<=DEFAULT_MAX_ARTIFACT_BYTES:
            raise ApiConfigError('响应大小上限须为正且不超过系统上限')

    # ---- 公共方法 ---------------------------------------------------------

    def health(self) -> dict:
        result=self._request_json('GET','/health')
        if result.get('status')!='ok':
            raise ApiError('protocol','识别服务健康状态响应不符合约定。')
        return result

    def list_jobs(self) -> list:
        payload = self._request_json("GET", "/jobs")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise ApiError("protocol", "作业列表响应缺少 jobs 字段")
        for job in jobs:
            self._validate_job(job)
        return jobs

    def get_job(self, job_id: int) -> dict:
        result=self._request_json('GET',f'/jobs/{self._job_id(job_id)}')
        self._validate_job(result)
        if result['job_id']!=job_id:
            raise ApiError('protocol','识别服务返回了其他任务的记录，已停止展示。')
        return result

    def get_fields(self, job_id: int) -> dict:
        result=self._request_json('GET',f'/jobs/{self._job_id(job_id)}/fields')
        rows=result.get('regions')
        if result.get('job_id')!=job_id or result.get('area_unit')!='pixel' or not isinstance(rows,list):
            raise ApiError('protocol','候选区域响应的任务编号或像素单位不符合约定。')
        required=('field_id','pixels','min_row','min_col','max_row','max_col')
        if result.get('region_count')!=len(rows) or any(not isinstance(r,dict) or
            any(type(r.get(k)) is not int or r[k]<0 for k in required) for r in rows):
            raise ApiError('protocol','候选区域记录不完整，请刷新后重试。')
        return result

    def submit_job(self, payload: dict, *, timeout: float | None = None) -> dict:
        """POST /jobs。payload 需已包含内联 content_base64 等字段。"""
        if not isinstance(payload, dict):
            raise ApiError("protocol", "提交内容必须是字典")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        result = self._request_json("POST", "/jobs", body=body, headers=headers,
                                    timeout=timeout if timeout is not None else SUBMIT_TIMEOUT)
        job_id = result.get("job_id")
        if type(job_id) is not int or not 0<job_id<=9223372036854775807:
            raise ApiError("protocol", "提交响应缺少 job_id")
        return result

    def get_artifact(self, job_id: int, name: str,
                     *, expected_sha256: str | None = None) -> bytes:
        """下载固定产物；仅成功作业可取，发现摘要不符或服务返回 409 时明确报错。"""
        if name not in FIXED_ARTIFACTS:
            raise ApiError("protocol", f"不支持的产物名：{name}")
        data = self._request_bytes(
            "GET", f"/jobs/{self._job_id(job_id)}/artifacts/{name}",
            max_bytes=self.max_artifact_bytes, timeout=self.timeout,
        )
        if expected_sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_sha256:
                raise ApiError(
                    "checksum",
                    f"下载产物 {name} 的摘要与作业记录不符，可能已被改动，"
                    "已拒绝使用该内容",
                    detail=f"expected={expected_sha256} actual={actual}",
                )
        return data

    # ---- 人工复核 ---------------------------------------------------------

    def submit_review(self, job_id: int, *, decision: str, reviewer: str, note: str,
                      request_id: str, field_id: int | None = None,
                      timeout: float | None = None) -> dict:
        """POST /jobs/<id>/reviews。本地先做与接口一致的基本校验，再提交。"""
        job_id = self._job_id(job_id)
        if decision not in REVIEW_DECISIONS:
            raise ApiError("protocol", f"复核结论非法：{decision!r}")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ApiError("protocol", "复核人不能为空")
        if len(reviewer.strip()) > MAX_REVIEWER_LEN:
            raise ApiError("protocol", f"复核人超过 {MAX_REVIEWER_LEN} 字符")
        if not isinstance(note, str) or not note.strip():
            raise ApiError("protocol", "复核原因不能为空")
        if len(note.strip()) > MAX_NOTE_LEN:
            raise ApiError("protocol", f"复核原因超过 {MAX_NOTE_LEN} 字符")
        if not isinstance(request_id, str) or not request_id:
            raise ApiError("protocol", "缺少请求号，无法保证幂等。")
        if field_id is not None and (type(field_id) is not int or field_id <= 0):
            raise ApiError("protocol", f"区域编号非法：{field_id!r}")

        payload = {
            "decision": decision,
            "reviewer": reviewer.strip(),
            "note": note.strip(),
            "request_id": request_id,
        }
        if field_id is not None:
            payload["field_id"] = field_id
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        result = self._request_json(
            "POST", f"/jobs/{job_id}/reviews", body=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=timeout if timeout is not None else self.timeout,
        )
        record = self._validate_review(result)
        expected = {'job_id':job_id, 'field_id':field_id, 'decision':decision,
                    'reviewer':reviewer.strip(), 'note':note.strip(), 'request_id':request_id}
        if any(record.get(k) != v for k, v in expected.items()) or type(result.get('created')) is not bool:
            raise ApiError('protocol', '复核回执与本次填写不一致，请使用同一请求号重试确认。')
        return result

    def list_reviews(self, job_id: int, *, after_id: int | None = None,
                     limit: int | None = None, until_id: int | None = None) -> dict:
        """GET /jobs/<id>/reviews（可带 after_id/limit 分页）。"""
        job_id = self._job_id(job_id)
        query = []
        if after_id is not None:
            if type(after_id) is not int or after_id < 0:
                raise ApiError("protocol", f"after_id 非法：{after_id!r}")
            query.append(f"after_id={after_id}")
        if limit is not None:
            if type(limit) is not int or not 0 < limit <= MAX_REVIEW_PAGE:
                raise ApiError("protocol", f"limit 非法：{limit!r}")
            query.append(f"limit={limit}")
        if until_id is not None:
            if type(until_id) is not int or not 0 <= until_id <= 9223372036854775807:
                raise ApiError('protocol', '复核快照编号非法。')
            query.append(f'until_id={until_id}')
        path = f"/jobs/{job_id}/reviews" + (("?" + "&".join(query)) if query else "")
        result = self._request_json("GET", path)
        if result.get("job_id") != job_id:
            raise ApiError("protocol", "识别服务返回了其他任务的复核记录，已停止展示。")
        reviews = result.get("reviews")
        pagination = result.get("pagination")
        if not isinstance(reviews, list) or not isinstance(pagination, dict):
            raise ApiError("protocol", "复核记录响应结构不符合约定。")
        previous = after_id or 0
        for row in reviews:
            self._validate_review(row)
            if row['job_id'] != job_id or row['review_id'] <= previous:
                raise ApiError('protocol', '复核历史的任务或记录顺序不符合约定。')
            previous = row['review_id']
        bound = pagination.get('until_id')
        total = result.get('total')
        has_more = pagination.get('has_more')
        next_id = pagination.get('next_after_id')
        if (type(bound) is not int or not 0 <= bound <= 9223372036854775807
                or (until_id is not None and bound != until_id)
                or any(r['review_id'] > bound for r in reviews)
                or type(total) is not int or total < len(reviews)
                or type(has_more) is not bool
                or pagination.get('after_id') != (after_id or 0)
                or pagination.get('limit') != (limit or MAX_REVIEW_PAGE)
                or pagination.get('returned') != len(reviews)
                or len(reviews) > (limit or MAX_REVIEW_PAGE)
                or (has_more and (not reviews or type(next_id) is not int or next_id != previous))
                or (not has_more and next_id is not None)):
            raise ApiError('protocol', '复核分页响应不完整，不能确认已取全。')
        latest = result.get('latest')
        if not isinstance(latest, dict) or not isinstance(latest.get('regions'), dict):
            raise ApiError('protocol', '复核当前意见响应结构不符合约定。')
        task = latest.get('task')
        current = []
        if task is not None:
            self._validate_review(task)
            if task['field_id'] is not None:
                raise ApiError('protocol', '任务整体意见混入了区域意见。')
            current.append(task)
        for key, row in latest['regions'].items():
            self._validate_review(row)
            if row['field_id'] is None or key != str(row['field_id']):
                raise ApiError('protocol', '复核区域意见与区域编号不一致。')
            current.append(row)
        if any(r['job_id'] != job_id or r['review_id'] > bound for r in current):
            raise ApiError('protocol', '当前意见不属于本任务的复核快照。')
        return result

    # ---- 治理登记与血缘（DEV-20261015-014）--------------------------------

    def register_governance(self, job_id: int, *, request_id: str,
                            timeout: float | None = None) -> dict:
        """POST /jobs/<id>/governance：为成功作业登记识别输入资料（固定请求号幂等）。"""
        job_id = self._job_id(job_id)
        if not isinstance(request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9._:-]{1,128}", request_id
        ):
            raise ApiError("protocol", "缺少合法请求号，无法保证幂等。")
        body = json.dumps({"request_id": request_id}, ensure_ascii=False).encode("utf-8")
        result = self._request_json(
            "POST", f"/jobs/{job_id}/governance", body=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=timeout if timeout is not None else self.timeout,
        )
        record = self._validate_registration(result)
        if (type(result.get("created")) is not bool or record["job_id"] != job_id
                or record["request_id"] != request_id):
            raise ApiError("protocol", "登记回执与本次请求不一致，请使用同一请求号重试确认。")
        return result

    def get_lineage(self, job_id: int) -> dict:
        """GET /jobs/<id>/lineage：读取治理登记与识别血缘（未登记也可读）。"""
        job_id = self._job_id(job_id)
        result = self._request_json("GET", f"/jobs/{job_id}/lineage")
        if result.get("job_id") != job_id:
            raise ApiError("protocol", "识别服务返回了其他任务的血缘记录，已停止展示。")
        registered = result.get("registered")
        if type(registered) is not bool:
            raise ApiError("protocol", "血缘响应缺少登记状态。")
        reviews = result.get("reviews")
        artifacts = result.get("artifacts")
        integrity = result.get("integrity")
        if not isinstance(reviews, dict) or not isinstance(artifacts, list) \
                or not isinstance(integrity, dict):
            raise ApiError("protocol", "血缘响应结构不符合约定。")
        if any(type(reviews.get(k)) is not int for k in ("total", "returned", "limit")) \
                or type(reviews.get("truncated")) is not bool:
            raise ApiError("protocol", "血缘复核分页结构不符合约定。")
        items = reviews.get("items")
        if not isinstance(items, list) or reviews["returned"] != len(items) \
                or reviews["total"] < reviews["returned"] \
                or reviews["returned"] > reviews["limit"]:
            raise ApiError("protocol", "血缘复核分页计数不符合约定。")
        for row in items:
            self._validate_review(row)
            if row["job_id"] != job_id:
                raise ApiError("protocol", "血缘复核记录混入了其他任务。")
        for art in artifacts:
            if (not isinstance(art, dict) or not isinstance(art.get("name"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(art.get("sha256", "")))
                    or type(art.get("size_bytes")) is not int or art["size_bytes"] < 0):
                raise ApiError("protocol", "血缘产物摘要结构不符合约定。")
        if registered:
            reg = result.get("registration")
            if not isinstance(reg, dict):
                raise ApiError("protocol", "血缘登记内容缺失。")
            self._validate_registration(reg)
            if (reg.get("job_id") != job_id
                    or not isinstance(reg.get("quality_at_registration"), dict)
                    or not isinstance(reg.get("current"), dict)
                    or not isinstance((reg.get("current") or {}).get("quality"), dict)):
                raise ApiError("protocol", "血缘登记内容不完整。")
        elif result.get("registration") is not None:
            raise ApiError("protocol", "未登记状态却包含登记内容。")
        self._validate_lineage_facts(result, job_id)
        return result

    @staticmethod
    def _validate_lineage_page(page) -> None:
        if not isinstance(page, dict) or not isinstance(page.get('items'), list):
            raise ApiError('protocol', '血缘明细结构不完整。')
        if (any(type(page.get(k)) is not int or page[k] < 0 for k in ('total','returned','limit'))
                or not 1 <= page['limit'] <= 500
                or type(page.get('truncated')) is not bool
                or page['returned'] != len(page['items'])
                or page['returned'] != min(page['total'],page['limit'])
                or page['truncated'] != (page['total'] > page['returned'])):
            raise ApiError('protocol', '血缘明细计数或截断标记不一致。')

    @staticmethod
    def _validate_lineage_facts(result, job_id) -> None:
        def require(ok):
            if not ok:
                raise ApiError('protocol', '血缘对象、来源快照或关联编号不一致，已停止展示。')
        sha = lambda x: isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) is not None
        positive = lambda x: type(x) is int and 0 < x <= 9223372036854775807
        job=result.get('job'); run=result.get('run')
        require(isinstance(job,dict))
        require(positive(job.get('input_id')) and sha(job.get('input_sha256'))
                and job.get('status') in ('queued','running','succeeded','failed','interrupted')
                and type(job.get('region_count')) is int and job['region_count'] >= 0
                and result.get('area_unit') == 'pixel')
        require(isinstance(result.get('integrity'),dict)
                and type(result['integrity'].get('input_object_intact')) is bool)
        if job['status']=='succeeded':
            require(isinstance(run,dict) and positive(run.get('run_seq')) and sha(job.get('model_sha256')))
            require(run.get('status')=='succeeded' and run.get('input_sha256')==job['input_sha256']
                    and run.get('model_sha256')==job['model_sha256'])
            names=[a['name'] for a in result['artifacts']]
            require(len(names)==len(FIXED_ARTIFACTS) and set(names)==set(FIXED_ARTIFACTS))
        RecognitionClient._validate_lineage_page(result['reviews'])
        for row in result['reviews']['items']:
            require(row['input_sha256']==job['input_sha256']
                    and row['model_sha256']==job['model_sha256'] and isinstance(run,dict)
                    and row['run_seq']==run['run_seq'])
        if not result['registered']:
            return
        reg=result['registration'];cur=reg['current'];ver=cur.get('version');asset=cur.get('asset')
        require(reg['input_sha256']==job['input_sha256'] and isinstance(ver,dict) and isinstance(asset,dict))
        require(ver.get('version_id')==reg['version_id'] and ver.get('asset_id')==reg['asset_id']
                and asset.get('asset_id')==reg['asset_id'] and ver.get('object_id')==reg['manifest_sha256']
                and ver.get('content_type')=='text')
        snap=reg.get('manifest_at_registration')
        require(isinstance(snap,dict) and snap.get('job_id')==job_id
                and snap.get('input_sha256')==job['input_sha256']
                and snap.get('model_sha256')==job['model_sha256'])
        for key in ('manifest_service_object_intact','manifest_p0_object_intact'):
            require(type(result['integrity'].get(key)) is bool)
        quality=cur['quality']
        require(quality.get('overall') in ('pass','warn','block','na'))
        for page in (quality.get('checks'),cur.get('dispositions'),cur.get('history')):
            RecognitionClient._validate_lineage_page(page)
            for row in page['items']:
                require(isinstance(row,dict) and row.get('version_id')==reg['version_id'])
        snap_quality=reg['quality_at_registration']
        require(isinstance(snap_quality.get('checks'),list))
        for row in snap_quality['checks']:
            require(isinstance(row,dict) and isinstance(row.get('rule_id'),str)
                    and row.get('status') in ('pass','warn','block','na') and isinstance(row.get('reason'),str))

    @staticmethod
    def _validate_registration(record) -> dict:
        if not isinstance(record, dict):
            raise ApiError("protocol", "登记回执不是对象。")
        if (type(record.get("link_id")) is not int or record["link_id"] <= 0
                or type(record.get("job_id")) is not int
                or not 0 < record["job_id"] <= 9223372036854775807
                or type(record.get("asset_id")) is not int or record["asset_id"] <= 0
                or type(record.get("version_id")) is not int or record["version_id"] <= 0
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(record.get(k, "")))
                       for k in ("manifest_sha256", "input_sha256"))
                or record.get("manifest_sha256") == record.get("input_sha256")
                or not isinstance(record.get("request_id"), str) or not record["request_id"]
                or not isinstance(record.get("created_at"), str) or not record["created_at"]):
            raise ApiError("protocol", "登记回执缺少完整字段或摘要不合法。")
        return record

    @staticmethod
    def _validate_review(record) -> dict:
        if not isinstance(record, dict):
            raise ApiError("protocol", "复核记录不是对象，已停止展示。")
        review_id = record.get("review_id")
        if type(review_id) is not int or review_id <= 0:
            raise ApiError("protocol", "复核记录缺少合法 review_id。")
        if type(record.get("job_id")) is not int or not 0 < record['job_id'] <= 9223372036854775807:
            raise ApiError("protocol", "复核记录缺少合法 job_id。")
        if record.get("decision") not in REVIEW_DECISIONS:
            raise ApiError("protocol", "复核记录的结论不符合约定。")
        field_id = record.get("field_id")
        if field_id is not None and (type(field_id) is not int or field_id <= 0):
            raise ApiError("protocol", "复核记录的区域编号不符合约定。")
        if (record.get('target') != ('task' if field_id is None else 'region')
                or any(not isinstance(record.get(k), str) or not record[k].strip() for k in
                       ('reviewer', 'note', 'request_id', 'created_at'))
                or type(record.get('run_seq')) is not int or record['run_seq'] <= 0
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(record.get(k,'')))
                       for k in ('input_sha256', 'model_sha256'))):
            raise ApiError('protocol', '复核记录缺少完整内容或识别来源快照。')
        return record

    # ---- 整改复跑与前后对照（DEV-20261015-016）----------------------------

    def submit_correction(self, parent_job_id: int, *, request_id: str, review_id: int,
                          operator: str, reason: str, changes: dict | None = None,
                          replacement: dict | None = None,
                          timeout: float | None = None) -> dict:
        """POST /jobs/<parent>/corrections：显式创建一条不可覆盖的整改并复跑。

        仅创建回一条整改记录（记录里带子作业编号）；“已提交”不等于“已识别成功”。
        """
        parent_job_id = self._job_id(parent_job_id)
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise ApiError("protocol", "缺少合法请求号，无法保证幂等。")
        if type(review_id) is not int or not 0 < review_id <= 9223372036854775807:
            raise ApiError("protocol", "整改来源复核编号非法。")
        if not isinstance(operator, str) or not operator.strip():
            raise ApiError("protocol", "操作者署名不能为空。")
        if len(operator.strip()) > _OPERATOR_MAX_LEN:
            raise ApiError("protocol", f"操作者署名超过 {_OPERATOR_MAX_LEN} 字符")
        if not isinstance(reason, str) or not reason.strip():
            raise ApiError("protocol", "整改原因不能为空。")
        if len(reason.strip()) > _REASON_MAX_LEN:
            raise ApiError("protocol", f"整改原因超过 {_REASON_MAX_LEN} 字符")

        payload: dict = {
            "request_id": request_id, "review_id": review_id,
            "operator": operator.strip(), "reason": reason.strip(),
        }
        sent_changes = self._normalize_correction_changes(changes)
        if sent_changes:
            payload["changes"] = sent_changes
        if replacement is not None:
            if (not isinstance(replacement, dict)
                    or set(replacement) != {"filename", "content_base64"}):
                raise ApiError("protocol", "替换文件必须且只能包含 filename 与 content_base64。")
            payload["replacement"] = {
                "filename": replacement["filename"], "content_base64": replacement["content_base64"],
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        result = self._request_json(
            "POST", f"/jobs/{parent_job_id}/corrections", body=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=timeout if timeout is not None else self.timeout,
        )
        record = self._validate_correction_record(result)
        if (type(result.get("created")) is not bool
                or record["parent_job_id"] != parent_job_id
                or record["review_id"] != review_id
                or record["request_id"] != request_id
                or record["operator"] != operator.strip()
                or record["reason"] != reason.strip()
                or record["changes"] != sent_changes):
            raise ApiError("protocol", "整改回执与本次填写不一致，请使用同一请求号重试确认。")
        self._validate_correction_links(record)
        return result

    def list_corrections(self, parent_job_id: int) -> dict:
        """GET /jobs/<id>/corrections：只读列出改正本作业的整改与 origin。"""
        parent_job_id = self._job_id(parent_job_id)
        result = self._request_json("GET", f"/jobs/{parent_job_id}/corrections")
        if result.get("job_id") != parent_job_id:
            raise ApiError("protocol", "识别服务返回了其他任务的整改列表，已停止展示。")
        corrections = result.get("corrections")
        total = result.get("total")
        limit = result.get("limit")
        truncated = result.get("truncated")
        if (not isinstance(corrections, list) or type(total) is not int or total < 0
                or type(limit) is not int or not 0 < limit <= 500
                or type(truncated) is not bool
                or len(corrections) > limit or total < len(corrections)
                or truncated != (total > len(corrections))):
            raise ApiError("protocol", "整改列表结构不完整，不能确认已取全。")
        previous = 0
        for row in corrections:
            self._validate_correction_record(row)
            self._validate_correction_links(row)
            if row["parent_job_id"] != parent_job_id or row["correction_id"] <= previous:
                raise ApiError("protocol", "整改列表的任务或记录顺序不符合约定。")
            previous = row["correction_id"]
        origin = result.get("origin")
        if origin is not None:
            self._validate_correction_record(origin)
            self._validate_correction_links(origin)
            if origin["child_job_id"] != parent_job_id:
                raise ApiError("protocol", "整改来源记录与当前作业不一致。")
        if result.get("note") is not None and not isinstance(result.get("note"), str):
            raise ApiError("protocol", "整改列表说明字段不符合约定。")
        return result

    def get_correction(self, correction_id: int) -> dict:
        """GET /corrections/<id>：只读返回整改记录、父/子摘要与事实对照。"""
        correction_id = self._job_id(correction_id)
        result = self._request_json("GET", f"/corrections/{correction_id}")
        record = self._validate_correction_record(result.get("correction"))
        if record["correction_id"] != correction_id:
            raise ApiError("protocol", "整改详情返回了其他记录，已停止展示。")
        self._validate_correction_links(record)
        snapshot = self._validate_snapshot(result.get("snapshot"), record)
        parent = self._validate_job_summary(result.get("parent"),
                                            expect_job_id=record["parent_job_id"])
        child = result.get("child")
        if record["child_job_id"] is None:
            if child is not None:
                raise ApiError("protocol", "整改尚未链接子作业却返回了子摘要。")
            child = None
        else:
            child = self._validate_job_summary(child, expect_job_id=record["child_job_id"])
        self._validate_snapshot_facts(snapshot, record, parent, child)
        self._validate_comparison(result.get("comparison"), record, parent, child, snapshot)
        return result

    def resume_correction(self, correction_id: int, *, request_id: str,
                          timeout: float | None = None) -> dict:
        """POST /corrections/<id>/resume：显式确认已保存的同一整改意图（同号）。"""
        correction_id = self._job_id(correction_id)
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise ApiError("protocol", "缺少合法请求号，无法确认原提交。")
        body = json.dumps({"request_id": request_id}, ensure_ascii=False).encode("utf-8")
        result = self._request_json(
            "POST", f"/corrections/{correction_id}/resume", body=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=timeout if timeout is not None else self.timeout,
        )
        record = self._validate_correction_record(result)
        if (result.get("created") is not False
                or record["correction_id"] != correction_id
                or record["request_id"] != request_id):
            raise ApiError("protocol", "整改确认回执与本次请求不一致，请保留原请求号重试。")
        self._validate_correction_links(record)
        return result

    @staticmethod
    def _normalize_correction_changes(changes) -> dict:
        """本地先按后端同一规则规范化 changes：白名单、类型与空白→None。"""
        if changes is None:
            return {}
        if not isinstance(changes, dict):
            raise ApiError("protocol", "changes 必须是字典。")
        unknown = sorted(set(changes) - set(_CHANGE_FIELDS))
        if unknown:
            raise ApiError("protocol", f"changes 不接受字段 {unknown}。")
        out: dict = {}
        for key in _CHANGE_FIELDS:
            if key not in changes:
                continue
            value = changes[key]
            if value is None:
                if key in ("source", "scene_id"):
                    raise ApiError("protocol", f"changes.{key} 不能为空。")
                out[key] = None
                continue
            if not isinstance(value, str):
                raise ApiError("protocol", f"changes.{key} 必须是字符串或 null。")
            text = value.strip()
            if key in ("source", "scene_id") and not text:
                raise ApiError("protocol", f"changes.{key} 不能为空。")
            out[key] = text or None
        return out

    @staticmethod
    def _validate_correction_record(record) -> dict:
        if not isinstance(record, dict):
            raise ApiError("protocol", "整改记录不是对象，已停止展示。")

        def positive(value) -> bool:
            return type(value) is int and 0 < value <= 9223372036854775807

        if (not positive(record.get("correction_id"))
                or not positive(record.get("root_job_id"))
                or not positive(record.get("parent_job_id"))
                or not positive(record.get("review_id"))
                or not positive(record.get("revision_no"))
                or (record.get("child_job_id") is not None and not positive(record.get("child_job_id")))
                or not isinstance(record.get("request_id"), str) or not record["request_id"]
                or not isinstance(record.get("derived_job_request_id"), str)
                or not record["derived_job_request_id"]
                or not isinstance(record.get("operator"), str) or not record["operator"].strip()
                or not isinstance(record.get("reason"), str) or not record["reason"].strip()
                or not isinstance(record.get("changes"), dict)
                or type(record.get("is_synthetic_case")) is not bool
                or record.get("submission_state") not in CORRECTION_STATES
                or not isinstance(record.get("prepared_at"), str) or not record["prepared_at"]
                or (record.get("submitted_at") is not None
                    and not isinstance(record["submitted_at"], str))
                or (record.get("error") is not None and not isinstance(record["error"], str))):
            raise ApiError("protocol", "整改记录缺少完整字段或类型不符合约定。")
        return record

    @staticmethod
    def _validate_correction_links(record) -> None:
        parent = record["parent_job_id"]
        root = record["root_job_id"]
        child = record["child_job_id"]
        if not 0 < root <= parent:
            raise ApiError("protocol", "整改的根/父作业关系不符合约定。")
        if child is not None and child <= parent:
            raise ApiError("protocol", "整改的子作业编号不符合约定。")
        if record["submission_state"] == "linked" and child is None:
            raise ApiError("protocol", "整改为已链接状态却缺少子作业编号。")
        if record["submission_state"] == "prepared" and child is not None:
            raise ApiError("protocol", "整改仍为待提交状态却已带子作业编号。")

    @staticmethod
    def _evidence_complete(summary) -> bool:
        """按事实重算证据完整性：成功运行 + 四类非负整数正合计 + 固定 5 产物唯一。"""
        if summary.get("status") != "succeeded":
            return False
        if summary.get("run_seq") is None or summary.get("run_status") != "succeeded":
            return False
        counts = summary.get("class_counts")
        if not isinstance(counts, dict) or set(counts) != set(_CLASS_KEYS):
            return False
        if any(type(v) is not int or v < 0 for v in counts.values()):
            return False
        if sum(counts[k] for k in _CLASS_KEYS) <= 0:
            return False
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(FIXED_ARTIFACTS):
            return False
        names = [a.get("name") for a in artifacts if isinstance(a, dict)]
        if len(names) != len(artifacts) or set(names) != set(FIXED_ARTIFACTS):
            return False
        return True

    @staticmethod
    def _compute_nodata(summary) -> dict | None:
        """NoData 占比：pixels=255 类、denominator=四类和、ratio=round(pixels/denom,6)。"""
        if not isinstance(summary, dict) or not summary.get("evidence_complete"):
            return None
        counts = summary.get("class_counts")
        if not isinstance(counts, dict) or set(counts) != set(_CLASS_KEYS):
            return None
        if any(type(v) is not int or v < 0 for v in counts.values()):
            return None
        total = sum(counts[k] for k in _CLASS_KEYS)
        if total <= 0:
            return None
        pixels = counts["255"]
        return {"pixels": pixels, "denominator": total, "ratio": round(pixels / total, 6)}

    @classmethod
    def _validate_job_summary(cls, summary, *, expect_job_id: int) -> dict:
        if not isinstance(summary, dict):
            raise ApiError("protocol", "整改对照的任务摘要不是对象。")
        # 已知必需字段必须存在（键缺失属于协议错误，与显式 null 区分）。
        if any(key not in summary for key in (
                "job_id", "status", "input_sha256", "model_sha256", "original_filename",
                "source", "scene_id", "source_link", "license_status", "sample_kind",
                "attempt_count", "run_seq", "run_status", "class_counts", "evidence_complete",
                "artifacts", "region_count", "governance", "latest_task_review")):
            raise ApiError("protocol", "整改对照的任务摘要缺少必需字段。")
        # SHA 摘要只接受字符串再匹配十六进制，不做 str 强转（避免把整数当摘要）。
        hex64 = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None
        if (type(summary.get("job_id")) is not int or summary["job_id"] != expect_job_id
                or summary.get("status") not in _JOB_STATES
                or not hex64(summary.get("input_sha256"))
                or not (summary.get("model_sha256") is None or hex64(summary.get("model_sha256")))
                or not isinstance(summary.get("original_filename"), str) or not summary["original_filename"]
                or not isinstance(summary.get("source"), str)
                or not isinstance(summary.get("scene_id"), str)
                or (summary.get("source_link") is not None and not isinstance(summary["source_link"], str))
                or (summary.get("license_status") is not None and not isinstance(summary["license_status"], str))
                or not isinstance(summary.get("sample_kind"), str)
                or type(summary.get("attempt_count")) is not int or summary["attempt_count"] < 0
                or (summary.get("run_seq") is not None
                    and (type(summary["run_seq"]) is not int or summary["run_seq"] <= 0))
                or (summary.get("run_status") is not None and summary["run_status"] not in _JOB_STATES)
                or type(summary.get("region_count")) is not int or summary["region_count"] < 0
                or type(summary.get("evidence_complete")) is not bool):
            raise ApiError("protocol", "整改对照的任务摘要不完整。")
        counts = summary.get("class_counts")
        if counts is not None and (not isinstance(counts, dict)
                                   or any(k not in _CLASS_KEYS for k in counts)
                                   or any(type(v) is not int or v < 0 for v in counts.values())):
            raise ApiError("protocol", "任务摘要的类别像素计数不符合约定。")
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, list):
            raise ApiError("protocol", "任务摘要的产物列表不符合约定。")
        names = []
        for art in artifacts:
            if (not isinstance(art, dict) or not isinstance(art.get("name"), str)
                    or not hex64(art.get("sha256"))
                    or type(art.get("size_bytes")) is not int or art["size_bytes"] < 0):
                raise ApiError("protocol", "任务摘要的产物摘要不符合约定。")
            names.append(art["name"])
        if len(set(names)) != len(names):
            raise ApiError("protocol", "任务摘要的产物名重复。")
        gov = summary.get("governance")
        if gov is not None and (not isinstance(gov, dict)
                                or type(gov.get("asset_id")) is not int or gov["asset_id"] <= 0
                                or type(gov.get("version_id")) is not int or gov["version_id"] <= 0
                                or not hex64(gov.get("manifest_sha256"))):
            raise ApiError("protocol", "任务摘要的治理登记编号不符合约定。")
        latest = summary.get("latest_task_review")
        if latest is not None:
            RecognitionClient._validate_review(latest)
            if (latest["job_id"] != expect_job_id or latest["field_id"] is not None
                    or latest["input_sha256"] != summary["input_sha256"]
                    or latest["model_sha256"] != summary["model_sha256"]
                    or latest["run_seq"] != summary.get("run_seq")):
                raise ApiError("protocol", "任务摘要的最新整体意见不属于该作业的运行。")
        if summary["evidence_complete"] != cls._evidence_complete(summary):
            raise ApiError("protocol", "任务摘要的证据完整性标记与运行/产物/计数不一致。")
        return summary

    @staticmethod
    def _validate_snapshot(snapshot, record) -> dict:
        if not isinstance(snapshot, dict):
            raise ApiError("protocol", "整改快照不是对象。")
        parent_facts = snapshot.get("parent_facts")
        revised = snapshot.get("revised")
        problem = snapshot.get("problem_review")
        registration = snapshot.get("parent_registration")
        changes = snapshot.get("changes")
        # SHA 摘要只接受字符串再匹配十六进制，不做 str 强转（避免把整数当摘要）。
        hex64 = lambda x: isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None
        if (not isinstance(parent_facts, dict) or not isinstance(revised, dict)
                or not isinstance(problem, dict) or not isinstance(changes, dict)):
            raise ApiError("protocol", "整改快照结构不完整。")
        # 已知必需字段必须存在（键缺失属于协议错误，与显式 null 区分）。
        if any(key not in parent_facts for key in (
                "job_id", "status", "input_sha256", "model_sha256", "original_filename",
                "source", "scene_id", "source_link", "license_status", "sample_kind")):
            raise ApiError("protocol", "整改快照的父作业事实缺少必需字段。")
        if any(key not in revised for key in (
                "replaced", "input_sha256", "filename", "size_bytes", "source", "scene_id",
                "source_link", "license_status", "changed_fields", "bytes_changed")):
            raise ApiError("protocol", "整改快照的修订输入摘要缺少必需字段。")
        if (type(parent_facts.get("job_id")) is not int
                or parent_facts["job_id"] != record["parent_job_id"]
                or parent_facts.get("status") not in _JOB_STATES
                or not hex64(parent_facts.get("input_sha256"))
                or not hex64(parent_facts.get("model_sha256"))
                or not isinstance(parent_facts.get("original_filename"), str)
                or not parent_facts["original_filename"]
                or not isinstance(parent_facts.get("source"), str)
                or not isinstance(parent_facts.get("scene_id"), str)
                or (parent_facts.get("source_link") is not None
                    and not isinstance(parent_facts["source_link"], str))
                or (parent_facts.get("license_status") is not None
                    and not isinstance(parent_facts["license_status"], str))
                or not isinstance(parent_facts.get("sample_kind"), str)):
            raise ApiError("protocol", "整改快照的父作业事实与记录不一致。")
        if registration is not None and (not isinstance(registration, dict)
                or type(registration.get("asset_id")) is not int or registration["asset_id"] <= 0
                or type(registration.get("version_id")) is not int or registration["version_id"] <= 0
                or not hex64(registration.get("manifest_sha256"))):
            raise ApiError("protocol", "整改快照的父治理登记不符合约定。")
        RecognitionClient._validate_review(problem)
        if (problem["job_id"] != record["parent_job_id"]
                or problem["review_id"] != record["review_id"]
                or problem["decision"] not in _PROBLEM_DECISIONS):
            raise ApiError("protocol", "整改快照的问题意见与记录不一致。")
        if (not hex64(revised.get("input_sha256")) or not isinstance(revised.get("filename"), str)
                or not revised["filename"]
                or type(revised.get("size_bytes")) is not int or revised["size_bytes"] < 0
                or not isinstance(revised.get("source"), str)
                or not isinstance(revised.get("scene_id"), str)
                or (revised.get("source_link") is not None and not isinstance(revised["source_link"], str))
                or (revised.get("license_status") is not None
                    and not isinstance(revised["license_status"], str))
                or type(revised.get("replaced")) is not bool
                or type(revised.get("bytes_changed")) is not bool
                or not isinstance(revised.get("changed_fields"), list)
                or any(not isinstance(f, str) or f not in _CHANGE_FIELDS
                       for f in revised["changed_fields"])):
            raise ApiError("protocol", "整改快照的修订输入摘要不完整。")
        if (any(k not in _CHANGE_FIELDS for k in changes)
                or any(not (v is None or isinstance(v, str)) for v in changes.values())):
            raise ApiError("protocol", "整改快照的变更字段不符合约定。")
        # 快照内部一致性：changes 与整改记录一致；修订元数据=父事实叠加 changes；标记为实际差异。
        if changes != record["changes"]:
            raise ApiError("protocol", "整改快照的变更与整改记录不一致。")
        base = {k: parent_facts.get(k) for k in _CHANGE_FIELDS}
        expected_meta = dict(base)
        expected_meta.update(changes)
        if any(revised.get(k) != expected_meta[k] for k in _CHANGE_FIELDS):
            raise ApiError("protocol", "整改快照的修订元数据与父事实/变更不一致。")
        expected_changed = [k for k in _CHANGE_FIELDS if expected_meta[k] != base[k]]
        if revised["changed_fields"] != expected_changed:
            raise ApiError("protocol", "整改快照的变更字段与实际差异不一致。")
        if revised["bytes_changed"] != (revised["input_sha256"] != parent_facts["input_sha256"]):
            raise ApiError("protocol", "整改快照的字节变更标记与实际摘要不一致。")
        if not revised["replaced"] and (revised["input_sha256"] != parent_facts["input_sha256"]
                                        or revised["filename"] != parent_facts["original_filename"]):
            raise ApiError("protocol", "整改快照未替换字节却改动了输入。")
        return snapshot

    @classmethod
    def _validate_snapshot_facts(cls, snapshot, record, parent, child) -> None:
        """把快照的父/修订事实与父/子摘要逐项对照，不只校验 JSON 类型。"""
        pf = snapshot["parent_facts"]
        revised = snapshot["revised"]
        problem = snapshot["problem_review"]
        for key in ("job_id", "status", "input_sha256", "model_sha256", "original_filename",
                    "source", "scene_id", "source_link", "license_status", "sample_kind"):
            if pf.get(key) != parent.get(key):
                raise ApiError("protocol", "整改快照的父作业事实与父摘要不一致。")
        if snapshot.get("parent_registration") != parent.get("governance"):
            raise ApiError("protocol", "整改快照的父治理登记与父摘要不一致。")
        if (problem["job_id"] != parent["job_id"]
                or problem["input_sha256"] != parent["input_sha256"]
                or problem["model_sha256"] != parent["model_sha256"]
                or problem["run_seq"] != parent.get("run_seq")):
            raise ApiError("protocol", "整改快照的问题意见与父作业运行不一致。")
        if parent.get("sample_kind") == _SYNTHETIC_KIND and not record["is_synthetic_case"]:
            raise ApiError("protocol", "父作业为合成测试样本，整改记录必须保留人工背景标记。")
        if child is not None:
            if (revised["input_sha256"] != child["input_sha256"]
                    or revised["filename"] != child["original_filename"]
                    or revised["source"] != child["source"]
                    or revised["scene_id"] != child["scene_id"]
                    or revised["source_link"] != child["source_link"]
                    or revised["license_status"] != child["license_status"]):
                raise ApiError("protocol", "整改快照的修订事实与子作业摘要不一致。")
            if child["model_sha256"] != pf["model_sha256"]:
                raise ApiError("protocol", "整改子作业未使用与父作业相同的模型。")

    @staticmethod
    def _validate_nodata(block, expected) -> None:
        if expected is None:
            if block is not None:
                raise ApiError("protocol", "整改证据不齐却给出 NoData 占比。")
            return
        if not isinstance(block, dict) or set(block) != {"pixels", "denominator", "ratio"}:
            raise ApiError("protocol", "整改对照的 NoData 结构不符合约定。")
        if type(block["pixels"]) is not int or type(block["denominator"]) is not int:
            raise ApiError("protocol", "整改对照的 NoData 计数不是整数。")
        ratio = block["ratio"]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            raise ApiError("protocol", "整改对照的 NoData 占比不是有限数值。")
        if block != expected:
            raise ApiError("protocol", "整改对照的 NoData 分母或占比与像素计数不符。")

    @classmethod
    def _validate_pixel_block(cls, pixel, parent, child) -> None:
        if (not isinstance(pixel, dict)
                or type(pixel.get("comparable_pixel_space")) is not bool
                or type(pixel.get("same_total_pixels")) is not bool):
            raise ApiError("protocol", "整改对照的像素比较标记不符合约定。")
        delta = pixel.get("delta")
        if (not isinstance(delta, dict) or set(delta) != set(_CLASS_KEYS)
                or any(type(delta.get(k)) is not int for k in _CLASS_KEYS)):
            raise ApiError("protocol", "整改对照的像素差值不完整。")
        totals: dict = {}
        for side, summary in (("parent", parent), ("child", child)):
            block = pixel.get(side)
            if (not isinstance(block, dict) or set(block) != set(_CLASS_KEYS) | {"total"}
                    or any(type(block.get(k)) is not int or block[k] < 0 for k in _CLASS_KEYS)
                    or type(block.get("total")) is not int):
                raise ApiError("protocol", "整改对照的像素计数不完整。")
            counts = summary.get("class_counts")
            if (not isinstance(counts, dict) or set(counts) != set(_CLASS_KEYS)
                    or any(block.get(k) != counts.get(k) for k in _CLASS_KEYS)):
                raise ApiError("protocol", "整改对照的像素计数与任务摘要不一致。")
            if block["total"] != sum(block[k] for k in _CLASS_KEYS):
                raise ApiError("protocol", "整改对照的像素合计与分类计数不一致。")
            totals[side] = block["total"]
        if (any(delta[k] != pixel["child"][k] - pixel["parent"][k] for k in _CLASS_KEYS)
                or pixel.get("same_total_pixels") != (totals["parent"] == totals["child"])
                or pixel.get("comparable_pixel_space")
                != (parent["input_sha256"] == child["input_sha256"])):
            raise ApiError("protocol", "整改对照的同总量/可比空间标记与实际不符。")

    @classmethod
    def _validate_comparison(cls, comparison, record, parent, child, snapshot) -> None:
        if not isinstance(comparison, dict):
            raise ApiError("protocol", "整改对照不是对象。")
        revised = snapshot["revised"]
        state = comparison.get("child_state")
        if state not in _JOB_STATES + ("not_submitted",) \
                or type(comparison.get("child_succeeded")) is not bool:
            raise ApiError("protocol", "整改对照的子任务状态不符合约定。")
        if (type(comparison.get("same_input_sha256")) not in (bool, type(None))
                or type(comparison.get("same_model_sha256")) not in (bool, type(None))):
            raise ApiError("protocol", "整改对照的一致性标记不符合约定。")
        # 复制字段必须与快照逐项一致。
        if (comparison.get("revised_input_sha256") != revised["input_sha256"]
                or comparison.get("revised_filename") != revised["filename"]
                or comparison.get("changed_fields") != revised["changed_fields"]
                or comparison.get("bytes_changed") != revised["bytes_changed"]
                or comparison.get("parent_problem_review") != snapshot["problem_review"]
                or comparison.get("parent_registration") != snapshot.get("parent_registration")):
            raise ApiError("protocol", "整改对照与快照不一致。")
        problem = comparison.get("parent_problem_review")
        if (not isinstance(problem, dict) or problem.get("review_id") != record["review_id"]
                or problem.get("job_id") != record["parent_job_id"]):
            raise ApiError("protocol", "整改对照的问题意见与记录不一致。")
        # 输入/模型映射与同异标记按事实计算。
        expected_input = {"parent": parent["input_sha256"],
                          "child": None if child is None else child["input_sha256"]}
        expected_model = {"parent": parent["model_sha256"],
                          "child": None if child is None else child["model_sha256"]}
        if comparison.get("input_sha256") != expected_input \
                or comparison.get("model_sha256") != expected_model:
            raise ApiError("protocol", "整改对照的输入/模型摘要映射不符合事实。")
        if child is None:
            if comparison.get("same_input_sha256") is not None \
                    or comparison.get("same_model_sha256") is not None:
                raise ApiError("protocol", "整改对照无子作业却给出同异标记。")
        elif (comparison.get("same_input_sha256") != (child["input_sha256"] == parent["input_sha256"])
              or comparison.get("same_model_sha256") != (child["model_sha256"] == parent["model_sha256"])):
            raise ApiError("protocol", "整改对照的同异标记与事实不一致。")
        if comparison.get("child_latest_task_review") != (None if child is None else child["latest_task_review"]):
            raise ApiError("protocol", "整改对照的子最新意见与子作业不一致。")
        # NoData：pixels=255、denominator=四类和、ratio=round(pixels/denom,6)；缺证据必须为 null。
        nodata = comparison.get("nodata")
        if not isinstance(nodata, dict) or "parent" not in nodata or "child" not in nodata:
            raise ApiError("protocol", "整改对照缺少 NoData 分母说明。")
        cls._validate_nodata(nodata.get("parent"), cls._compute_nodata(parent))
        cls._validate_nodata(nodata.get("child"),
                             None if child is None else cls._compute_nodata(child))
        # 像素对照：仅两次成功且证据齐全才有值，且与父子摘要逐项一致。
        pixel = comparison.get("class_pixel_comparison")
        expected_present = bool(child is not None and child.get("status") == "succeeded"
                                and parent.get("evidence_complete") and child.get("evidence_complete"))
        if pixel is None:
            if expected_present:
                raise ApiError("protocol", "整改证据齐全却缺少像素对照。")
        else:
            if not expected_present:
                raise ApiError("protocol", "整改证据不齐却给出像素对照。")
            cls._validate_pixel_block(pixel, parent, child)
        if child is None:
            if state != "not_submitted" or comparison["child_succeeded"]:
                raise ApiError("protocol", "整改缺少子作业摘要却给出已提交对照。")
        elif state != child["status"] \
                or comparison["child_succeeded"] != (child["status"] == "succeeded"):
            raise ApiError("protocol", "对照状态与子作业摘要不一致。")

    # ---- 内部实现 ---------------------------------------------------------

    @staticmethod
    def _job_id(job_id) -> int:
        if type(job_id) is not int or not 0<job_id<=9223372036854775807:
            raise ApiError("protocol", f"作业编号非法：{job_id!r}")
        return job_id

    @classmethod
    def _validate_job(cls,job):
        if not isinstance(job,dict) or job.get('status') not in ('queued','running','succeeded','failed','interrupted'):
            raise ApiError('protocol','识别服务返回的任务状态不符合约定。')
        cls._job_id(job.get('job_id'))
        for key in ('run','class_counts','artifact_checksums'):
            if job.get(key) is not None and not isinstance(job[key],dict):
                raise ApiError('protocol','识别服务任务结果的结构不符合约定。')
        checks=job.get('artifact_checksums') or {}
        if any(not isinstance(v,dict) for v in checks.values()):
            raise ApiError('protocol','识别服务产物校验记录的结构不符合约定。')

    def _connect(self, timeout: float):
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def _send(self, method: str, path: str, body, headers, timeout: float):
        conn = self._connect(timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            return conn, conn.getresponse()
        except socket.timeout as exc:
            conn.close()
            raise ApiError("timeout", "请求识别服务超时，结果未知，请稍后刷新或重试。",
                           detail=str(exc)) from None
        except (ConnectionRefusedError, socket.gaierror, ConnectionError, OSError) as exc:
            conn.close()
            raise ApiError(
                "offline",
                "无法连接识别服务，可能尚未启动或地址不正确。",
                detail=f"{type(exc).__name__}: {exc}",
            ) from None
        except http.client.HTTPException as exc:
            conn.close()
            raise ApiError('protocol','识别服务响应不完整，请刷新或以同一请求号重试。',detail=str(exc)) from None

    def _read_bounded(self, response, limit: int) -> bytes:
        raw_len = response.getheader("Content-Length")
        length=None
        if raw_len is not None:
            try:
                length=int(raw_len)
                if length<0:
                    raise ValueError('负长度')
                if length > limit:
                    raise ApiError(
                        "toolarge",
                        f"识别服务响应 {raw_len} 字节，超过本地读取上限 {limit} 字节",
                    )
            except ValueError:
                raise ApiError('protocol','识别服务响应长度非法。') from None
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ApiError("toolarge", f"识别服务响应超过本地读取上限 {limit} 字节")
        if length is not None and len(data)!=length:
            raise ApiError('protocol','识别服务响应传输中断，请刷新或以同一请求号重试。')
        return data

    def _request_raw(self, method: str, path: str, *, body=None, headers=None,
                     timeout: float, max_bytes: int):
        conn, response = self._send(method, path, body, headers, timeout)
        try:
            data = self._read_bounded(response, max_bytes)
            status = response.status
            ctype = response.getheader("Content-Type", "") or ""
        except socket.timeout as exc:
            raise ApiError('timeout','读取识别服务响应超时，提交结果可能尚未确认。',detail=str(exc)) from None
        except (http.client.HTTPException,ConnectionError,OSError) as exc:
            raise ApiError('protocol','识别服务响应传输中断，请刷新或以同一请求号重试。',detail=str(exc)) from None
        finally:
            conn.close()
        return status, ctype, data

    def _error_from_status(self, status: int, data: bytes) -> ApiError:
        message = ""
        try:
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, dict) and parsed.get("error"):
                message = str(parsed["error"])
        except (ValueError, UnicodeDecodeError):
            pass
        summary = message or data.decode("utf-8", "replace")[:200]
        if status == 409:
            return ApiError(
                "conflict",
                f"识别服务拒绝了该请求（HTTP 409）：{summary or '资源冲突或产物已变化'}",
                status=status, detail=summary,
            )
        return ApiError("http", f"识别服务返回错误（HTTP {status}）：{summary}",
                        status=status, detail=summary)

    def _request_json(self, method: str, path: str, *, body=None, headers=None,
                      timeout: float | None = None) -> dict:
        status, ctype, data = self._request_raw(
            method, path, body=body, headers=headers,
            timeout=self.timeout if timeout is None else timeout,
            max_bytes=self.max_json_bytes,
        )
        if status >= 400:
            raise self._error_from_status(status, data)
        try:
            result=json.loads(data.decode('utf-8'),parse_constant=lambda s: (_ for _ in ()).throw(ValueError('非法JSON数值')))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ApiError("badjson", "识别服务返回的内容不是合法 JSON。",
                           detail=f"{type(exc).__name__}: {exc}") from None
        if not isinstance(result,dict):
            raise ApiError('protocol','识别服务响应须为JSON对象，请检查本机服务。')
        return result

    def _request_bytes(self, method: str, path: str, *, max_bytes: int,
                       timeout: float | None = None) -> bytes:
        status, _ctype, data = self._request_raw(
            method, path, timeout=self.timeout if timeout is None else timeout,
            max_bytes=max_bytes,
        )
        if status >= 400:
            raise self._error_from_status(status, data)
        return data
