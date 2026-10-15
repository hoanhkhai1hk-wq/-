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
