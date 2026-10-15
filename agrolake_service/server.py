"""标准库 HTTP 服务（DEV-20261015-009）。

仅绑定本机地址，只接受 JSON POST，校验 Host（拒绝非本机）与 Origin（拒绝外站浏览器来源）。
不暴露任意文件下载、不支持目录浏览、拒绝路径穿越与未知作业/文件名。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

from . import SERVICE_NAME, __version__
from . import governance
from . import reviews
from . import validation
from .jobs import IdempotencyConflict,ArtifactIntegrityError
from .validation import ValidationError

LOG = logging.getLogger("agrolake_service")

LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def _hostname_from_host_header(host_header: str) -> str:
    value = (host_header or "").strip()
    if not value:
        return ""
    if value.startswith("["):
        return value[1:].split("]", 1)[0].lower()
    return value.split(":", 1)[0].lower()


class RecognitionRequestHandler(BaseHTTPRequestHandler):
    server_version = "AgroLakeRecognition/" + __version__
    protocol_version = "HTTP/1.1"

    # ---- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # route BaseHTTPRequestHandler logs to logging
        LOG.info("%s - %s", self.client_address[0], fmt % args)

    @property
    def config(self):
        return self.server.config

    @property
    def manager(self):
        return self.server.manager

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self.close_connection = True  # 拒绝正文后不把尚未读取的内容当作下一请求
        self._send_json(status, {"error": message, "status": status})

    def _send_bytes(self, status: int, data: bytes, content_type: str, filename: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self) -> bool:
        return _hostname_from_host_header(self.headers.get("Host", "")) in LOCAL_HOSTNAMES

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # 非浏览器客户端不带 Origin
        try:
            parsed = urlsplit(origin)
            _ = parsed.port
            return (parsed.scheme in ('http','https') and parsed.hostname in LOCAL_HOSTNAMES
                    and parsed.username is None and parsed.password is None
                    and not parsed.path and not parsed.query and not parsed.fragment)
        except ValueError:
            return False

    def _read_json_body(self, max_bytes: int | None = None):
        ctype = (self.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ValidationError("仅接受 application/json 请求体", status=415)
        if self.headers.get('Transfer-Encoding') is not None:
            raise ValidationError('不支持Transfer-Encoding',400)
        if len(self.headers.get_all('Content-Length',[])) != 1:
            raise ValidationError('须提供唯一Content-Length',411)
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise ValidationError("缺少 Content-Length", status=411)
        try:
            length = int(raw_len)
        except ValueError:
            raise ValidationError("Content-Length 非法") from None
        if length < 0:
            raise ValidationError("Content-Length 非法")
        limit = self.config.max_body_bytes if max_bytes is None else max_bytes
        if length > limit:
            raise ValidationError(
                f"请求正文 {length} 字节超过上限 {limit} 字节",
                status=413,
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValidationError("请求正文长度与 Content-Length 不符")
        try:
            return json.loads(raw.decode('utf-8'),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('非法JSON数值')))
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ValidationError(f"请求体不是合法 JSON：{exc}") from exc

    # ---- routing ----------------------------------------------------------

    def _segments(self) -> list:
        # origin-form路由无需解析任意外部URL，原样保留编码以拒绝穿越。
        path = self.path.split('?',1)[0]
        return [seg for seg in path.split("/") if seg != ""]

    def do_GET(self):  # noqa: N802 - stdlib naming
        if not self._host_ok():
            return self._send_error_json(403, "拒绝非本机 Host 的请求")
        segs = self._segments()
        try:
            if segs == ["health"]:
                return self._handle_health()
            if segs == ["jobs"]:
                return self._send_json(200, {"jobs": self.manager.list_jobs()})
            if len(segs) == 2 and segs[0] == "jobs":
                return self._handle_job_detail(segs[1])
            if len(segs) == 3 and segs[0] == "jobs" and segs[2] == "fields":
                return self._handle_job_fields(segs[1])
            if len(segs) == 3 and segs[0] == "jobs" and segs[2] == "reviews":
                return self._handle_review_list(segs[1])
            if len(segs) == 3 and segs[0] == "jobs" and segs[2] == "lineage":
                return self._handle_lineage(segs[1])
            if len(segs) == 4 and segs[0] == "jobs" and segs[2] == "artifacts":
                return self._handle_artifact(segs[1], segs[3])
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except ArtifactIntegrityError as exc:
            return self._send_error_json(409,str(exc))
        except (OSError,sqlite3.Error,ValueError):
            LOG.exception('读取识别记录失败')
            return self._send_error_json(500,'读取识别记录失败，详情请查看本地服务日志')
        return self._send_error_json(404, "未知路径")

    def do_POST(self):  # noqa: N802 - stdlib naming
        if not self._host_ok():
            return self._send_error_json(403, "拒绝非本机 Host 的请求")
        if not self._origin_ok():
            return self._send_error_json(403, "拒绝非本机或外站浏览器来源（Origin）")
        segs = self._segments()
        if segs == ["jobs"]:
            return self._handle_job_submit()
        if len(segs) == 3 and segs[0] == "jobs" and segs[2] == "reviews":
            return self._handle_review_submit(segs[1])
        if len(segs) == 3 and segs[0] == "jobs" and segs[2] == "governance":
            return self._handle_governance_submit(segs[1])
        return self._send_error_json(404, "未知路径")

    def _handle_job_submit(self):
        try:
            payload = self._read_json_body()
            normalized = validation.validate_job_payload(
                payload, max_content_bytes=self.config.max_content_bytes
            )
            result = self.manager.submit(normalized)
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except IdempotencyConflict as exc:
            return self._send_error_json(409, str(exc))
        except (OSError,sqlite3.Error,ValueError):
            LOG.exception('作业提交失败')
            return self._send_error_json(500,'作业未能提交，已保留已有数据，请查看本地服务日志')
        job = self.manager.get_job(result["job_id"]) or {}
        body = {
            "job_id": result["job_id"],
            "status": job.get("status", "queued"),
            "input_sha256": result["input_sha256"],
            "sample_kind": result["sample_kind"],
            "request_id": result["request_id"],
            "created": result["created"],
        }
        return self._send_json(202 if result["created"] else 200, body)

    # ---- handlers ---------------------------------------------------------

    def _handle_health(self):
        readiness = self.config.readiness()
        return self._send_json(200, {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": __version__,
            "worker_python_ready": readiness["worker_python_ready"],
            "cli_script_ready": readiness["cli_script_ready"],
            "model_ready": readiness["model_ready"],
        })

    def _parse_job_id(self, raw: str) -> int:
        if (not raw.isascii() or not raw.isdigit() or len(raw)>19
                or not 0 < int(raw) <= 9223372036854775807):
            raise ValidationError("作业编号非法", status=404)
        return int(raw)

    def _handle_job_detail(self, raw_id: str):
        job_id = self._parse_job_id(raw_id)
        job = self.manager.get_job(job_id)
        if job is None:
            return self._send_error_json(404, f"未知作业 {job_id}")
        return self._send_json(200, job)

    def _handle_job_fields(self, raw_id: str):
        job_id = self._parse_job_id(raw_id)
        fields = self.manager.get_fields(job_id)
        if fields is None:
            return self._send_error_json(404, f"未知作业 {job_id}")
        return self._send_json(200, {
            "job_id": job_id,
            "area_unit": "pixel",
            "note": "候选连通区域像素，不代表经验证平方米或真实田块数",
            "region_count": len(fields),
            "regions": fields,
        })

    def _handle_artifact(self, raw_id: str, name: str):
        job_id = self._parse_job_id(raw_id)
        from . import config as cfg

        if name not in cfg.FIXED_ARTIFACTS:
            return self._send_error_json(404, "未知产物名")
        try:
            data = self.manager.read_artifact(job_id,name)
        except OSError:
            return self._send_error_json(404, "产物不存在")
        if data is None:
            return self._send_error_json(404, '产物不存在或作业尚未成功')
        return self._send_bytes(200, data, cfg.ARTIFACT_CONTENT_TYPES[name], name)

    def _query_params(self) -> dict:
        if "?" not in self.path:
            return {}
        try:
            pairs = parse_qsl(self.path.split('?', 1)[1], keep_blank_values=True, max_num_fields=10)
        except ValueError:
            raise ValidationError('复核分页参数过多') from None
        if any(k not in {'after_id', 'limit', 'until_id'} for k, _ in pairs):
            raise ValidationError('未知复核分页参数')
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValidationError('复核分页参数不能重复')
        return dict(pairs)

    def _handle_review_submit(self, raw_id: str):
        try:
            job_id = self._parse_job_id(raw_id)
            payload = self._read_json_body(reviews.MAX_REVIEW_BODY_BYTES)
            outcome = reviews.submit_review(self.config.db_path, job_id, payload)
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except (OSError, sqlite3.Error, ValueError):
            LOG.exception('记录人工复核失败')
            return self._send_error_json(
                500, '人工复核未能保存，已保留已有数据，请查看本地服务日志'
            )
        body = dict(outcome["record"])
        body["created"] = outcome["created"]
        return self._send_json(201 if outcome["created"] else 200, body)

    def _handle_review_list(self, raw_id: str):
        try:
            job_id = self._parse_job_id(raw_id)
            params = self._query_params()
            result = reviews.list_reviews(
                self.config.db_path, job_id,
                after_id=params.get("after_id"), limit=params.get("limit"), until_id=params.get('until_id'),
            )
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except (OSError, sqlite3.Error, ValueError):
            LOG.exception('读取复核记录失败')
            return self._send_error_json(
                500, '读取复核记录失败，详情请查看本地服务日志'
            )
        return self._send_json(200, result)

    def _handle_governance_submit(self, raw_id: str):
        try:
            job_id = self._parse_job_id(raw_id)
            payload = self._read_json_body(governance.MAX_GOVERNANCE_BODY_BYTES)
            outcome = governance.register_governance(self.config, job_id, payload)
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except (OSError, sqlite3.Error, ValueError):
            LOG.exception('登记治理资料失败')
            return self._send_error_json(
                500, '登记未能保存，已保留已有数据，请查看本地服务日志'
            )
        body = dict(outcome["registration"])
        body["created"] = outcome["created"]
        return self._send_json(201 if outcome["created"] else 200, body)

    def _handle_lineage(self, raw_id: str):
        try:
            job_id = self._parse_job_id(raw_id)
            result = governance.get_lineage(self.config, job_id)
        except ValidationError as exc:
            return self._send_error_json(exc.status, exc.message)
        except (OSError, sqlite3.Error, ValueError):
            LOG.exception('读取治理血缘失败')
            return self._send_error_json(
                500, '读取治理血缘失败，详情请查看本地服务日志'
            )
        return self._send_json(200, result)


class RecognitionServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, manager, config):
        super().__init__(address, handler)
        self.manager = manager
        self.config = config

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

    def get_request(self):
        client,address = super().get_request()
        client.settimeout(30)
        return client,address


def make_server(config, manager, *, host: str | None = None, port: int | None = None) -> RecognitionServer:
    bind_host = host if host is not None else config.host
    bind_port = port if port is not None else config.port
    if bind_host not in ("127.0.0.1", "localhost"):
        raise ValueError("服务只允许绑定本机地址（127.0.0.1/localhost）")
    return RecognitionServer((bind_host, bind_port), RecognitionRequestHandler, manager, config)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
