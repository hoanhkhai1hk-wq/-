"""请求体校验（DEV-20261015-009）。

只接受 ``content_base64`` 内联内容，绝不接受任意磁盘路径或可执行命令。
所有上限在解码/分配大对象之前生效。
"""
from __future__ import annotations

import base64
import binascii
import re

from . import config
from .store import safe_name, sha256_bytes

# 明确拒绝的字段名：出现即视为越权尝试。
FORBIDDEN_KEYS = {
    "path", "input_path", "input-path", "file", "filepath", "file_path",
    "command", "cmd", "argv", "exec", "shell", "output_dir", "output-dir",
    "url", "download_url", "checkpoint",
}

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % config.MAX_REQUEST_ID_LEN)


class ValidationError(Exception):
    """带 HTTP 状态码的请求校验失败。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _require_str(payload: dict, key: str, *, max_len: int, required: bool = True,
                 default: str | None = None) -> str | None:
    value = payload.get(key, default)
    if value is None:
        if required:
            raise ValidationError(f"缺少必填字段：{key}")
        return None
    if not isinstance(value, str):
        raise ValidationError(f"字段 {key} 必须是字符串")
    if required and value.strip() == "":
        raise ValidationError(f"字段 {key} 不能为空")
    if len(value) > max_len:
        raise ValidationError(f"字段 {key} 超过长度上限 {max_len}")
    return value.strip()


def decode_base64_strict(content_base64: str, *, max_bytes: int) -> bytes:
    """严格 base64 解码，并在解码前按长度上限拒绝过大内容。"""
    # base64 文本长度对应原始字节上限（4 个字符 -> 3 字节）。
    max_encoded = ((max_bytes + 2) // 3) * 4 + 4
    if len(content_base64) > max_encoded:
        raise ValidationError(
            f"content_base64 编码长度 {len(content_base64)} 超过上限"
            f"（原始内容上限 {max_bytes} 字节）", status=413,
        )
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(f"content_base64 不是合法 base64：{exc}") from exc
    if not data:
        raise ValidationError("content_base64 解码后为空内容")
    if len(data) > max_bytes:
        raise ValidationError(
            f"原始内容 {len(data)} 字节超过上限 {max_bytes} 字节", status=413
        )
    return data


def validate_job_payload(payload, *, max_content_bytes: int = config.MAX_CONTENT_BYTES) -> dict:
    """校验并规范化 POST /jobs 请求体，返回规范化字典（含 content 字节与摘要）。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")

    bad = sorted(k for k in payload if k in FORBIDDEN_KEYS)
    if bad:
        raise ValidationError(
            f"不接受字段 {bad}：本接口只接受内联内容，不读取磁盘路径或执行命令"
        )

    filename = _require_str(payload, "filename", max_len=config.MAX_FILENAME_LEN)
    base = safe_name(filename)
    filename = base
    lower = filename.lower()
    if not lower.endswith(config.ALLOWED_TIF_SUFFIXES):
        raise ValidationError("仅接受 .tif / .tiff 文件")

    source = _require_str(payload, "source", max_len=config.MAX_SOURCE_LEN)
    scene_id = _require_str(payload, "scene_id", max_len=config.MAX_SCENE_ID_LEN)
    source_link = _require_str(
        payload, "source_link", max_len=config.MAX_SOURCE_LINK_LEN, required=False
    )
    license_status = _require_str(
        payload, "license_status", max_len=config.MAX_LICENSE_LEN, required=False
    )

    request_id = payload.get("request_id")
    if request_id is not None:
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise ValidationError(
                f"request_id 必须匹配 ^[A-Za-z0-9._:-]{{1,{config.MAX_REQUEST_ID_LEN}}}$"
            )

    content_base64 = _require_str(payload, "content_base64", max_len=10 ** 12)
    data = decode_base64_strict(content_base64, max_bytes=max_content_bytes)

    return {
        "filename": filename,
        "source": source,
        "source_link": source_link,
        "license_status": license_status,
        "scene_id": scene_id,
        "request_id": request_id,
        "content": data,
        "content_sha256": sha256_bytes(data),
    }


def fingerprint(normalized: dict) -> str:
    """规范化请求指纹，用于幂等键比较：同 key 不同内容必须冲突。"""
    import hashlib
    import json

    payload = {
        "content_sha256": normalized["content_sha256"],
        "filename": normalized["filename"],
        "source": normalized["source"],
        "source_link": normalized["source_link"],
        "license_status": normalized["license_status"],
        "scene_id": normalized["scene_id"],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
