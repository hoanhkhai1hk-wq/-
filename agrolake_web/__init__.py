"""网页田块识别入口（DEV-20261015-012）。

本包只承载“网页 → 本机识别服务”的客户端与界面：
- ``client``：纯标准库 HTTP 客户端，调用已审识别服务（agrolake_service），不新增依赖。
- ``recognition_page``：Streamlit 页面，仅在被 app.py 导入时加载 streamlit。

本包不导入 torch，不直接运行识别 CLI，也不写识别数据库；真实推理始终由
已审识别服务在独立子进程中完成。``__init__`` 只导出客户端，避免测试在未
加载 streamlit 时被迫导入界面模块。
"""
from __future__ import annotations

from .client import (
    DEFAULT_API_URL,
    ApiConfigError,
    ApiError,
    RecognitionClient,
    resolve_base_url,
)

WEB_VERSION = "0.1.0"

__all__ = [
    "DEFAULT_API_URL",
    "ApiConfigError",
    "ApiError",
    "RecognitionClient",
    "resolve_base_url",
    "WEB_VERSION",
]
