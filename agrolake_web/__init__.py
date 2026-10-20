"""网页田块识别入口（DEV-20261015-012；RGB图片识别见 INT-20261020-002）。

本包只承载“网页 → 本机识别服务/CLI”的客户端与界面：
- ``client``：纯标准库 HTTP 客户端，调用已审识别服务（agrolake_service），不新增依赖。
- ``recognition_page``：Streamlit 页面（田块识别，8 波段），仅在被 app.py 导入时加载 streamlit。
- ``rgb_results_page``：RGB 识别实验只读历史页。
- ``rgb_candidate_page``：RGB 图片识别页（INT-20261020-002），显式调用
  ``agrolake.rgb_jobs.RgbCandidateJobs`` 运行已审单图 CLI 并读写专用历史库；
  本包不导入 torch，也不在网页进程内加载任何模型。``__init__`` 只导出客户端，避免测试在未
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
