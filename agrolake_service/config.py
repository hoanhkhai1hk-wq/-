"""识别服务配置与路径解析（DEV-20261015-009）。

所有默认路径基于本包源码位置推导，绝不依赖 shell 默认工作目录。
环境变量可覆盖，便于测试使用独立目录；命令行参数最终覆盖。

本模块只使用标准库。
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# agrolake_service/ 的上一级即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8506

# 原始输入内容上限 40 MiB（与已审 CLI 一致）。
MAX_CONTENT_BYTES = 40 * 1024 * 1024
# base64(40 MiB) ≈ 53.34 MiB；请求正文上限留出 JSON 与元数据余量并保持有界。
MAX_BODY_BYTES = 56 * 1024 * 1024

# HTTP 层允许的元数据长度上限（避免超长字段）。
MAX_FILENAME_LEN = 255
MAX_SOURCE_LEN = 500
MAX_SOURCE_LINK_LEN = 2048
MAX_LICENSE_LEN = 500
MAX_SCENE_ID_LEN = 200
MAX_REQUEST_ID_LEN = 128

ALLOWED_TIF_SUFFIXES = (".tif", ".tiff")

# CLI 固定产出的五个文件名（成功判定依据）。
FIXED_ARTIFACTS = (
    "mask.tif",
    "preview.png",
    "class_pixels.csv",
    "field_pixels.csv",
    "run.json",
)

ARTIFACT_CONTENT_TYPES = {
    "mask.tif": "image/tiff",
    "preview.png": "image/png",
    "class_pixels.csv": "text/csv; charset=utf-8",
    "field_pixels.csv": "text/csv; charset=utf-8",
    "run.json": "application/json; charset=utf-8",
}

# 已审模型（FTW v1 CC-BY）预期 SHA256，用于启动前与运行后校验。
EXPECTED_MODEL_SHA256 = (
    "d04671c498c181a2bca4b9c1a82fb8dcc93384eaf55841cb6672692f6fb21d3c"
)
# 公开上游冒烟样本 SHA256，用于把「公开适配样例」与用户上传显著区分。
PUBLIC_SAMPLE_SHA256 = (
    "da482e7987b36898bd9f8381e7c7fe95c05dfc5ccbba19a58eb65e544d1af750"
)

SAMPLE_KIND_PUBLIC = "public_adapter_sample"
SAMPLE_KIND_UPLOAD = "user_upload"
SAMPLE_KIND_SYNTHETIC = "synthetic_test"


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def default_db_path() -> Path:
    """与 P0/网页共享 AGROLAKE_DB；识别表以 rec_ 前缀独立新增，保留既有表与数据。"""
    return Path(_env("AGROLAKE_DB", str(PROJECT_ROOT / "data" / "p0" / "agrolake.sqlite")))


def default_store_root() -> Path:
    return Path(_env("AGROLAKE_SERVICE_STORE", str(PROJECT_ROOT / "data" / "service" / "store")))


def default_runs_root() -> Path:
    return Path(_env("AGROLAKE_SERVICE_RUNS", str(PROJECT_ROOT / "data" / "service" / "runs")))


def default_worker_python() -> Path:
    return Path(
        _env(
            "AGROLAKE_SERVICE_WORKER_PYTHON",
            str(PROJECT_ROOT / ".venv-recognition" / "Scripts" / "python.exe"),
        )
    )


def default_cli_script() -> Path:
    return Path(
        _env(
            "AGROLAKE_SERVICE_CLI",
            str(PROJECT_ROOT / "scripts" / "recognition" / "run_inference.py"),
        )
    )


def default_model_path() -> Path:
    return Path(
        _env(
            "AGROLAKE_SERVICE_MODEL",
            str(PROJECT_ROOT / "models" / "ftw-v1" / "3_Class_CCBY_FTW_Pretrained.ckpt"),
        )
    )


def default_public_sample_path() -> Path:
    return Path(
        _env(
            "AGROLAKE_SERVICE_PUBLIC_SAMPLE",
            str(PROJECT_ROOT / "samples" / "ftw_smoke" / "inference-img.tif"),
        )
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ServiceConfig:
    """一次服务进程的完整配置。路径一律为绝对路径。"""

    db_path: Path = field(default_factory=default_db_path)
    store_root: Path = field(default_factory=default_store_root)
    runs_root: Path = field(default_factory=default_runs_root)
    worker_python: Path = field(default_factory=default_worker_python)
    cli_script: Path = field(default_factory=default_cli_script)
    model_path: Path = field(default_factory=default_model_path)
    model_sha256: str = EXPECTED_MODEL_SHA256
    public_sample_path: Path | None = field(default_factory=default_public_sample_path)
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_seconds: float = 600.0
    max_body_bytes: int = MAX_BODY_BYTES
    max_content_bytes: int = MAX_CONTENT_BYTES
    # 部署级标记：为真时把非公开样例的上传标记为合成测试，避免把合成数据当作真实用户数据。
    synthetic_mode: bool = False

    def __post_init__(self):
        for name in ('db_path', 'store_root', 'runs_root', 'worker_python', 'cli_script',
                     'model_path', 'public_sample_path'):
            value = getattr(self, name)
            if value is not None:
                path = Path(value).expanduser()
                setattr(self, name, (path if path.is_absolute() else PROJECT_ROOT / path).resolve())
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('作业超时必须是大于零的有限秒数')
        if not 0 <= self.port <= 65535:
            raise ValueError('端口须在0至65535之间')
        if not 0 < self.max_content_bytes <= MAX_CONTENT_BYTES or not 0 < self.max_body_bytes <= MAX_BODY_BYTES:
            raise ValueError('请求/内容上限必须为正且不超过系统上限')

    def public_sample_sha256(self) -> str | None:
        """仅返回本地文件摘要（缓存），不能作为公开来源证明；分类只认固定上游摘要。"""
        if getattr(self, "_public_sha_computed", False):
            return getattr(self, "_public_sha_cache", None)
        path = self.public_sample_path
        result = None
        if path is not None and Path(path).is_file():
            try:
                result = sha256_file(Path(path))
            except OSError:
                result = None
        self._public_sha_cache = result
        self._public_sha_computed = True
        return result

    def classify_sample(self, content_sha256: str) -> str:
        """依据内容摘要把输入显著区分为公开适配样例 / 用户上传 / 合成测试。"""
        if content_sha256 == PUBLIC_SAMPLE_SHA256:
            return SAMPLE_KIND_PUBLIC
        if self.synthetic_mode:
            return SAMPLE_KIND_SYNTHETIC
        return SAMPLE_KIND_UPLOAD

    def readiness(self) -> dict:
        """启动前/健康检查使用的真实可用性（不伪装成功）。"""
        worker_ok = Path(self.worker_python).is_file()
        cli_ok = Path(self.cli_script).is_file()
        model_ok = Path(self.model_path).is_file()
        model_sha_note = None
        if model_ok:
            # 仅声明文件存在；完整可信哈希由CLI及运行后校验，不冒充模型成功运行。
            model_sha_note = "model file present"
        return {
            "worker_python": str(self.worker_python),
            "worker_python_ready": worker_ok,
            "cli_script": str(self.cli_script),
            "cli_script_ready": cli_ok,
            "model_path": str(self.model_path),
            "model_ready": model_ok,
            "model_note": model_sha_note,
        }

    def missing_requirements(self) -> list[str]:
        """返回面向零基础用户的中文缺失说明；空列表表示就绪。"""
        problems: list[str] = []
        if not Path(self.worker_python).is_file():
            problems.append(
                f"未找到识别子进程解释器 .venv-recognition：{self.worker_python}"
            )
        if not Path(self.cli_script).is_file():
            problems.append(f"未找到已审识别 CLI：{self.cli_script}")
        if not Path(self.model_path).is_file():
            problems.append(f"未找到模型权重文件：{self.model_path}")
        return problems

    def with_overrides(self, **kwargs) -> "ServiceConfig":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})
