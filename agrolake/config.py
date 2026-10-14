"""AgroLake P0 配置与路径解析。

所有路径基于包位置推导，避免依赖 shell 默认工作目录。
环境变量可覆盖存储位置，便于测试使用独立目录。
"""
from __future__ import annotations

import os
from pathlib import Path

# agrolake/ 的上一级即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
SEMANTIC_RULES_PATH = CONFIG_DIR / "semantic_rules.json"
QUALITY_RULES_PATH = CONFIG_DIR / "quality_rules.json"

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "p0"


def data_dir() -> Path:
    return Path(os.environ.get("AGROLAKE_DATA_DIR", DEFAULT_DATA_DIR))


def storage_root() -> Path:
    """内容对象存储根目录。"""
    return Path(os.environ.get("AGROLAKE_STORE", data_dir() / "store"))


def db_path() -> Path:
    """SQLite 数据库路径。"""
    return Path(os.environ.get("AGROLAKE_DB", data_dir() / "agrolake.sqlite"))


def reports_dir() -> Path:
    return Path(os.environ.get("AGROLAKE_REPORTS", data_dir() / "reports"))


CONTENT_TYPES = {
    "image": {".jpg", ".jpeg", ".png"},
    "attribute": {".csv"},
    "text": {".json", ".txt"},
}


def classify_extension(ext: str) -> str | None:
    ext = ext.lower()
    for ctype, exts in CONTENT_TYPES.items():
        if ext in exts:
            return ctype
    return None
