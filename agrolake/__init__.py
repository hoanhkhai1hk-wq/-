"""AgroLake P0 轻量数据湖治理底座。

模块：
- storage   内容寻址对象存储与路径边界
- db        SQLite 模式与连接
- semantics 语义字典、字段别名与单位口径
- quality   质量检查规则
- ingest    影像/CSV/JSON/TXT 统一接入
- disposition 处置与复检
- scenes    场景关联与统计
- reporting 报告与清单导出
"""
from __future__ import annotations

from . import (  # noqa: F401
    config,
    db,
    disposition,
    ingest,
    quality,
    reporting,
    scenes,
    semantics,
    storage,
)

__all__ = [
    "config", "db", "disposition", "ingest", "quality", "reporting",
    "scenes", "semantics", "storage",
]

__version__ = "0.1.0-p0"
