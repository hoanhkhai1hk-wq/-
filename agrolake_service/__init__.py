"""AgroLake 识别后端服务（DEV-20261015-009）。

独立本地后端：标准库 HTTP 服务 + 独立 SQLite 识别表 + 通过已审 CLI 子进程做真实推理。
本包只使用标准库，不导入 torch；真正的模型推理在 .venv-recognition 子进程中执行。

它不是网页（下一轮才接 Streamlit），也不修改已审 CLI、app.py、.env.example 或 P0 模块。
"""

__version__ = "0.1.0"
SERVICE_NAME = "agrolake-recognition"
