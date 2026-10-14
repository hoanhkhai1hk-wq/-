"""Streamlit 健康检查（标准库 urllib，不依赖 requests）。

用法：
  .venv-p0/Scripts/python.exe scripts/p0/health_check.py [url]
"""
from __future__ import annotations

import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8505/_stcore/health"


def main() -> int:
    try:
        with urllib.request.urlopen(URL, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
            print(f"HTTP {resp.status} body={body!r}")
            return 0 if resp.status == 200 else 1
    except Exception as exc:
        print(f"健康检查失败：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
