"""tests/web 合成fixture（DEV-20261015-012）。

所有内容都是人工合成，绝不代表真实模型、真实TIFF解码或农业效果。
- :class:`WebFixtureRunner` 给识别服务用：写出**真实可解码的小PNG**预览，
  使 Streamlit 能渲染；其余产物为占位内容。run.json 明确标注 SYNTHETIC。
- :class:`FakeHTTPServer` 给客户端单测用：由测试提供 responder 决定返回。
"""
from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from agrolake_service.runner import RunResult


class WebFixtureRunner:
    """合成运行器：产生可被网页展示的假结果，明确不是真实模型输出。"""

    def __init__(self, mode: str = "success"):
        self.mode = mode
        self.calls = 0

    def run(self, input_path, input_sha256, out_dir, log_path):
        self.calls += 1
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data = b""
        try:
            data = Path(input_path).read_bytes()
        except OSError:
            data = b""
        # 内容含 FAIL 即失败：让“失败路径”判定与工作线程时序无关。
        if self.mode == "failure" or b"FAIL" in data:
            return RunResult(False, "failed", str(out), exit_code=7,
                             error="SYNTHETIC 失败fixture, NOT REAL MODEL")

        # 真PNG三面板占位（左/中/右各11x12），保证可解码。
        panel = Image.new("RGB", (33, 12), (30, 120, 90))
        panel.save(out / "preview.png")
        (out / "mask.tif").write_bytes(b"SYNTHETIC MASK FIXTURE, NOT REAL MODEL")
        counts = {"background_0": 4, "field_interior_1": 4, "boundary_2": 2, "nodata_255": 2}
        with (out / "class_pixels.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "meaning", "pixels"])
            for c, key, meaning in ((0, "background_0", "background"),
                                    (1, "field_interior_1", "field_interior"),
                                    (2, "boundary_2", "boundary"),
                                    (255, "nodata_255", "nodata/unlabelled")):
                w.writerow([c, meaning, counts[key]])
        region = {"field_id": 1, "pixels": 4, "min_row": 0, "min_col": 0,
                  "max_row": 1, "max_col": 1}
        with (out / "field_pixels.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(region))
            w.writeheader()
            w.writerow(region)
        run = {
            "status": "success",
            "adapter": "SYNTHETIC FIXTURE, NOT REAL MODEL",
            "input": {"sha256": input_sha256, "shape_hw": [3, 4]},
            "checkpoint": {"sha256": "f" * 64},
            "versions": {"fixture": True},
            "inference": {"fixture": True},
            "class_counts": counts,
            "field_count": 1,
        }
        (out / "run.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        return RunResult(
            True, "succeeded", str(out), exit_code=0, run_json=run,
            class_counts=counts, field_count=1, regions=[region],
            model_sha256="f" * 64, input_sha256=input_sha256,
            versions=run["versions"], params=run["inference"], adapter=run["adapter"],
        )


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CannedTest/1.0"

    def log_message(self, *args):  # silence
        pass

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        status, headers, payload = self.server.responder(
            self.command, self.path, dict(self.headers), body
        )
        headers = dict(headers or {})
        self.send_response(status)
        headers.setdefault("Content-Length", str(len(payload)))
        headers.setdefault("Content-Type", "application/json; charset=utf-8")
        headers.setdefault('Connection','close')
        self.close_connection=True
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if payload:
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # 客户端可能在读取上限/超时后已关闭连接，属预期，不使测试崩溃。
                self.close_connection = True

    do_GET = _handle
    do_POST = _handle


class FakeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, responder):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.responder = responder

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://127.0.0.1:{port}"


def start_fake_server(responder):
    """在后台线程启动，返回 (server, url, stop)。"""
    server = FakeHTTPServer(responder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        server.server_close()
        thread.join(5)

    return server, server.url, stop


def free_closed_port() -> int:
    """返回一个刚被释放、当前无监听的端口（用于离线测试）。"""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
