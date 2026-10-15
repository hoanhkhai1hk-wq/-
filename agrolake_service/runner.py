"""真实识别 CLI 子进程执行器（DEV-20261015-009）。

用明确参数数组启动已审 CLI（scripts/recognition/run_inference.py）于
``.venv-recognition`` 解释器；本进程绝不导入 torch。

失败绝不伪装成功：
- 超时：终止自己启动的进程并记为失败；
- 非零退出：失败；
- 缺失任一固定产物 / run.json 事实不符：失败。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .db import CLASS_PIXEL_KEY
from .processes import ProcessTree


@dataclass
class RunResult:
    success: bool
    status: str                       # succeeded | failed
    output_dir: str
    exit_code: int | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    class_counts: dict | None = None
    field_count: int | None = None
    regions: list | None = None
    run_json: dict | None = None
    model_sha256: str | None = None
    input_sha256: str | None = None
    adapter: str | None = None
    versions: dict | None = None
    params: dict | None = None
    command: list = field(default_factory=list)


def _read_class_pixels_csv(path: Path) -> dict:
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row['class']) in out:
                raise ValueError('class_pixels.csv存在重复类别')
            out[int(row["class"])] = int(row["pixels"])
    return out


def _read_field_pixels_csv(path: Path) -> list:
    regions = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            regions.append(
                {
                    "field_id": int(row["field_id"]),
                    "pixels": int(row["pixels"]),
                    "min_row": int(row["min_row"]),
                    "min_col": int(row["min_col"]),
                    "max_row": int(row["max_row"]),
                    "max_col": int(row["max_col"]),
                }
            )
    return regions


class CliRunner:
    """真实 CLI 执行器。接口与测试用假 runner 一致：run(input_path, input_sha256, out_dir, log_path)。"""

    def __init__(
        self,
        worker_python: Path,
        cli_script: Path,
        model_path: Path,
        model_sha256: str,
        timeout_seconds: float = 600.0,
    ):
        self.worker_python = Path(worker_python)
        self.cli_script = Path(cli_script)
        self.model_path = Path(model_path)
        self.model_sha256 = model_sha256
        self.timeout_seconds = float(timeout_seconds)
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def build_command(self, input_path: Path, input_sha256: str, out_dir: Path) -> list:
        return [
            str(self.worker_python),
            str(self.cli_script),
            "--input", str(input_path),
            "--checkpoint", str(self.model_path),
            "--output-dir", str(out_dir),
            "--expected-input-sha256", input_sha256,
        ]

    def run(self, input_path, input_sha256, out_dir, log_path) -> RunResult:
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 关键：输出目录必须不存在（CLI 会以独占方式创建）；父目录可存在。
        if out_dir.exists():
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                error=f"输出目录已存在，拒绝执行以免覆盖历史：{out_dir}",
            )

        cmd = self.build_command(input_path, input_sha256, out_dir)
        started = time.monotonic()
        if self._cancel.is_set():
            return RunResult(False,'failed',str(out_dir),error='服务已停止，未启动识别子进程')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        timed_out = False
        cancelled = False
        with open(log_path,'x',encoding='utf-8') as log:
            log.write('$ '+subprocess.list2cmdline(cmd)+'\n\n')
            log.flush()
            tree = ProcessTree(cmd,cwd=str(config.PROJECT_ROOT),env=env,stdout=log)
            proc = tree.process
            try:
                while proc.poll() is None:
                    cancelled = self._cancel.is_set()
                    timed_out = time.monotonic()-started >= self.timeout_seconds
                    if cancelled or timed_out:
                        tree.terminate()
                        break
                    self._cancel.wait(0.05)
            finally:
                tree.close()
                proc.wait(timeout=15)
        elapsed = round(time.monotonic() - started, 3)

        if timed_out or cancelled:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=proc.returncode, elapsed_seconds=elapsed,
                error=('服务停止，已终止本次识别进程树' if cancelled else
                       f'CLI超时（>{self.timeout_seconds}s），已终止本次识别进程树'),
                command=cmd,
            )
        if proc.returncode != 0:
            detail = ''
            error_file = out_dir/'error.json'
            if error_file.is_file():
                try:
                    if error_file.stat().st_size <= 16384:
                        detail = str(json.loads(error_file.read_text(encoding='utf-8')).get('error',''))[:1000]
                except (OSError,ValueError,AttributeError):
                    pass
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=proc.returncode, elapsed_seconds=elapsed,
                error=f'CLI退出码非零：{proc.returncode}；{detail}',
                command=cmd,
            )

        return self._verify_success(input_path, input_sha256, out_dir, elapsed, cmd)

    def _verify_success(self, input_path, input_sha256, out_dir, elapsed, cmd) -> RunResult:
        try:
            return self._read_success(input_path,input_sha256,out_dir,elapsed,cmd)
        except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError) as exc:
            return RunResult(False,'failed',str(out_dir),exit_code=0,elapsed_seconds=elapsed,
                             error=f'识别产物校验失败：{type(exc).__name__}: {exc}',command=cmd)

    def _read_success(self, input_path, input_sha256, out_dir, elapsed, cmd) -> RunResult:
        missing = [n for n in config.FIXED_ARTIFACTS if not (out_dir / n).is_file()]
        if missing:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"CLI 退出码为 0 但缺少固定产物：{missing}",
                command=cmd,
            )

        try:
            with open(out_dir / "run.json", encoding="utf-8") as fh:
                run = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"run.json 读取失败：{exc}", command=cmd,
            )

        if run.get("status") != "success":
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"run.json status={run.get('status')!r}，不作为成功",
                command=cmd,
            )

        in_sha = (run.get("input") or {}).get("sha256")
        if in_sha != input_sha256:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"run.json 输入摘要 {in_sha!r} 与提交内容 {input_sha256!r} 不符",
                command=cmd,
            )

        ck_sha = (run.get("checkpoint") or {}).get("sha256")
        if self.model_sha256 and ck_sha != self.model_sha256:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"run.json 模型摘要 {ck_sha!r} 与预期 {self.model_sha256!r} 不符",
                command=cmd,
            )

        shape = (run.get("input") or {}).get("shape_hw")
        counts = run.get("class_counts")
        if not (isinstance(counts, dict) and isinstance(shape, list) and len(shape) == 2):
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error="run.json 缺少 class_counts 或 shape_hw", command=cmd,
            )
        try:
            if any(type(n) is not int or n <= 0 for n in shape) or shape[0]*shape[1]>1048576:
                raise ValueError('图像尺寸非法')
            if any(type(counts[k]) is not int or counts[k]<0 for k in CLASS_PIXEL_KEY.values()):
                raise ValueError('类像素计数须为非负整数')
            total = sum(counts[k] for k in CLASS_PIXEL_KEY.values())
        except (KeyError, TypeError, ValueError) as exc:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"run.json class_counts 字段异常：{exc}", command=cmd,
            )
        if total != int(shape[0]) * int(shape[1]):
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"class_counts 合计 {total} 与 shape {shape} 不符", command=cmd,
            )

        try:
            csv_counts = _read_class_pixels_csv(out_dir / "class_pixels.csv")
            regions = _read_field_pixels_csv(out_dir / "field_pixels.csv")
        except (OSError, KeyError, ValueError) as exc:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"CSV 产物解析失败：{exc}", command=cmd,
            )

        expected_csv = {c: int(counts[key]) for c, key in CLASS_PIXEL_KEY.items()}
        if csv_counts != expected_csv:
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"class_pixels.csv {csv_counts} 与 run.json {expected_csv} 不符",
                command=cmd,
            )

        field_count = run.get("field_count")
        if type(field_count) is not int or field_count != len(regions):
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error=f"field_count {field_count} 与 field_pixels.csv 行数 {len(regions)} 不符",
                command=cmd,
            )
        if sum(r["pixels"] for r in regions) != int(counts["field_interior_1"]):
            return RunResult(
                success=False, status="failed", output_dir=str(out_dir),
                exit_code=0, elapsed_seconds=elapsed,
                error="候选区域像素合计与 field_interior_1 不符", command=cmd,
            )

        ids = set()
        for r in regions:
            if (r['field_id'] in ids or r['field_id']<=0 or r['pixels']<=0
                or not 0<=r['min_row']<=r['max_row']<shape[0]
                or not 0<=r['min_col']<=r['max_col']<shape[1]
                or r['pixels']>(r['max_row']-r['min_row']+1)*(r['max_col']-r['min_col']+1)):
                raise ValueError('候选区域编号/像素/边框非法')
            ids.add(r['field_id'])

        return RunResult(
            success=True,
            status="succeeded",
            output_dir=str(out_dir),
            exit_code=0,
            elapsed_seconds=elapsed,
            class_counts={k: int(counts[k]) for k in CLASS_PIXEL_KEY.values()},
            field_count=int(field_count),
            regions=regions,
            run_json=run,
            model_sha256=ck_sha,
            input_sha256=in_sha,
            adapter=run.get("adapter"),
            versions=run.get("versions"),
            params=run.get("inference"),
            command=cmd,
        )
