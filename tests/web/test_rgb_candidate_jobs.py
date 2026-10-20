"""INT-20261020-002 RGB候选作业后端测试（.venv-p0，合成fixture，离线）。

全部使用 mock 的 subprocess.run 与合成产物，不加载模型、不真实推理、
不写业务库、不访问网络。真实模型网页流程由 Root 独立验收。

用法：
  .venv-p0/Scripts/python.exe tests/web/test_rgb_candidate_jobs.py
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import agrolake.rgb_jobs as rgb_jobs  # noqa: E402
from agrolake.rgb_jobs import (  # noqa: E402
    ARTIFACT_NAMES,
    DEFAULT_MANIFEST,
    DEFAULT_SEED,
    RgbArtifactError,
    RgbCandidateInputError,
    RgbCandidateJobs,
    verify_outputs,
)

RID = "a" * 32
RID2 = "b" * 32


def png_bytes(mode: str = "RGB", size: tuple[int, int] = (512, 512)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (10, 120, 30)).save(buf, "PNG")
    return buf.getvalue()


def tiff_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (10, 120, 30)).save(buf, "TIFF")
    return buf.getvalue()


# ---------- 合成产物 fixture：与已审 CLI 输出格式一致，内容人工构造 ----------

def _build_mask256() -> np.ndarray:
    m = np.zeros((256, 256), dtype=np.uint8)      # A: class0 上半幅
    m[128:200, 0:128] = 1                          # C: class1
    m[200:256, 0:128] = 2                          # D: class2
    m[128:256, 128:256] = 0                        # B: class0 右下块
    return m


def _regions() -> list[dict]:
    return [
        {"field_id": 1, "area_pixels": 131072, "bbox_x0": 0, "bbox_y0": 0,
         "bbox_x1": 512, "bbox_y1": 256, "centroid_x": 255.5, "centroid_y": 255.5,
         "edge_touch": True, "small_patch": False},
        {"field_id": 2, "area_pixels": 65536, "bbox_x0": 256, "bbox_y0": 256,
         "bbox_x1": 512, "bbox_y1": 512, "centroid_x": 383.5, "centroid_y": 383.5,
         "edge_touch": False, "small_patch": False},
    ]


def _regions_csv() -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["field_id", "area_pixels", "bbox_x0", "bbox_y0", "bbox_x1",
                     "bbox_y1", "centroid_x", "centroid_y", "edge_touch", "small_patch"])
    for r in _regions():
        writer.writerow([r["field_id"], r["area_pixels"], r["bbox_x0"], r["bbox_y0"],
                         r["bbox_x1"], r["bbox_y1"], r["centroid_x"], r["centroid_y"],
                         "true" if r["edge_touch"] else "false",
                         "true" if r["small_patch"] else "false"])
    return buf.getvalue().encode("utf-8")


def write_synthetic_outputs(result_dir: Path, image_bytes: bytes) -> None:
    """写出六份与真实 CLI 同构的合成产物（明确不是真实模型输出）。"""
    result_dir.mkdir(parents=True)
    mask256 = _build_mask256()
    mask512 = np.repeat(np.repeat(mask256, 2, axis=0), 2, axis=1)
    Image.fromarray(mask256, mode="L").save(result_dir / "mask256.png")
    Image.fromarray(mask512, mode="L").save(result_dir / "mask512.png")
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay[:, :, 1] = 120
    Image.fromarray(overlay, mode="RGB").save(result_dir / "overlay.png")
    (result_dir / "candidate_fields.csv").write_bytes(_regions_csv())

    import hashlib

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    hashed = {name: sha((result_dir / name).read_bytes())
              for name in ("mask256.png", "mask512.png", "overlay.png",
                           "candidate_fields.csv")}
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8-sig"))
    group = next(m for m in manifest["models"] if m["seed"] == DEFAULT_SEED)
    quality = {
        "grid_class_counts": {"0": 49152, "1": 9216, "2": 7168},
        "native_class_counts": {"0": 196608, "1": 36864, "2": 28672},
        "grid_total_65536": True,
        "native_total_262144": True,
        "candidate_area_sum_equals_native_field_count": True,
        "connectivity_rule": "4-neighborhood (up/down/left/right) on the 256 grid",
        "candidate_count": 2,
        "edge_touch_count": 1,
        "small_patch_count": 0,
        "small_patch_rule": "native area_pixels < 200",
        "notes": "SYNTHETIC FIXTURE, NOT REAL MODEL",
    }
    metadata = {
        "task": "SYNTHETIC FIXTURE, NOT REAL MODEL",
        "seed": DEFAULT_SEED,
        "manifest_path": DEFAULT_MANIFEST.name,
        "manifest_sha256": sha(DEFAULT_MANIFEST.read_bytes()),
        "weights_sha256": {role: group[role]["sha256"]
                           for role in ("baseline_W5", "retention_003", "feedback_005")},
        "image_sha256": sha(image_bytes),
        "input": {"original_size": [512, 512], "mode": "RGB uint8"},
        "output": {"grid": [256, 256]},
        "runtime": {"fixture": True},
        "versions": {"fixture": True},
        "grid_class_counts": quality["grid_class_counts"],
        "native_class_counts": quality["native_class_counts"],
        "candidate_count": 2,
        "output_files_sha256": hashed,
        "candidate_scope": "class0 four-connected components on the 256 grid, all kept",
        "independent_test_available": False,
        "notes": "SYNTHETIC FIXTURE, NOT REAL MODEL",
    }
    (result_dir / "quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- mock 的 subprocess.run ----------

FIXTURE = {"mode": "success", "calls": []}


def fake_run(cmd, **kwargs):
    FIXTURE["calls"].append(list(cmd))
    mode = FIXTURE["mode"]
    image = Path(cmd[cmd.index("--image") + 1])
    out = Path(cmd[cmd.index("--output-dir") + 1])
    if mode == "timeout":
        raise subprocess.TimeoutExpired(cmd, 120)
    if mode == "fail_exit":
        return mock.Mock(returncode=3, stdout=b"", stderr=b"boom")
    if mode == "exit0_corrupt":
        out.mkdir(parents=True)
        (out / "metadata.json").write_bytes(b"{}")
        return mock.Mock(returncode=0, stdout=b"", stderr=b"")
    if mode == "success":
        write_synthetic_outputs(out, image.read_bytes())
        return mock.Mock(returncode=0, stdout=b"2 candidates", stderr=b"")
    raise AssertionError(f"unknown fixture mode {mode}")


class JobsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "rgb"
        self.jobs = RgbCandidateJobs(data_dir=self.data_dir)
        FIXTURE["mode"] = "success"
        FIXTURE["calls"] = []
        self._patcher = mock.patch("subprocess.run", fake_run)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.tmp.cleanup()

    def calls(self) -> int:
        return len(FIXTURE["calls"])


class InputValidationTest(JobsTestBase):
    def test_empty_and_oversize_rejected(self):
        for bad in (b"", b"x" * (4 * 1024 * 1024 + 1)):
            with self.assertRaises(RgbCandidateInputError):
                self.jobs.submit(bad, "x.png", RID)
        self.assertEqual(self.jobs.list_jobs(), [])
        self.assertEqual(self.calls(), 0)

    def test_undecodable_and_wrong_shape_rejected(self):
        cases = [
            b"not an image at all",
            png_bytes(size=(256, 256)),          # 尺寸不符
            png_bytes(mode="RGBA"),              # 模式不符
            b"GIF89a" + b"\x00" * 64,            # 不支持的格式开头
        ]
        for bad in cases:
            with self.assertRaises(RgbCandidateInputError):
                self.jobs.submit(bad, "x.png", RID)
        self.assertEqual(self.jobs.list_jobs(), [])
        self.assertEqual(self.calls(), 0)

    def test_bad_request_ids_rejected(self):
        img = png_bytes()
        for rid in ("", "short", "A" * 32, "g" * 32, "a" * 31, "a" * 33, 12345):
            with self.assertRaises(RgbCandidateInputError):
                self.jobs.submit(img, "a.png", rid)
        self.assertEqual(self.jobs.list_jobs(), [])
        self.assertEqual(self.calls(), 0)


class SuccessAndIdempotencyTest(JobsTestBase):
    def test_success_persists_regions_and_artifacts(self):
        img = png_bytes()
        job = self.jobs.submit(img, "a.png", RID)
        self.assertEqual(job["status"], "succeeded")
        self.assertIsNone(job["error"])
        self.assertEqual(job["source_name"], "a.png")
        import hashlib
        self.assertEqual(job["image_sha256"], hashlib.sha256(img).hexdigest())
        self.assertEqual(self.calls(), 1)
        # 六产物 SHA 已存库
        self.assertEqual(set(job["artifact_sha256"]), set(ARTIFACT_NAMES))

        result = self.jobs.result(job["id"])
        self.assertEqual(result["metadata"]["seed"], DEFAULT_SEED)
        self.assertEqual(len(result["regions"]), 2)
        self.assertEqual([r["field_id"] for r in result["regions"]], [1, 2])
        self.assertEqual(result["quality"]["native_class_counts"]["0"], 196608)

        # 下载与原始字节
        data = self.jobs.artifact(job["id"], "mask256.png")
        self.assertTrue(data[:8] == b"\x89PNG\r\n\x1a\n")
        self.assertEqual(self.jobs.source_bytes(job["id"]), img)

        # 重建实例后仍可读（持久化）
        fresh = RgbCandidateJobs(data_dir=self.data_dir)
        again = fresh.get_job(job["id"])
        self.assertEqual(again["status"], "succeeded")
        self.assertEqual(len(fresh.result(job["id"])["regions"]), 2)

    def test_tiff_input_accepted(self):
        img = tiff_bytes()
        job = self.jobs.submit(img, "a.tif", RID2)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["input_format"], "TIFF")
        self.assertTrue(job["source_path"].endswith("input.tiff"))

    def test_same_request_same_content_idempotent(self):
        img = png_bytes()
        job1 = self.jobs.submit(img, "a.png", RID)
        job2 = self.jobs.submit(img, "a.png", RID)
        self.assertEqual(job1["id"], job2["id"])
        self.assertEqual(self.calls(), 1)

    def test_same_request_different_content_rejected(self):
        img1 = png_bytes()
        buf = io.BytesIO()
        Image.new("RGB", (512, 512), (200, 10, 10)).save(buf, "PNG")
        img3 = buf.getvalue()
        self.jobs.submit(img1, "a.png", RID)
        with self.assertRaises(RgbCandidateInputError):
            self.jobs.submit(img3, "b.png", RID)
        # 原作业不受影响，也不触发新推理
        self.assertEqual(self.calls(), 1)
        self.assertEqual(len(self.jobs.list_jobs()), 1)


class FailureTest(JobsTestBase):
    def test_corrupt_output_with_exit0_marks_failed(self):
        FIXTURE["mode"] = "exit0_corrupt"
        job = self.jobs.submit(png_bytes(), "a.png", RID)
        self.assertEqual(job["status"], "failed")
        self.assertIn("产物校验", job["error"])
        self.assertTrue(all(j["status"] != "succeeded" for j in self.jobs.list_jobs()))

    def test_timeout_marks_failed(self):
        FIXTURE["mode"] = "timeout"
        job = self.jobs.submit(png_bytes(), "a.png", RID)
        self.assertEqual(job["status"], "failed")
        self.assertIn("超时", job["error"])

    def test_nonzero_exit_marks_failed(self):
        FIXTURE["mode"] = "fail_exit"
        job = self.jobs.submit(png_bytes(), "a.png", RID)
        self.assertEqual(job["status"], "failed")
        self.assertIn("退出码 3", job["error"])

    def test_missing_environment_marks_failed(self):
        with mock.patch.object(rgb_jobs, "INTERPRETER",
                               Path(self.tmp.name) / "no-python.exe"):
            job = self.jobs.submit(png_bytes(), "a.png", RID)
        self.assertEqual(job["status"], "failed")
        self.assertIn("解释器", job["error"])
        self.assertEqual(self.calls(), 0)

    def test_failed_job_with_same_request_is_returned_not_rerun(self):
        FIXTURE["mode"] = "fail_exit"
        job1 = self.jobs.submit(png_bytes(), "a.png", RID)
        FIXTURE["mode"] = "success"
        job2 = self.jobs.submit(png_bytes(), "a.png", RID)
        self.assertEqual(job1["id"], job2["id"])
        self.assertEqual(job2["status"], "failed")
        self.assertEqual(self.calls(), 1)


class ArtifactSafetyTest(JobsTestBase):
    def setUp(self):
        super().setUp()
        self.img = png_bytes()
        self.job = self.jobs.submit(self.img, "a.png", RID)
        self.assertEqual(self.job["status"], "succeeded")

    def test_tampered_artifact_rejected(self):
        result_dir = Path(self.job["result_dir"])
        (result_dir / "mask256.png").write_bytes(b"tampered")
        with self.assertRaises(RgbArtifactError):
            self.jobs.artifact(self.job["id"], "mask256.png")
        # 未被改动的产物仍可读
        self.jobs.artifact(self.job["id"], "metadata.json")

    def test_inconsistent_quality_detected_by_verify(self):
        result_dir = Path(self.job["result_dir"])
        doc = json.loads((result_dir / "quality.json").read_text(encoding="utf-8"))
        doc["candidate_count"] = 99
        (result_dir / "quality.json").write_text(
            json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_outputs(result_dir, self.job["image_sha256"],
                           DEFAULT_MANIFEST, DEFAULT_SEED)

    def test_bad_names_and_paths_rejected(self):
        jid = self.job["id"]
        for name in ("input.png", "../secret.png", "", "mask256.png/../../x"):
            with self.assertRaises(RgbArtifactError):
                self.jobs.artifact(jid, name)
        for bad_id in ("not-a-uuid", "", "z" * 32, f"{jid}/../{jid}"):
            with self.assertRaises(RgbArtifactError):
                self.jobs.artifact(bad_id, "mask256.png")
        with self.assertRaises(RgbArtifactError):
            self.jobs.artifact("c" * 32, "mask256.png")  # 不存在的作业

    def test_failed_job_has_no_artifacts(self):
        FIXTURE["mode"] = "fail_exit"
        bad = self.jobs.submit(png_bytes(), "b.png", RID2)
        with self.assertRaises(RgbArtifactError):
            self.jobs.artifact(bad["id"], "mask256.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
