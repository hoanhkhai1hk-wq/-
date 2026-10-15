"""Regression tests for the DEV-20261015-007 review fixes.

Covers Codex REV-20261015-001 blocking issues:
- numeric GDAL NoData (42113) is excluded from the valid mask;
- NaN/Inf never reach the model and non-finite logits fail hard;
- an existing output directory is refused and left byte-for-byte unchanged;
- oversize file / pixel / side inputs are refused before decode;
- spatial-tag and NoData read-back mismatches hard-fail;
- a plain unrelated valid 8-band raster still passes validation;
- an input with no valid pixels is rejected.

Artifacts (fixtures, log, summary) are written under
reports/recognition/DEV-20261015-007/<run-id>/ and are never deleted.
Synthetic results never represent real agricultural accuracy.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import tifffile
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition import (  # noqa: E402
    read_image, write_mask, read_mask_back, verify_mask_readback, predict,
)

RUN_INFER = os.path.join(PROJECT_ROOT, "scripts", "recognition", "run_inference.py")
DEV007_ROOT = os.path.join(PROJECT_ROOT, "reports", "recognition", "DEV-20261015-007")


def _run_dir():
    rid = os.environ.get("DEV007_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(DEV007_ROOT, f"tests-{rid}")
    os.makedirs(d, exist_ok=True)
    return d


RUN = _run_dir()


def _eight_band(path, h, w, fill=1):
    arr = np.full((h, w, 8), fill, dtype=np.uint16)
    return arr


def test_numeric_nodata_42113_excluded():
    """A pixel equal to the declared GDAL NoData must be invalid, not kept valid."""
    d = os.path.join(RUN, "t1_nodata42113")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "nodata42113.tif")
    arr = np.zeros((20, 24, 8), dtype=np.uint16)
    arr[5, 6, :] = 65535          # declared NoData in all bands
    arr[7, 8, 3] = 65535          # declared NoData in only one band -> invalid
    extratags = [(42113, 2, 0, "65535", True)]
    tifffile.imwrite(path, arr, planarconfig="contig", photometric="minisblack", extratags=extratags)
    image, valid, meta = read_image(path)
    assert meta["gdal_nodata_source"] == "65535"
    assert meta["gdal_nodata_value"] == 65535.0
    assert not valid[5, 6], "numeric-NoData pixel must be excluded from valid"
    assert not valid[7, 8], "any-band NoData pixel must be excluded from valid"
    assert int(valid.sum()) == 20 * 24 - 2


def test_no_nodata_declared_none_assumed():
    """A plain 8-band raster with no NoData tag: no value (not even 0) is assumed."""
    d = os.path.join(RUN, "t6_plain8band")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "plain8.tif")
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 4000, size=(40, 55, 8)).astype(np.uint16)
    arr[0, 0, :] = 0               # a legitimate zero is NOT NoData when undeclared
    tifffile.imwrite(path, arr, planarconfig="contig", photometric="minisblack")
    image, valid, meta = read_image(path)
    assert meta["gdal_nodata_source"] is None
    assert meta["gdal_nodata_value"] is None
    assert image.shape == (40, 55, 8)
    assert valid.all(), "no NoData declared -> zero pixels stay valid"


class _RecordingModel:
    """Counts calls and records whether any non-finite value reached it."""

    def __init__(self):
        self.calls = 0
        self.saw_nonfinite = False

    def __call__(self, x):
        self.calls += 1
        if not bool(torch.isfinite(x).all()):
            self.saw_nonfinite = True
        out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        out[:, 1] = 1.0
        return out


def test_nonfinite_never_reaches_model():
    """NaN/Inf are neutral-filled before the model and become 255 in the mask."""
    rng = np.random.default_rng(11)
    image = rng.integers(0, 4000, size=(60, 70, 8)).astype(np.float32)
    image[30:36, 30:36, :] = np.nan
    image[10, 10, 2] = np.inf
    valid = np.isfinite(image).all(axis=2)
    m = _RecordingModel()
    mask, stats = predict(m, image, valid)
    assert m.calls > 0, "model was never called"
    assert not m.saw_nonfinite, "non-finite values reached the model"
    assert (mask[30:36, 30:36] == 255).all(), "NaN block must be 255"
    assert mask[10, 10] == 255, "Inf pixel must be 255"
    assert (mask[valid] != 255).all(), "valid pixels must not be faked as NoData"
    assert stats["coverage_ratio_valid"] == 1.0
    assert stats["uncovered_valid_pixels"] == 0


def test_nonfinite_logits_fail_hard():
    """A model emitting non-finite logits must fail loudly, not silently corrupt output."""
    image = np.ones((40, 40, 8), dtype=np.float32)
    valid = np.ones((40, 40), dtype=bool)

    class _NanLogit:
        def __call__(self, x):
            out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
            out[:, 0] = float("nan")
            return out

    try:
        predict(_NanLogit(), image, valid)
    except RuntimeError as e:
        assert "non-finite" in str(e), e
        return
    raise AssertionError("non-finite logits did not raise")


def test_empty_valid_rejected():
    image = np.full((20, 20, 8), np.nan, dtype=np.float32)
    valid = np.zeros((20, 20), dtype=bool)
    try:
        predict(_RecordingModel(), image, valid)
    except ValueError as e:
        assert "no valid pixels" in str(e), e
        return
    raise AssertionError("input with no valid pixels was not rejected")


def test_valid_mask_shape_mismatch_rejected():
    image = np.ones((30, 30, 8), dtype=np.float32)
    valid = np.ones((30, 31), dtype=bool)
    try:
        predict(_RecordingModel(), image, valid)
    except ValueError as e:
        assert "does not match" in str(e), e
        return
    raise AssertionError("valid-mask shape mismatch did not raise")


def test_existing_output_dir_refused_and_unchanged():
    """CLI must refuse an existing result dir and leave its files byte-for-byte."""
    import hashlib

    d = os.path.join(RUN, "t3_existing_dir")
    os.makedirs(d, exist_ok=True)
    sentinel = os.path.join(d, "keep.txt")
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write("ORIGINAL-RESULT")
    h_before = hashlib.sha256(open(sentinel, "rb").read()).hexdigest()

    proc = subprocess.run(
        [sys.executable, RUN_INFER,
         "--input", os.path.join(d, "does-not-exist.tif"),
         "--checkpoint", os.path.join(d, "does-not-exist.ckpt"),
         "--output-dir", d],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, "existing output dir must exit non-zero"
    assert "already exists" in proc.stdout, proc.stdout
    assert sorted(os.listdir(d)) == ["keep.txt"], "existing dir contents changed"
    h_after = hashlib.sha256(open(sentinel, "rb").read()).hexdigest()
    assert h_after == h_before, "original file was modified"


def test_limits_rejected_before_decode():
    """Oversize file / side / pixel inputs are refused without allocating big arrays."""
    d = os.path.join(RUN, "t4_limits")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "ok8.tif")
    tifffile.imwrite(path, _eight_band(path, 20, 24), planarconfig="contig", photometric="minisblack")

    # file-size bound
    try:
        read_image(path, max_bytes=10)
    except ValueError as e:
        assert "too large" in str(e), e
    else:
        raise AssertionError("oversize file was not rejected")

    # per-side bound (must raise, never attempt a huge decode)
    try:
        read_image(path, max_side=5)
    except ValueError as e:
        assert "side exceeds" in str(e), e
    else:
        raise AssertionError("oversize side was not rejected")

    # total-pixel bound
    try:
        read_image(path, max_pixels=10)
    except ValueError as e:
        assert "pixel count exceeds" in str(e), e
    else:
        raise AssertionError("oversize pixel count was not rejected")

    # decode-byte bound
    try:
        read_image(path, max_decode_bytes=8)
    except ValueError as e:
        assert "decoded size exceeds" in str(e), e
    else:
        raise AssertionError("oversize decoded estimate was not rejected")


def test_readback_verify_hard_fails():
    mask = np.ones((8, 8), dtype=np.uint8)
    back = mask.copy()
    good_tags = {"shape": (8, 8), "dtype": "uint8", "gdal_nodata": "255", "tags": {}}

    # spatial tag claimed in source but absent on read-back -> hard failure
    src_with_spatial = {33550: (1.0, 1.0, 0.0)}
    try:
        verify_mask_readback(mask, back, good_tags, src_with_spatial)
    except ValueError as e:
        assert "spatial tag" in str(e), e
    else:
        raise AssertionError("spatial-tag mismatch did not raise")

    # NoData read-back mismatch -> hard failure
    bad_nodata = dict(good_tags, gdal_nodata="0")
    try:
        verify_mask_readback(mask, back, bad_nodata, {})
    except ValueError as e:
        assert "NoData" in str(e), e
    else:
        raise AssertionError("NoData mismatch did not raise")

    # pixel mismatch -> hard failure
    tampered = back.copy()
    tampered[0, 0] = 7
    try:
        verify_mask_readback(mask, tampered, good_tags, {})
    except ValueError as e:
        assert "pixel mismatch" in str(e), e
    else:
        raise AssertionError("pixel mismatch did not raise")

    # honest success when the source had no spatial reference
    res = verify_mask_readback(mask, back, good_tags, {})
    assert res["spatial_reference_state"] == "absent_in_source"
    assert res["spatial_tags_preserved"] is False


def test_plain_8band_roundtrip_ok():
    """A fresh unrelated valid 8-band raster round-trips mask tags via the helper."""
    d = os.path.join(RUN, "t6_roundtrip")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "src8.tif")
    tags = {33550: (10.0, 10.0, 0.0), 33922: (0.0, 0.0, 0.0, 359130.0, 5352340.0, 0.0)}
    extratags = [(33550, 12, 3, list(tags[33550]), True), (33922, 12, 6, list(tags[33922]), True)]
    tifffile.imwrite(src, _eight_band(src, 32, 32), planarconfig="contig",
                     photometric="minisblack", extratags=extratags)
    image, valid, meta = read_image(src)
    mask = np.full((32, 32), 1, dtype=np.uint8)
    out = os.path.join(d, "mask.tif")
    write_mask(out, mask, meta["geotiff_tags"])
    back, back_tags = read_mask_back(out)
    res = verify_mask_readback(mask, back, back_tags, meta["geotiff_tags"])
    assert res["spatial_reference_state"] == "preserved"
    assert res["spatial_tags_preserved"] is True


def main():
    tests = [
        test_numeric_nodata_42113_excluded,
        test_no_nodata_declared_none_assumed,
        test_nonfinite_never_reaches_model,
        test_nonfinite_logits_fail_hard,
        test_empty_valid_rejected,
        test_valid_mask_shape_mismatch_rejected,
        test_existing_output_dir_refused_and_unchanged,
        test_limits_rejected_before_decode,
        test_readback_verify_hard_fails,
        test_plain_8band_roundtrip_ok,
    ]
    results = []
    log_path = os.path.join(RUN, "tests.log")
    with open(log_path, "w", encoding="utf-8") as log:
        for t in tests:
            try:
                t()
                results.append({"test": t.__name__, "status": "PASS"})
                line = f"PASS {t.__name__}"
            except Exception as e:  # noqa: BLE001 - record then re-raise
                results.append({"test": t.__name__, "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
                line = f"FAIL {t.__name__}: {type(e).__name__}: {e}"
                log.write(line + "\n")
                log.flush()
                with open(os.path.join(RUN, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"run_dir": RUN, "results": results}, f, ensure_ascii=False, indent=2)
                print(line)
                raise
            log.write(line + "\n")
            log.flush()
            print(line)
        summary = {"run_dir": RUN, "results": results}
        with open(os.path.join(RUN, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"{passed}/{len(tests)} regression tests passed (run dir: {RUN})")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
