"""Synthetic rule tests for the recognition adapter (DEV-20261015-006).

Self-contained: run with the .venv-recognition interpreter.
`python tests/recognition/test_recognition.py`
Synthetic results never represent real agricultural accuracy.
"""

import os
import sys
import tempfile
import importlib.util
import warnings

import numpy as np
import tifffile
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition import read_image, write_mask, read_mask_back, predict, class_counts, field_table  # noqa: E402


class StubModel:
    """Returns fixed logits: class 1 everywhere (input already upsampled)."""

    def __call__(self, x):
        out = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        out[:, 1] = 1.0
        return out


def _tmp_dir():
    base = os.path.join(PROJECT_ROOT, "reports", "recognition")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="recognition-test-", dir=base)


def _preview_writer():
    path = os.path.join(PROJECT_ROOT, "scripts", "recognition", "run_inference.py")
    spec = importlib.util.spec_from_file_location("preview_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_preview


def test_preview_partial_nodata_keeps_contrast():
    from PIL import Image
    image = np.arange(32 * 32 * 8, dtype=np.float32).reshape(32, 32, 8)
    image[5, 5] = np.nan
    image[6, 6] = np.inf
    image[7, 7] = 65535
    mask = np.ones((32, 32), dtype=np.uint8)
    mask[5, 5] = mask[6, 6] = mask[7, 7] = 255
    path = os.path.join(_tmp_dir(), "partial.png")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            _preview_writer()(path, image, mask)
    with Image.open(path) as im:
        panel = np.asarray(im)
    rgb = panel[18:50, :32]
    assert rgb.std() > 10, "valid preview pixels lost their contrast"
    for r, c in ((5, 5), (6, 6), (7, 7)):
        assert (rgb[r, c] == (90, 90, 90)).all(), "NoData must be visibly grey"
    assert np.count_nonzero(rgb) > 1000


def test_preview_all_invalid_is_gray():
    from PIL import Image
    image = np.full((16, 16, 8), np.nan, dtype=np.float32)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    path = os.path.join(_tmp_dir(), "all-invalid.png")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            _preview_writer()(path, image, mask)
    with Image.open(path) as im:
        panel = np.asarray(im)
    assert (panel[18:34, :16] == 90).all(), "empty preview must show NoData grey"


def test_preview_normal_output_size():
    from PIL import Image
    image = np.arange(24 * 40 * 8, dtype=np.float32).reshape(24, 40, 8)
    path = os.path.join(_tmp_dir(), "normal.png")
    _preview_writer()(path, image, np.ones((24, 40), dtype=np.uint8))
    with Image.open(path) as im:
        assert im.size == (40 * 3 + 20, 24 + 18)
        assert np.asarray(im)[18:, :40].std() > 10


def test_reject_wrong_band_count():
    p = os.path.join(_tmp_dir(), "7band.tif")
    tifffile.imwrite(p, np.zeros((10, 12, 7), dtype=np.uint16), planarconfig="contig")
    try:
        read_image(p)
    except ValueError as e:
        assert "8-band" in str(e) or "(H, W, 8)" in str(e), e
        return
    raise AssertionError("7-band raster was not rejected")


def test_reject_bad_file():
    p = os.path.join(_tmp_dir(), "bad.tif")
    with open(p, "wb") as f:
        f.write(b"this is not a tiff file")
    try:
        read_image(p)
    except Exception as e:  # noqa: BLE001 - any structured failure is correct
        assert not isinstance(e, AssertionError), e
        return
    raise AssertionError("corrupt file was not rejected")


def test_nodata_and_coverage():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 4000, size=(60, 70, 8)).astype(np.float32)
    image[30:36, 30:36, :] = np.nan  # NoData block
    valid = np.isfinite(image).all(axis=2)
    mask, stats = predict(StubModel(), image, valid)
    assert mask.shape == (60, 70)
    assert set(np.unique(mask)).issubset({0, 1, 2, 255})
    assert (mask[30:36, 30:36] == 255).all(), "NoData block must be 255"
    assert stats["uncovered_valid_pixels"] == 0
    assert stats["coverage_ratio_valid"] == 1.0
    assert (mask[valid] != 255).all(), "valid pixels must not be faked as NoData"


def test_class_counts_and_field_table():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    mask[20:30, 20:30] = 1
    mask[40, 40] = 2
    mask[49, 49] = 255
    counts = class_counts(mask)
    assert sum(counts.values()) == 50 * 50
    assert counts["field_interior_1"] == 25 + 100
    rows = field_table(mask)
    assert len(rows) == 2
    assert sum(r["pixels"] for r in rows) == counts["field_interior_1"]
    r1 = rows[0]
    assert (r1["min_row"], r1["min_col"], r1["max_row"], r1["max_col"]) == (5, 5, 9, 9)
    for r in rows:
        for k in ("field_id", "pixels", "min_row", "min_col", "max_row", "max_col"):
            assert isinstance(r[k], int) and np.isfinite(r[k])
            assert 0 <= r[k] < 50 or k in ("field_id", "pixels")


def test_mask_geotiff_tags_preserved():
    d = _tmp_dir()
    src = os.path.join(d, "src8.tif")
    img = np.zeros((32, 32, 8), dtype=np.uint16)
    tags = {
        33550: (10.0, 10.0, 0.0),
        33922: (0.0, 0.0, 0.0, 359130.0, 5352340.0, 0.0),
        34735: (1, 1, 0, 7, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026, 34737, 22, 0,
                2049, 34737, 7, 22, 2054, 0, 1, 9102, 3072, 0, 1, 32633, 3076, 0, 1, 9001),
        34737: "WGS 84 / UTM zone 33N|WGS 84|",
    }
    extratags = []
    for code, v in tags.items():
        if isinstance(v, str):
            extratags.append((code, 2, 0, v, True))
        elif code == 34735:
            extratags.append((code, 3, len(v), list(v), True))
        else:
            extratags.append((code, 12, len(v), list(v), True))
    tifffile.imwrite(src, img, planarconfig="contig", photometric="minisblack", extratags=extratags)
    image, valid, meta = read_image(src)
    mask = np.full((32, 32), 1, dtype=np.uint8)
    out = os.path.join(d, "mask.tif")
    write_mask(out, mask, meta["geotiff_tags"])
    back, back_tags = read_mask_back(out)
    assert back.shape == (32, 32) and back.dtype == np.uint8
    for code, v in meta["geotiff_tags"].items():
        got = back_tags["tags"].get(code)
        assert got == v, f"tag {code}: {got!r} != {v!r}"
    assert back_tags["gdal_nodata"] == "255"


def main():
    tests = [
        test_reject_wrong_band_count,
        test_reject_bad_file,
        test_nodata_and_coverage,
        test_class_counts_and_field_table,
        test_mask_geotiff_tags_preserved,
        test_preview_partial_nodata_keeps_contrast,
        test_preview_all_invalid_is_gray,
        test_preview_normal_output_size,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} synthetic rule tests passed")


if __name__ == "__main__":
    main()
