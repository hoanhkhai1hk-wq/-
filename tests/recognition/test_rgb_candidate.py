"""Meaning tests for the candidate RGB offline flow (INF-20261020-002).

Pure-function tests only: no real weights, no model loading, no GPU, no network.
Run with the .venv-recognition interpreter:
  .venv-recognition/Scripts/python.exe -B tests/recognition/test_rgb_candidate.py
or via pytest. Synthetic results never represent real agricultural accuracy.
"""
import sys
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recognition.rgb_candidate import (  # noqa: E402
    CSV_HEADER, candidate_csv, extract_candidate_fields, fuse_rule,
    load_rgb_512, native_repeat2, network_input_256)


def _probs(shape, cls):
    """float32 probability image (3,H,W) with certainty on cls."""
    p = np.zeros((3, *shape), dtype=np.float32)
    p[cls] = np.float32(1.0)
    return p


def test_fusion_tie_takes_field():
    base = np.zeros((256, 256), dtype=np.uint8)
    p003 = np.full((3, 256, 256), np.float32(1.0 / 3), dtype=np.float32)
    p005 = p003.copy()
    out = fuse_rule(base, p003, p005)
    assert out.dtype == np.uint8 and out.shape == (256, 256)
    assert (out == 0).all()  # equal 0/1 mean -> argmax picks Field


def test_fusion_bank_and_road_support():
    base = np.zeros((4, 4), dtype=np.uint8)
    base[3, 3] = 2  # W5 Road support
    p003 = _probs((4, 4), 1)  # fusion alone would say ridge everywhere
    p005 = _probs((4, 4), 1)
    out = fuse_rule(base, p003, p005)
    assert out[0, 0] == 1 and out[3, 3] == 2  # Road only where base==2
    p003 = _probs((4, 4), 1)
    p003[:, 2, 2] = _probs((4, 4), 0)[:, 2, 2]  # one pixel: certain Field
    out2 = fuse_rule(base, p003, _probs((4, 4), 1))
    assert out2[2, 2] == 0  # high-confidence Field survives


def test_native_repeat2_is_nearest_no_interpolation():
    m = np.zeros((256, 256), dtype=np.uint8)
    m[10:12, 20:22] = 2
    up = native_repeat2(m)
    assert up.shape == (512, 512)
    assert (up[20:24, 40:44] == 2).all() and up.sum() == m.sum() * 4
    assert up[19, 40] == 0 and up[24, 44] == 0  # hard edges, no blending


def test_candidate_diagonal_pixels_are_separate():
    m = np.full((256, 256), 1, dtype=np.uint8)
    m[5, 5] = 0
    m[6, 6] = 0  # diagonal only -> two 4-connected components
    rows = extract_candidate_fields(m)
    assert len(rows) == 2
    assert {r['area_pixels'] for r in rows} == {4}


def test_candidate_geometry_with_2x_scaling():
    m = np.ones((256, 256), dtype=np.uint8)
    m[7, 11] = 0
    rows = extract_candidate_fields(m)
    row = rows[0]
    assert row['area_pixels'] == 4
    assert (row['bbox_x0'], row['bbox_y0'], row['bbox_x1'], row['bbox_y1']) == (22, 14, 24, 16)
    assert row['centroid_x'] == 2 * 11 + 0.5 and row['centroid_y'] == 2 * 7 + 0.5
    assert row['edge_touch'] is False and row['small_patch'] is True


def test_candidate_edge_touch_on_border():
    m = np.ones((256, 256), dtype=np.uint8)
    m[0, 100] = 0
    m[255, 100] = 0
    rows = extract_candidate_fields(m)
    assert all(r['edge_touch'] for r in rows) and len(rows) == 2


def test_empty_table_when_no_field():
    m = np.full((256, 256), 1, dtype=np.uint8)
    assert extract_candidate_fields(m) == []
    lines = candidate_csv([]).strip().splitlines()
    assert len(lines) == 1 and lines[0].split(',') == CSV_HEADER  # header even when empty


def test_input_rejection_and_acceptance():
    with tempfile.TemporaryDirectory() as tmp:
        ok = Path(tmp) / 'ok.png'
        Image.new('RGB', (512, 512), (10, 20, 30)).save(ok)
        arr = load_rgb_512(ok)
        assert arr.dtype == np.uint8 and arr.shape == (512, 512, 3)
        small = network_input_256(arr)
        assert small.dtype == np.uint8 and small.shape == (256, 256, 3)
        wrong_size = Path(tmp) / 'small.png'
        Image.new('RGB', (256, 256)).save(wrong_size)
        try:
            load_rgb_512(wrong_size)
            raise AssertionError('256x256 accepted')
        except ValueError:
            pass
        rgba = Path(tmp) / 'rgba.png'
        Image.new('RGBA', (512, 512)).save(rgba)
        try:
            load_rgb_512(rgba)
            raise AssertionError('RGBA accepted')
        except ValueError:
            pass
        gray = Path(tmp) / 'gray.png'
        Image.new('L', (512, 512)).save(gray)
        try:
            load_rgb_512(gray)
            raise AssertionError('L accepted')
        except ValueError:
            pass


def test_preloaded_wrong_threads_rejected_without_mutation():
    # The public constructor must not silently rewrite a loaded process's
    # thread count before checking whether its runtime can be reused.
    code = '''
import torch
from unittest.mock import patch
from recognition.rgb_candidate import _init_runtime
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
with patch('recognition.rgb_candidate._apply_affinity', side_effect=AssertionError('affinity changed')):
    try:
        _init_runtime(20261017)
    except RuntimeError as error:
        assert 'threads!=1' in str(error)
    else:
        raise AssertionError('preloaded two-thread runtime was accepted')
assert torch.get_num_threads() == 2
'''
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONHASHSEED='0')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env, check=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for test in tests:
        try:
            test()
            print('PASS', test.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL', test.__name__, repr(exc))
    print(f'{len(tests) - failed}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
