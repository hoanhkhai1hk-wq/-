"""Independent 2-layer boundary construction and class counts from saved arrays."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def independent_band(labels, include_unknown_margin):
    """Union of 4x3 / 3x4 pixel boxes around each known annotation interface.

    This does not call the producer's seed/dilate implementation.
    """
    h, w = labels.shape
    horizontal = labels[:, :-1] != labels[:, 1:]
    vertical = labels[:-1] != labels[1:]
    if include_unknown_margin:
        horizontal &= (labels[:, :-1] != 255) | (labels[:, 1:] != 255)
        vertical &= (labels[:-1] != 255) | (labels[1:] != 255)
    else:
        horizontal &= (labels[:, :-1] != 255) & (labels[:, 1:] != 255)
        vertical &= (labels[:-1] != 255) & (labels[1:] != 255)
    band = np.zeros((h, w), dtype=bool)
    for y, x in zip(*np.where(horizontal)):
        band[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 3)] = True
    for y, x in zip(*np.where(vertical)):
        band[max(0, y - 1):min(h, y + 3), max(0, x - 1):min(w, x + 2)] = True
    return band & (labels != 255)


if __name__ == '__main__':
    # Synthetic proof of the saved-model Road gate; no torch/runtime/model load.
    from recognition.rgb_candidate import fuse_rule
    base = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    probabilities = np.zeros((3, 2, 2), dtype=np.float32)
    probabilities[2] = 1  # Even 003/005 Road=100% cannot create unsupported Road.
    fused = fuse_rule(base, probabilities, probabilities)
    assert np.array_equal(fused == 2, base == 2)
    assert 'torch' not in sys.modules
    frozen = load(ROOT / 'reports/data/Codex-DATA-20261021-001/frozen-evaluation.json')
    source = ROOT / 'recognition/rgb_candidate.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen['pinned_files']['recognition/rgb_candidate.py']
    measured = load(OUT / 'frozen-statistics.json')
    saved = load(ROOT / 'reports/rgb/NEW-EVAL-20261021-001/summary.json')
    rows, total = [], Counter()
    for rec, official, initial in zip(frozen['records'], saved['records'], measured['per_image']):
        with Image.open(ROOT / rec['mask_path']) as im:
            truth = np.array(im)
        with Image.open(ROOT / 'reports/rgb/NEW-EVAL-20261021-001' / official['prediction_path']) as im:
            prediction = np.array(im)
        known_band = independent_band(truth, False)
        upper_band = independent_band(truth, True)
        assert not (known_band & ~upper_band).any()
        error = (truth != 255) & (truth != prediction)
        assert int(np.count_nonzero(error & known_band)) == initial['c_boundary_band_error']
        joint = Counter(zip(truth.flatten().tolist(), prediction.flatten().tolist()))
        assert [[joint[(a, b)] for b in range(3)] for a in range(3)] == official['confusion_truth_rows_prediction_columns']
        a = (truth == 0) & (prediction == 1)
        b = (truth == 0) & (prediction == 2)
        row = {'sample_id': rec['sample_id'], 'error_pixels': int(error.sum()),
               'a_non_boundary': int(np.count_nonzero(a & ~upper_band)),
               'b_non_boundary': int(np.count_nonzero(b & ~upper_band)),
               'c_boundary_associated': int(np.count_nonzero(error & upper_band)),
               'd_other_non_boundary': int(np.count_nonzero(error & ~upper_band & ~a & ~b)),
               'a_boundary_subset': int(np.count_nonzero(a & upper_band)),
               'b_boundary_subset': int(np.count_nonzero(b & upper_band)),
               'pure_boundary_only': bool(error.any() and not (error & ~upper_band).any())}
        assert sum(row[k] for k in ['a_non_boundary', 'b_non_boundary', 'c_boundary_associated', 'd_other_non_boundary']) == row['error_pixels']
        rows.append(row)
        total.update({key: int(value) for key, value in row.items() if key != 'sample_id'})
    images = {key: sum(int(row[key] > 0) for row in rows) for key in total}
    result = {'passed': True, 'saved_confusion_recounted': 24, 'new_forwards': 0,
              'road_gate_source_sha_pinned': True, 'road_gate_synthetic_invariant_test': 'passed without importing torch',
              'known_known_band_independent_equality': '24/24 counts match',
              'main_band_definition': '2 layers per side at any saved label boundary, including known/255 margins; no image frame; only known target pixels scored',
              'known_known_only_boundary_errors': measured['totals']['c_boundary_band_error'],
              'totals': dict(total), 'images': images, 'records': rows,
              'boundary_error_fraction': total['c_boundary_associated'] / total['error_pixels'],
              'caution': 'Including255 margins gives an inclusive geometric boundary estimate, never a proof of a pure 2px shift; values255 themselves remain unscored'}
    with (OUT / 'independent-audit.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False, indent=2))
