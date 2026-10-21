"""Hand-calculated metric and input-rejection checks; no model or source photos."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('frozen_eval', ROOT / 'scripts/rgb/evaluate_frozen.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FrozenEvaluationTests(unittest.TestCase):
    def test_unknown_pixels_excluded_and_axes_correct(self):
        truth = np.array([[0, 0, 1], [2, 255, 255]], dtype=np.uint8)
        pred = np.array([[0, 1, 1], [0, 1, 2]], dtype=np.uint8)
        mat = MOD.confusion(truth, pred)
        np.testing.assert_array_equal(mat, [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        result = MOD.metrics(mat)
        self.assertEqual(result['valid_pixels'], 4)
        self.assertEqual(result['pixel_accuracy'], 0.5)
        self.assertAlmostEqual(result['classes'][0]['iou'], 1 / 3)
        self.assertEqual(result['classes'][1]['iou'], 0.5)
        self.assertEqual(result['classes'][2]['iou'], 0)
        self.assertAlmostEqual(result['macro_iou_nonempty_union'], 5 / 18)

    def test_absent_class_is_null_not_perfect(self):
        result = MOD.metrics(np.diag([3, 0, 0]))
        self.assertEqual(result['macro_iou_nonempty_union'], 1.0)
        self.assertIsNone(result['classes'][1]['iou'])
        self.assertIsNone(result['classes'][1]['recall'])
        self.assertIsNone(result['classes'][1]['precision'])

    def test_all_ignored_has_no_score(self):
        mat = MOD.confusion(np.full((2, 2), 255), np.zeros((2, 2), dtype=np.uint8))
        result = MOD.metrics(mat)
        self.assertEqual(result['valid_pixels'], 0)
        self.assertIsNone(result['pixel_accuracy'])
        self.assertIsNone(result['macro_iou_nonempty_union'])

    def test_predicted_only_class_counts_as_false_positive(self):
        result = MOD.metrics(np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]))
        self.assertEqual(result['classes'][1]['iou'], 0)
        self.assertEqual(result['macro_iou_nonempty_union'], 0)

    def test_invalid_classes_shapes_and_counts_rejected(self):
        for a, b in [(np.array([[7]]), np.array([[0]])), (np.array([[0]]), np.array([[255]])),
                     (np.array([[0]]), np.array([[0, 1]]))]:
            with self.assertRaises(ValueError):
                MOD.confusion(a, b)
        for mat in [np.zeros((3, 3), dtype=float), np.full((3, 3), -1), np.zeros((2, 2), dtype=int)]:
            with self.assertRaises(ValueError):
                MOD.metrics(mat)

    def test_changed_freeze_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'frozen.json'
            p.write_text('not valid json', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                MOD.verify_frozen(p, '0' * 64)

    def test_output_refuses_overwrite_without_starting_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            marker = out / 'preserved.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                MOD.run('unused', 'unused', out)
            self.assertEqual(marker.read_text(), 'keep')

    def test_duplicate_freeze_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'frozen.json'
            data = {'schema': MOD.SCHEMA, 'seed': MOD.SEED, 'size': MOD.SIZE,
                    'records': [{'sample_id': 'same'}] * MOD.SIZE}
            p.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                MOD.verify_frozen(p, MOD.digest(p))


if __name__ == '__main__':
    unittest.main()
