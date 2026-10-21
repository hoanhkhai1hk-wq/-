"""Deterministic, hand-counted in-memory tests; no source data or model access."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('diagnose', Path(__file__).with_name('diagnose.py'))
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


class DiagnosisTests(unittest.TestCase):
    def test_exact_two_layers_on_both_sides(self):
        truth = np.zeros((10, 20), dtype=np.uint8)
        truth[:, 10:] = 1
        expected = np.zeros_like(truth, dtype=bool)
        expected[:, 8:12] = True
        np.testing.assert_array_equal(D.boundary_band(truth), expected)

    def test_no_artificial_image_frame(self):
        self.assertFalse(D.boundary_band(np.zeros((8, 8), dtype=np.uint8)).any())

    def test_unknown_interfaces_not_class_boundaries(self):
        truth = np.zeros((8, 8), dtype=np.uint8)
        truth[:, 4:] = 255
        self.assertFalse(D.boundary_band(truth).any())

    def test_four_categories_are_disjoint_and_sum_to_errors(self):
        truth = np.zeros((4, 12), dtype=np.uint8)
        truth[:, 6:] = 1
        pred = truth.copy()
        pred[0, 0] = 1  # a outside band
        pred[1, 0] = 2  # b outside band
        pred[2, 5] = 1  # a inside band -> c
        pred[0, 6] = 0  # other inside band -> c
        pred[1, 11] = 2  # other outside band -> d
        truth[3, 0] = 255
        pred[3, 0] = 2  # ignored
        result = D.classify(truth, pred)
        self.assertEqual(result['errors'], 5)
        self.assertEqual(result['gross_a_field_to_ridge'], 2)
        self.assertEqual(result['gross_b_field_to_road'], 1)
        self.assertEqual([result[k] for k in ['a_field_to_ridge_outside_band', 'b_field_to_road_outside_band', 'c_boundary_band_error', 'd_other_error_outside_band']], [1, 1, 2, 1])
        self.assertFalse(result['all_errors_confined_to_band'])

    def test_pure_boundary_and_zero_error_differ(self):
        truth = np.zeros((4, 12), dtype=np.uint8)
        truth[:, 6:] = 1
        self.assertFalse(D.classify(truth, truth)['all_errors_confined_to_band'])
        pred = truth.copy()
        pred[:, 5] = 1
        self.assertTrue(D.classify(truth, pred)['all_errors_confined_to_band'])

    def test_width_proxy_is_hand_calculable(self):
        self.assertIsNone(D.equivalent_width(np.zeros((3, 3), dtype=bool)))
        self.assertEqual(D.equivalent_width(np.ones((2, 6), dtype=bool)), 1.5)

    def test_invalid_classes_and_shape_rejected(self):
        with self.assertRaises(ValueError):
            D.boundary_band(np.array([[3]]))
        with self.assertRaises(ValueError):
            D.classify(np.zeros((2, 2), dtype=int), np.zeros((2, 3), dtype=int))
        with self.assertRaises(ValueError):
            D.classify(np.zeros((2, 2), dtype=int), np.full((2, 2), 255))


if __name__ == '__main__':
    unittest.main(verbosity=2)
