"""Hand-counted checks for inclusion of unknown-region annotation margins."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('independent_audit', Path(__file__).with_name('audit.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class BoundaryMarginTests(unittest.TestCase):
    def test_unknown_margin_has_two_known_layers_only(self):
        a = np.zeros((5, 10), dtype=np.uint8)
        a[:, 5:] = 255
        expected = np.zeros_like(a, dtype=bool)
        expected[:, 3:5] = True
        np.testing.assert_array_equal(A.independent_band(a, True), expected)
        self.assertFalse(A.independent_band(a, False).any())

    def test_known_interface_two_sided_exact_layers(self):
        a = np.zeros((5, 10), dtype=np.uint8)
        a[:, 5:] = 2
        expected = np.zeros_like(a, dtype=bool)
        expected[:, 3:7] = True
        np.testing.assert_array_equal(A.independent_band(a, False), expected)
        np.testing.assert_array_equal(A.independent_band(a, True), expected)

    def test_no_frame_and_no_all_unknown_boundary(self):
        for a in [np.zeros((5, 10), dtype=np.uint8), np.full((5, 10), 255)]:
            self.assertFalse(A.independent_band(a, True).any())

    def test_horizontal_interface_obeys_same_rule(self):
        a = np.zeros((10, 5), dtype=np.uint8)
        a[5:] = 1
        expected = np.zeros_like(a, dtype=bool)
        expected[3:7] = True
        np.testing.assert_array_equal(A.independent_band(a, False), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
