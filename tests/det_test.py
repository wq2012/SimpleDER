from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from simpleder import der


class TestComputeIntersectionLength(unittest.TestCase):
    """Tests for the compute_intersection_length function."""

    def test_include(self):
        A = ("A", 1.0, 5.0)
        B = ("B", 2.0, 3.0)
        self.assertEqual(1.0, der.compute_intersection_length(A, B))

    def test_separate(self):
        A = ("A", 1.0, 3.0)
        B = ("B", 5.0, 7.0)
        self.assertEqual(0.0, der.compute_intersection_length(A, B))

    def test_overlap(self):
        A = ("A", 1.0, 5.0)
        B = ("B", 2.0, 9.0)
        self.assertEqual(3.0, der.compute_intersection_length(A, B))


class TestcomputeTotalLength(unittest.TestCase):
    """Tests for the compute_total_length function."""

    def test_example(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 5.0),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        self.assertEqual(8.0, der.compute_total_length(hyp))


if __name__ == "__main__":
    unittest.main()
