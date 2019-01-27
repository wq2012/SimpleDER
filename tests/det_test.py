from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest

from simpleder import der


class TestCheckInput(unittest.TestCase):
    """Tests for the check_input function."""

    def test_valid(self):
        ref = [("A", 1.0, 2.0),
               ("B", 4.0, 5.0),
               ("A", 6.7, 9.0),
               ("C", 10.0, 12.0),
               ("D", 12.0, 13.0)]
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 4.8),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        der.check_input(ref)
        der.check_input(hyp)


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


class TestComputeTotalLength(unittest.TestCase):
    """Tests for the compute_total_length function."""

    def test_example(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 5.0),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        self.assertEqual(8.0, der.compute_total_length(hyp))


class TestComputeMergedTotalLength(unittest.TestCase):
    """Tests for the compute_merged_total_length function."""

    def test_example(self):
        ref = [("A", 1.0, 2.0),
               ("B", 4.0, 5.0),
               ("A", 6.7, 9.0),
               ("C", 10.0, 12.0),
               ("D", 12.0, 13.0)]
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 4.8),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        merged_total_length = der.compute_merged_total_length(ref, hyp)
        self.assertEqual(8.3, merged_total_length)


class TestBuildSpeakerIndex(unittest.TestCase):
    """Tests for the build_speaker_index function."""

    def test_example(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 5.0),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        expected = {
            "A": 0,
            "B": 1,
            "C": 2
        }
        hyp_index = der.build_speaker_index(hyp)
        self.assertDictEqual(expected, hyp_index)


class TestBuildCostMatrix(unittest.TestCase):
    """Tests for the build_cost_matrix function."""

    def test_example(self):
        ref = [("A", 1.0, 2.0),
               ("B", 4.0, 4.8),
               ("A", 6.7, 9.0),
               ("C", 10.0, 12.0),
               ("D", 12.0, 13.0)]
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 5.0),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        expected = np.array(
            [[3.0,  0.0,  0.0],
             [0.0,  0.8, 0.0],
             [0.0,  0.0, 2.0],
             [0.0, 0.0,  1.0]])
        cost_matrix = der.build_cost_matrix(ref, hyp)
        self.assertTrue(np.allclose(expected, cost_matrix, atol=0.0001))


if __name__ == "__main__":
    unittest.main()
