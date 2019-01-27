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

    def test_missing_speaker(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 7.1),
               ("A", 7.0, 9.0),
               (10.0, 13.0)]
        with self.assertRaises(TypeError):
            der.check_input(hyp)

    def test_wrong_speaker_type(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 7.1),
               ("A", 7.0, 9.0),
               (3, 10.0, 13.0)]
        with self.assertRaises(TypeError):
            der.check_input(hyp)

    def test_overlap(self):
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 7.1),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        with self.assertRaises(ValueError):
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


class TestDER(unittest.TestCase):
    """Tests for the DER function."""

    def test_single_tuple_same(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_single_tuple_all_miss(self):
        ref = [("A", 0.0, 1.0)]
        hyp = []
        self.assertEqual(1.0, der.DER(ref, hyp))

    def test_single_tuple_separate(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 1.0, 2.0)]
        self.assertEqual(2.0, der.DER(ref, hyp))

    def test_single_tuple_half_miss(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 0.5)]
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_single_tuple_half_different(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 0.5),
               ("C", 0.5, 1.0)]
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_two_tuples_same(self):
        ref = [("A", 0.0, 0.5),
               ("B", 0.5, 1.0)]
        hyp = [("B", 0.0, 0.5),
               ("C", 0.5, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_two_tuples_half_miss(self):
        ref = [("A", 0.0, 0.5),
               ("B", 0.5, 1.0)]
        hyp = [("B", 0.0, 0.25),
               ("C", 0.5, 0.75)]
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_two_tuples_one_quarter_correct(self):
        ref = [("A", 0.0, 0.5),
               ("B", 0.5, 1.0)]
        hyp = [("B", 0.0, 0.25),
               ("C", 0.25, 0.5)]
        self.assertEqual(0.75, der.DER(ref, hyp))

    def test_two_tuples_half_correct(self):
        ref = [("A", 0.0, 0.5),
               ("B", 0.5, 1.0)]
        hyp = [("B", 0.0, 0.25),
               ("C", 0.25, 0.5),
               ("D", 0.5, 0.75),
               ("E", 0.75, 1.0)]
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_three_tuples(self):
        ref = [("A", 0.0, 1.0),
               ("B", 1.0, 1.5),
               ("A", 1.6, 2.1)]
        hyp = [("1", 0.0, 0.8),
               ("2", 0.8, 1.4),
               ("3", 1.5, 1.8),
               ("1", 1.8, 2.0)]
        self.assertAlmostEqual(0.35, der.DER(ref, hyp), delta=0.0001)


if __name__ == "__main__":
    unittest.main()
