import unittest
import numpy as np

import simpleder
from simpleder import der


class TestCheckInput(unittest.TestCase):
    """Tests for the check_input function."""

    def test_valid(self):
        ref = [("A", 1.0, 2.0),
               ("B", 4.0, 5.0),
               ("A", 6.7, 9.0),
               ("C", 10.0, 12.0),
               ("D", 12, 13)]
        hyp = [("A", 1.0, 3.0),
               ("B", 4.0, 4.8),
               ("A", 7.0, 9.0),
               ("C", 10.0, 13.0)]
        der.check_input(ref)
        der.check_input(hyp)

    def test_not_a_list(self):
        with self.assertRaises(TypeError):
            der.check_input(("A", 1.0, 2.0))

    def test_element_not_a_tuple(self):
        with self.assertRaises(TypeError):
            der.check_input([["A", 1.0, 2.0]])

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

    def test_wrong_timestamp_type(self):
        with self.assertRaises(TypeError):
            der.check_input([("A", "1.0", 2.0)])
        with self.assertRaises(TypeError):
            der.check_input([("A", True, 2.0)])

    def test_non_finite_timestamp(self):
        with self.assertRaises(ValueError):
            der.check_input([("A", float("nan"), 2.0)])
        with self.assertRaises(ValueError):
            der.check_input([("A", 1.0, float("inf"))])

    def test_start_greater_than_end(self):
        with self.assertRaises(ValueError):
            der.check_input([("A", 3.0, 2.0)])


class TestMergeSpeakerIntervals(unittest.TestCase):
    """Tests for the merge_speaker_intervals function."""

    def test_overlap_and_contiguous_same_speaker(self):
        segments = [
            ("A", 0.0, 2.0),
            ("A", 1.0, 3.0),
            ("A", 3.0, 4.0),
            ("B", 1.5, 2.5),
            ("A", 5.0, 5.0),
        ]
        expected = [
            ("A", 0.0, 4.0),
            ("B", 1.5, 2.5),
        ]
        self.assertEqual(expected, der.merge_speaker_intervals(segments))


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

    def test_same_speaker_overlap_not_double_counted(self):
        hyp = [("A", 0.0, 2.0), ("A", 1.0, 3.0)]
        self.assertEqual(3.0, der.compute_total_length(hyp))


class TestComputeLoadLength(unittest.TestCase):
    """Tests for the compute_load_length function."""

    def test_example(self):
        ref = [("A", 0.0, 1.0),
               ("B", 1.0, 2.0)]
        hyp = [("A", 0.0, 1.0),
               ("B", 1.0, 2.0)]
        self.assertEqual(2.0, der.compute_load_length(ref, hyp))

    def test_overlap(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        self.assertEqual(2.0, der.compute_load_length(ref, hyp))

    def test_overlap_miss(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0)]
        self.assertEqual(2.0, der.compute_load_length(ref, hyp))


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
    """Tests for the DER and corpus_der functions."""

    def test_single_tuple_same(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 1.0)]
        self.assertEqual(0.0, simpleder.DER(ref, hyp))

    def test_single_tuple_all_miss(self):
        ref = [("A", 0.0, 1.0)]
        hyp = []
        self.assertEqual(1.0, der.DER(ref, hyp))

    def test_empty_ref(self):
        self.assertEqual(0.0, der.DER([], []))
        self.assertEqual(0.0, der.DER([], [("A", 0.0, 1.0)]))
        details = der.DER([], [("A", 0.0, 1.0)], detailed=True)
        self.assertEqual(0.0, details["der"])
        self.assertEqual(1.0, details["false_alarm"])
        self.assertEqual(0.0, details["ref_length"])

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

    def test_three_tuples_detailed_breakdown(self):
        ref = [("A", 0.0, 1.0),
               ("B", 1.0, 1.5),
               ("A", 1.6, 2.1)]
        hyp = [("1", 0.0, 0.8),
               ("2", 0.8, 1.4),
               ("3", 1.5, 1.8),
               ("1", 1.8, 2.0)]
        details = der.DER(ref, hyp, detailed=True)
        self.assertAlmostEqual(0.35, details["der"], delta=0.0001)
        self.assertAlmostEqual(0.2, details["miss"], delta=0.0001)
        self.assertAlmostEqual(0.1, details["false_alarm"], delta=0.0001)
        self.assertAlmostEqual(0.4, details["confusion"], delta=0.0001)
        self.assertAlmostEqual(2.0, details["ref_length"], delta=0.0001)
        self.assertEqual({"A": "1", "B": "2"}, details["speaker_mapping"])

    def test_hyp_has_more_labels(self):
        ref = [("0", 0.0, 1.0),
               ("1", 1.0, 2.0),
               ("1", 2.0, 3.0),
               ("1", 3.0, 4.0),
               ("0", 4.0, 5.0)]
        hyp = [("0", 0.0, 1.0),
               ("2", 1.0, 2.0),
               ("2", 2.0, 3.0),
               ("0", 3.0, 4.0),
               ("2", 4.0, 5.0)]
        self.assertAlmostEqual(0.4, der.DER(ref, hyp), delta=0.0001)

    def test_ref_has_more_labels(self):
        ref = [("0", 0.0, 1.0),
               ("2", 1.0, 2.0),
               ("2", 2.0, 3.0),
               ("0", 3.0, 4.0),
               ("1", 4.0, 5.0)]
        hyp = [("0", 0.0, 1.0),
               ("1", 1.0, 2.0),
               ("1", 2.0, 3.0),
               ("1", 3.0, 4.0),
               ("0", 4.0, 5.0)]
        self.assertAlmostEqual(0.4, der.DER(ref, hyp), delta=0.0001)

    def test_overlap_perfect(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0),
               ("2", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_overlap_miss(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0)]
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_overlap_fa(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0),
               ("2", 0.0, 1.0)]
        self.assertEqual(1.0, der.DER(ref, hyp))

    def test_overlap_confusion(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_overlap_confusion_enforced(self):
        ref = [("A", 0.0, 10.0), ("A", 20.0, 21.0)]
        hyp = [("1", 0.0, 10.0), ("2", 20.0, 21.0)]
        self.assertAlmostEqual(1.0 / 11.0, der.DER(ref, hyp), delta=0.0001)

    def test_many_speakers_overlap(self):
        ref = [("A", 0.0, 1.0), ("B", 0.0, 1.0), ("C", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0), ("2", 0.0, 1.0)]
        self.assertAlmostEqual(1.0 / 3.0, der.DER(ref, hyp), delta=0.0001)

    def test_same_speaker_duplicate_segments_no_negative_der(self):
        ref = [("A", 0.0, 1.0), ("A", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0), ("1", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_collar_simple(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.1))

    def test_collar_miss_boundary(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 0.8)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.25))

    def test_collar_false_alarm_boundary(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.2)]
        self.assertAlmostEqual(0.125,
                               der.DER(ref, hyp, collar=0.1),
                               delta=0.0001)

    def test_collar_adjacent(self):
        ref = [("A", 0.0, 1.0), ("B", 1.0, 2.0)]
        hyp = [("A", 0.0, 1.05), ("B", 1.05, 2.0)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.1))

    def test_collar_same_speaker_contiguous_not_excluded(self):
        ref = [("A", 0.0, 1.0), ("A", 1.0, 2.0)]
        hyp = [("1", 0.0, 0.9), ("1", 1.1, 2.0)]
        self.assertAlmostEqual(
            0.2 / 1.5, der.DER(ref, hyp, collar=0.25), delta=0.0001)

    def test_skip_overlap(self):
        ref = [("A", 0.0, 2.0), ("B", 1.0, 3.0)]
        hyp = [("1", 0.0, 2.0)]
        details = der.DER(ref, hyp, skip_overlap=True, detailed=True)
        self.assertAlmostEqual(0.5, details["der"], delta=0.0001)
        self.assertAlmostEqual(1.0, details["miss"], delta=0.0001)
        self.assertAlmostEqual(0.0, details["false_alarm"], delta=0.0001)
        self.assertAlmostEqual(0.0, details["confusion"], delta=0.0001)
        self.assertAlmostEqual(2.0, details["ref_length"], delta=0.0001)
        self.assertEqual(
            0.0, der.DER([("A", 0.0, 1.0)], [("1", 0.0, 1.0)],
                         skip_overlap=True))

    def test_invalid_options(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0)]
        with self.assertRaises(ValueError):
            der.DER(ref, hyp, collar=-0.1)
        with self.assertRaises(TypeError):
            der.DER(ref, hyp, collar="0.1")
        with self.assertRaises(TypeError):
            der.DER(ref, hyp, skip_overlap="yes")
        with self.assertRaises(TypeError):
            der.DER(ref, hyp, detailed="yes")

    def test_corpus_der(self):
        refs = {
            "rec1": [("A", 0.0, 1.0)],
            "rec2": [("B", 0.0, 9.0)],
        }
        hyps = {
            "rec1": [],
            "rec2": [("1", 0.0, 9.0)],
        }
        self.assertAlmostEqual(
            0.1, simpleder.corpus_der(refs, hyps), delta=1e-6)
        details = simpleder.corpus_der(refs, hyps, detailed=True)
        self.assertAlmostEqual(0.1, details["der"], delta=1e-6)
        self.assertAlmostEqual(1.0, details["miss"], delta=1e-6)
        self.assertAlmostEqual(10.0, details["ref_length"], delta=1e-6)
        self.assertIn("rec1", details["per_recording"])
        self.assertIn("rec2", details["per_recording"])

        refs_list = [[("A", 0.0, 1.0)], [("B", 0.0, 1.0)]]
        hyps_list = [[("1", 0.0, 1.0)], []]
        details_list = simpleder.corpus_der(
            refs_list, hyps_list, detailed=True)
        self.assertAlmostEqual(0.5, details_list["der"], delta=1e-6)
        self.assertIsInstance(details_list["per_recording"], list)
        self.assertEqual(0.0, simpleder.corpus_der([], []))

        with self.assertRaises(ValueError):
            simpleder.corpus_der({"a": []}, {"b": []})
        with self.assertRaises(ValueError):
            simpleder.corpus_der([[]], [[], []])
        with self.assertRaises(TypeError):
            simpleder.corpus_der({"a": []}, [[]])
        with self.assertRaises(TypeError):
            simpleder.corpus_der([], [], detailed="yes")


class TestComputeMergedExclusionIntervals(unittest.TestCase):
    def test_basic(self):
        ref = [("A", 10.0, 20.0)]
        collar = 1.0
        expected = [(9.0, 11.0), (19.0, 21.0)]
        self.assertEqual(expected,
                         der.compute_merged_exclusion_intervals(ref, collar))

    def test_merge(self):
        ref = [("A", 10.0, 12.0)]
        collar = 1.5
        expected = [(8.5, 13.5)]
        self.assertEqual(expected,
                         der.compute_merged_exclusion_intervals(ref, collar))

    def test_zero_and_invalid_collar(self):
        ref = [("A", 10.0, 12.0)]
        self.assertEqual([], der.compute_merged_exclusion_intervals(ref, 0.0))
        with self.assertRaises(TypeError):
            der.compute_merged_exclusion_intervals(ref, True)
        with self.assertRaises(ValueError):
            der.compute_merged_exclusion_intervals(ref, -1.0)


class TestSubtractIntervals(unittest.TestCase):
    def test_basic(self):
        segments = [("A", 0.0, 10.0)]
        exclusions = [(0.0, 1.0), (9.0, 10.0)]
        expected = [("A", 1.0, 9.0)]
        self.assertEqual(expected,
                         der.subtract_intervals(segments, exclusions))

    def test_split(self):
        segments = [("A", 0.0, 10.0)]
        exclusions = [(4.0, 6.0)]
        expected = [("A", 0.0, 4.0), ("A", 6.0, 10.0)]
        self.assertEqual(expected,
                         der.subtract_intervals(segments, exclusions))

    def test_empty_exclusions_and_non_overlapping(self):
        segments = [("A", 2.0, 4.0)]
        self.assertEqual(segments, der.subtract_intervals(segments, []))
        exclusions = [(0.0, 1.0), (5.0, 6.0)]
        self.assertEqual(segments,
                         der.subtract_intervals(segments, exclusions))


if __name__ == "__main__":
    unittest.main()
