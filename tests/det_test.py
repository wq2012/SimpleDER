import numpy as np
import unittest

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
    """Tests for the DER function."""

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
        # Speaker A speaks continuously from 0.0 to 2.0 (represented as two
        # contiguous segments). With collar=0.25, only 0.0 and 2.0 are
        # boundaries (not 1.0). Scored region is [0.25, 1.75] (length 1.5).
        # Hypothesis misses [0.9, 1.1] (length 0.2 inside scored region).
        ref = [("A", 0.0, 1.0), ("A", 1.0, 2.0)]
        hyp = [("1", 0.0, 0.9), ("1", 1.1, 2.0)]
        self.assertAlmostEqual(
            0.2 / 1.5, der.DER(ref, hyp, collar=0.25), delta=0.0001)

    def test_skip_overlap(self):
        # Ref: A(0-2), B(1-3). Overlap is [1, 2].
        # With skip_overlap=True, scored regions are [0, 1] (A) and [2, 3] (B),
        # total ref_length = 2.0.
        # Hyp has 1(0-2): after removing [1, 2], hyp has 1(0-1).
        # Match = 1.0 (A->1), Miss = 1.0 (B on [2, 3]), DER = 1.0 / 2.0 = 0.5.
        ref = [("A", 0.0, 2.0), ("B", 1.0, 3.0)]
        hyp = [("1", 0.0, 2.0)]
        details = der.DER(ref, hyp, skip_overlap=True, detailed=True)
        self.assertAlmostEqual(0.5, details["der"], delta=0.0001)
        self.assertAlmostEqual(1.0, details["miss"], delta=0.0001)
        self.assertAlmostEqual(0.0, details["false_alarm"], delta=0.0001)
        self.assertAlmostEqual(0.0, details["confusion"], delta=0.0001)
        self.assertAlmostEqual(2.0, details["ref_length"], delta=0.0001)
        # Also test skip_overlap=True when there is no overlap in ref
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


class TestJER(unittest.TestCase):
    """Tests for the JER function."""

    def test_perfect_match(self):
        ref = [("A", 0.0, 1.0), ("B", 1.0, 2.0)]
        hyp = [("1", 0.0, 1.0), ("2", 1.0, 2.0)]
        self.assertEqual(0.0, simpleder.JER(ref, hyp))

    def test_empty_ref_and_hyp(self):
        self.assertEqual(0.0, simpleder.JER([], []))
        self.assertEqual(
            {"jer": 0.0, "speaker_jer": {}, "speaker_mapping": {}},
            simpleder.JER([], [("1", 0.0, 1.0)], detailed=True))

    def test_all_miss(self):
        ref = [("A", 0.0, 1.0), ("B", 1.0, 2.0)]
        hyp = []
        details = simpleder.JER(ref, hyp, detailed=True)
        self.assertEqual(1.0, details["jer"])
        self.assertEqual({"A": 1.0, "B": 1.0}, details["speaker_jer"])
        self.assertEqual({}, details["speaker_mapping"])

    def test_single_speaker_miss_and_fa(self):
        # Ref A: [0.0, 1.0], Hyp 1: [0.5, 1.5]
        # Intersection = 0.5, Union = 1.5 -> Jaccard = 1/3 -> JER = 2/3.
        ref = [("A", 0.0, 1.0)]
        hyp = [("1", 0.5, 1.5)]
        self.assertAlmostEqual(2.0 / 3.0, simpleder.JER(ref, hyp), delta=1e-6)

    def test_jer_vs_der_optimal_mapping_difference(self):
        # Ref A: [0, 10] (dur 10), Ref B: [10, 11] (dur 1)
        # Hyp 1: [8, 11] (dur 3)
        # J(A, 1) = 2 / 11; J(B, 1) = 1 / 3.
        # Since 1/3 > 2/11, JER maps B -> 1, giving JER = (1 + 2/3) / 2 = 5/6.
        ref = [("A", 0.0, 10.0), ("B", 10.0, 11.0)]
        hyp = [("1", 8.0, 11.0)]
        details = simpleder.JER(ref, hyp, detailed=True)
        self.assertAlmostEqual(5.0 / 6.0, details["jer"], delta=1e-6)
        self.assertEqual({"B": "1"}, details["speaker_mapping"])
        self.assertAlmostEqual(1.0, details["speaker_jer"]["A"], delta=1e-6)
        self.assertAlmostEqual(
            2.0 / 3.0, details["speaker_jer"]["B"], delta=1e-6)

    def test_invalid_detailed(self):
        with self.assertRaises(TypeError):
            simpleder.JER([("A", 0.0, 1.0)], [("1", 0.0, 1.0)], detailed=1)


class TestCorpusEvaluation(unittest.TestCase):
    """Tests for corpus_der and corpus_jer."""

    def test_corpus_der_dict_micro_average(self):
        # Rec 1: ref length 1.0, DER = 1.0 (error = 1.0)
        # Rec 2: ref length 9.0, DER = 0.0 (error = 0.0)
        # Micro-averaged Corpus DER = (1.0 + 0.0) / (1.0 + 9.0) = 0.1
        # (whereas unweighted mean would be 0.5).
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

    def test_corpus_der_list_and_empty(self):
        refs = [[("A", 0.0, 1.0)], [("B", 0.0, 1.0)]]
        hyps = [[("1", 0.0, 1.0)], []]
        details = simpleder.corpus_der(refs, hyps, detailed=True)
        self.assertAlmostEqual(0.5, details["der"], delta=1e-6)
        self.assertIsInstance(details["per_recording"], list)
        self.assertEqual(0.0, simpleder.corpus_der([], []))

    def test_corpus_jer_speaker_weighted(self):
        # Rec 1: 1 speaker with JER = 1.0
        # Rec 2: 3 speakers all with JER = 0.0
        # Corpus JER across all 4 ref speakers = 1.0 / 4 = 0.25.
        refs = [
            [("A", 0.0, 1.0)],
            [("B", 0.0, 1.0), ("C", 1.0, 2.0), ("D", 2.0, 3.0)],
        ]
        hyps = [
            [],
            [("1", 0.0, 1.0), ("2", 1.0, 2.0), ("3", 2.0, 3.0)],
        ]
        self.assertAlmostEqual(
            0.25, simpleder.corpus_jer(refs, hyps), delta=1e-6)
        details = simpleder.corpus_jer(refs, hyps, detailed=True)
        self.assertAlmostEqual(0.25, details["jer"], delta=1e-6)
        self.assertEqual(4, details["num_ref_speakers"])
        self.assertEqual(0.0, simpleder.corpus_jer({}, {}))

    def test_corpus_input_errors(self):
        with self.assertRaises(ValueError):
            simpleder.corpus_der({"a": []}, {"b": []})
        with self.assertRaises(ValueError):
            simpleder.corpus_der([[]], [[], []])
        with self.assertRaises(TypeError):
            simpleder.corpus_der({"a": []}, [[]])
        with self.assertRaises(TypeError):
            simpleder.corpus_der([], [], detailed="yes")
        with self.assertRaises(TypeError):
            simpleder.corpus_jer([], [], detailed="yes")


if __name__ == "__main__":
    unittest.main()
