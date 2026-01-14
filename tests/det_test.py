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


class TestComputeLoadLength(unittest.TestCase):
    """Tests for the compute_load_length function."""

    def test_example(self):
        ref = [("A", 0.0, 1.0),
               ("B", 1.0, 2.0)]
        hyp = [("A", 0.0, 1.0),
               ("B", 1.0, 2.0)]
        # Load length should be 2.0
        self.assertEqual(2.0, der.compute_load_length(ref, hyp))

    def test_overlap(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        # Two speakers active from 0 to 1. Max active is 2.
        # Length is 1 * 2 = 2.0
        self.assertEqual(2.0, der.compute_load_length(ref, hyp))

    def test_overlap_miss(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0)]
        # Ref has 2 speakers, Hyp has 1. Max is 2.
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
        # Perfect match. DER = 0.
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_overlap_miss(self):
        ref = [("A", 0.0, 1.0),
               ("B", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0)]
        # Load length is 2.0.
        # Match is 1.0 (one speaker matched).
        # Ref total length is 2.0.
        # DER = (2.0 - 1.0) / 2.0 = 0.5
        self.assertEqual(0.5, der.DER(ref, hyp))

    def test_overlap_fa(self):
        ref = [("A", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0),
               ("2", 0.0, 1.0)]
        # Load length is 2.0 (Hyp has 2 speakers).
        # Match is 1.0.
        # Ref total length is 1.0.
        # DER = (2.0 - 1.0) / 1.0 = 1.0
        self.assertEqual(1.0, der.DER(ref, hyp))

    def test_overlap_confusion(self):
        # Case where optimal mapping cannot match everyone
        ref = [("A", 0.0, 1.0)]
        hyp = [("B", 0.0, 1.0)]
        # If A can map to B, DER is 0.
        self.assertEqual(0.0, der.DER(ref, hyp))

    def test_overlap_confusion_enforced(self):
        # Enforce confusion by having A match 1 elsewhere better
        # Ref: A(0-10), A(20-21)
        # Hyp: 1(0-10), 2(20-21)
        # Mapping A->1 gives match 10.
        # Then at 20-21: Ref A vs Hyp 2.
        # Since A->1, A cannot match 2.
        # So at 20-21: N_ref=1, N_hyp=1. Match=0.
        # Load=1.
        # Error = 1.
        # Segment 0-10: Error = 0.
        # Total Error = 1.
        # Ref Total = 11.
        # DER = 1/11.
        ref = [("A", 0.0, 10.0), ("A", 20.0, 21.0)]
        hyp = [("1", 0.0, 10.0), ("2", 20.0, 21.0)]
        self.assertAlmostEqual(1.0 / 11.0, der.DER(ref, hyp), delta=0.0001)

    def test_many_speakers_overlap(self):
        # Ref: A, B, C (all 0-1)
        # Hyp: 1, 2 (all 0-1)
        # Match: 2 (A->1, B->2). C is Missed.
        # Load: 3 (max(3, 2)).
        # DER = (3 - 2) / 3 = 1/3.
        ref = [("A", 0.0, 1.0), ("B", 0.0, 1.0), ("C", 0.0, 1.0)]
        hyp = [("1", 0.0, 1.0), ("2", 0.0, 1.0)]
        self.assertAlmostEqual(1.0 / 3.0, der.DER(ref, hyp), delta=0.0001)

    def test_collar_simple(self):
        # Ref: 0.0-1.0. Collar 0.1 -> Exclude [-0.1, 0.1] and [0.9, 1.1]
        # Valid Ref: 0.1-0.9 (Length 0.8)
        # Hyp: 0.0-1.0. Valid Hyp: 0.1-0.9 (Length 0.8)
        # DER = 0.
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.0)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.1))

    def test_collar_miss_boundary(self):
        # Ref: 0.0-1.0. Collar 0.25 -> Exclude [-0.25, 0.25] and [0.75, 1.25].
        # Valid Ref: 0.25-0.75 (Length 0.5)
        # Hyp: 0.0-0.8.
        # Valid Hyp: [0.0, 0.8] - Exclusions
        # = [0.25, 0.75] intersect [0.0, 0.8] = [0.25, 0.75].
        # (Hyp also trimmed by 0.75-1.25 exclusion)
        # Hyp 0.75-0.8 is in exclusion.
        # So Matches = 0.5. Load = 0.5. DER = 0.
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 0.8)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.25))

    def test_collar_false_alarm_boundary(self):
        # Ref: 0.0-1.0. Collar 0.1 -> Valid 0.1-0.9.
        # Hyp: 0.0-1.2.
        # Valid Hyp: 0.1-0.9 from Ref(0-1), and 1.1-1.2?
        # Exclusion: [-0.1, 0.1], [0.9, 1.1].
        # Hyp has 1.1-1.2 left?
        # WAIT: Exclusion is based on Ref boundaries.
        # Ref boundaries: 0.0, 1.0.
        # Hyp part 1.1-1.2 is NOT in exclusion. So it is FA.
        # Matches: 0.1-0.9 (0.8).
        # Load: 0.1-0.9 (0.8) + 1.1-1.2 (0.1) = 0.9.
        # DER = (0.9 - 0.8) / 0.8 = 0.1/0.8 = 0.125
        ref = [("A", 0.0, 1.0)]
        hyp = [("A", 0.0, 1.2)]
        self.assertAlmostEqual(0.125,
                               der.DER(ref, hyp, collar=0.1),
                               delta=0.0001)

    def test_collar_adjacent(self):
        # Ref: A(0-1), B(1-2). Collar 0.1.
        # Boundaries: 0, 1, 2.
        # Exclusions: [-0.1, 0.1], [0.9, 1.1], [1.9, 2.1].
        # Ref Valid: A(0.1-0.9), B(1.1-1.9). Total 1.6.
        # Hyp: A(0-1.05), B(1.05-2).
        # Hyp Valid:
        # A: (0.1-0.9). 0.9-1.05 is excluded (0.9-1.1). So A contributes 0.8.
        # B: 1.05-2. 1.05-1.1 excluded. 1.1-1.9 valid. 1.9-2 excluded.
        # B contributes 0.8.
        # Total Match: 1.6.
        # DER = 0.
        ref = [("A", 0.0, 1.0), ("B", 1.0, 2.0)]
        hyp = [("A", 0.0, 1.05), ("B", 1.05, 2.0)]
        self.assertEqual(0.0, der.DER(ref, hyp, collar=0.1))


class TestComputeMergedExclusionIntervals(unittest.TestCase):
    def test_basic(self):
        ref = [("A", 10.0, 20.0)]
        collar = 1.0
        # 10->[9,11], 20->[19,21]
        expected = [(9.0, 11.0), (19.0, 21.0)]
        self.assertEqual(expected,
                         der.compute_merged_exclusion_intervals(ref, collar))

    def test_merge(self):
        ref = [("A", 10.0, 12.0)]
        collar = 1.5
        # 10->[8.5, 11.5], 12->[10.5, 13.5]
        # Overlap: 10.5 < 11.5. Merge to [8.5, 13.5]
        expected = [(8.5, 13.5)]
        self.assertEqual(expected,
                         der.compute_merged_exclusion_intervals(ref, collar))


class TestSubtractIntervals(unittest.TestCase):
    def test_basic(self):
        segments = [("A", 0.0, 10.0)]
        exclusions = [(0.0, 1.0), (9.0, 10.0)]
        # Should result in (1.0, 9.0)
        expected = [("A", 1.0, 9.0)]
        self.assertEqual(expected,
                         der.subtract_intervals(segments, exclusions))

    def test_split(self):
        segments = [("A", 0.0, 10.0)]
        exclusions = [(4.0, 6.0)]
        # Should result in (0.0, 4.0), (6.0, 10.0)
        expected = [("A", 0.0, 4.0), ("A", 6.0, 10.0)]
        self.assertEqual(expected,
                         der.subtract_intervals(segments, exclusions))


if __name__ == "__main__":
    unittest.main()
