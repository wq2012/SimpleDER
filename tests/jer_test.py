import unittest

import simpleder


class TestJER(unittest.TestCase):
    """Tests for the JER and corpus_jer functions."""

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
        ref = [("A", 0.0, 1.0)]
        hyp = [("1", 0.5, 1.5)]
        self.assertAlmostEqual(2.0 / 3.0, simpleder.JER(ref, hyp), delta=1e-6)

    def test_jer_vs_der_optimal_mapping_difference(self):
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

    def test_corpus_jer_speaker_weighted(self):
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
        with self.assertRaises(TypeError):
            simpleder.corpus_jer([], [], detailed="yes")


if __name__ == "__main__":
    unittest.main()
