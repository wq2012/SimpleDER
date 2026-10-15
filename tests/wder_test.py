import unittest

import simpleder
from simpleder import wder


class TestWDER(unittest.TestCase):
    """Tests for WDER, check_word_input, normalize_text, and corpus_wder."""

    def test_word_tuples_and_phrase_tuples(self):
        ref_words = [
            ("A", "hello"),
            ("A", "world"),
            ("B", "how"),
            ("B", "are"),
            ("B", "you"),
        ]
        hyp_words = [
            ("1", "hello"),
            ("2", "world"),
            ("2", "how"),
            ("2", "are"),
            ("2", "you"),
        ]
        self.assertAlmostEqual(0.2, simpleder.WDER(ref_words, hyp_words))

        ref_phrases = [("A", "Hello, world!"), ("B", "How are you?")]
        hyp_phrases = [("1", "hello"), ("2", "world, how are you.")]
        details = simpleder.WDER(ref_phrases, hyp_phrases, detailed=True)
        self.assertAlmostEqual(0.2, details["wder"])
        self.assertEqual(1, details["wder_sub"])
        self.assertEqual(4, details["wder_correct"])
        self.assertEqual(5, details["wder_total"])
        self.assertEqual(0.0, details["wer"])
        self.assertEqual({"A": "1", "B": "2"}, details["speaker_mapping"])

    def test_wder_with_asr_ins_del_sub(self):
        ref = [("A", "very good morning"), ("B", "everyone here")]
        hyp = [("1", "good evening"), ("2", "everyone right here")]
        details = simpleder.WDER(ref, hyp, detailed=True)
        self.assertEqual(0.0, details["wder"])
        self.assertEqual(4, details["wder_total"])
        self.assertEqual(1, details["wer_insert"])
        self.assertEqual(1, details["wer_delete"])
        self.assertEqual(1, details["wer_sub"])
        self.assertEqual(3, details["wer_correct"])
        self.assertEqual(5, details["wer_total"])
        self.assertAlmostEqual(3.0 / 5.0, details["wer"])

    def test_wder_empty_and_standalone_punctuation(self):
        self.assertEqual(0.0, simpleder.WDER([], []))
        self.assertEqual(0.0, simpleder.WDER([("A", "hello")], []))
        self.assertEqual("...", wder.normalize_text("..."))

    def test_wder_input_validation(self):
        with self.assertRaises(TypeError):
            wder.check_word_input("not a list")
        with self.assertRaises(TypeError):
            wder.check_word_input([["A", "hello"]])
        with self.assertRaises(TypeError):
            wder.check_word_input([("A", "hello", 1.0)])
        with self.assertRaises(TypeError):
            wder.check_word_input([(1, "hello")])
        with self.assertRaises(TypeError):
            wder.check_word_input([("A", 123)])
        with self.assertRaises(TypeError):
            simpleder.WDER([], [], detailed="yes")

    def test_corpus_wder(self):
        refs = {
            "utt1": [("A", "hello world"), ("B", "how are you")],
            "utt2": [("A", "good morning"), ("B", "thank you")],
        }
        hyps = {
            "utt1": [("1", "hello"), ("2", "world how are you")],
            "utt2": [("1", "good morning"), ("2", "thank you")],
        }
        self.assertAlmostEqual(1.0 / 9.0, simpleder.corpus_wder(refs, hyps))
        details = simpleder.corpus_wder(refs, hyps, detailed=True)
        self.assertAlmostEqual(1.0 / 9.0, details["wder"])
        self.assertEqual(1, details["wder_sub"])
        self.assertEqual(8, details["wder_correct"])
        self.assertEqual(9, details["wder_total"])
        self.assertEqual(0.0, simpleder.corpus_wder([], []))
        with self.assertRaises(TypeError):
            simpleder.corpus_wder([], [], detailed="yes")


if __name__ == "__main__":
    unittest.main()
