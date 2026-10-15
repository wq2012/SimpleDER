import io
import os
import tempfile
import unittest

import simpleder
from simpleder import cli


class TestRTTMAndCLI(unittest.TestCase):
    """Tests for load_rttm, dump_rttm, and the command-line interface."""

    def test_dump_and_load_rttm_stream_and_file(self):
        import simpleder.__main__  # noqa: F401
        ref_segments = [("A", 0.0, 1.0), ("B", 1.0, 2.5)]
        buf = io.StringIO()
        simpleder.dump_rttm(ref_segments, buf, recording_id="rec1")
        buf.write(
            "# comment line\n\nSPKR-INFO rec1 1 <NA> <NA> <NA> unknown\n")
        buf.seek(0)
        loaded = simpleder.load_rttm(buf)
        self.assertEqual({"rec1": ref_segments}, loaded)

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.rttm")
            hyp_path = os.path.join(tmpdir, "hyp.rttm")
            simpleder.dump_rttm(
                {"rec1": ref_segments, "rec2": [("A", 0.0, 2.0)]}, ref_path)
            simpleder.dump_rttm(
                {"rec1": [("1", 0.0, 1.0), ("2", 1.0, 2.5)]}, hyp_path)

            res = cli.main(
                ["--ref", ref_path, "--hyp", hyp_path, "--detailed"])
            self.assertAlmostEqual(2.0 / 4.5, res["der"], delta=1e-5)

            res_json = cli.main(
                ["--ref", ref_path, "--hyp", hyp_path, "--json"])
            self.assertAlmostEqual(2.0 / 4.5, res_json["der"], delta=1e-5)

            bad_hyp_path = os.path.join(tmpdir, "bad_hyp.rttm")
            simpleder.dump_rttm({"unknown_rec": ref_segments}, bad_hyp_path)
            with self.assertRaises(ValueError):
                cli.main(["--ref", ref_path, "--hyp", bad_hyp_path])

    def test_rttm_invalid_inputs(self):
        with self.assertRaises(TypeError):
            simpleder.load_rttm(123)
        with self.assertRaises(ValueError):
            simpleder.load_rttm(["SPEAKER rec1 1 0.0\n"])
        with self.assertRaises(ValueError):
            simpleder.load_rttm(
                ["SPEAKER rec1 1 0.0 -1.0 <NA> <NA> spk1 <NA>\n"])
        with self.assertRaises(TypeError):
            simpleder.dump_rttm("invalid", io.StringIO())
        with self.assertRaises(TypeError):
            simpleder.dump_rttm([("A", 0.0, 1.0)], 123)


if __name__ == "__main__":
    unittest.main()
