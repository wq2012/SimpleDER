import argparse
import json

from . import der
from . import jer
from . import rttm


def main(argv=None):
    """Run the SimpleDER command-line interface on reference/hypothesis RTTMs.

    Args:
        argv: optional list of command-line argument strings

    Returns:
        a dict containing the computed corpus DER and JER results
    """
    parser = argparse.ArgumentParser(
        prog="simpleder",
        description="Compute DER and JER from NIST RTTM files.",
    )
    parser.add_argument(
        "-r", "--ref", required=True, help="Path to reference RTTM file.")
    parser.add_argument(
        "-p", "--hyp", required=True, help="Path to hypothesis RTTM file.")
    parser.add_argument(
        "-c", "--collar", type=float, default=0.0,
        help="No-score collar (in seconds) around reference boundaries.")
    parser.add_argument(
        "-s", "--skip-overlap", action="store_true",
        help="Exclude regions where multiple reference speakers are active.")
    parser.add_argument(
        "-d", "--detailed", action="store_true",
        help="Show detailed component breakdown and per-recording results.")
    parser.add_argument(
        "--json", action="store_true",
        help="Format output as JSON.")

    args = parser.parse_args(argv)
    refs = rttm.load_rttm(args.ref)
    hyps = rttm.load_rttm(args.hyp)

    unknown_recs = set(hyps.keys()) - set(refs.keys())
    if unknown_recs:
        raise ValueError(
            f"Hypothesis contains recordings not in reference: "
            f"{sorted(unknown_recs)}"
        )
    aligned_hyps = {rec_id: hyps.get(rec_id, []) for rec_id in refs}

    der_res = der.corpus_der(
        refs,
        aligned_hyps,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
        detailed=True,
    )
    jer_res = jer.corpus_jer(
        refs,
        aligned_hyps,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
        detailed=True,
    )

    output = {
        "der": der_res["der"],
        "jer": jer_res["jer"],
        "miss": der_res["miss"],
        "false_alarm": der_res["false_alarm"],
        "confusion": der_res["confusion"],
        "ref_length": der_res["ref_length"],
        "num_ref_speakers": jer_res["num_ref_speakers"],
    }
    if args.detailed:
        output["per_recording_der"] = der_res["per_recording"]
        output["per_recording_jer"] = jer_res["per_recording"]

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"DER: {output['der']:.4f} | JER: {output['jer']:.4f}")
        if args.detailed:
            print(
                f"Miss: {output['miss']:.3f}s | "
                f"False Alarm: {output['false_alarm']:.3f}s | "
                f"Confusion: {output['confusion']:.3f}s | "
                f"Ref Length: {output['ref_length']:.3f}s"
            )
            for rec_id in sorted(refs.keys()):
                r_der = der_res["per_recording"][rec_id]["der"]
                r_jer = jer_res["per_recording"][rec_id]["jer"]
                print(f"  [{rec_id}] DER={r_der:.4f}, JER={r_jer:.4f}")

    return output
