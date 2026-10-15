import numpy as np
from scipy import optimize

from . import der


def JER(ref, hyp, collar=0.0, skip_overlap=False, detailed=False):
    """Compute Jaccard Error Rate (JER).

    JER is the average over all reference speakers of (1 - Jaccard similarity)
    under an optimal 1-to-1 mapping between reference and hypothesis speakers:

        JER_r = 1 - |r \\cap h| / |r \\cup h|
        JER = (1 / |S_ref|) * \\sum_{r \\in S_ref} JER_r

    Unmapped reference speakers have JER_r = 1.0.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
        collar: float, no-score zone (in seconds) around each reference
            speaker boundary ([boundary - collar, boundary + collar])
        skip_overlap: bool, if True, exclude regions where more than one
            reference speaker is active
        detailed: bool, if True, return a dict containing overall JER,
            per-reference-speaker JER, and the optimal speaker_mapping

    Returns:
        If `detailed=False`, a float number for the Jaccard Error Rate.
        If `detailed=True`, a dict with keys:
            - "jer": float, Jaccard Error Rate
            - "speaker_jer": dict mapping each ref speaker -> speaker JER
            - "speaker_mapping": dict mapping matched ref -> hyp speaker
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    ref, hyp = der._preprocess_segments(
        ref, hyp, collar=collar, skip_overlap=skip_overlap)

    ref_index = der.build_speaker_index(ref)
    hyp_index = der.build_speaker_index(hyp)

    if not ref_index:
        if detailed:
            return {
                "jer": 0.0,
                "speaker_jer": {},
                "speaker_mapping": {},
            }
        return 0.0

    ref_durations = np.zeros(len(ref_index))
    for spk, start, end in ref:
        ref_durations[ref_index[spk]] += end - start

    hyp_durations = np.zeros(len(hyp_index))
    for spk, start, end in hyp:
        hyp_durations[hyp_index[spk]] += end - start

    intersection_matrix = der.build_cost_matrix(ref, hyp)
    union_matrix = (
        ref_durations[:, None] + hyp_durations[None, :] - intersection_matrix
    )
    jaccard_matrix = np.divide(
        intersection_matrix,
        union_matrix,
        out=np.zeros_like(intersection_matrix),
        where=union_matrix > 0.0,
    )

    row_index, col_index = optimize.linear_sum_assignment(-jaccard_matrix)

    inv_ref = {i: spk for spk, i in ref_index.items()}
    inv_hyp = {j: spk for spk, j in hyp_index.items()}
    speaker_jer = {spk: 1.0 for spk in ref_index}
    speaker_mapping = {}

    for r, c in zip(row_index, col_index):
        ref_spk = inv_ref[r]
        speaker_jer[ref_spk] = float(1.0 - jaccard_matrix[r, c])
        if intersection_matrix[r, c] > 0.0:
            speaker_mapping[ref_spk] = inv_hyp[c]

    jer_val = float(sum(speaker_jer.values()) / len(ref_index))

    if detailed:
        return {
            "jer": jer_val,
            "speaker_jer": speaker_jer,
            "speaker_mapping": speaker_mapping,
        }
    return jer_val


def corpus_jer(refs, hyps, collar=0.0, skip_overlap=False, detailed=False):
    """Compute Corpus Jaccard Error Rate across multiple recordings.

    Corpus JER is the average of per-speaker Jaccard Error Rates across all
    reference speakers in all recordings.

    Args:
        refs: a dict mapping recording_id -> ref segments, or a list of
            ref segment lists
        hyps: a dict mapping recording_id -> hyp segments, or a list of
            hyp segment lists (matching `refs`)
        collar: float, no-score zone (in seconds) around each reference
            speaker boundary
        skip_overlap: bool, if True, exclude regions where more than one
            reference speaker is active
        detailed: bool, if True, return a dict with corpus JER, total number of
            reference speakers, and per-recording results

    Returns:
        If `detailed=False`, a float number for the Corpus JER.
        If `detailed=True`, a dict with keys:
            - "jer": float, Corpus JER
            - "num_ref_speakers": int, total reference speakers across corpus
            - "per_recording": dict or list of per-recording detailed results
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    pairs, is_dict = der._normalize_corpus_inputs(refs, hyps)

    total_speaker_jer = 0.0
    total_ref_speakers = 0
    per_recording = {}

    for key, ref, hyp in pairs:
        res = JER(
            ref, hyp, collar=collar, skip_overlap=skip_overlap, detailed=True)
        total_speaker_jer += sum(res["speaker_jer"].values())
        total_ref_speakers += len(res["speaker_jer"])
        per_recording[key] = res

    if total_ref_speakers == 0:
        jer_val = 0.0
    else:
        jer_val = total_speaker_jer / total_ref_speakers

    if detailed:
        return {
            "jer": float(jer_val),
            "num_ref_speakers": total_ref_speakers,
            "per_recording": per_recording if is_dict else [
                per_recording[i] for i in range(len(pairs))
            ],
        }
    return float(jer_val)
