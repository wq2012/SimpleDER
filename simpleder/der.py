import numpy as np
from scipy import optimize


def check_input(hyp):
    """Check whether a hypothesis/reference is valid.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Raises:
        TypeError: if the type of `hyp` or its elements is incorrect
        ValueError: if some tuple has non-finite timestamps or start > end
    """
    if not isinstance(hyp, list):
        raise TypeError("Input must be a list.")
    for element in hyp:
        if not isinstance(element, tuple):
            raise TypeError("Input must be a list of tuples.")
        if len(element) != 3:
            raise TypeError(
                "Each tuple must have the elements: (speaker, start, end).")
        if not isinstance(element[0], str):
            raise TypeError("Speaker must be a string.")
        if (isinstance(element[1], bool) or isinstance(element[2], bool) or
                not isinstance(element[1], (int, float, np.number)) or
                not isinstance(element[2], (int, float, np.number))):
            raise TypeError("Start and end must be float numbers.")
        if not np.isfinite(element[1]) or not np.isfinite(element[2]):
            raise ValueError("Start and end must be finite numbers.")
        if element[1] > element[2]:
            raise ValueError("Start must not be larger than end.")


def _merge_intervals(intervals):
    """Sort and merge a list of (start, end) intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = []
    curr_start, curr_end = sorted_intervals[0]
    for next_start, next_end in sorted_intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged


def merge_speaker_intervals(segments):
    """Merge overlapping or contiguous intervals for each speaker.

    Also removes zero-length intervals where start == end.

    Args:
        segments: a list of (speaker, start, end) tuples

    Returns:
        a list of (speaker, start, end) tuples with disjoint, non-adjacent
        intervals per speaker, sorted by (start, end, speaker)
    """
    by_speaker = {}
    for speaker, start, end in segments:
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        by_speaker.setdefault(speaker, []).append((start_f, end_f))

    merged_segments = []
    for speaker, intervals in by_speaker.items():
        for start_f, end_f in _merge_intervals(intervals):
            merged_segments.append((speaker, start_f, end_f))

    merged_segments.sort(key=lambda x: (x[1], x[2], x[0]))
    return merged_segments


def compute_total_length(hyp):
    """Compute total length of a hypothesis/reference.

    Overlapping intervals belonging to the same speaker are merged before
    summing durations.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Returns:
        a float number for the total length
    """
    merged_hyp = merge_speaker_intervals(hyp)
    total_length = 0.0
    for element in merged_hyp:
        total_length += element[2] - element[1]
    return total_length


def compute_intersection_length(A, B):
    """Compute the intersection length of two tuples.

    Args:
        A: a (speaker, start, end) tuple of type (string, float, float)
        B: a (speaker, start, end) tuple of type (string, float, float)

    Returns:
        a float number of the intersection between `A` and `B`
    """
    max_start = max(A[1], B[1])
    min_end = min(A[2], B[2])
    return max(0.0, min_end - max_start)


def compute_error_components(ref, hyp):
    """Compute elementary duration components between reference and hypothesis.

    Args:
        ref: a list of (speaker, start, end) tuples for the ground truth
        hyp: a list of (speaker, start, end) tuples for the hypothesis

    Returns:
        a tuple of (miss, false_alarm, min_overlap, load_length) in seconds,
        where:
            miss = \\int max(0, N_ref(t) - N_hyp(t)) dt
            false_alarm = \\int max(0, N_hyp(t) - N_ref(t)) dt
            min_overlap = \\int min(N_ref(t), N_hyp(t)) dt
            load_length = \\int max(N_ref(t), N_hyp(t)) dt
    """
    ref = merge_speaker_intervals(ref)
    hyp = merge_speaker_intervals(hyp)

    boundaries = set()
    for element in ref + hyp:
        boundaries.add(element[1])
        boundaries.add(element[2])
    boundaries = sorted(list(boundaries))

    miss = 0.0
    false_alarm = 0.0
    min_overlap = 0.0
    load_length = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start

        ref_count = 0
        for element in ref:
            if element[1] <= start and end <= element[2]:
                ref_count += 1

        hyp_count = 0
        for element in hyp:
            if element[1] <= start and end <= element[2]:
                hyp_count += 1

        miss += dur * max(0, ref_count - hyp_count)
        false_alarm += dur * max(0, hyp_count - ref_count)
        min_overlap += dur * min(ref_count, hyp_count)
        load_length += dur * max(ref_count, hyp_count)

    return miss, false_alarm, min_overlap, load_length


def compute_load_length(ref, hyp):
    """Compute the load length (integrated maximum number of speakers).

    The "load" generalizes the union of reference and hypothesis when
    overlapped speech is present. It is defined as the integral of the
    maximum number of speakers at any given time point:

        Load = \\int max(N_ref(t), N_hyp(t)) dt

    where N_ref(t) is the number of speakers in reference at time t, and
    N_hyp(t) is the number of speakers in hypothesis at time t.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a float number for the load length
    """
    _, _, _, load_length = compute_error_components(ref, hyp)
    return load_length


def build_speaker_index(hyp):
    """Build the index for the speakers.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Returns:
        a dict from speaker to integer
    """
    speaker_set = sorted({element[0] for element in hyp})
    index = {speaker: i for i, speaker in enumerate(speaker_set)}
    return index


def build_cost_matrix(ref, hyp):
    """Build the cost (intersection) matrix between reference and hypothesis.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a 2-dim numpy array, whose element (i, j) is the overlap duration
        between `i`th reference speaker and `j`th hypothesis speaker
    """
    ref = merge_speaker_intervals(ref)
    hyp = merge_speaker_intervals(hyp)
    ref_index = build_speaker_index(ref)
    hyp_index = build_speaker_index(hyp)
    cost_matrix = np.zeros((len(ref_index), len(hyp_index)))
    for ref_element in ref:
        for hyp_element in hyp:
            i = ref_index[ref_element[0]]
            j = hyp_index[hyp_element[0]]
            cost_matrix[i, j] += compute_intersection_length(
                ref_element, hyp_element)
    return cost_matrix


def compute_merged_exclusion_intervals(ref, collar):
    """Compute merged exclusion intervals based on reference boundaries.

    Args:
        ref: a list of tuples for the ground truth
        collar: float, no-score tolerance around each reference boundary
            ([boundary - collar, boundary + collar])

    Returns:
        a list of (start, end) tuples, sorted and merged
    """
    if isinstance(collar, bool) or not isinstance(
            collar, (int, float, np.number)):
        raise TypeError("Collar must be a number.")
    if not np.isfinite(collar) or collar < 0.0:
        raise ValueError("Collar must be a non-negative finite number.")
    collar = float(collar)
    if collar == 0.0:
        return []

    ref = merge_speaker_intervals(ref)
    intervals = []
    for element in ref:
        start, end = element[1], element[2]
        intervals.append((start - collar, start + collar))
        intervals.append((end - collar, end + collar))

    return _merge_intervals(intervals)


def compute_overlap_exclusion_intervals(ref):
    """Compute merged intervals where multiple reference speakers are active.

    Args:
        ref: a list of (speaker, start, end) tuples for the ground truth

    Returns:
        a list of (start, end) tuples where N_ref(t) > 1, sorted and merged
    """
    ref = merge_speaker_intervals(ref)
    boundaries = set()
    for element in ref:
        boundaries.add(element[1])
        boundaries.add(element[2])
    boundaries = sorted(list(boundaries))

    overlap_intervals = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        ref_count = 0
        for element in ref:
            if element[1] <= start and end <= element[2]:
                ref_count += 1
        if ref_count > 1:
            overlap_intervals.append((start, end))

    return _merge_intervals(overlap_intervals)


def subtract_intervals(segments, exclusions):
    """Subtract exclusion intervals from segments.

    Args:
        segments: a list of (speaker, start, end)
        exclusions: a list of (start, end) tuples, sorted and merged

    Returns:
        a list of (speaker, start, end) with exclusions removed
    """
    if not exclusions:
        return segments

    new_segments = []
    for speaker, start, end in segments:
        current_time = start
        for ex_start, ex_end in exclusions:
            if ex_end <= current_time:
                continue
            if ex_start >= end:
                break

            if ex_start > current_time:
                new_segments.append((speaker, current_time, ex_start))

            current_time = max(current_time, ex_end)

            if current_time >= end:
                break

        if current_time < end:
            new_segments.append((speaker, current_time, end))

    return new_segments


def _preprocess_segments(ref, hyp, collar=0.0, skip_overlap=False):
    """Validate and preprocess reference and hypothesis segments."""
    check_input(ref)
    check_input(hyp)
    if not isinstance(skip_overlap, bool):
        raise TypeError("skip_overlap must be a boolean.")
    if not isinstance(collar, bool) and isinstance(
            collar, (int, float, np.number)):
        if not np.isfinite(collar) or collar < 0.0:
            raise ValueError("Collar must be a non-negative finite number.")
    else:
        raise TypeError("Collar must be a number.")

    ref = merge_speaker_intervals(ref)
    hyp = merge_speaker_intervals(hyp)

    exclusions = []
    if collar > 0.0:
        exclusions.extend(compute_merged_exclusion_intervals(ref, collar))
    if skip_overlap:
        exclusions.extend(compute_overlap_exclusion_intervals(ref))
    if exclusions:
        exclusions = _merge_intervals(exclusions)
        ref = subtract_intervals(ref, exclusions)
        hyp = subtract_intervals(hyp, exclusions)

    return ref, hyp


def DER(ref, hyp, collar=0.0, skip_overlap=False, detailed=False):
    """Compute Diarization Error Rate (DER).

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
        collar: float, no-score zone (in seconds) around each reference
            speaker boundary ([boundary - collar, boundary + collar])
        skip_overlap: bool, if True, exclude regions where more than one
            reference speaker is active
        detailed: bool, if True, return a dict containing DER, error component
            durations (miss, false_alarm, confusion, ref_length), and the
            optimal speaker_mapping

    Returns:
        If `detailed=False`, a float number for the Diarization Error Rate.
        If `detailed=True`, a dict with keys:
            - "der": float, Diarization Error Rate
            - "miss": float, missed speech duration in seconds
            - "false_alarm": float, false alarm duration in seconds
            - "confusion": float, speaker confusion duration in seconds
            - "ref_length": float, scored reference length in seconds
            - "speaker_mapping": dict mapping matched ref -> hyp speaker
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    ref, hyp = _preprocess_segments(
        ref, hyp, collar=collar, skip_overlap=skip_overlap)

    ref_total_length = compute_total_length(ref)
    ref_index = build_speaker_index(ref)
    hyp_index = build_speaker_index(hyp)
    cost_matrix = build_cost_matrix(ref, hyp)
    row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
    optimal_match_overlap = float(cost_matrix[row_index, col_index].sum())

    inv_ref = {i: spk for spk, i in ref_index.items()}
    inv_hyp = {j: spk for spk, j in hyp_index.items()}
    speaker_mapping = {
        inv_ref[r]: inv_hyp[c]
        for r, c in zip(row_index, col_index)
        if cost_matrix[r, c] > 0.0
    }

    miss, false_alarm, min_overlap, load_length = compute_error_components(
        ref, hyp)
    confusion = max(0.0, min_overlap - optimal_match_overlap)

    if ref_total_length == 0.0:
        der_val = 0.0
    else:
        der_val = (load_length - optimal_match_overlap) / ref_total_length

    if detailed:
        return {
            "der": float(der_val),
            "miss": float(miss),
            "false_alarm": float(false_alarm),
            "confusion": float(confusion),
            "ref_length": float(ref_total_length),
            "speaker_mapping": speaker_mapping,
        }
    return float(der_val)


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
    ref, hyp = _preprocess_segments(
        ref, hyp, collar=collar, skip_overlap=skip_overlap)

    ref_index = build_speaker_index(ref)
    hyp_index = build_speaker_index(hyp)

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

    intersection_matrix = build_cost_matrix(ref, hyp)
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


def _normalize_corpus_inputs(refs, hyps):
    """Validate and normalize multi-recording inputs (dicts or lists)."""
    if isinstance(refs, dict) and isinstance(hyps, dict):
        if set(refs.keys()) != set(hyps.keys()):
            raise ValueError(
                "Reference and hypothesis dicts must have the same keys.")
        pairs = [(k, refs[k], hyps[k]) for k in sorted(refs.keys())]
        return pairs, True
    if isinstance(refs, list) and isinstance(hyps, list):
        if len(refs) != len(hyps):
            raise ValueError(
                "Reference and hypothesis lists must have the same length.")
        pairs = [(i, refs[i], hyps[i]) for i in range(len(refs))]
        return pairs, False
    raise TypeError("refs and hyps must both be dicts or both be lists.")


def corpus_der(refs, hyps, collar=0.0, skip_overlap=False, detailed=False):
    """Compute micro-averaged Corpus Diarization Error Rate across recordings.

    Corpus DER aggregates total missed speech, false alarm, and speaker
    confusion durations across all recordings and divides by the total
    reference duration across all recordings.

    Args:
        refs: a dict mapping recording_id -> ref segments, or a list of
            ref segment lists
        hyps: a dict mapping recording_id -> hyp segments, or a list of
            hyp segment lists (matching `refs`)
        collar: float, no-score zone (in seconds) around each reference
            speaker boundary
        skip_overlap: bool, if True, exclude regions where more than one
            reference speaker is active
        detailed: bool, if True, return a dict with corpus-level totals and
            per-recording results

    Returns:
        If `detailed=False`, a float number for the Corpus DER.
        If `detailed=True`, a dict with keys:
            - "der": float, Corpus DER
            - "miss": float, total missed speech duration in seconds
            - "false_alarm": float, total false alarm duration in seconds
            - "confusion": float, total speaker confusion duration in seconds
            - "ref_length": float, total scored reference length in seconds
            - "per_recording": dict or list of per-recording detailed results
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    pairs, is_dict = _normalize_corpus_inputs(refs, hyps)

    total_miss = 0.0
    total_fa = 0.0
    total_confusion = 0.0
    total_ref_length = 0.0
    per_recording = {}

    for key, ref, hyp in pairs:
        res = DER(
            ref, hyp, collar=collar, skip_overlap=skip_overlap, detailed=True)
        total_miss += res["miss"]
        total_fa += res["false_alarm"]
        total_confusion += res["confusion"]
        total_ref_length += res["ref_length"]
        per_recording[key] = res

    if total_ref_length == 0.0:
        der_val = 0.0
    else:
        der_val = (total_miss + total_fa + total_confusion) / total_ref_length

    if detailed:
        return {
            "der": float(der_val),
            "miss": float(total_miss),
            "false_alarm": float(total_fa),
            "confusion": float(total_confusion),
            "ref_length": float(total_ref_length),
            "per_recording": per_recording if is_dict else [
                per_recording[i] for i in range(len(pairs))
            ],
        }
    return float(der_val)


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
    pairs, is_dict = _normalize_corpus_inputs(refs, hyps)

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
