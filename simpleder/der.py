import argparse
import json
import os
import numpy as np
from scipy import optimize

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]

# Edit operations for word-level Levenshtein alignment.
_CORRECT = 0
_SUBSTITUTION = 1
_INSERTION = 2
_DELETION = 3


def check_input(hyp):
    """Check whether a hypothesis/reference segment list is valid.

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
        hyp: a list of tuples whose first element is speaker (string)

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


def check_word_input(hyp):
    """Check whether a word-level hypothesis/reference list is valid.

    Args:
        hyp: a list of (speaker, text) tuples of type (str, str), where `text`
            can be a single word or whitespace-separated words spoken by
            `speaker`

    Raises:
        TypeError: if `hyp` or any of its elements has an invalid type
    """
    if not isinstance(hyp, list):
        raise TypeError("Input must be a list.")
    for element in hyp:
        if not isinstance(element, tuple):
            raise TypeError("Input must be a list of tuples.")
        if len(element) != 2:
            raise TypeError(
                "Each tuple must have the elements: (speaker, word).")
        if not isinstance(element[0], str):
            raise TypeError("Speaker must be a string.")
        if not isinstance(element[1], str):
            raise TypeError("Word must be a string.")


def normalize_text(text):
    """Normalize text by lowercasing and stripping punctuation.

    Follows the normalization convention in Google's DiarizationLM: converts
    to lowercase and removes punctuation characters while preserving token
    boundaries.

    Args:
        text: input string

    Returns:
        normalized string
    """
    text_lower = text.lower().strip()
    words = text_lower.split()
    new_words = []
    for word in words:
        new_word = word
        for punc in PUNCTUATIONS:
            replaced = new_word.replace(punc, "")
            if len(replaced.split()) != 1:
                continue
            new_word = replaced
        new_words.append(new_word)
    return " ".join(new_words)


def _expand_word_tuples(segments):
    """Expand a list of (speaker, text) tuples into normalized word tuples."""
    expanded = []
    for speaker, text in segments:
        norm = normalize_text(text)
        for word in norm.split():
            expanded.append((speaker, word))
    return expanded


def levenshtein_with_edits(ref_words, hyp_words):
    """Compute word-level Levenshtein distance and alignment.

    Args:
        ref_words: a list of reference word strings
        hyp_words: a list of hypothesis word strings

    Returns:
        a tuple (edit_distance, align), where `align` is a list of
        (ref_pos, hyp_pos) index pairs. Insertions have ref_pos == -1 and
        deletions have hyp_pos == -1.
    """
    n1 = len(ref_words)
    n2 = len(hyp_words)
    costs = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    backptr = [[_CORRECT] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(1, n1 + 1):
        costs[i][0] = i
        backptr[i][0] = _DELETION

    for j in range(n2):
        costs[0][j + 1] = j + 1
        backptr[0][j + 1] = _INSERTION
        w2 = hyp_words[j]
        for i in range(n1):
            ins = costs[i + 1][j] + 1
            del_ = costs[i][j + 1] + 1
            same = (ref_words[i] == w2)
            sub = costs[i][j] + (0 if same else 1)
            best = min(ins, del_, sub)
            costs[i + 1][j + 1] = best
            if best == ins:
                backptr[i + 1][j + 1] = _INSERTION
            elif best == del_:
                backptr[i + 1][j + 1] = _DELETION
            elif same:
                backptr[i + 1][j + 1] = _CORRECT
            else:
                backptr[i + 1][j + 1] = _SUBSTITUTION

    align = []
    i = n1
    j = n2
    while i > 0 or j > 0:
        ed_op = backptr[i][j]
        if ed_op == _INSERTION:
            align.append((-1, j - 1))
            j -= 1
        elif ed_op == _DELETION:
            align.append((i - 1, -1))
            i -= 1
        else:
            align.append((i - 1, j - 1))
            i -= 1
            j -= 1

    align.reverse()
    return costs[n1][n2], align


def WDER(ref, hyp, detailed=False):
    """Compute Word Diarization Error Rate (WDER).

    WDER measures the fraction of aligned words (correct ASR words plus
    substituted ASR words after Levenshtein alignment between reference and
    hypothesis transcripts) that are assigned to the wrong speaker after
    optimal 1-to-1 speaker mapping (Shafey et al., Interspeech 2019).

    Args:
        ref: a list of (speaker, word) tuples of type (str, str) for the
            ground truth. Each tuple's text may be a single word or
            whitespace-separated words spoken by that speaker.
        hyp: a list of (speaker, word) tuples of type (str, str) for the
            hypothesis, same format as `ref`.
        detailed: bool, if True, return a dict containing WDER, word counts
            (wder_sub, wder_correct, wder_total), WER breakdown, and the
            optimal speaker_mapping.

    Returns:
        If `detailed=False`, a float number for the Word Diarization Error
        Rate.
        If `detailed=True`, a dict with keys:
            - "wder": float, Word Diarization Error Rate
            - "wder_sub": int, aligned words with wrong speaker label
            - "wder_correct": int, aligned words with correct speaker label
            - "wder_total": int, total aligned words (wer_correct + wer_sub)
            - "wer": float, Word Error Rate of the transcript
            - "wer_insert": int, inserted words
            - "wer_delete": int, deleted words
            - "wer_sub": int, substituted words
            - "wer_correct": int, correct words
            - "wer_total": int, total reference words
            - "speaker_mapping": dict mapping matched ref -> hyp speaker
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    check_word_input(ref)
    check_word_input(hyp)

    ref_expanded = _expand_word_tuples(ref)
    hyp_expanded = _expand_word_tuples(hyp)

    ref_spks = [spk for spk, _ in ref_expanded]
    ref_words = [w for _, w in ref_expanded]
    hyp_spks = [spk for spk, _ in hyp_expanded]
    hyp_words = [w for _, w in hyp_expanded]

    _, align = levenshtein_with_edits(ref_words, hyp_words)

    wer_insert = 0
    wer_delete = 0
    wer_sub = 0
    wer_correct = 0
    aligned_spk_pairs = []

    for i, j in align:
        if i == -1:
            wer_insert += 1
        elif j == -1:
            wer_delete += 1
        else:
            if ref_words[i] == hyp_words[j]:
                wer_correct += 1
            else:
                wer_sub += 1
            aligned_spk_pairs.append((ref_spks[i], hyp_spks[j]))

    wer_total = len(ref_words)
    wer_val = (
        0.0 if wer_total == 0
        else float((wer_insert + wer_delete + wer_sub) / wer_total)
    )

    wder_total = len(aligned_spk_pairs)
    if wder_total == 0:
        wder_correct = 0
        wder_sub = 0
        wder_val = 0.0
        speaker_mapping = {}
    else:
        ref_index = build_speaker_index(ref_expanded)
        hyp_index = build_speaker_index(hyp_expanded)
        cost_matrix = np.zeros((len(ref_index), len(hyp_index)), dtype=int)
        for r_spk, h_spk in aligned_spk_pairs:
            cost_matrix[ref_index[r_spk], hyp_index[h_spk]] += 1

        row_index, col_index = optimize.linear_sum_assignment(
            cost_matrix, maximize=True)
        wder_correct = int(cost_matrix[row_index, col_index].sum())
        wder_sub = wder_total - wder_correct
        wder_val = float(wder_sub / wder_total)

        inv_ref = {i: spk for spk, i in ref_index.items()}
        inv_hyp = {j: spk for spk, j in hyp_index.items()}
        speaker_mapping = {
            inv_ref[r]: inv_hyp[c]
            for r, c in zip(row_index, col_index)
            if cost_matrix[r, c] > 0
        }

    if detailed:
        return {
            "wder": wder_val,
            "wder_sub": wder_sub,
            "wder_correct": wder_correct,
            "wder_total": wder_total,
            "wer": wer_val,
            "wer_insert": wer_insert,
            "wer_delete": wer_delete,
            "wer_sub": wer_sub,
            "wer_correct": wer_correct,
            "wer_total": wer_total,
            "speaker_mapping": speaker_mapping,
        }
    return wder_val


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


def corpus_wder(refs, hyps, detailed=False):
    """Compute micro-averaged Corpus Word Diarization Error Rate.

    Aggregates `wder_sub` and `wder_total` across all recordings in the corpus.

    Args:
        refs: a dict mapping recording_id -> ref word tuples, or a list of
            ref word tuple lists
        hyps: a dict mapping recording_id -> hyp word tuples, or a list of
            hyp word tuple lists (matching `refs`)
        detailed: bool, if True, return a dict with corpus-level totals and
            per-recording results

    Returns:
        If `detailed=False`, a float number for the Corpus WDER.
        If `detailed=True`, a dict with keys:
            - "wder": float, Corpus WDER
            - "wder_sub": int, total aligned words with wrong speaker
            - "wder_correct": int, total aligned words with correct speaker
            - "wder_total": int, total aligned words across corpus
            - "wer": float, Corpus WER
            - "wer_insert": int, total inserted words
            - "wer_delete": int, total deleted words
            - "wer_sub": int, total substituted words
            - "wer_correct": int, total correct words
            - "wer_total": int, total reference words
            - "per_recording": dict or list of per-recording detailed results
    """
    if not isinstance(detailed, bool):
        raise TypeError("detailed must be a boolean.")
    pairs, is_dict = _normalize_corpus_inputs(refs, hyps)

    total_wder_sub = 0
    total_wder_correct = 0
    total_wder_total = 0
    total_wer_insert = 0
    total_wer_delete = 0
    total_wer_sub = 0
    total_wer_correct = 0
    total_wer_total = 0
    per_recording = {}

    for key, ref, hyp in pairs:
        res = WDER(ref, hyp, detailed=True)
        total_wder_sub += res["wder_sub"]
        total_wder_correct += res["wder_correct"]
        total_wder_total += res["wder_total"]
        total_wer_insert += res["wer_insert"]
        total_wer_delete += res["wer_delete"]
        total_wer_sub += res["wer_sub"]
        total_wer_correct += res["wer_correct"]
        total_wer_total += res["wer_total"]
        per_recording[key] = res

    wder_val = (
        0.0 if total_wder_total == 0
        else float(total_wder_sub / total_wder_total)
    )
    wer_val = (
        0.0 if total_wer_total == 0
        else float(
            (total_wer_insert + total_wer_delete + total_wer_sub)
            / total_wer_total
        )
    )

    if detailed:
        return {
            "wder": wder_val,
            "wder_sub": total_wder_sub,
            "wder_correct": total_wder_correct,
            "wder_total": total_wder_total,
            "wer": wer_val,
            "wer_insert": total_wer_insert,
            "wer_delete": total_wer_delete,
            "wer_sub": total_wer_sub,
            "wer_correct": total_wer_correct,
            "wer_total": total_wer_total,
            "per_recording": per_recording if is_dict else [
                per_recording[i] for i in range(len(pairs))
            ],
        }
    return wder_val


def load_rttm(rttm_source):
    """Load diarization segments from a NIST RTTM file or line iterable.

    Each `SPEAKER` line in an RTTM file has the format:
        SPEAKER <rec_id> <channel> <start> <duration> <NA> <NA> <spk> <NA> <NA>

    Args:
        rttm_source: path to an RTTM file (str or os.PathLike) or an iterable
            of RTTM lines (such as an open file object)

    Returns:
        a dict mapping `recording_id` (str) to a list of
        `(speaker, start, end)` tuples
    """
    if isinstance(rttm_source, (str, os.PathLike)):
        with open(rttm_source, "r", encoding="utf-8") as f:
            lines = f.readlines()
    elif hasattr(rttm_source, "__iter__") and not isinstance(
            rttm_source, (bytes, bytearray)):
        lines = list(rttm_source)
    else:
        raise TypeError(
            "rttm_source must be a file path or an iterable of lines.")

    recordings = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        parts = stripped.split()
        if parts[0].upper() != "SPEAKER":
            continue
        if len(parts) < 8:
            raise ValueError(f"Invalid RTTM SPEAKER line: {stripped}")
        rec_id = parts[1]
        start = float(parts[3])
        duration = float(parts[4])
        if not np.isfinite(start) or not np.isfinite(duration) or duration < 0:
            raise ValueError(
                f"Invalid start or duration in RTTM line: {stripped}")
        end = start + duration
        speaker = parts[7]
        recordings.setdefault(rec_id, []).append((speaker, start, end))

    return recordings


def dump_rttm(data, rttm_target, recording_id="recording", channel="1"):
    """Write diarization segments to a NIST RTTM file or writable stream.

    Args:
        data: either a dict mapping `recording_id -> list of (speaker, start,
            end)` tuples, or a single list of `(speaker, start, end)` tuples
        rttm_target: output file path (str or os.PathLike) or a writable
            file-like object
        recording_id: recording ID to use when `data` is a single list of
            segments (default: "recording")
        channel: channel ID field in RTTM output (default: "1")
    """
    if isinstance(data, list):
        check_input(data)
        recordings = {str(recording_id): data}
    elif isinstance(data, dict):
        recordings = {}
        for rec_id, segments in data.items():
            check_input(segments)
            recordings[str(rec_id)] = segments
    else:
        raise TypeError("data must be a list of segments or a dict.")

    lines = []
    for rec_id, segments in recordings.items():
        for speaker, start, end in segments:
            start_f = float(start)
            dur_f = float(end) - start_f
            lines.append(
                f"SPEAKER {rec_id} {channel} {start_f:.6f} {dur_f:.6f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    if isinstance(rttm_target, (str, os.PathLike)):
        with open(rttm_target, "w", encoding="utf-8") as f:
            f.writelines(lines)
    elif hasattr(rttm_target, "write"):
        rttm_target.writelines(lines)
    else:
        raise TypeError(
            "rttm_target must be a file path or writable file-like object.")


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
    refs = load_rttm(args.ref)
    hyps = load_rttm(args.hyp)

    unknown_recs = set(hyps.keys()) - set(refs.keys())
    if unknown_recs:
        raise ValueError(
            f"Hypothesis contains recordings not in reference: "
            f"{sorted(unknown_recs)}"
        )
    aligned_hyps = {rec_id: hyps.get(rec_id, []) for rec_id in refs}

    der_res = corpus_der(
        refs,
        aligned_hyps,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
        detailed=True,
    )
    jer_res = corpus_jer(
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
