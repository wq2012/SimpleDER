import numpy as np
from scipy import optimize


def check_input(hyp):
    """Check whether a hypothesis/reference is valid.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Raises:
        TypeError: if the type of `hyp` is incorrect
        ValueError: if some tuple has start > end; or if two tuples intersect
            with each other
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
        if not isinstance(element[1], float) or not isinstance(
                element[2], float):
            raise TypeError("Start and end must be float numbers.")
        if element[1] > element[2]:
            raise ValueError("Start must not be larger than end.")


def compute_total_length(hyp):
    """Compute total length of a hypothesis/reference.

    Args:
        hyp: a list of tuples, where each tuple is (speaker, start, end)
            of type (string, float, float)

    Returns:
        a float number for the total length
    """
    total_length = 0.0
    for element in hyp:
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


def compute_load_length(ref, hyp):
    """Compute the load length (integrated maximum number of speakers).

    The "load" is a concept to generalize the union of reference and hypothesis
    when overlapped speech is present. It is defined as the integral of the
    maximum number of speakers at any given time point.

    Equation:
        Load = \\int max(N_ref(t), N_hyp(t)) dt

    where N_ref(t) is the number of speakers in reference at time t, and
    N_hyp(t) is the number of speakers in hypothesis at time t.

    This is equivalent to the "Union" length when there is no overlap (N <= 1).

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a float number for the load length
    """
    boundaries = set()
    for element in ref + hyp:
        boundaries.add(element[1])
        boundaries.add(element[2])
    boundaries = sorted(list(boundaries))

    load_length = 0.0
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        mid = (start + end) / 2.0

        ref_count = 0
        for element in ref:
            if element[1] <= mid <= element[2]:
                ref_count += 1

        hyp_count = 0
        for element in hyp:
            if element[1] <= mid <= element[2]:
                hyp_count += 1

        load_length += (end - start) * max(ref_count, hyp_count)

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
    """Build the cost matrix.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a 2-dim numpy array, whose element (i, j) is the overlap between
            `i`th reference speaker and `j`th hypothesis speaker
    """
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
        collar: float, tolerance

    Returns:
        a list of (start, end) tuples, sorted and merged
    """
    if collar == 0.0:
        return []

    intervals = []
    for element in ref:
        start, end = element[1], element[2]
        intervals.append((start - collar, start + collar))
        intervals.append((end - collar, end + collar))

    # Sort and merge
    intervals.sort()
    merged = []
    if not intervals:
        return []

    curr_start, curr_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
    merged.append((curr_start, curr_end))
    return merged


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
        # We need to intersect [start, end] with NOT exclusions
        # Iterate through exclusions and cut [start, end]
        current_time = start
        for ex_start, ex_end in exclusions:
            if ex_end <= current_time:
                continue
            if ex_start >= end:
                break

            # Now we have overlap between
            # [current_time, end] and [ex_start, ex_end]
            # The valid part is [current_time, ex_start] (if valid)
            if ex_start > current_time:
                new_segments.append((speaker, current_time, ex_start))

            # Advance current_time to after exclusion
            current_time = max(current_time, ex_end)

            if current_time >= end:
                break

        # If there is remaining time after all exclusions
        if current_time < end:
            new_segments.append((speaker, current_time, end))

    return new_segments


def DER(ref, hyp, collar=0.0):
    """Compute Diarization Error Rate.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`
        collar: float, tolerance allowing for some mismatch in speaker borders

    Returns:
        a float number for the Diarization Error Rate
    """
    check_input(ref)
    check_input(hyp)

    if collar > 0.0:
        exclusions = compute_merged_exclusion_intervals(ref, collar)
        ref = subtract_intervals(ref, exclusions)
        hyp = subtract_intervals(hyp, exclusions)

    ref_total_length = compute_total_length(ref)
    cost_matrix = build_cost_matrix(ref, hyp)
    row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
    optimal_match_overlap = cost_matrix[row_index, col_index].sum()
    load_length = compute_load_length(ref, hyp)
    if ref_total_length == 0.0:
        return 0.0
    der = (load_length - optimal_match_overlap) / ref_total_length
    return der
