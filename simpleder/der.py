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
    num_elements = len(hyp)



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
        Load = \int max(N_ref(t), N_hyp(t)) dt

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


def DER(ref, hyp):
    """Compute Diarization Error Rate.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a float number for the Diarization Error Rate
    """
    check_input(ref)
    check_input(hyp)
    ref_total_length = compute_total_length(ref)
    cost_matrix = build_cost_matrix(ref, hyp)
    row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
    optimal_match_overlap = cost_matrix[row_index, col_index].sum()
    load_length = compute_load_length(ref, hyp)
    der = (load_length - optimal_match_overlap) / ref_total_length
    return der
