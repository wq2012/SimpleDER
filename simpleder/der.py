from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    for i in range(num_elements - 1):
        for j in range(i + 1, num_elements):
            if compute_intersection_length(hyp[i], hyp[j]) > 0.0:
                raise ValueError(
                    "Input must not contain overlapped speech.")


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


def compute_merged_total_length(ref, hyp):
    """Compute the total length of the union of reference and hypothesis.

    Args:
        ref: a list of tuples for the ground truth, where each tuple is
            (speaker, start, end) of type (string, float, float)
        hyp: a list of tuples for the diarization result hypothesis, same type
            as `ref`

    Returns:
        a float number for the union total length
    """
    # Remove speaker label and merge.
    merged = [(element[1], element[2]) for element in (ref + hyp)]
    # Sort by start.
    merged = sorted(merged, key=lambda element: element[0])
    i = len(merged) - 2
    while i >= 0:
        if merged[i][1] >= merged[i + 1][0]:
            max_end = max(merged[i][1], merged[i + 1][1])
            merged[i] = (merged[i][0], max_end)
            del merged[i + 1]
            if i == len(merged) - 1:
                i -= 1
        else:
            i -= 1
    total_length = 0.0
    for element in merged:
        total_length += element[1] - element[0]
    return total_length


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
    union_total_length = compute_merged_total_length(ref, hyp)
    der = (union_total_length - optimal_match_overlap) / ref_total_length
    return der
