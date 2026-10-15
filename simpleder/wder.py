import numpy as np
from scipy import optimize

from . import der

PUNCTUATIONS = [",", ".", "_", "?", "!", "-", '"', "'"]

# Edit operations for word-level Levenshtein alignment.
_CORRECT = 0
_SUBSTITUTION = 1
_INSERTION = 2
_DELETION = 3


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
        ref_index = der.build_speaker_index(ref_expanded)
        hyp_index = der.build_speaker_index(hyp_expanded)
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
    pairs, is_dict = der._normalize_corpus_inputs(refs, hyps)

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
