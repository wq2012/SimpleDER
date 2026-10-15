# SimpleDER ![Python package](https://github.com/wq2012/SimpleDER/workflows/Python%20package/badge.svg) [![PyPI Version](https://img.shields.io/pypi/v/simpleder.svg)](https://pypi.python.org/pypi/simpleder) [![Python Versions](https://img.shields.io/pypi/pyversions/simpleder.svg)](https://pypi.org/project/simpleder) [![Downloads](https://static.pepy.tech/badge/simpleder)](https://pepy.tech/project/simpleder) [![codecov](https://codecov.io/gh/wq2012/SimpleDER/branch/master/graph/badge.svg)](https://codecov.io/gh/wq2012/SimpleDER) [![Documentation](https://img.shields.io/badge/api-documentation-blue.svg)](https://wq2012.github.io/SimpleDER)

## Table of Contents

* [Overview](#overview)
* [Diarization Error Rate](#diarization-error-rate)
* [Jaccard Error Rate](#jaccard-error-rate)
* [Word Diarization Error Rate](#word-diarization-error-rate)
* [Implementation](#implementation)
* [Tutorial](#tutorial)
* [Citation](#citation)

## Overview

This is a lightweight library (depending only on `numpy` and `scipy`) to compute speaker diarization evaluation metrics:

* **Diarization Error Rate (DER)** with full component breakdown (`Miss`, `False Alarm`, `Confusion`) and optimal speaker mapping
* **Jaccard Error Rate (JER)** with per-speaker breakdown and optimal speaker mapping
* **Word Diarization Error Rate (WDER)** with word-level alignment breakdown and optimal speaker mapping
* **Overlapped speech** support, plus optional `collar` tolerance and `skip_overlap` filtering
* **Corpus-level evaluation** (`corpus_der`, `corpus_jer`, and `corpus_wder`) across multiple recordings
* **NIST RTTM I/O & CLI** (`load_rttm`, `dump_rttm`, and `python3 -m simpleder`)

To learn more about speaker diarization, here is a curated list of resources:
[awesome-diarization](https://github.com/wq2012/awesome-diarization).

## Diarization Error Rate

Diarization Error Rate (DER) is the most commonly used metric for
[speaker diarization](https://en.wikipedia.org/wiki/Speaker_diarisation):

```
       False Alarm + Miss + Confusion
DER = --------------------------------
              Reference Length
```

The definition of each term (all measured in *seconds*, accounting for overlapped speech):

* `Reference Length`: Total speaker-time duration in the reference (ground truth), $\int N_{\text{ref}}(t) \, dt$.
* `False Alarm`: Duration of hypothesis speakers in excess of reference speakers, $\int \max(0, N_{\text{hyp}}(t) - N_{\text{ref}}(t)) \, dt$.
* `Miss`: Duration of reference speakers missed by the hypothesis, $\int \max(0, N_{\text{ref}}(t) - N_{\text{hyp}}(t)) \, dt$.
* `Confusion`: Duration of active speakers assigned to the wrong speaker after applying an optimal 1-to-1 speaker mapping.

Note that DER can be larger than `1.0`.

References:

* [pyannote-metrics documentation](https://pyannote.github.io/pyannote-metrics/reference.html)
* [Xavier Anguera's thesis](http://www.xavieranguera.com/phdthesis/node108.html)

## Jaccard Error Rate

Jaccard Error Rate (JER), introduced for the DIHARD II challenge, evaluates diarization with equal weight for each reference speaker. Under an optimal 1-to-1 mapping between reference speakers $r \in S_{\text{ref}}$ and hypothesis speakers $h \in S_{\text{hyp}}$:

$$\text{JER} = \frac{1}{|S_{\text{ref}}|} \sum_{r \in S_{\text{ref}}} \left(1 - \frac{\text{duration}(r \cap h)}{\text{duration}(r \cup h)}\right)$$

Unmapped reference speakers receive a speaker error of `1.0`. Unlike DER, JER is always bounded in `[0.0, 1.0]`.

## Word Diarization Error Rate

Word Diarization Error Rate (WDER), proposed by [Shafey et al. (Interspeech 2019)](https://arxiv.org/pdf/1907.05337) and used in [DiarizationLM](https://github.com/google/speaker-id/tree/master/DiarizationLM), evaluates joint speech recognition and speaker diarization at the word level. After aligning the normalized reference and hypothesis words via Levenshtein edit distance, WDER computes the fraction of aligned words (correct words $C$ plus substituted words $S$) assigned to the wrong speaker under an optimal 1-to-1 speaker mapping:

$$\text{WDER} = \frac{S_{\text{IS}} + C_{\text{IS}}}{S + C}$$

## Implementation

1. **Per-Speaker Canonicalization**: Overlapping or contiguous segments belonging to the same speaker are merged so each speaker's timeline consists of disjoint intervals.
2. **Pre-processing (`collar` and `skip_overlap`)**:
   * If `collar > 0`: For every reference boundary $t$, the interval $[t - \text{collar}, t + \text{collar}]$ (NIST `md-eval` convention) is excluded from both reference and hypothesis.
   * If `skip_overlap=True`: Regions where $N_{\text{ref}}(t) > 1$ are excluded from both reference and hypothesis.
3. **Optimal Mapping**: We solve the linear sum assignment problem via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)—maximizing total overlap duration $\text{Match}$ for DER, total Jaccard similarity for JER, or aligned word count for WDER.
4. **Error & Load Calculation**: Let $\text{Load} = \int \max(N_{\text{ref}}(t), N_{\text{hyp}}(t)) \, dt$. Then:
   $$\text{Miss} + \text{False Alarm} + \text{Confusion} = \text{Load} - \text{Match}$$
   $$\text{DER} = \frac{\text{Load} - \text{Match}}{\text{Reference Length}}$$

## Tutorial

### Install

```bash
pip3 install simpleder
```

### API

#### Basic DER and JER

```python
import simpleder

# reference (ground truth)
ref = [("A", 0.0, 1.0),
       ("B", 1.0, 1.5),
       ("A", 1.6, 2.1)]

# hypothesis (diarization result from your algorithm)
hyp = [("1", 0.0, 0.8),
       ("2", 0.8, 1.4),
       ("3", 1.5, 1.8),
       ("1", 1.8, 2.0)]

der = simpleder.DER(ref, hyp)
jer = simpleder.JER(ref, hyp)
print(f"DER={der:.3f}, JER={jer:.3f}")
# DER=0.350, JER=0.381
```

#### Word Diarization Error Rate (WDER)

`WDER` accepts a list of `(speaker, word_or_phrase)` tuples for both `ref` and `hyp`:

```python
ref_words = [("A", "hello"), ("A", "world"), ("B", "how"), ("B", "are"), ("B", "you")]
hyp_words = [("1", "hello"), ("2", "world"), ("2", "how"), ("2", "are"), ("2", "you")]

wder = simpleder.WDER(ref_words, hyp_words)
print(f"WDER={wder:.3f}")
# WDER=0.200

# Multi-word utterances with punctuation work identically:
ref_turns = [("A", "Hello, world!"), ("B", "How are you?")]
hyp_turns = [("1", "hello"), ("2", "world, how are you.")]
print(simpleder.WDER(ref_turns, hyp_turns, detailed=True))
```

#### Options: `collar`, `skip_overlap`, and `detailed` Breakdown

```python
details = simpleder.DER(ref, hyp, collar=0.0, skip_overlap=False, detailed=True)
# {
#   'der': 0.35,
#   'miss': 0.2,
#   'false_alarm': 0.1,
#   'confusion': 0.4,
#   'ref_length': 2.0,
#   'speaker_mapping': {'A': '1', 'B': '2'}
# }
```

#### Corpus-Level Evaluation (Multiple Recordings)

`corpus_der`, `corpus_jer`, and `corpus_wder` evaluate across multiple recordings (accepting either `dict` or `list` inputs):

```python
refs = {"rec1": ref, "rec2": [("A", 0.0, 2.0)]}
hyps = {"rec1": hyp, "rec2": [("1", 0.0, 2.0)]}

total_der = simpleder.corpus_der(refs, hyps)
total_jer = simpleder.corpus_jer(refs, hyps)
```

#### RTTM File I/O and Command-Line Interface

```python
simpleder.dump_rttm(refs, "ref.rttm")
simpleder.dump_rttm(hyps, "hyp.rttm")

loaded_refs = simpleder.load_rttm("ref.rttm")
```

From the terminal:

```bash
python3 -m simpleder --ref ref.rttm --hyp hyp.rttm --collar 0.25 --detailed
```

## Citation

We developed this package as part of the following work:

```
@inproceedings{wang2018speaker,
  title={{Speaker Diarization with LSTM}},
  author={Wang, Quan and Downey, Carlton and Wan, Li and Mansfield, Philip Andrew and Moreno, Ignacio Lopz},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5239--5243},
  year={2018},
  organization={IEEE}
}

@inproceedings{xia2022turn,
  title={{Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection}},
  author={Wei Xia and Han Lu and Quan Wang and Anshuman Tripathi and Yiling Huang and Ignacio Lopez Moreno and Hasim Sak},
  booktitle={2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8077--8081},
  year={2022},
  organization={IEEE}
}

@inproceedings{huang24d_interspeech,
  title={{On the Success and Limitations of Auxiliary Network Based Word-Level End-to-End Neural Speaker Diarization}},
  author={Yiling Huang and Weiran Wang and Guanlong Zhao and Hank Liao and Wei Xia and Quan Wang},
  year={2024},
  booktitle={Interspeech 2024},
  pages={32--36},
}
```
