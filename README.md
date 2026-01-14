# SimpleDER ![Python package](https://github.com/wq2012/SimpleDER/workflows/Python%20package/badge.svg) [![PyPI Version](https://img.shields.io/pypi/v/simpleder.svg)](https://pypi.python.org/pypi/simpleder) [![Python Versions](https://img.shields.io/pypi/pyversions/simpleder.svg)](https://pypi.org/project/simpleder) [![Downloads](https://static.pepy.tech/badge/simpleder)](https://pepy.tech/project/simpleder) [![codecov](https://codecov.io/gh/wq2012/SimpleDER/branch/master/graph/badge.svg)](https://codecov.io/gh/wq2012/SimpleDER) [![Documentation](https://img.shields.io/badge/api-documentation-blue.svg)](https://wq2012.github.io/SimpleDER)

## Table of Contents

* [Overview](#overview)
* [Diarization Error Rate](#diarization-error-rate)
* [Implementation](#implementation)
* [Tutorial](#tutorial)
* [Citation](#citation)

## Overview

This is a single-file lightweight library to compute Diarization Error Rate (DER).

Features **NOT** supported in this library:

* A full breakdown of the different components of DER (False Alarm, Miss, Overlap, Confusion).

For more sophisticated metrics with this support, please use
[pyannote-metrics](https://github.com/pyannote/pyannote-metrics) instead.

To learn more about speaker diarization, here is a curated list of resources:
[awesome-diarization](https://github.com/wq2012/awesome-diarization).

## Diarization Error Rate

Diarization Error Rate (DER) is the most commonly used metrics for
[speaker diarization](https://en.wikipedia.org/wiki/Speaker_diarisation).

Its strict form is:

```
       False Alarm + Miss + Overlap + Confusion
DER = ------------------------------------------
                   Reference Length
```

The definition of each term:

* `Reference Length:` The total length of the reference (ground truth).
* `False Alarm`: Length of segments which are considered as speech in
  hypothesis, but not in reference.
* `Miss`: Length of segments which are considered as speech in
  reference, but not in hypothesis.
* `Overlap`: Length of segments which are considered as overlapped speech
  in hypothesis, but not in reference.
* `Confusion`: Length of segments which are assigned to different speakers
  in hypothesis and reference (after applying an optimal assignment).

The unit of each term is *seconds*.

Note that DER can theoretically be larger than 1.0.

References:

* [pyannote-metrics documentation](https://pyannote.github.io/pyannote-metrics/reference.html)
* [Xavier Anguera's thesis](http://www.xavieranguera.com/phdthesis/node108.html)

## Implementation

This library allows efficient computation of DER including support for overlapped speech.

The algorithm works as follows:

1.  **Collar Pre-processing** (if `collar > 0`): We remove regions around reference boundaries from both the reference and the hypothesis. For every start/end time $t$ in the reference, the interval $[t - \text{collar}, t + \text{collar}]$ is excluded from scoring.

2.  **Optimal Mapping**: We first align the speakers in the hypothesis to the reference by maximizing the total overlap duration between them. This is a linear sum assignment problem (also known as the weighted bipartite matching problem), which we solve using the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`). Let `Match` be the total overlap duration of this optimal mapping.

3.  **Load Calculation**: We calculate a value called "Load", representing the total duration of speech that *requires* being matched. This accounts for overlapped speech.

    Mathematically:
    `Load` = $\int \max(N_{\text{ref}}(t), N_{\text{hyp}}(t)) dt$
    where $N_{\text{ref}}(t)$ and $N_{\text{hyp}}(t)$ are the number of active speakers at time $t$ in reference and hypothesis, respectively.

4.  **DER Calculation**:
    `DER = (Load - Match) / Reference Length`

    This formulation is mathematically equivalent to the standard definition `(Miss + False Alarm + Confusion) / Reference Length`.

## Tutorial

### Install

Install the package by:

```bash
pip3 install simpleder
```

or

```bash
python3 -m pip install simpleder
```

### API

Here is a minimal example:

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

error = simpleder.DER(ref, hyp)

print("DER={:.3f}".format(error))
```

This should output:

```
DER=0.350
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
