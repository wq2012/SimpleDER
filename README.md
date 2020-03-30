# SimpleDER [![Build Status](https://travis-ci.org/wq2012/SimpleDER.svg?branch=master)](https://travis-ci.org/wq2012/SimpleDER) [![PyPI Version](https://img.shields.io/pypi/v/simpleder.svg)](https://pypi.python.org/pypi/simpleder) [![Python Versions](https://img.shields.io/pypi/pyversions/simpleder.svg)](https://pypi.org/project/simpleder) [![Downloads](https://pepy.tech/badge/simpleder)](https://pepy.tech/project/simpleder) [![codecov](https://codecov.io/gh/wq2012/SimpleDER/branch/master/graph/badge.svg)](https://codecov.io/gh/wq2012/SimpleDER) [![Documentation](https://img.shields.io/badge/api-documentation-blue.svg)](https://wq2012.github.io/SimpleDER)

## Overview

This is a lightweight library to compute Diarization Error Rate (DER).

Features **NOT** supported:

* Handling overlapped speech, *i.e.* two speakers speaking at the same time.
* Allowing segment boundary tolerance, *a.k.a.* the `collar` value.

For more sophisticated metrics with these supports, please use
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
  **This library does NOT support overlap.**
* `Confusion`: Length of segments which are assigned to different speakers
  in hypothesis and reference (after applying an optimal assignment).

The unit of each term is *seconds*.

Note that DER can theoretically be larger than 1.0.

References:

* [pyannote-metrics documentation](https://pyannote.github.io/pyannote-metrics/reference.html)
* [Xavier Anguera's thesis](http://www.xavieranguera.com/phdthesis/node108.html)

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