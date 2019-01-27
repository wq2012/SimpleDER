# SimpleDER [![Build Status](https://travis-ci.org/wq2012/SimpleDER.svg?branch=master)](https://travis-ci.org/wq2012/SimpleDER) [![PyPI Version](https://img.shields.io/pypi/v/simpleder.svg)](https://pypi.python.org/pypi/simpleder) [![Python Versions](https://img.shields.io/pypi/pyversions/simpleder.svg)](https://pypi.org/project/simpleder)

## Overview

This is a lightweight library to compute Diarization Error Rate (DER).

Features **NOT** supported:

* Handling overlapped speech, *i.e.* two speakers speaking at the same time.
* Allowing segment boundary tolerance, *a.k.a.* the `collar` value.

For more sophisticated with these supports, please use
[pyannote-metrics](https://github.com/pyannote/pyannote-metrics) instead.

To learn more about speaker diarization, here is a curated list of resources:
[awesome-diarization](https://github.com/wq2012/awesome-diarization).

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
import der

# reference (ground truth)
ref = [("A", 0.0, 1.0),
       ("B", 1.0, 1.5),
       ("A", 1.6, 2.1)]

# hypothesis (diarization result from your algorithm)
hyp = [("1", 0.0, 0.8),
       ("2", 0.8, 1.4),
       ("3", 1.5, 1.8),
       ("1", 1.8, 2.0)]

error = der.DER(ref, hyp)

print("DER={:.3f}".format(error))
```

This should output:

```
DER=0.350
```