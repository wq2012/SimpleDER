import os
import numpy as np

from . import der


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
        der.check_input(data)
        recordings = {str(recording_id): data}
    elif isinstance(data, dict):
        recordings = {}
        for rec_id, segments in data.items():
            der.check_input(segments)
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
