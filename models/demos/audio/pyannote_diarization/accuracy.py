# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Shared diarization accuracy scoring for the community-1 port.

Two questions get asked about this port and they are not the same:

* does the device run reproduce the host run? -- a *fidelity* check, which
  catches a ttnn kernel that drifts from the reference implementation;
* is the pipeline right in absolute terms? -- an *accuracy* check against a
  human annotation, which catches the pipeline being misconfigured in a way
  that both host and device share.

Both are diarization error rates, differing only in what is used as the
reference, so they live together here. tt-inference-server's eval workflow
imports this module too, so the number it reports for the served model is
produced by exactly the same code as the tt-metal test.

Note that a healthy pipeline is not expected to score zero against the human
annotation: the reference marks speech the way a listener hears it, and any
diarizer differs from that at turn boundaries. The published DER figures for
community-1 are of that kind. Fidelity, by contrast, is expected to be ~0.
"""

import os

# Threshold for the device-vs-host fidelity check. Device arithmetic is bf16,
# so a turn boundary can move by a frame without changing who is speaking.
FIDELITY_DER_MAX = 0.05

# Threshold for the accuracy check against the shipped human annotation. The
# pipeline scores ~0.05 on this clip -- it is not a fidelity failure but the
# ordinary gap between any diarizer and a human transcript -- so the gate sits
# below community-1's own published DER while leaving room for that gap.
ACCURACY_DER_MAX = 0.15

# community-1's published DER, for reporting context. Measured on full corpora
# (AMI IHM 17.0%, DIHARD3 20.2%, VoxConverse 11.2%), not on the 30 s sample, so
# it is recorded next to the measured value rather than used as a threshold.
PUBLISHED_DER = 0.170
PUBLISHED_DER_REF = "https://huggingface.co/pyannote/speaker-diarization-community-1"


def sample_audio_path() -> str:
    """The 30 s two-speaker recording shipped inside ``pyannote.audio``.

    Built from the package path rather than by importing
    ``pyannote.audio.sample``, which decodes the file eagerly at import time
    and therefore needs a working torchcodec.
    """
    import pyannote.audio

    return os.path.join(os.path.dirname(pyannote.audio.__file__), "sample", "sample.wav")


def sample_reference_path() -> str:
    """The human annotation shipped beside :func:`sample_audio_path`."""
    import pyannote.audio

    return os.path.join(os.path.dirname(pyannote.audio.__file__), "sample", "sample.rttm")


def load_rttm(path: str):
    """Read an RTTM file into a ``pyannote.core.Annotation``."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            # RTTM stores onset and duration; Annotation wants onset and offset.
            start, duration, speaker = float(fields[3]), float(fields[4]), fields[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation


def turns_to_annotation(turns):
    """Convert ``[{speaker, start, end}, ...]`` into an ``Annotation``.

    This is the shape the served API returns, so a served result and a local
    pipeline result can be scored by the same code.
    """
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for turn in turns:
        annotation[Segment(turn["start"], turn["end"])] = turn["speaker"]
    return annotation


def diarization_error_rate(reference, hypothesis) -> float:
    """DER of ``hypothesis`` against ``reference``."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    return float(DiarizationErrorRate()(reference, hypothesis))


def speaker_count(annotation) -> int:
    """Number of distinct speakers in an ``Annotation``."""
    return len(annotation.labels())


def score_against_reference(hypothesis, reference) -> dict:
    """Score one diarization, returning the DER and the speaker counts.

    The speaker count is returned separately rather than folded into the DER:
    a pipeline that splits or merges speakers can still post an acceptable DER,
    so callers gate on both.
    """
    return {
        "der": diarization_error_rate(reference, hypothesis),
        "num_speakers": speaker_count(hypothesis),
        "reference_num_speakers": speaker_count(reference),
        "speaker_count_matches": speaker_count(hypothesis) == speaker_count(reference),
    }
