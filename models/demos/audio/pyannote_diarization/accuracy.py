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

# Published DER per corpus, which is what a corpus run can actually be compared
# against. The single 30 s sample cannot: it is one clean two-speaker clip,
# while these are hours of meeting and broadcast audio.
#
# The split matters and is part of the key. pyannote reports these on the
# evaluation split, so scoring a development split and comparing against the
# same number is not a like-for-like check: dev is the easier half, and a run
# that lands *below* the published figure is a sign the comparison is wrong
# rather than that the port is good. Measured here: VoxConverse dev scores
# 0.0705 while the published 0.112 is a test-split figure.
PUBLISHED_CORPUS_DER = {
    "ami": 0.170,  # AMI IHM, test
    "dihard3": 0.202,  # DIHARD 3 full, eval
    "voxconverse-test": 0.112,  # VoxConverse v0.3, test -- the published split
}

# Splits with no published figure of their own. They are still worth scoring --
# a regression shows up on any audio -- but a run must not be compared against
# another split's number, so they carry no target and the test reports the DER
# without gating on it.
CORPUS_NO_PUBLISHED_FIGURE = {
    "voxconverse-dev",  # measured 0.0705; easier than test, not comparable
}

# How far a corpus run may sit above the published figure before it counts as a
# regression. The published numbers come from pyannote's own harness, so an
# exact match is not expected -- resampling or overlap handling move the third
# decimal -- but a real break moves it much further.
CORPUS_DER_TOLERANCE = 0.05

# Measured on the published split: all 232 VoxConverse test recordings through
# the p150, embedding on device, scored 0.1113 against the published 0.112 --
# 0.6% apart, so the port reproduces the model's published accuracy. The dev
# split scores 0.0705 on the same code, which is why the split is part of the
# key: landing under a published figure means the comparison is wrong.
#
# Per-recording DERs can exceed 1.0 where a recording holds only seconds of
# annotated speech (the denominator is small), which is why the metric is
# accumulated by speech time rather than averaged per file.


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


def corpus_root(name: str = "voxconverse"):
    """Local directory holding a diarization corpus, or ``None`` if unset.

    Corpora are hundreds of files and gigabytes, so they are never downloaded
    by a test run. Point ``DIARIZATION_CORPUS_DIR`` (or the per-corpus
    ``DIARIZATION_<NAME>_DIR``) at a prepared directory holding ``audio/*.wav``
    and ``rttm/*.rttm``, and the corpus test runs; otherwise it skips.
    """
    env_name = name.upper().replace("-", "_")
    specific = os.environ.get(f"DIARIZATION_{env_name}_DIR")
    generic = os.environ.get("DIARIZATION_CORPUS_DIR")
    for candidate in (specific, generic):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def published_corpus_der(name: str):
    """Published DER for a corpus split, or ``None`` when there is not one.

    Returning ``None`` rather than falling back to a neighbouring split's figure
    is deliberate: comparing a dev-split run against a test-split number reads
    as a pass while measuring the wrong thing.
    """
    if name in CORPUS_NO_PUBLISHED_FIGURE:
        return None
    return PUBLISHED_CORPUS_DER.get(name)


def corpus_files(root: str, limit=None):
    """Pair ``audio/<id>.wav`` with ``rttm/<id>.rttm`` under ``root``.

    Returns ``[(recording_id, wav_path, rttm_path), ...]`` sorted by id, so a
    ``limit`` selects the same subset on every run rather than a random one.
    """
    audio_dir = os.path.join(root, "audio")
    rttm_dir = os.path.join(root, "rttm")
    pairs = []
    for entry in sorted(os.listdir(audio_dir)):
        if not entry.endswith(".wav"):
            continue
        recording_id = entry[: -len(".wav")]
        rttm = os.path.join(rttm_dir, recording_id + ".rttm")
        if os.path.exists(rttm):
            pairs.append((recording_id, os.path.join(audio_dir, entry), rttm))
    return pairs[:limit] if limit else pairs


def corpus_der(diarize, root: str, limit=None) -> dict:
    """Score a whole corpus, accumulating one DER over every recording.

    ``diarize(wav_path) -> [{speaker, start, end}, ...]``.

    The metric is accumulated rather than averaged per file, which is how the
    published figures are computed: a five-minute recording should weigh more
    than a thirty-second one. Per-recording DERs are returned alongside so a
    regression can be traced to the file that caused it.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate()
    per_recording = {}
    for recording_id, wav, rttm in corpus_files(root, limit):
        reference = load_rttm(rttm)
        hypothesis = turns_to_annotation(diarize(wav))
        per_recording[recording_id] = float(metric(reference, hypothesis))
    if not per_recording:
        raise RuntimeError(f"no audio/rttm pairs found under {root}")
    return {
        "der": abs(metric),
        "num_recordings": len(per_recording),
        "per_recording": per_recording,
    }
