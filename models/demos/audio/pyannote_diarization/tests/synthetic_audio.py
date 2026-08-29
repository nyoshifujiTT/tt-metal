# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Deterministic multi-speaker audio built from pyannote's bundled sample.

The bundled 30 s sample is one conversation with no overlap, so on its own it
never drives the segmentation net down its multi-active-speaker path and never
gives the clustering more than two well-separated embeddings.

Rather than download a corpus to reach those cases, this rearranges the sample
itself: take the longest turn of each annotated speaker and lay them out as
"A alone / A and B together / B alone". The result contains real speech from
real, distinguishable speakers -- which synthetic tones do not: pyannote hears
a pitched tone sequence as a single speaker, so a fixture built that way would
silently test nothing.

Everything here is a deterministic function of the shipped wav and rttm, so
there is no seed to drift and no dataset to fetch.
"""

import numpy as np
import torch

from models.demos.audio.pyannote_diarization import accuracy


def _longest_turn_per_speaker(reference):
    """Longest (start, end) annotated for each speaker, by speaker label."""
    longest = {}
    for segment, _, speaker in reference.itertracks(yield_label=True):
        duration = segment.end - segment.start
        if speaker not in longest or duration > longest[speaker][2]:
            longest[speaker] = (segment.start, segment.end, duration)
    return longest


def overlapping_speech():
    """Return ``{"waveform", "sample_rate"}`` holding two overlapping speakers.

    Layout is A alone, then A and B summed, then B alone, so the middle third
    has two speakers active at once. Both halves of the pipeline see the same
    audio, so the on-device comparison is exact regardless of how pyannote
    happens to label the result.
    """
    import soundfile as sf

    samples, sample_rate = sf.read(
        accuracy.sample_audio_path(), dtype="float32", always_2d=True
    )
    mono = samples[:, 0]
    reference = accuracy.load_rttm(accuracy.sample_reference_path())

    longest = _longest_turn_per_speaker(reference)
    if len(longest) < 2:
        raise RuntimeError(
            f"the bundled annotation must hold at least two speakers, found "
            f"{sorted(longest)}"
        )

    (start_a, end_a, _), (start_b, end_b, _) = (
        longest[speaker] for speaker in sorted(longest)[:2]
    )
    first = mono[int(start_a * sample_rate) : int(end_a * sample_rate)]
    second = mono[int(start_b * sample_rate) : int(end_b * sample_rate)]

    # Trim to a common length so the overlapped section has both speakers
    # throughout rather than one of them trailing off.
    width = min(len(first), len(second))
    first, second = first[:width], second[:width]

    mixed = np.concatenate([first, first + second, second])
    return {
        "waveform": torch.from_numpy(mixed).unsqueeze(0),
        "sample_rate": sample_rate,
    }


def overlap_window(audio):
    """(start, end) seconds of the section where both speakers are active."""
    total = audio["waveform"].shape[-1] / audio["sample_rate"]
    third = total / 3.0
    return third, 2.0 * third
