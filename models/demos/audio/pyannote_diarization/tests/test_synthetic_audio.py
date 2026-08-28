# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Guard the synthetic audio the on-device overlap test depends on."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from models.demos.audio.pyannote_diarization.tests.synthetic_audio import (  # noqa: E402
    overlapping_speech,
)


OVERLAP_WINDOWS = ((8.0, 9.0), (11.0, 15.0))
SINGLE_SPEAKER_WINDOWS = ((3.0, 7.0), (9.5, 10.5), (16.0, 18.0))


def _rms(waveform, sample_rate, start, end):
    span = waveform[int(start * sample_rate) : int(end * sample_rate)]
    return float(span.pow(2).mean().sqrt())


def test_synthetic_audio_really_overlaps_and_is_reproducible():
    """Guard the fixture the overlap test depends on.

    If the voice schedule is ever edited so the windows stop overlapping, the
    on-device test would still pass while silently no longer covering overlap.
    Two speakers at once are louder than one, so the RMS separates the cases;
    and the audio must be identical run to run, which is the whole reason for
    generating it instead of downloading a corpus.
    """
    first = overlapping_speech()
    second = overlapping_speech()
    assert torch.equal(first["waveform"], second["waveform"])

    waveform, sample_rate = first["waveform"][0], first["sample_rate"]
    quietest_overlap = min(
        _rms(waveform, sample_rate, *w) for w in OVERLAP_WINDOWS
    )
    loudest_single = max(
        _rms(waveform, sample_rate, *w) for w in SINGLE_SPEAKER_WINDOWS
    )
    assert quietest_overlap > loudest_single, (
        f"overlap windows {OVERLAP_WINDOWS} are not louder than the "
        f"single-speaker ones: {quietest_overlap} vs {loudest_single}"
    )


