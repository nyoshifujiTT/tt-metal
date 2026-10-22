# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Guard the multi-speaker audio the on-device overlap test depends on.

The overlap test is only worth running if its input really holds two
distinguishable speakers talking over each other. An earlier version of this
fixture synthesised pitched tones, which pyannote labelled as a single speaker
-- the on-device test still passed while covering neither overlap nor speaker
separation. These checks make that failure mode visible.
"""
import pytest

import torch

from models.demos.audio.pyannote_diarization.tests.synthetic_audio import (
    overlap_window,
    overlapping_speech,
)


def _rms(waveform, sample_rate, start, end):
    span = waveform[int(start * sample_rate) : int(end * sample_rate)]
    return float(span.pow(2).mean().sqrt())


def test_audio_is_reproducible():
    """Derived from the shipped wav, so two calls must be bit-identical."""
    first = overlapping_speech()
    second = overlapping_speech()
    assert torch.equal(first["waveform"], second["waveform"])
    assert first["sample_rate"] == second["sample_rate"]


def test_middle_section_really_has_two_speakers_mixed():
    """Two voices summed are louder than either alone."""
    audio = overlapping_speech()
    waveform, sample_rate = audio["waveform"][0], audio["sample_rate"]
    total = waveform.shape[-1] / sample_rate
    third = total / 3.0

    first_alone = _rms(waveform, sample_rate, 0.0, third)
    overlapped = _rms(waveform, sample_rate, *overlap_window(audio))
    second_alone = _rms(waveform, sample_rate, 2.0 * third, total)

    assert overlapped > first_alone
    assert overlapped > second_alone


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_pyannote_hears_two_speakers(device, model_location_generator):
    """The fixture is worthless unless pyannote itself separates the speakers.

    This is the check that would have caught the tone-based fixture: it found
    exactly one speaker, so the overlap test it fed was comparing two runs on
    audio that exercised none of what it claimed to.
    """
    from models.demos.audio.pyannote_diarization import pipeline as diar_pipeline

    diarization = diar_pipeline.load_pipeline(model_location_generator)(overlapping_speech()).speaker_diarization

    speakers = diar_pipeline.speakers(diarization)
    assert len(speakers) >= 2, f"the fixture must hold speakers pyannote can tell apart, got {speakers}"
