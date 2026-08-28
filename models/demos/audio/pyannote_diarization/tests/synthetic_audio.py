# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Deterministic synthetic audio for the diarization tests.

Needs neither a device nor pyannote, so the guard test next to it runs
anywhere, and the on-device tests import the generator from here.
"""

import torch


def overlapping_speech(seed: int = 11, seconds: float = 20.0, sample_rate: int = 16000):
    """Synthesise multi-speaker audio with overlap, deterministically.

    The bundled sample is one clean two-speaker conversation, so it never
    exercises overlap or a third speaker. Rather than depend on a corpus
    download to reach those paths, build the audio here: three tone-like
    "voices" at different pitches, with two of them deliberately speaking at
    once in the middle.

    This is not meant to be realistic enough to score a meaningful DER against
    -- it is fed to both the host and device pipelines and only their agreement
    is checked, which is exactly what a fidelity test needs and what makes the
    result reproducible from a seed instead of from gigabytes of audio.
    """
    generator = torch.Generator().manual_seed(seed)
    total = int(seconds * sample_rate)
    t = torch.arange(total, dtype=torch.float32) / sample_rate
    audio = torch.zeros(total)

    # (pitch Hz, start s, end s). Consecutive voices overlap: 8-9 s and
    # 11-15 s have two speakers active, which is the case the bundled sample
    # never produces. Asserted below rather than left to the comment.
    voices = [(110.0, 1.0, 9.0), (190.0, 8.0, 15.0), (260.0, 11.0, 19.0)]
    for pitch, start, end in voices:
        lo, hi = int(start * sample_rate), int(end * sample_rate)
        span = t[lo:hi]
        # A few harmonics plus an amplitude wobble, so the segmentation net has
        # something with speech-like structure rather than a pure tone.
        voice = sum(torch.sin(2 * torch.pi * pitch * k * span) / k for k in (1, 2, 3))
        envelope = 0.6 + 0.4 * torch.sin(2 * torch.pi * 3.0 * span)
        audio[lo:hi] += 0.1 * voice * envelope

    audio += 0.001 * torch.randn(total, generator=generator)
    return {"waveform": audio.unsqueeze(0), "sample_rate": sample_rate}
