# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Corpus-scale diarization accuracy for the community-1 port.

The 30 s sample that ships with pyannote is enough to catch a pipeline that is
plainly broken, but it is one clean two-speaker clip: it cannot tell whether
the port still handles overlapping speech, many speakers, or noisy audio, and
its DER cannot be compared against any published figure. This test scores a
real corpus so the measured DER can be checked against what the model is
published as scoring.

Corpora are gigabytes and are never downloaded by a test run. Prepare one as

    <root>/audio/<id>.wav
    <root>/rttm/<id>.rttm

and point ``DIARIZATION_CORPUS_DIR`` (or ``DIARIZATION_VOXCONVERSE_DIR``) at
it. VoxConverse is the practical choice -- audio and annotations are both CC-BY
downloads -- whereas DIHARD needs an LDC licence. Without the variable the test
skips rather than failing, so CI without the data is unaffected.

Set ``DIARIZATION_CORPUS_LIMIT`` to score only the first N recordings; the
files are sorted, so a limited run is reproducible, but note that a subset does
not reproduce the published DER exactly.
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("ttnn")
pytest.importorskip("pyannote.audio")

import torch  # noqa: E402

from models.demos.audio.pyannote_diarization import accuracy  # noqa: E402
from models.demos.audio.pyannote_diarization.tests.test_diarization_e2e_ondevice import (  # noqa: E402
    _load_pipeline,
    _offload_embedding,
    _offload_segmentation,
)

CORPUS_NAME = os.environ.get("DIARIZATION_CORPUS_NAME", "voxconverse-test")


def _corpus_root():
    root = accuracy.corpus_root(CORPUS_NAME)
    if root is None:
        pytest.skip(
            "no diarization corpus available; set DIARIZATION_CORPUS_DIR to a "
            "directory holding audio/<id>.wav and rttm/<id>.rttm"
        )
    return root


def _limit():
    raw = os.environ.get("DIARIZATION_CORPUS_LIMIT")
    return int(raw) if raw else None


def _diarize_with(pipeline):
    """Return ``diarize(wav_path) -> turns`` for accuracy.corpus_der."""
    import soundfile as sf

    def diarize(wav_path):
        # Hand pyannote an in-memory waveform: decoding a path goes through
        # torchcodec, whose wheels are tied to a torch release the image does
        # not carry.
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = {
            "waveform": torch.from_numpy(samples.T.copy()),
            "sample_rate": sample_rate,
        }
        annotation = pipeline(audio).speaker_diarization
        return [
            {"speaker": speaker, "start": segment.start, "end": segment.end}
            for segment, _, speaker in annotation.itertracks(yield_label=True)
        ]

    return diarize


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_corpus_der_matches_the_published_figure(device, model_location_generator):
    """The device pipeline must score what this model is published as scoring.

    A port can pass the single-sample checks while having quietly lost, say,
    overlap handling; only a corpus with varied audio exposes that. The gate is
    the published DER plus a tolerance, because the published number comes from
    pyannote's own harness and small harness differences move it.

    The split has to be the published one. Scoring a development split against
    a test-split figure passes easily -- dev is the easier half -- and would
    report success while measuring the wrong thing, so a split with no published
    number of its own is reported and not gated.
    """
    root = _corpus_root()

    pipeline = _load_pipeline(model_location_generator)
    _offload_embedding(pipeline, device)
    _offload_segmentation(pipeline, device)

    scored = accuracy.corpus_der(_diarize_with(pipeline), root, limit=_limit())

    published = accuracy.published_corpus_der(CORPUS_NAME)
    worst = max(scored["per_recording"].items(), key=lambda kv: kv[1])
    if published is None:
        pytest.skip(
            f"{CORPUS_NAME} DER {scored['der']:.4f} over "
            f"{scored['num_recordings']} recordings (worst {worst[0]} at "
            f"{worst[1]:.4f}); no published figure for this split, so there is "
            "nothing to gate against -- use the published split to assert"
        )

    ceiling = published + accuracy.CORPUS_DER_TOLERANCE
    assert scored["der"] <= ceiling, (
        f"{CORPUS_NAME} DER {scored['der']:.4f} over {scored['num_recordings']} "
        f"recordings exceeds the published {published:.4f} + "
        f"{accuracy.CORPUS_DER_TOLERANCE:.2f}; worst recording {worst[0]} "
        f"at {worst[1]:.4f}"
    )
