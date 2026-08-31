# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""End-to-end community-1 diarization with the neural nets on the p150.

Two things are checked, and they answer different questions:

* *fidelity* -- the pipeline is run twice over the same recording, once
  entirely on host and once with the WeSpeaker embedding (and optionally the
  PyanNet segmentation) executed through ttnn, and the device run must
  reproduce the host run. This catches a ttnn kernel drifting from the
  reference implementation.
* *accuracy* -- the device run is scored against the human annotation shipped
  with the sample. This catches the pipeline being wrong in absolute terms, in
  a way host and device would share and fidelity alone would never reveal.

Both use diarization error rate, the standard metric, differing only in what
serves as the reference; the scoring lives in ``common``'s sibling
``accuracy`` module so tt-inference-server's eval workflow reports the same
number for the served model. Raw segment counts are not used: bf16 arithmetic
can split or merge a turn at a boundary without changing who is speaking when.
The audio and its annotation both ship with pyannote.audio, so no external
fixture is needed.
"""
import pytest

from models.demos.audio.pyannote_diarization import accuracy
from models.demos.audio.pyannote_diarization.tests.synthetic_audio import (
    overlapping_speech,
)
from models.demos.audio.pyannote_diarization import pipeline as diar_pipeline
from models.demos.audio.pyannote_diarization.pipeline import (
    load_pipeline,
    sample_audio_path,
    speakers,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("offload_segmentation", [False, True], ids=["embedding_only", "both_nets"])
def test_diarization_matches_host_pipeline(device, model_location_generator, offload_segmentation):
    audio = sample_audio_path()

    host = load_pipeline(model_location_generator)(audio).speaker_diarization

    pipeline = load_pipeline(model_location_generator)
    diar_pipeline.offload_embedding(pipeline, device)
    if offload_segmentation:
        diar_pipeline.offload_segmentation(pipeline, device)
    on_device = pipeline(audio).speaker_diarization

    assert len(speakers(host)) >= 2, "host pipeline should find multiple speakers in the sample recording"
    assert len(speakers(on_device)) == len(speakers(host)), (
        f"speaker count differs: device={sorted(speakers(on_device))} " f"host={sorted(speakers(host))}"
    )

    der = accuracy.diarization_error_rate(host, on_device)
    assert der < accuracy.FIDELITY_DER_MAX, f"diarization error rate against the host pipeline too high: {der}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("offload_segmentation", [False, True], ids=["embedding_only", "both_nets"])
def test_diarization_matches_human_annotation(device, model_location_generator, offload_segmentation):
    """The device pipeline must be right in absolute terms, not just consistent.

    Fidelity against the host run cannot catch a pipeline that is misconfigured
    identically on both sides -- wrong clustering threshold, wrong checkpoint --
    because both runs would agree while both are wrong. Scoring against the
    human annotation that ships with the sample catches that.

    This is the same measurement tt-inference-server's eval workflow reports for
    the served model, through the same ``accuracy`` helpers.
    """
    pipeline = load_pipeline(model_location_generator)
    diar_pipeline.offload_embedding(pipeline, device)
    if offload_segmentation:
        diar_pipeline.offload_segmentation(pipeline, device)
    on_device = pipeline(sample_audio_path()).speaker_diarization

    reference = accuracy.load_rttm(accuracy.sample_reference_path())
    scored = accuracy.score_against_reference(on_device, reference)

    assert scored["speaker_count_matches"], (
        f"speaker count differs from the annotation: device={scored['num_speakers']} "
        f"reference={scored['reference_num_speakers']}"
    )
    assert scored["der"] < accuracy.ACCURACY_DER_MAX, (
        f"diarization error rate against the human annotation too high: {scored['der']} "
        f"(published DER for this model is {accuracy.PUBLISHED_DER}, see {accuracy.PUBLISHED_DER_REF})"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("offload_segmentation", [False, True], ids=["embedding_only", "both_nets"])
def test_diarization_matches_host_pipeline_on_overlapping_speech(
    device, model_location_generator, offload_segmentation
):
    """Fidelity must hold on overlapping speech, not just the clean sample.

    Overlap drives the segmentation net down a different path -- multiple
    speakers active in one frame -- which the bundled sample never does, so a
    port that broke it would still pass the sample test.

    The audio is the sample's own two speakers rearranged to talk over each
    other, so it needs no corpus download and stays real speech; see
    synthetic_audio for why tones do not work here.
    """
    audio = overlapping_speech()

    host = load_pipeline(model_location_generator)(audio).speaker_diarization

    pipeline = load_pipeline(model_location_generator)
    diar_pipeline.offload_embedding(pipeline, device)
    if offload_segmentation:
        diar_pipeline.offload_segmentation(pipeline, device)
    on_device = pipeline(audio).speaker_diarization

    assert len(speakers(on_device)) == len(speakers(host)), (
        f"speaker count differs on overlapping speech: "
        f"device={sorted(speakers(on_device))} host={sorted(speakers(host))}"
    )

    der = accuracy.diarization_error_rate(host, on_device)
    assert der < accuracy.FIDELITY_DER_MAX, (
        f"diarization error rate against the host pipeline on overlapping " f"speech too high: {der}"
    )
