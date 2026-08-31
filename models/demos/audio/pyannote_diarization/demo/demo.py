# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Speaker diarization demo: community-1 with the neural nets on a p150.

Prints the ``{start, end, speaker}`` turns the pipeline produces with the
WeSpeaker embedding -- and optionally the PyanNet segmentation -- executed
through ttnn.

Run it on pyannote's bundled 30 s sample:

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/demo/demo.py
```

Run it on your own recording, and put both nets on the device:

```sh
pytest --disable-warnings --input-path=/path/to/audio.wav \\
    models/demos/audio/pyannote_diarization/demo/demo.py::test_demo_both_nets
```
"""
import pytest
from loguru import logger

from models.demos.audio.pyannote_diarization import pipeline as diar_pipeline
from models.demos.audio.pyannote_diarization.pipeline import (
    load_pipeline,
    sample_audio_path,
)


def _audio(input_path):
    """The recording to diarize: the caller's, else pyannote's bundled sample."""
    return input_path or sample_audio_path()


def _run(device, model_location_generator, input_path, offload_segmentation):
    audio = _audio(input_path)
    pipeline = load_pipeline(model_location_generator)
    diar_pipeline.offload_embedding(pipeline, device)
    if offload_segmentation:
        diar_pipeline.offload_segmentation(pipeline, device)

    diarization = pipeline(audio).speaker_diarization
    turns = [
        (round(segment.start, 2), round(segment.end, 2), speaker)
        for segment, _, speaker in diarization.itertracks(yield_label=True)
    ]

    nets = "embedding + segmentation" if offload_segmentation else "embedding"
    logger.info(f"{audio}: {len(turns)} turns, {nets} on device")
    for start, end, speaker in turns:
        logger.info(f"  {start:7.2f} - {end:7.2f}  {speaker}")

    assert turns, "diarization produced no speaker turns"
    return turns


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_demo(device, model_location_generator, input_path):
    """WeSpeaker embedding on device, segmentation on host."""
    _run(device, model_location_generator, input_path, offload_segmentation=False)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_demo_both_nets(device, model_location_generator, input_path):
    """Both neural nets on device."""
    _run(device, model_location_generator, input_path, offload_segmentation=True)
