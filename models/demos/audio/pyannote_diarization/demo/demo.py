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

pytest.importorskip("torch")
pytest.importorskip("ttnn")
pytest.importorskip("pyannote.audio")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.demos.audio.pyannote_diarization import common  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import TTNNPyanNet  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker  # noqa: E402


def _install_torch_load_shim():
    """pyannote checkpoints hold non-tensor objects; lightning>=2.6 flips the default."""
    original = torch.load
    if getattr(original, "_diar_shim", False):
        return

    def patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    patched._diar_shim = True
    torch.load = patched


def _load_pipeline(model_location_generator):
    _install_torch_load_shim()
    from pyannote.audio import Pipeline

    config = common.resolve_weights(common.PIPELINE_RELPATH, model_location_generator)
    pipeline = Pipeline.from_pretrained(config)
    pipeline.to(torch.device("cpu"))
    return pipeline


def _audio(input_path):
    if input_path:
        return input_path
    from pyannote.audio.sample import SAMPLE_FILE

    return str(SAMPLE_FILE["audio"])


def _offload_embedding(pipeline, device):
    """Run the WeSpeaker ResNet34 conv backbone through ttnn."""
    wespeaker = pipeline._embedding.model_
    resnet = wespeaker.resnet
    tt = TTNNWeSpeaker(wespeaker.state_dict(), device)
    tt.use_device_elementwise = True

    def backbone(feats1):
        x = tt._relu_dev(tt._conv(feats1, tt.folded["conv1"], 1))
        for layer, blocks in enumerate(tt.BLOCKS, start=1):
            for block in range(blocks):
                stride = 2 if (block == 0 and layer > 1) else 1
                x = tt._block(x, f"resnet.layer{layer}.{block}", stride)
        return x

    def forward(fbank, weights=None):
        feats = fbank.permute(0, 2, 1).unsqueeze(1).float()
        outs = [backbone(feats[i : i + 1]) for i in range(feats.shape[0])]
        frames = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :frames] for o in outs], dim=0)
        return torch.tensor(0.0), resnet.seg_1(resnet.pool(x, weights=weights))

    resnet.forward = forward


def _offload_segmentation(pipeline, device):
    """Run the PyanNet segmentation net through ttnn."""
    model = pipeline._segmentation.model
    sinc = model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    tt = TTNNPyanNet(model.state_dict(), sinc, device)

    def forward(waveforms):
        logits = tt.forward_batch(waveforms.detach().numpy())
        return F.log_softmax(torch.from_numpy(logits).float(), dim=-1)

    model.forward = forward


def _run(device, model_location_generator, input_path, offload_segmentation):
    audio = _audio(input_path)
    pipeline = _load_pipeline(model_location_generator)
    _offload_embedding(pipeline, device)
    if offload_segmentation:
        _offload_segmentation(pipeline, device)

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
