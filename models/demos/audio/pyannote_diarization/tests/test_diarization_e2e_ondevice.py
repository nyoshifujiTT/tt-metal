# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""End-to-end community-1 diarization with the neural nets on the p150.

Runs the real pyannote pipeline twice over the same recording -- once entirely
on host, once with the WeSpeaker embedding (and optionally the PyanNet
segmentation) executed through ttnn -- and requires the device run to reproduce
the host diarization.

Agreement is measured with diarization error rate, the standard metric for
comparing two diarizations, taking the host run as the reference. Raw segment
counts are not used: bf16 arithmetic can split or merge a turn at a boundary
without changing who is speaking when. The audio is the 30 s two-speaker sample
that ships with pyannote.audio, so no external fixture is needed.
"""
import pytest

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


def _sample_audio():
    from pyannote.audio.sample import SAMPLE_FILE

    return str(SAMPLE_FILE["audio"])


def _speakers(diarization):
    return {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "offload_segmentation", [False, True], ids=["embedding_only", "both_nets"]
)
def test_diarization_matches_host_pipeline(
    device, model_location_generator, offload_segmentation
):
    from pyannote.metrics.diarization import DiarizationErrorRate

    audio = _sample_audio()

    host = _load_pipeline(model_location_generator)(audio).speaker_diarization

    pipeline = _load_pipeline(model_location_generator)
    _offload_embedding(pipeline, device)
    if offload_segmentation:
        _offload_segmentation(pipeline, device)
    on_device = pipeline(audio).speaker_diarization

    assert len(_speakers(host)) >= 2, (
        "host pipeline should find multiple speakers in the sample recording"
    )
    assert len(_speakers(on_device)) == len(_speakers(host)), (
        f"speaker count differs: device={sorted(_speakers(on_device))} "
        f"host={sorted(_speakers(host))}"
    )

    der = DiarizationErrorRate()(host, on_device)
    assert der < 0.05, f"diarization error rate against the host pipeline too high: {der}"
