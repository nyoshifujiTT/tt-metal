# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Build the community-1 pipeline and offload its neural nets onto a device.

The pipeline is pyannote's own; what this module adds is the wiring that swaps
the two neural nets for their ttnn ports. It lives beside the ports rather than
inside a test so that tests, the demo and the corpus benchmark all drive the
same code -- previously the benchmark had to import from a test module, which
dragged pytest's ``importorskip`` into a plain CLI run.

Clustering, the speaker-count decision and all the pre/post-processing stay on
host: they are not neural nets, and pyannote runs them on CPU on GPU too.
"""

import torch
import torch.nn.functional as F

from models.demos.audio.pyannote_diarization import common
from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import TTNNPyanNet
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker


def install_torch_load_shim():
    """Restore ``torch.load(weights_only=False)`` for pyannote checkpoints.

    PyTorch 2.6 flipped the default and lightning>=2.6 propagated it; pyannote
    checkpoints legitimately hold non-tensor objects and fail to unpickle under
    the strict default. Idempotent, so repeated pipeline loads are safe.
    """
    original = torch.load
    if getattr(original, "_diar_shim", False):
        return

    def patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    patched._diar_shim = True
    torch.load = patched


def load_pipeline(model_location_generator=None):
    """Load community-1 on host. Weights resolve through :mod:`common`."""
    install_torch_load_shim()
    from pyannote.audio import Pipeline

    config = common.resolve_weights(common.PIPELINE_RELPATH, model_location_generator)
    pipeline = Pipeline.from_pretrained(config)
    pipeline.to(torch.device("cpu"))
    return pipeline


def sample_audio_path():
    """pyannote's bundled 30 s two-speaker sample."""
    from pyannote.audio.sample import SAMPLE_FILE

    return str(SAMPLE_FILE["audio"])


def speakers(diarization):
    """Distinct speaker labels in an ``Annotation``."""
    return {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}


def offload_embedding(pipeline, device):
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
        # Conv rounding can leave the chunks a frame apart; align before concat.
        frames = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :frames] for o in outs], dim=0)
        return torch.tensor(0.0), resnet.seg_1(resnet.pool(x, weights=weights))

    resnet.forward = forward


def offload_segmentation(pipeline, device):
    """Run the PyanNet segmentation net through ttnn."""
    model = pipeline._segmentation.model
    sinc = model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    tt = TTNNPyanNet(model.state_dict(), sinc, device)

    def forward(waveforms):
        logits = tt.forward_batch(waveforms.detach().numpy())
        return F.log_softmax(torch.from_numpy(logits).float(), dim=-1)

    model.forward = forward
