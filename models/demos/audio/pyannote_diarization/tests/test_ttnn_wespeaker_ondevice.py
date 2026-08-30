# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""On-device parity: TTNNWeSpeaker embedding == torch WeSpeakerResNet34.

Covers the non-resident WeSpeaker port end to end (convs, residual adds, relu,
TSTP pooling and the seg_1 linear). The device-offload toggles are parametrized
so both the host-pooling path and the fully on-device path
(``use_device_elementwise`` + ``use_device_pool``) are exercised.

Inputs are generated from a fixed seed and the torch reference is computed in
process, so the test needs no checked-in fixture data.
"""
import pytest

import numpy as np
import torch
from pyannote.audio import Model

from models.demos.audio.pyannote_diarization import common
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker


@pytest.fixture(scope="module")
def wespeaker(model_location_generator):
    model = Model.from_pretrained(common.resolve_weights(common.EMBEDDING_RELPATH, model_location_generator))
    model.eval()
    return model


def _torch_reference(model, seconds):
    """Fbank features and the torch embedding for a deterministic waveform."""
    samples = int(seconds * 16000)
    generator = torch.Generator().manual_seed(samples)
    wav = (torch.rand(1, 1, samples, generator=generator) * 2 - 1) * 0.1
    with torch.no_grad():
        fbank = model.compute_fbank(wav)
        embedding = model.resnet(fbank)[1].numpy()
    feats = fbank.permute(0, 2, 1).unsqueeze(1).float()
    return feats, embedding


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "use_device_elementwise, use_device_pool",
    [(False, False), (True, False), (True, True)],
    ids=["convs_only", "device_elementwise", "device_elementwise_and_pool"],
)
def test_ttnn_wespeaker_matches_torch(device, wespeaker, use_device_elementwise, use_device_pool):
    feats, embedding_torch = _torch_reference(wespeaker, seconds=3.0)

    tt = TTNNWeSpeaker(wespeaker.state_dict(), device)
    tt.use_device_elementwise = use_device_elementwise
    tt.use_device_pool = use_device_pool
    embedding_tt = tt.forward(feats).numpy()

    assert embedding_tt.shape == embedding_torch.shape

    a = embedding_tt[0]
    b = embedding_torch[0]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"embedding cosine too low: {cos}"
