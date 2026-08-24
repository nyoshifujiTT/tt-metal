# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""On-device parity: TTNNPyanNet segmentation logits == torch PyanNet.

Exercises the segmentation net the diarization pipeline runs on device: the
SincNet frontend, the four BiLSTM layers implemented as a device-resident
recurrence, the leaky-relu linears and the powerset classifier. Both the
single-window ``forward`` and the batched ``forward_batch`` path used by the
diarization accelerator are covered.

The waveform comes from a fixed seed and the torch reference is computed in
process, so the test needs no checked-in fixture data.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("ttnn")
pytest.importorskip("pyannote.audio")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from pyannote.audio import Model  # noqa: E402

from models.demos.audio.pyannote_diarization import common  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import TTNNPyanNet  # noqa: E402


@pytest.fixture(scope="module")
def pyannet(model_location_generator):
    model = Model.from_pretrained(common.resolve_weights(common.SEGMENTATION_RELPATH, model_location_generator))
    model.eval()
    return model


def _waveform(seed, samples=16000):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 1, samples, generator=generator) * 2 - 1) * 0.1


def _torch_logits(model, wav):
    with torch.no_grad():
        out = model.sincnet(wav)
        out, _ = model.lstm(out.transpose(1, 2))
        for linear in model.linear:
            out = F.leaky_relu(linear(out))
        return model.classifier(out).numpy()


def _assert_matches(logits_tt, logits_torch):
    frames = min(logits_torch.shape[1], logits_tt.shape[1])
    a = logits_torch[0, :frames]
    b = logits_tt[0, :frames]
    cos = float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"segmentation logits cosine too low: {cos}"
    agreement = float((a.argmax(1) == b.argmax(1)).mean())
    assert agreement > 0.95, f"powerset argmax agreement too low: {agreement}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_pyannet_matches_torch(device, pyannet):
    wav = _waveform(seed=3)
    sinc_kernel = pyannet.sincnet.conv1d[0].filterbank.filters().detach().numpy()

    net = TTNNPyanNet(pyannet.state_dict(), sinc_kernel, device)
    logits_tt = net.forward(wav.numpy())

    _assert_matches(logits_tt, _torch_logits(pyannet, wav))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_pyannet_batched_matches_torch(device, pyannet):
    """forward_batch is the path the diarization accelerator actually uses."""
    wavs = [_waveform(seed=s) for s in (3, 5)]
    sinc_kernel = pyannet.sincnet.conv1d[0].filterbank.filters().detach().numpy()

    net = TTNNPyanNet(pyannet.state_dict(), sinc_kernel, device)
    batch = np.concatenate([w.numpy() for w in wavs], axis=0)
    logits_tt = net.forward_batch(batch)

    assert logits_tt.shape[0] == len(wavs)
    for index, wav in enumerate(wavs):
        _assert_matches(logits_tt[index : index + 1], _torch_logits(pyannet, wav))
