# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""On-device parity for the narrow-width (pad+crop) conv path of the resident
WeSpeaker backbone.

The last chunks of a recording produce very small conv input time-widths (W as
low as 1). The default ttnn.conv2d auto-shard estimate trips a reader-index CB
assert for those (a known ttnn bug, tt-metal #35207 / #43193), so
``TTNNWeSpeakerResident._conv_dev`` zero-pads the time axis up to ``SAFE_W`` on
device, runs the conv, and crops the output back. This test asserts that path is
numerically faithful to the numpy reference backbone for degenerate widths and
that the whole backbone still runs on the p150.

Skipped automatically when ttnn / a Tenstorrent device / the community-1
embedding weights are unavailable, so it is safe in the media-server suite.
"""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("ttnn")
pytest.importorskip("pyannote.audio")

# WeSpeaker ResNet34 weights come straight from the community-1 embedding
# checkpoint (same state_dict the test used to slice out of the dev emb_all.npz
# fixture). Override the path with PYANNOTE_COMMUNITY1_WEIGHTS for a non-default
# install; skip cleanly when the weights are not present.
WEIGHTS = os.environ.get(
    "PYANNOTE_COMMUNITY1_WEIGHTS",
    "/data/pyannote-community-1/weights/embedding/pytorch_model.bin",
)
if not os.path.exists(WEIGHTS):
    pytest.skip(
        f"community-1 embedding weights not present at {WEIGHTS}",
        allow_module_level=True,
    )

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
from pyannote.audio import Model  # noqa: E402

from models.demos.audio.pyannote_diarization.reference.wespeaker_numpy_ref import WeSpeakerNumpyRef  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker_resident import TTNNWeSpeakerResident  # noqa: E402


@pytest.fixture(scope="module")
def device():
    try:
        dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no Tenstorrent device available: {exc}")
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def state_dict():
    model = Model.from_pretrained(WEIGHTS)
    model.eval()
    return model.state_dict()


@pytest.mark.parametrize("W", [1, 2, 4, 8, 12])
def test_resident_narrow_matches_numpy_backbone(device, state_dict, W):
    sdn = {k: v.numpy() for k, v in state_dict.items()}
    ref = WeSpeakerNumpyRef(sdn)
    tt = TTNNWeSpeakerResident(state_dict, device)

    feat = np.random.RandomState(W).randn(1, 1, 80, W).astype(np.float32)
    ref_map = ref.backbone_numpy(feat)                       # (1,C,H,Wref)
    dev_map = tt.backbone(torch.from_numpy(feat).float()).numpy()

    # the pad+crop path must reproduce the exact unpadded output width
    assert dev_map.shape == ref_map.shape, (dev_map.shape, ref_map.shape)

    a = dev_map.flatten()
    b = ref_map.flatten()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"narrow-width backbone parity too low for W={W}: cos={cos}"
