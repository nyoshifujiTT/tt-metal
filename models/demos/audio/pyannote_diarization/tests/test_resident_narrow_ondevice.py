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

The device comes from the shared tt-metal ``device`` fixture, so ``--device-id``
and the CI SKU/topology handling apply; ``l1_small_size`` is requested through
``device_params``.
"""
import pytest

import numpy as np
import torch
from pyannote.audio import Model

from models.demos.audio.pyannote_diarization import common
from models.demos.audio.pyannote_diarization.reference.wespeaker_numpy_ref import WeSpeakerNumpyRef
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker_resident import TTNNWeSpeakerResident


@pytest.fixture(scope="module")
def state_dict(model_location_generator):
    """WeSpeaker ResNet34 weights from the community-1 embedding checkpoint."""
    weights = common.resolve_weights(common.EMBEDDING_RELPATH, model_location_generator)
    model = Model.from_pretrained(weights)
    model.eval()
    return model.state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("W", [1, 2, 4, 8, 12])
def test_resident_narrow_matches_numpy_backbone(device, state_dict, W):
    sdn = {k: v.numpy() for k, v in state_dict.items()}
    ref = WeSpeakerNumpyRef(sdn)
    tt = TTNNWeSpeakerResident(state_dict, device)

    feat = np.random.RandomState(W).randn(1, 1, 80, W).astype(np.float32)
    ref_map = ref.backbone_numpy(feat)  # (1,C,H,Wref)
    dev_map = tt.backbone(torch.from_numpy(feat).float()).numpy()

    # the pad+crop path must reproduce the exact unpadded output width
    assert dev_map.shape == ref_map.shape, (dev_map.shape, ref_map.shape)

    a = dev_map.flatten()
    b = ref_map.flatten()
    cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos > 0.99, f"narrow-width backbone parity too low for W={W}: cos={cos}"
