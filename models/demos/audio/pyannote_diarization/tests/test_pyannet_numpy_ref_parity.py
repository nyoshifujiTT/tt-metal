# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Parity: numpy PyanNet reference == real torch PyanNet (community-1 segmentation).

Establishes the golden op-graph (SincNet + BiLSTM4 + linear + classifier) for the
ttnn port. Skips without torch/pyannote/weights so it is CI-safe.
"""
import os
import pytest

WEIGHTS = "/data/pyannote-community-1/weights/segmentation/pytorch_model.bin"
pytest.importorskip("torch")
pytest.importorskip("pyannote.audio")
if not os.path.exists(WEIGHTS):
    pytest.skip("community-1 segmentation weights not present", allow_module_level=True)

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pyannote.audio import Model

from models.demos.audio.pyannote_diarization.reference.pyannet_numpy_ref import PyanNetNumpyRef


def test_numpy_ref_matches_torch_pyannet():
    m = Model.from_pretrained(WEIGHTS)
    m.eval()
    sinc_kernel = m.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    ref = PyanNetNumpyRef(m.state_dict(), sinc_kernel)
    wav = (torch.rand(1, 1, 16000, generator=torch.Generator().manual_seed(3)) * 2 - 1) * 0.1
    with torch.no_grad():
        o = m.sincnet(wav)
        o, _ = m.lstm(o.transpose(1, 2))
        for lin in m.linear:
            o = F.leaky_relu(lin(o))
        logits_torch = m.classifier(o).numpy()
    logits_np = ref.forward(wav.numpy())
    T = min(logits_torch.shape[1], logits_np.shape[1])
    a = logits_torch[0, :T]
    b = logits_np[0, :T]
    cos = float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.999, f"cosine too low: {cos}"
    assert np.max(np.abs(a - b)) < 1e-3
