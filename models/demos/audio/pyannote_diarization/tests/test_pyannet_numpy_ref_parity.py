# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Parity: numpy PyanNet reference == real torch PyanNet (community-1 segmentation).

Establishes the golden op-graph (SincNet + BiLSTM4 + linear + classifier) for the
ttnn port. Weights are resolved through the shared helper
(model_location_generator, else a Hugging Face download). Skips without
torch/pyannote so it is CI-safe.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyannote.audio")

import numpy as np
import torch
import torch.nn.functional as F
from pyannote.audio import Model

from models.demos.audio.pyannote_diarization import common
from models.demos.audio.pyannote_diarization.reference.pyannet_numpy_ref import PyanNetNumpyRef


def test_numpy_ref_matches_torch_pyannet(model_location_generator):
    m = Model.from_pretrained(
        common.resolve_weights(common.SEGMENTATION_RELPATH, model_location_generator)
    )
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
