# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Parity: numpy reference forward == real torch WeSpeakerResNet34 (community-1).

Establishes the golden op-graph for the ttnn port. Weights are resolved through
the shared helper (model_location_generator, else a Hugging Face download).
Skipped automatically when torch/pyannote are unavailable.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyannote.audio")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pyannote.audio import Model  # noqa: E402

from models.demos.audio.pyannote_diarization import common  # noqa: E402
from models.demos.audio.pyannote_diarization.reference.wespeaker_numpy_ref import WeSpeakerNumpyRef  # noqa: E402


def test_numpy_ref_matches_torch_wespeaker_resnet34(model_location_generator):
    m = Model.from_pretrained(
        common.resolve_weights(common.EMBEDDING_RELPATH, model_location_generator)
    )
    m.eval()
    ref = WeSpeakerNumpyRef(m.state_dict())
    wav = (torch.rand(1, 1, 48000, generator=torch.Generator().manual_seed(7)) * 2 - 1) * 0.1
    with torch.no_grad():
        fbank = m.compute_fbank(wav)
        emb_torch = m.resnet(fbank)[1].numpy()
    feats = fbank.permute(0, 2, 1).unsqueeze(1).numpy()
    emb_np = ref.forward(feats)
    cos = float(
        np.dot(emb_torch[0], emb_np[0])
        / (np.linalg.norm(emb_torch[0]) * np.linalg.norm(emb_np[0]))
    )
    max_abs = float(np.max(np.abs(emb_torch - emb_np)))
    assert cos > 0.999, f"cosine too low: {cos}"
    assert max_abs < 1e-2, f"max_abs too high: {max_abs}"
