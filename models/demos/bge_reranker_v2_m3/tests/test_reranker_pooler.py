# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test for the TT-native reranker ClassifierPooler.

``RerankerClassifierPooler`` reproduces vLLM's two-stage cross-encoder pooling
(CLS extract + classification head) on device: it slices the ``<s>`` (position
0) hidden row with ``ttnn.slice`` and runs the fp32 device head. This test
drives the shared ``score_cls_on_device`` path on a synthetic ``[B, S, D]``
encoder hidden state and asserts the resulting ``[B, 1]`` logits match a host
fp32 reference that takes position 0 and runs ``dense -> tanh -> out_proj``.

Only the CLS row must influence the score, so the reference intentionally
ignores the non-CLS positions; a device slice that picked the wrong position
(or pooled across the sequence) would diverge here.

Requires a single Tenstorrent device.
"""

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import (
    XLMRobertaClassificationHead,
)
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head_tt import (
    XLMRobertaClassificationHeadTT,
)
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import score_cls_on_device

HIDDEN_SIZE = 1024
# Same fp32-dest-acc + HiFi4 device head as test_xlm_roberta_classification_head_tt:
# logits agree with the host fp32 reference to ~1e-3 (tile rounding).
LOGIT_ATOL = 5e-3
LOGIT_RTOL = 1e-2


def _random_state_dict(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "classifier.dense.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.02,
        "classifier.dense.bias": torch.randn(HIDDEN_SIZE) * 0.01,
        "classifier.out_proj.weight": torch.randn(1, HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.bias": torch.randn(1) * 0.01,
    }


@pytest.mark.parametrize("batch_size", [1, 8], ids=["B1", "B8"])
@pytest.mark.parametrize("seq_len", [16], ids=["S16"])
def test_pooler_scores_cls_on_device(device, batch_size, seq_len, reset_seeds):
    require_single_device(device)

    state_dict = _random_state_dict()
    # Full [B, S, D] hidden; only position 0 (CLS) should drive the score.
    hidden = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.float32)

    # Host fp32 reference: take the CLS row, then dense -> tanh -> out_proj.
    host_head = XLMRobertaClassificationHead.from_state_dict(state_dict)
    ref_logits = host_head(hidden[:, 0, :]).to(torch.float32)

    # Device path: same slice + head via the shared pooler helper.
    tt_head = XLMRobertaClassificationHeadTT.from_state_dict(device, state_dict)
    hidden_tt = to_ttnn_tensor(hidden, device, dtype=ttnn.float32)
    logits_tt = score_cls_on_device(hidden_tt, tt_head)
    cand_logits = to_torch(logits_tt, (batch_size, 1))

    assert cand_logits.shape == (batch_size, 1)
    torch.testing.assert_close(
        cand_logits, ref_logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
    )
