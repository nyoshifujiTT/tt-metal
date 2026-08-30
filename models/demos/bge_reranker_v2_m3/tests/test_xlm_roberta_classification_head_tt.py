# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device PCC test for the ttnn XLM-RoBERTa classification head.

Runs ``XLMRobertaClassificationHeadTT`` (dense -> tanh -> out_proj on device in
fp32) and compares its ``[batch, 1]`` logits against the host fp32 reference
``XLMRobertaClassificationHead`` on the same random weights and CLS hidden
states. The device head uses ``fp32_dest_acc_en`` + ``HiFi4``, so it must match
the host fp32 reference to a tight absolute tolerance (tile-rounding ~1e-3).

Note on the metric: the head emits a single logit per row (``[batch, 1]``), so
a correlation metric (PCC) is ill-defined across the width-1 axis. The score is
a scalar per (query, document) pair, so correctness is asserted directly on the
logit values with an absolute+relative tolerance that comfortably covers the
~1e-3 fp32 tile rounding while catching any real regression.

Requires a single Tenstorrent device.
"""

import pytest
import torch

import ttnn
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import XLMRobertaClassificationHead
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head_tt import XLMRobertaClassificationHeadTT
from models.demos.wormhole.bge_m3.tests.test_utils import require_single_device, to_torch, to_ttnn_tensor

HIDDEN_SIZE = 1024
# fp32-dest-acc + HiFi4 on device vs host fp32: logits agree to ~1e-3 (tile
# rounding). Tolerances chosen well above that noise floor, far below any
# meaningful score difference (real /score gaps are O(1)-O(10)).
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
def test_device_head_matches_host_reference(device, batch_size, reset_seeds):
    require_single_device(device)

    state_dict = _random_state_dict()
    cls_hidden = torch.randn(batch_size, HIDDEN_SIZE, dtype=torch.float32)

    # Host fp32 reference (the numerics we must reproduce on device).
    host_head = XLMRobertaClassificationHead.from_state_dict(state_dict)
    ref_logits = host_head(cls_hidden).to(torch.float32)

    # Device head.
    tt_head = XLMRobertaClassificationHeadTT.from_state_dict(device, state_dict)
    cls_tt = to_ttnn_tensor(cls_hidden, device, dtype=ttnn.float32)
    logits_tt = tt_head(cls_tt)
    cand_logits = to_torch(logits_tt, (batch_size, 1))

    assert cand_logits.shape == (batch_size, 1)
    torch.testing.assert_close(cand_logits, ref_logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
