# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the reranker sequence-classification head (host, no device).

Verifies that XLMRobertaClassificationHead reproduces the transformers
RobertaClassificationHead numerics: dense -> tanh -> out_proj on the CLS
hidden state. Runs on CPU in fp32 and asserts an exact (tight) match.
"""

import pytest
import torch

from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import (
    XLMRobertaClassificationHead,
)

HIDDEN_SIZE = 1024


def _reference_head(cls_hidden, sd):
    # transformers RobertaClassificationHead at inference (dropout = identity):
    # dense -> tanh -> out_proj, applied to the <s> (CLS) token hidden state.
    x = torch.nn.functional.linear(cls_hidden, sd["classifier.dense.weight"], sd["classifier.dense.bias"])
    x = torch.tanh(x)
    x = torch.nn.functional.linear(x, sd["classifier.out_proj.weight"], sd["classifier.out_proj.bias"])
    return x


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_head_matches_reference(batch):
    torch.manual_seed(0)
    sd = {
        "classifier.dense.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.02,
        "classifier.dense.bias": torch.randn(HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.weight": torch.randn(1, HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.bias": torch.randn(1) * 0.02,
    }
    head = XLMRobertaClassificationHead.from_state_dict(sd)
    cls_hidden = torch.randn(batch, HIDDEN_SIZE)

    got = head(cls_hidden)
    expected = _reference_head(cls_hidden, sd)

    assert got.shape == (batch, 1)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


def test_head_missing_keys_raises():
    with pytest.raises(KeyError):
        XLMRobertaClassificationHead.from_state_dict({"classifier.dense.weight": torch.zeros(1)})


def test_head_known_vector_regression():
    # Hand-set 2-dim weights so the expected logit is computed by hand, pinning
    # the exact formula (dense -> tanh -> out_proj). Guards against silent
    # changes to the head math.
    sd = {
        "classifier.dense.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "classifier.dense.bias": torch.tensor([0.0, 0.0]),
        "classifier.out_proj.weight": torch.tensor([[1.0, -1.0]]),
        "classifier.out_proj.bias": torch.tensor([0.5]),
    }
    head = XLMRobertaClassificationHead.from_state_dict(sd)
    cls_hidden = torch.tensor([[0.5, -0.25]])
    # dense -> [0.5, -0.25]; tanh -> [tanh(0.5), tanh(-0.25)];
    # out_proj -> tanh(0.5) - tanh(-0.25) + 0.5
    import math

    expected = math.tanh(0.5) - math.tanh(-0.25) + 0.5
    got = head(cls_hidden)
    assert got.shape == (1, 1)
    torch.testing.assert_close(got.view(-1), torch.tensor([expected], dtype=torch.float32), rtol=0, atol=1e-6)
