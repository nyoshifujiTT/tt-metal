# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the reranker sequence-classification head (host, no device).

Verifies that RerankerClassifierHead reproduces the transformers
RobertaClassificationHead numerics: dense -> tanh -> out_proj on the CLS
hidden state. Runs on CPU in fp32 and asserts an exact (tight) match.
"""

import pytest
import torch

from models.demos.bge_reranker_v2_m3.tt.classifier_head import (
    CLASSIFIER_KEYS,
    RerankerClassifierHead,
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
def test_classifier_head_matches_reference(batch):
    torch.manual_seed(0)
    sd = {
        "classifier.dense.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.02,
        "classifier.dense.bias": torch.randn(HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.weight": torch.randn(1, HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.bias": torch.randn(1) * 0.02,
    }
    head = RerankerClassifierHead.from_state_dict(sd)
    cls_hidden = torch.randn(batch, HIDDEN_SIZE)

    got = head(cls_hidden)
    expected = _reference_head(cls_hidden, sd)

    assert got.shape == (batch, 1)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


def test_classifier_head_missing_keys_raises():
    with pytest.raises(KeyError):
        RerankerClassifierHead.from_state_dict({"classifier.dense.weight": torch.zeros(1)})


def test_classifier_keys_constant():
    assert CLASSIFIER_KEYS == (
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    )
