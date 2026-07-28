# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the reranker vLLM generator wiring (no device required).

Checks the class-level declarations vLLM relies on to route bge-reranker-v2-m3
through the cross-encoding score/rerank path, and that forward exposes the
vLLM-required keyword arguments.
"""

import inspect

import torch

from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3
from models.demos.bge_reranker_v2_m3.demo import generator_vllm as gen_mod


def test_cross_encoder_class_flags():
    # vLLM is_pooling_model() requires the attribute AND is_vllm_model(); the
    # cross-encoder routing additionally needs supports_cross_encoding.
    assert BgeRerankerV2M3.is_pooling_model is True
    assert BgeRerankerV2M3.supports_cross_encoding is True
    assert BgeRerankerV2M3.default_pooling_type == "CLS"


def test_forward_exposes_vllm_kwargs():
    # is_vllm_model() checks forward accepts input_ids and positions.
    params = inspect.signature(BgeRerankerV2M3.forward).parameters
    assert "input_ids" in params
    assert "positions" in params


def test_vllm_interface_methods_present():
    for name in ("embed_input_ids", "initialize_vllm_model", "get_embedding_dim"):
        assert hasattr(BgeRerankerV2M3, name)


class _StubTokenizer:
    pad_token_id = 1


class _IdentityHead:
    """Returns the first hidden dim as the logit so forward output is checkable."""

    def __call__(self, cls_hidden):
        return cls_hidden[:, :1]


def test_forward_extracts_cls_and_returns_logits(monkeypatch):
    # Device-free: stub the shared _encode_to_last_hidden to return a known
    # [B,S,D] tensor and verify forward() takes the CLS (position 0) hidden and
    # runs the classifier, returning one logit per input row [B,1].
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.max_batch_size = 32
    model.max_seq_len = 8192
    model.tokenizer = _StubTokenizer()
    model.classifier = _IdentityHead()
    model._is_initialized = True
    model.model = object()
    model.device = object()

    batch, seq, hidden = 3, 7, 4
    fake_hidden = torch.zeros(batch, seq, hidden)
    for r in range(batch):
        fake_hidden[r, 0, 0] = r + 1  # CLS marker at position 0
        fake_hidden[r, 1, 0] = -99  # non-CLS position, must be ignored

    seen = {}

    def fake_encode(self, input_ids, attention_mask=None):
        # Bound method: records that forward() delegated to the instance.
        seen["self"] = self
        return fake_hidden

    monkeypatch.setattr(BgeRerankerV2M3, "_encode_to_last_hidden", fake_encode)
    monkeypatch.setattr(gen_mod, "get_padded_sequence_length", lambda s: s)

    input_ids = torch.ones(batch, seq, dtype=torch.long)
    out = model.forward(input_ids=input_ids)

    assert out.shape == (batch, 1)
    # identity head returns CLS[:, :1] => [1, 2, 3]
    torch.testing.assert_close(out.view(-1), torch.tensor([1.0, 2.0, 3.0]))
    # forward() must delegate to the instance's shared encoder entry point.
    assert seen["self"] is model
