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


def test_forward_concatenates_per_chunk_logits(monkeypatch):
    # Device-free: forward() now delegates the encoder + device CLS/head to the
    # shared _encode_in_chunks template (which calls _forward_chunk per chunk and
    # returns a [chunk, 1] logit per chunk). Stub _encode_in_chunks to return
    # known per-chunk logits and verify forward() concatenates them into [B, 1].
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.max_batch_size = 32
    model.max_seq_len = 8192
    model.tokenizer = _StubTokenizer()
    model._is_initialized = True
    model.model = object()
    model.device = object()

    seen = {}
    per_chunk = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])]

    def fake_encode_in_chunks(self, input_ids, attention_mask=None):
        seen["self"] = self
        return per_chunk

    monkeypatch.setattr(BgeRerankerV2M3, "_encode_in_chunks", fake_encode_in_chunks)
    monkeypatch.setattr(gen_mod, "get_padded_sequence_length", lambda s: s)

    input_ids = torch.ones(3, 7, dtype=torch.long)
    out = model.forward(input_ids=input_ids)

    # Concatenated per-chunk logits: [[1],[2]] + [[3]] -> [B=3, 1].
    assert out.shape == (3, 1)
    torch.testing.assert_close(out.view(-1), torch.tensor([1.0, 2.0, 3.0]))
    # forward() must delegate to the instance's shared chunking entry point.
    assert seen["self"] is model
