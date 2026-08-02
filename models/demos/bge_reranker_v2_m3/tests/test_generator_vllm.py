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


def test_forward_exposes_return_full_hidden_states_kwarg():
    # The canonical pooling runner calls forward with
    # return_full_hidden_states=True to get the un-pooled hidden for model.pooler.
    params = inspect.signature(BgeRerankerV2M3.forward).parameters
    assert "return_full_hidden_states" in params
    # Default must be off so the fork runner (which never sets it) keeps the
    # scored-logit pass-through unchanged.
    assert params["return_full_hidden_states"].default is False


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


def test_forward_full_hidden_returns_chunked_device_hidden(monkeypatch):
    # Device-free: with return_full_hidden_states=True, forward must NOT score;
    # it collects each chunk's (encoder hidden, mask, real rows) into a
    # RerankerChunkedHidden for model.pooler. Stub the encoder chunk runner so no
    # device is needed and assert the per-chunk step keeps the raw hidden.
    from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import RerankerChunkedHidden

    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.max_batch_size = 32
    model.max_seq_len = 8192
    model.tokenizer = _StubTokenizer()
    model._is_initialized = True
    model.model = object()
    model.device = object()
    model._collect_hidden = False

    # Fake the encoder chunk output so _forward_chunk (collect branch) runs
    # device-free; return a sentinel "hidden" object per chunk.
    def fake_run_encoder_chunk(self, padded_inputs):
        return ("HIDDEN", padded_inputs["attention_mask"].shape)

    monkeypatch.setattr(BgeRerankerV2M3, "_run_encoder_chunk", fake_run_encoder_chunk)
    monkeypatch.setattr(gen_mod, "get_padded_sequence_length", lambda s: s)

    # One chunk of 3 rows, seq 7 (short-seq path pads batch to 32 but the real
    # row count is what forward passes through).
    input_ids = torch.ones(3, 7, dtype=torch.long)
    out = model.forward(input_ids=input_ids, return_full_hidden_states=True)

    assert isinstance(out, RerankerChunkedHidden)
    assert len(out.chunks) >= 1
    total_rows = sum(chunk_batch_size for _, _, chunk_batch_size in out.chunks)
    assert total_rows == 3  # real rows preserved across chunks, no scoring
    # Each chunk carries the raw encoder hidden (sentinel), not a scored logit.
    for hidden, mask, _ in out.chunks:
        assert hidden[0] == "HIDDEN"
    # The toggle is reset after the call.
    assert model._collect_hidden is False


def test_pooler_available_before_first_forward():
    # The canonical runner queries model.pooler right after load, BEFORE the
    # first forward builds the device head. So the pooler must exist from
    # construction and advertise classify/score without a head.
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    # Simulate the parts of __init__ that install the pooler (no device / super).
    from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import (
        RerankerClassifierPooler,
    )

    model.classifier = None
    model.pooler = RerankerClassifierPooler(model)
    model._collect_hidden = False

    assert isinstance(model.pooler, RerankerClassifierPooler)
    # get_supported_tasks works with no head built yet (static classify/score).
    assert model.pooler.get_supported_tasks() == {"classify", "score"}


def test_post_initialize_builds_device_head(monkeypatch):
    # _post_initialize builds the device classification head that the pooler
    # reads lazily at scoring time.
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.device = object()
    model.state_dict = {"unused": 0}

    sentinel_head = object()
    monkeypatch.setattr(
        gen_mod.XLMRobertaClassificationHeadTT,
        "from_state_dict",
        classmethod(lambda cls, device, sd: sentinel_head),
    )

    model._post_initialize()

    assert model.classifier is sentinel_head
