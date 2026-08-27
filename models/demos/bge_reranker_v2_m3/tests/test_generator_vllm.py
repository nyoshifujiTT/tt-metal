# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the reranker vLLM generator wiring (no device required).

Checks the class-level declarations vLLM relies on to route bge-reranker-v2-m3
through the cross-encoding score/rerank path, and that forward exposes the
vLLM-required keyword arguments.
"""

import inspect

import torch

from models.demos.bge_reranker_v2_m3.demo import generator_vllm as gen_mod
from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3


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


def test_first_token_indices_prefers_cursor_else_prompt_lens():
    # The pooler selects CLS rows via the cursor's first_token_indices (standard
    # CLSPool selector); when no cursor is present it falls back to prefix sums
    # of prompt_lens (the same quantity for single-shot prefill pooling).
    from types import SimpleNamespace

    from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import _first_token_indices_cpu

    # Cursor present: use it verbatim.
    cur = SimpleNamespace(first_token_indices_gpu=torch.tensor([0, 3, 5]))
    meta = SimpleNamespace(pooling_cursor=cur, prompt_lens=torch.tensor([3, 2, 4]))
    got = _first_token_indices_cpu(meta, 3)
    assert got.tolist() == [0, 3, 5]

    # No cursor: prefix sums of prompt_lens -> [0, 3, 5].
    meta2 = SimpleNamespace(pooling_cursor=None, prompt_lens=torch.tensor([3, 2, 4]))
    got2 = _first_token_indices_cpu(meta2, 3)
    assert got2.tolist() == [0, 3, 5]


def test_vllm_interface_methods_present():
    for name in ("embed_input_ids", "initialize_vllm_model", "get_embedding_dim"):
        assert hasattr(BgeRerankerV2M3, name)


def test_initialize_vllm_model_uses_served_checkpoint_path(monkeypatch):
    # vLLM points the model at a resolved checkpoint (hf_config._name_or_path);
    # the reranker must load weights from there, not the hardcoded HF id, so the
    # seq-classification loader works from a local dir / offline.
    captured = {}

    class _StubBase:
        @classmethod
        def initialize_vllm_model(cls, hf_config, *args, **kwargs):
            captured["model_name"] = kwargs.get("model_name")
            return "stub-model"

    # Patch the base initialize_vllm_model that BgeRerankerV2M3 delegates to via
    # super(); use the MRO parent (XlmRobertaEncoder).
    monkeypatch.setattr(
        gen_mod.XlmRobertaEncoder,
        "initialize_vllm_model",
        _StubBase.initialize_vllm_model,
    )

    class _HfCfg:
        _name_or_path = "/weights/bge-reranker-v2-m3"

    out = BgeRerankerV2M3.initialize_vllm_model(_HfCfg(), object(), 32)
    assert out == "stub-model"
    assert captured["model_name"] == "/weights/bge-reranker-v2-m3"

    # An explicit model_name wins over the checkpoint path.
    captured.clear()
    BgeRerankerV2M3.initialize_vllm_model(_HfCfg(), object(), 32, model_name="explicit")
    assert captured["model_name"] == "explicit"


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


def test_forward_full_hidden_returns_flat_device_hidden(monkeypatch):
    # Device-free: with return_full_hidden_states=True, forward must NOT score;
    # it appends each request's real rows (via _append_flat_rows) and returns the
    # flat [total_tokens, D] built by flatten_request_hidden_to_device. Stub both
    # so no device is needed and assert the flat layout is produced, not a score.
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.max_batch_size = 32
    model.max_seq_len = 8192
    model.tokenizer = _StubTokenizer()
    model._is_initialized = True
    model.model = object()
    model.device = object()
    model._collect_hidden = False
    model._flat_rows = []

    def fake_run_encoder_chunk(self, padded_inputs):
        return "HIDDEN"

    # Record the per-request rows the collect branch appends (one row per real
    # request), tagged with the chunk's real row count.
    def fake_append(self, output, attention_mask, chunk_batch_size):
        for row in range(chunk_batch_size):
            self._flat_rows.append((output, row))

    flattened = {}

    def fake_flatten(rows):
        flattened["rows"] = list(rows)
        return "FLAT_HIDDEN"

    monkeypatch.setattr(BgeRerankerV2M3, "_run_encoder_chunk", fake_run_encoder_chunk)
    monkeypatch.setattr(BgeRerankerV2M3, "_append_flat_rows", fake_append)
    monkeypatch.setattr(gen_mod, "flatten_request_hidden_to_device", fake_flatten)
    monkeypatch.setattr(gen_mod, "get_padded_sequence_length", lambda s: s)

    input_ids = torch.ones(3, 7, dtype=torch.long)
    out = model.forward(input_ids=input_ids, return_full_hidden_states=True)

    # forward returns the flat hidden (no scoring), built from 3 real rows.
    assert out == "FLAT_HIDDEN"
    assert len(flattened["rows"]) == 3
    # Accumulator and toggle are reset after the call.
    assert model._collect_hidden is False
    assert model._flat_rows == []


def test_pooler_available_before_first_forward():
    # The canonical runner queries model.pooler right after load, BEFORE the
    # first forward builds the device head. So the pooler must exist from
    # construction and advertise classify/score without a head.
    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    # Simulate the parts of __init__ that install the pooler (no device / super).
    from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import RerankerClassifierPooler

    model.classifier = None
    model.pooler = RerankerClassifierPooler(model)
    model._collect_hidden = False

    assert isinstance(model.pooler, RerankerClassifierPooler)
    # get_supported_tasks works with no head built yet (it is static).
    assert model.pooler.get_supported_tasks() == {"classify"}


def test_supported_tasks_are_real_vllm_pooling_tasks():
    """Every advertised task must be a name vLLM actually knows.

    The list is matched against vllm.tasks.PoolingTask -- get_pooling_task()
    selects from a fixed priority list and an explicit pooler_config.task is
    checked for membership. A made-up name is silently ignored by the former and
    produces a confusing error from the latter, so pin the set against upstream's
    own enumeration rather than against a hand-written copy of it.
    """
    from vllm.tasks import POOLING_TASKS

    from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import RerankerClassifierPooler

    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.vllm_config = None
    tasks = RerankerClassifierPooler(model).get_supported_tasks()

    assert tasks, "a pooler that advertises nothing can never be selected"
    unknown = set(tasks) - set(POOLING_TASKS)
    assert not unknown, f"not vLLM pooling tasks: {sorted(unknown)}"


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
