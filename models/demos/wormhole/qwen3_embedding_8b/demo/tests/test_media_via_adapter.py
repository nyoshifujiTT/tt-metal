# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's layout is for callers that own a Pooler, and only those.

``Qwen3EmbeddingForTTvLLM.forward`` returns the pre-pooling
``[total_tokens, hidden]`` layout, because vLLM's pooling runner indexes that
token axis before handing the tensor to ``model.pooler``.

A caller with no Pooler must therefore not route through this class. The one
that tried is tt-media-server: it calls ``forward`` directly on whichever class
its runner names, then slices ``result[:num_requests]`` as rows -- so a flat
tensor is read with the token axis where the batch axis should be, and every
served vector is silently wrong. It now names the base wrapper and asks for the
finished embedding by :meth:`Qwen3ForEmbedding.encode` instead.

These pin both halves of that split so neither drifts back: the adapter really
does return the pre-pooling layout, and it reaches it by naming the stage rather
than by flipping a flag on the shared ``forward``.

Device-free: the base wrapper is stubbed, so only the subclass's own logic runs.
"""

import sys
import types

import pytest
import torch


def _adapter_with_stub_base(monkeypatch):
    base_mod = types.ModuleType("models.demos.wormhole.qwen3_embedding_8b.demo.model")

    class _StubBase:
        def __init__(self, *args, **kwargs):
            self.forward_kwargs = None
            self.init_kwargs = kwargs
            self.pooler = None  # the real base does this too

        def forward(self, input_ids, **kwargs):
            self.forward_kwargs = kwargs
            # Pooled [batch, hidden], which is what MEDIA expects.
            return torch.zeros(int(input_ids.shape[0]), 8)

        def encode(self, input_ids, attention_mask=None):
            self.encoded = True
            return torch.zeros(int(input_ids.shape[0]), 8)

        def encode_token_hidden_states(self, input_ids, attention_mask=None, **kwargs):
            self.token_hidden_kwargs = kwargs
            self.token_hidden_mask = attention_mask
            # Flat [total_tokens, hidden].
            return torch.zeros(int(input_ids.numel()), 8)

    base_mod.Qwen3ForEmbedding = _StubBase
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)
    sys.modules.pop("models.demos.qwen3_embedding.tt.generator_vllm", None)
    from models.demos.qwen3_embedding.tt.generator_vllm import Qwen3EmbeddingForTTvLLM

    return Qwen3EmbeddingForTTvLLM


def test_media_style_construction_builds_no_pooler(monkeypatch):
    cls = _adapter_with_stub_base(monkeypatch)

    # Exactly the kwargs tt-media-server passes: no vllm_config.
    model = cls(
        device=types.SimpleNamespace(),
        model_location_generator=lambda *a, **k: None,
        max_batch_size=32,
        max_seq_len=8192,
        act_dtype=None,
        weight_dtype=None,
        model_name="Qwen/Qwen3-Embedding-8B",
    )

    assert model._pooler is None, (
        "no vllm_config is supplied on the media path; building a Pooler there "
        "would raise during construction"
    )


def test_adapter_forward_returns_the_pre_pooling_layout(monkeypatch):
    cls = _adapter_with_stub_base(monkeypatch)
    model = cls(device=types.SimpleNamespace(), model_name="Qwen/Qwen3-Embedding-8B")

    out = model.forward(torch.zeros(4, 16, dtype=torch.long), attention_mask=torch.ones(4, 16))

    # 4 requests x 16 tokens flattened onto the token axis -- what the pooling
    # cursor indexes. A pooled return would have 4 rows here.
    assert out.shape[0] == 64


def test_adapter_names_the_stage_instead_of_flagging_forward(monkeypatch):
    cls = _adapter_with_stub_base(monkeypatch)
    model = cls(device=types.SimpleNamespace(), model_name="Qwen/Qwen3-Embedding-8B")

    model.forward(torch.zeros(2, 8, dtype=torch.long), attention_mask=torch.ones(2, 8))

    # Setting return_full_hidden_states on the shared forward would make the
    # flat layout the behaviour of every call reaching this class, which is how
    # the media runner ended up misreading its result.
    assert model.forward_kwargs is None, "the adapter must not drive the base's forward flags"
    assert model.token_hidden_kwargs == {}
    assert model.token_hidden_mask is not None, "the attention mask must reach the model"
