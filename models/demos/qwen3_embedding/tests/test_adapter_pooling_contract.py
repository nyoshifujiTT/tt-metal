# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free checks on the vLLM pooling contract of Qwen3EmbeddingForTTvLLM.

The plugin's pooling runner mirrors upstream ``GPUModelRunner._pool``: it feeds
``model.forward``'s flat per-token hidden states to ``model.pooler`` along with a
``PoolingMetadata``. Two properties of the adapter make that work, and both are
silent failures if they regress:

  * ``forward`` must request the flat ``[total_tokens, hidden]`` layout. Returning
    a pooled ``[batch, hidden]`` tensor would be misread as token-major by the
    pooling cursor.
  * ``pooler`` must exist, and must be built from the vLLM-resolved PoolerConfig
    rather than a locally invented one, so normalization follows the served
    configuration.

These run without a device: forward is exercised against a stub base class.
"""

import sys
import types

import pytest
import torch


def _load_adapter_with_stub_base(monkeypatch):
    """Import the adapter with a stubbed base wrapper (no ttnn / no device)."""
    base_mod = types.ModuleType("models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm")

    class _StubBase:
        def __init__(self, *args, **kwargs):
            self.forward_kwargs = None
            self.pooler = None  # the real base does this too

        def forward(self, input_ids, **kwargs):
            self.forward_kwargs = kwargs
            # Stand in for the flat per-token hidden states.
            return torch.zeros(int(input_ids.numel()), 8)

    base_mod.Qwen3ForEmbedding = _StubBase
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)
    sys.modules.pop("models.demos.qwen3_embedding.tt.generator_vllm", None)
    from models.demos.qwen3_embedding.tt.generator_vllm import Qwen3EmbeddingForTTvLLM

    return Qwen3EmbeddingForTTvLLM


def test_forward_requests_the_flat_per_token_layout(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    model.forward(input_ids=torch.zeros(1, 4, dtype=torch.long))

    assert model.forward_kwargs["return_full_hidden_states"] is True, (
        "the pooling runner indexes the flat token axis; a pooled return would be misread"
    )


def test_forward_accepts_positions_for_the_vllm_signature_check(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # vLLM's _check_vllm_model_forward requires input_ids + positions kwargs.
    model.forward(input_ids=torch.zeros(1, 2, dtype=torch.long), positions=torch.zeros(2))


def test_caller_can_still_override_the_layout(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    model.forward(input_ids=torch.zeros(1, 2, dtype=torch.long), return_full_hidden_states=False)

    assert model.forward_kwargs["return_full_hidden_states"] is False


def test_is_pooling_model_is_advertised(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)

    # inspect_model_cls only enumerates the "embed" task when this is truthy.
    assert cls.is_pooling_model is True


def test_pooler_requires_the_vllm_resolved_config(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # No vllm_config: the adapter must refuse rather than invent a PoolerConfig,
    # whose field set differs across vLLM releases. Asserted on the resolver so the
    # check does not depend on the installed vLLM's Pooler factory API.
    with pytest.raises(RuntimeError, match="pooler_config"):
        model._resolve_pooler_config()


def test_pooler_uses_the_config_vllm_resolved(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    sentinel = object()
    model.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(pooler_config=sentinel)
    )

    # Taken as-is: vLLM derives it from the checkpoint plus --override-pooler-config,
    # so the adapter must not substitute its own.
    assert model._resolve_pooler_config() is sentinel


def test_base_assigning_pooler_none_does_not_break_the_property(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # The base wrapper sets self.pooler = None in __init__; the setter must absorb
    # that instead of shadowing the property (which would make `pooler` a plain
    # attribute and silently disable pooling).
    model.pooler = None
    assert type(model).pooler.fget is not None
    assert model._pooler is None
