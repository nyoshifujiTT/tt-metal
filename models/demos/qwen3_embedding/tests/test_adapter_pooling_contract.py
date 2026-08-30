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
    base_mod = types.ModuleType("models.demos.wormhole.qwen3_embedding_8b.demo.model")

    class _StubBase:
        def __init__(self, *args, **kwargs):
            self.forward_kwargs = None
            self.pooler = None  # the real base does this too

        def forward(self, input_ids, **kwargs):
            self.forward_kwargs = kwargs
            # Stand in for the flat per-token hidden states.
            return torch.zeros(int(input_ids.numel()), 8)

        def encode_token_hidden_states(self, input_ids, attention_mask=None, **kwargs):
            self.token_hidden_kwargs = kwargs
            return torch.zeros(int(input_ids.numel()), 8)

    base_mod.Qwen3ForEmbedding = _StubBase
    monkeypatch.setitem(sys.modules, base_mod.__name__, base_mod)
    sys.modules.pop("models.demos.qwen3_embedding.tt.generator_vllm", None)
    from models.demos.qwen3_embedding.tt.generator_vllm import Qwen3EmbeddingForTTvLLM

    return Qwen3EmbeddingForTTvLLM


def test_forward_requests_the_flat_per_token_layout(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    out = model.forward(input_ids=torch.zeros(1, 4, dtype=torch.long))

    # The pooling runner indexes the flat token axis; a pooled return (1 row
    # here) would be misread as if that axis were the tokens.
    assert out.shape[0] == 4


def test_forward_accepts_positions_for_the_vllm_signature_check(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # vLLM's _check_vllm_model_forward requires input_ids + positions kwargs.
    model.forward(input_ids=torch.zeros(1, 2, dtype=torch.long), positions=torch.zeros(2))
    # positions is accepted and dropped: the base never uses it.
    assert model.token_hidden_kwargs == {}


def test_forward_accepts_the_runners_explicit_request_for_full_hidden(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # The pooling runner states the same requirement on its side. Agreeing is
    # not a reason to route the call back through the base's flag.
    out = model.forward(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        return_full_hidden_states=True,
    )

    assert out.shape[0] == 4
    assert model.token_hidden_kwargs == {}


def test_forward_refuses_to_return_an_already_pooled_tensor(monkeypatch):
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    # This class exists to serve a runner that pools the hidden states itself.
    # Handing it a pooled tensor would be read as token-major, so say so rather
    # than silently obliging.
    with pytest.raises(ValueError, match="encode"):
        model.forward(
            input_ids=torch.zeros(1, 4, dtype=torch.long),
            return_full_hidden_states=False,
        )


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

def test_pooler_is_built_with_the_vllm_embedding_factory(monkeypatch):
    """The pooler must come from vLLM's own embedding-pooler constructor.

    Not a hand-rolled Pooler: ``DispatchPooler.for_embedding`` is what vLLM's own
    embedding adapters call (vllm/model_executor/models/adapters.py), so using it
    keeps pooling type, activation and normalization identical to every other
    backend. Patched here so the assertion holds without instantiating a real
    Pooler (whose constructor signature is vLLM-version specific).
    """
    cls = _load_adapter_with_stub_base(monkeypatch)
    model = cls()

    sentinel_config = object()
    sentinel_pooler = object()
    monkeypatch.setattr(model, "_resolve_pooler_config", lambda: sentinel_config)

    seen = {}

    class _StubDispatchPooler:
        @staticmethod
        def for_embedding(pooler_config):
            seen["config"] = pooler_config
            return sentinel_pooler

    pooler_mod = types.ModuleType("vllm.model_executor.layers.pooler")
    pooler_mod.DispatchPooler = _StubDispatchPooler
    monkeypatch.setitem(sys.modules, pooler_mod.__name__, pooler_mod)

    assert model.pooler is sentinel_pooler
    assert seen["config"] is sentinel_config
    # Built once and cached.
    assert model.pooler is sentinel_pooler


def test_pooler_is_built_during_construction_when_a_vllm_config_is_present(monkeypatch):
    """Pooler construction must happen while the model is being constructed.

    That is the only window in which vLLM's current-config context is set: the
    TT loader wraps ``initialize_vllm_model`` in ``set_current_vllm_config``,
    and the pooling methods underneath ``DispatchPooler`` resolve the config
    through ``get_current_vllm_config()`` in their ``__init__``. Building the
    pooler lazily on first access instead moved construction into
    ``get_supported_tasks``, outside that context, and serving
    Qwen3-Embedding-0.6B on p150 failed at startup with "Current vLLM config is
    not set".
    """
    cls = _load_adapter_with_stub_base(monkeypatch)

    sentinel_config = object()
    sentinel_pooler = object()
    seen = {}

    class _StubDispatchPooler:
        @staticmethod
        def for_embedding(pooler_config):
            seen["config"] = pooler_config
            return sentinel_pooler

    pooler_mod = types.ModuleType("vllm.model_executor.layers.pooler")
    pooler_mod.DispatchPooler = _StubDispatchPooler
    monkeypatch.setitem(sys.modules, pooler_mod.__name__, pooler_mod)

    class _WithVllmConfig(cls):
        def __init__(self):
            # The real base stores the vllm_config it was handed before the
            # adapter's __init__ body runs.
            self.vllm_config = types.SimpleNamespace(
                model_config=types.SimpleNamespace(pooler_config=sentinel_config)
            )
            super().__init__()

    model = _WithVllmConfig()

    assert seen["config"] is sentinel_config, "the pooler must be built in __init__"
    assert model._pooler is sentinel_pooler


def test_construction_without_a_vllm_config_defers_instead_of_failing(monkeypatch):
    """The metal-only demo builds this class with no vLLM config and never pools
    through vLLM, so construction must not try to build a Pooler; only an actual
    ``pooler`` access reports the missing config."""
    cls = _load_adapter_with_stub_base(monkeypatch)

    model = cls()

    assert model._pooler is None
    with pytest.raises(RuntimeError, match="pooler_config"):
        model.pooler
