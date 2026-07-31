# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit tests for the shared XLM-RoBERTa vLLM encoder base.

The base owns the vLLM plumbing shared by bge-m3 and the bge-reranker
cross-encoder: device / vllm_config resolution, initialize_vllm_model,
lazy model construction (with the _load_state_dict / _post_initialize hooks),
request validation and the get_max_* helpers. All of this is checked
without a device by stubbing create_tt_model.
"""

import sys
import types

import pytest

# Stub ttnn so the module imports without a real device stack.
if "ttnn" not in sys.modules:
    ttnn_stub = types.ModuleType("ttnn")
    ttnn_stub.Device = object
    ttnn_stub.bfloat16 = object()
    sys.modules["ttnn"] = ttnn_stub

from models.demos.wormhole.bge_m3.demo import xlm_roberta_encoder as base_mod
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import XlmRobertaEncoder


class _ConcreteEncoder(XlmRobertaEncoder):
    """Minimal concrete subclass so the abstract base's shared plumbing can be
    instantiated in tests. The two abstract primitives are stubbed."""

    def forward(self, input_ids, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        return 1


class _FakeModelArgs:
    def __init__(self):
        self.tokenizer = object()


@pytest.fixture(autouse=True)
def _stub_create_tt_model(monkeypatch):
    """Replace create_tt_model with a host stub recording the state_dict it got."""
    calls = {}

    def fake_create_tt_model(*, mesh_device, max_batch_size, max_seq_len, dtype, state_dict, hf_model_name):
        calls["state_dict"] = state_dict
        calls["hf_model_name"] = hf_model_name
        return _FakeModelArgs(), object(), state_dict if state_dict is not None else {"loaded": True}

    monkeypatch.setattr(base_mod, "create_tt_model", fake_create_tt_model)
    return calls


def test_abstract_base_cannot_be_instantiated():
    # forward / get_embedding_dim are abstract, so the base itself is abstract.
    with pytest.raises(TypeError):
        XlmRobertaEncoder(device=object())


def test_requires_device_or_vllm_config():
    with pytest.raises(ValueError):
        _ConcreteEncoder()


def test_device_and_defaults():
    dev = object()
    m = _ConcreteEncoder(device=dev, max_batch_size=8, max_seq_len=1024)
    assert m.device is dev
    assert m.get_max_batch_size() == 8
    assert m.get_max_seq_len() == 1024
    assert m._is_initialized is False


def test_device_resolved_from_vllm_config():
    dev = object()
    vllm_config = types.SimpleNamespace(device_config=types.SimpleNamespace(device=dev))
    m = _ConcreteEncoder(vllm_config=vllm_config)
    assert m.device is dev


def test_data_parallel_gt_one_is_rejected():
    # Execution is single-device only; a data-parallel request must fail loudly
    # instead of silently running on one device.
    with pytest.raises(NotImplementedError):
        _ConcreteEncoder(device=object(), tt_data_parallel=2)


def test_data_parallel_one_is_accepted():
    m = _ConcreteEncoder(device=object(), tt_data_parallel=1)
    assert m.tt_data_parallel == 1


def test_validate_request_bounds():
    m = _ConcreteEncoder(device=object(), max_batch_size=4, max_seq_len=128)
    m._validate_request(4, 128)  # ok at the boundary
    with pytest.raises(ValueError):
        m._validate_request(5, 128)
    with pytest.raises(ValueError):
        m._validate_request(4, 256)


def test_initialize_vllm_model_builds_from_vllm_config_and_rejects_optimizations():
    dev = object()
    model_config = types.SimpleNamespace(override_tt_config=None)
    vllm_config = types.SimpleNamespace(
        model_config=model_config, device_config=types.SimpleNamespace(device=dev)
    )
    m = _ConcreteEncoder.initialize_vllm_model(
        hf_config=None, mesh_device=dev, max_batch_size=2, vllm_config=vllm_config
    )
    assert isinstance(m, _ConcreteEncoder)
    assert m.device is dev

    with pytest.raises(ValueError):
        _ConcreteEncoder.initialize_vllm_model(
            hf_config=None, mesh_device=dev, max_batch_size=2, optimizations="perf"
        )


def test_initialize_model_runs_hooks_in_order(_stub_create_tt_model):
    """_load_state_dict feeds create_tt_model; _post_initialize runs after."""
    events = []

    class _Sub(_ConcreteEncoder):
        def _load_state_dict(self):
            events.append("load")
            return {"reranker": True}

        def _post_initialize(self):
            events.append("post")
            # tokenizer/model must already be set when _post_initialize runs
            assert self.tokenizer is not None
            assert self.model is not None

    m = _Sub(device=object(), model_name="BAAI/bge-reranker-v2-m3")
    m._initialize_model()

    assert events == ["load", "post"]
    assert _stub_create_tt_model["state_dict"] == {"reranker": True}
    assert m._is_initialized is True

    # Second call is a no-op (no extra hook events).
    m._initialize_model()
    assert events == ["load", "post"]


def test_initialize_model_default_hook_lets_backbone_load(_stub_create_tt_model):
    """Default _load_state_dict returns None so the backbone loads its own weights."""
    m = _ConcreteEncoder(device=object(), model_name="BAAI/bge-m3")
    m._initialize_model()
    assert _stub_create_tt_model["state_dict"] is None


def test_post_initialize_default_is_explicit_noop():
    # The default hook is an explicit no-op (returns None), not an abstract
    # method: bge-m3 deliberately does not override it.
    m = _ConcreteEncoder(device=object())
    assert m._post_initialize() is None
