# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``Qwen36ForCausalLM.decode_forward`` handling of the plugin's mRoPE slot contract.

The vLLM plugin stores each request's ``mrope_position_delta`` at prefill and passes the batch's
deltas as ``rope_deltas_all_users`` on the first decode after the batch composition changes, then
None while it is unchanged. These tests pin that the wrapper installs the list onto the model's
per-slot offsets, keeps them across the None steps, moves them with ``slot_remap``, and does not
leak the kwarg into the base ``Generator``.

The wrapper's bookkeeping runs against a stub model, so no weights or device allocation are
involved, but importing it pulls in ttnn and vllm — hence the serving-image requirement.
Run: pytest models/demos/blackhole/qwen36/tests/unit/test_vllm_slot_deltas.py -v
"""
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

pytest.importorskip("vllm", reason="the vLLM wrapper under test imports vllm and ttnn")

MAX_BATCH = 4


class _StubModel:
    """The parts of ``Qwen36Model`` that ``decode_forward`` touches before delegating."""

    def __init__(self, num_devices, max_batch_size):
        self.num_devices = num_devices
        self.args = SimpleNamespace(max_batch_size=max_batch_size)
        self.rope = Qwen36RoPESetup.__new__(Qwen36RoPESetup)
        self.rope.rope_delta = 0
        self.rope._slot_deltas = None
        self.sampling = None
        self.gdn_remaps = []

    def _remap_gdn_slots(self, remap):
        self.gdn_remaps.append(remap)


def _make_wrapper(monkeypatch, num_devices=4, max_batch_size=MAX_BATCH):
    """A real ``Qwen36ForCausalLM`` whose base ``decode_forward`` records what reaches it.

    ``__init__`` allocates device state, so the instance is built without it and given only the
    attributes ``decode_forward`` touches; the method under test is the unmodified original.
    """
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
    from models.tt_transformers.tt.generator import Generator

    wrapper = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    wrapper.model = [_StubModel(num_devices, max_batch_size)]
    forwarded = []
    monkeypatch.setattr(Generator, "decode_forward", lambda self, *a, **k: forwarded.append(k))
    wrapper.forwarded = forwarded
    return wrapper


@pytest.fixture
def wrapper(monkeypatch):
    return _make_wrapper(monkeypatch)


def _decode(wrapper, tokens_batch, **kwargs):
    tokens = torch.zeros(tokens_batch, dtype=torch.int32)
    start_pos = torch.zeros(tokens_batch, dtype=torch.int32)
    wrapper.decode_forward(tokens=tokens, start_pos=start_pos, **kwargs)


def test_installs_batch_deltas(wrapper):
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=[-5, -7, 0, 0])
    deltas = wrapper.model[0].rope.decode_delta_vec(MAX_BATCH)
    assert torch.equal(deltas, torch.tensor([-5, -7, 0, 0], dtype=torch.int32))


def test_none_keeps_the_installed_deltas(wrapper):
    # The plugin sends None while the batch composition is unchanged; dropping the vector there
    # would silently revert every slot to the text offset mid-generation.
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=[-5, -7, 0, 0])
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=None)
    deltas = wrapper.model[0].rope.decode_delta_vec(MAX_BATCH)
    assert torch.equal(deltas, torch.tensor([-5, -7, 0, 0], dtype=torch.int32))


def test_absent_kwarg_keeps_the_installed_deltas(wrapper):
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=[-5, -7, 0, 0])
    _decode(wrapper, MAX_BATCH)
    deltas = wrapper.model[0].rope.decode_delta_vec(MAX_BATCH)
    assert torch.equal(deltas, torch.tensor([-5, -7, 0, 0], dtype=torch.int32))


def test_kwarg_is_not_forwarded_to_the_base_generator(wrapper):
    # The base Generator does not accept the mRoPE kwarg; leaking it raises at the call.
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=[-5, -7, 0, 0])
    assert "rope_deltas_all_users" not in wrapper.forwarded[-1]


def test_slot_remap_moves_the_deltas_with_the_gdn_state(wrapper):
    _decode(wrapper, MAX_BATCH, rope_deltas_all_users=[-1, -2, -3, -4])
    _decode(wrapper, MAX_BATCH, slot_remap=torch.tensor([2, 0, 1, 3], dtype=torch.int32))
    deltas = wrapper.model[0].rope.decode_delta_vec(MAX_BATCH)
    assert torch.equal(deltas, torch.tensor([-3, -1, -2, -4], dtype=torch.int32))
    # The GDN state must be remapped too, or the offsets would describe a different request.
    assert len(wrapper.model[0].gdn_remaps) == 1


def test_single_sequence_model_skips_the_slot_remap(monkeypatch):
    single = _make_wrapper(monkeypatch, max_batch_size=1)
    _decode(single, 1, slot_remap=torch.tensor([0], dtype=torch.int32))
    assert single.model[0].gdn_remaps == []
