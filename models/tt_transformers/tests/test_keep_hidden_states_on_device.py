# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the opt-in "leave the hidden states on device" return.

``prefill_forward_text`` normally copies every result to host before returning,
which is the right default. For the flat per-token pooling contract that default
is expensive in a way the caller cannot avoid: the whole ``[seq, dim]`` hidden
crosses to host, while a pooling layer that runs on device only ever reads one
row of it. At 128 tokens that is 128x more transfer than the pooled paths do.

``keep_hidden_states_on_device=True`` lets such a caller take ownership of the
device tensor instead. These tests cover the batched-prefill extraction helper,
which is where the copy is issued.

The pooled paths are deliberately excluded: they already return a single
``[batch, dim]`` row, so there is no transfer to save and honouring the flag
there would only hand the caller an unexpected tensor type.
"""

import types

import pytest

from models.tt_transformers.tt.generator import Generator


@pytest.fixture(autouse=True)
def _stub_device_sync(monkeypatch):
    """The host path syncs the mesh device; there is no device in these tests.

    Patched for every test, including the keep-on-device ones, so that a
    regression which reintroduces the sync there cannot hide behind a stub that
    only exists on the host path.
    """
    from models.tt_transformers.tt import generator as generator_module

    calls = []
    monkeypatch.setattr(
        generator_module.ttnn,
        "synchronize_device",
        lambda device, *a, **k: calls.append(device),
    )
    return calls


class _FakeDeviceTensor:
    """Stands in for a ttnn tensor: records whether it was copied to host."""

    def __init__(self, name):
        self.name = name
        self.copied_to_host = False

    def cpu(self, blocking=False):
        self.copied_to_host = True
        return _FakeHostTensor(self.name)


class _FakeHostTensor:
    def __init__(self, name):
        self.name = name


class _FakeModel:
    """Just the two hooks the extraction helper calls."""

    def __init__(self):
        self.mesh_device = object()
        self.host_compositions = []

    def process_full_hidden_states_after_prefill_trace(self, user_hidden):
        # The real one applies the final norm and returns a device tensor.
        return _FakeDeviceTensor(user_hidden)

    def process_output_prefill_full_hidden_states(self, host_tensor, seq_len):
        self.host_compositions.append((host_tensor, seq_len))
        return ("host-composed", host_tensor.name, seq_len)


class _FakeLogits:
    """Sliceable like the trace output: ``logits[slot:slot+1, :, :, :]``."""

    def __getitem__(self, key):
        return f"slot{key[0].start}"


def _make_stub(model):
    stub = types.SimpleNamespace()
    stub.model = [model]
    # Bind the real (unbound) method so we exercise production code, not a copy.
    stub._extract_batched_prefill_full_hidden = types.MethodType(
        Generator._extract_batched_prefill_full_hidden, stub
    )
    return stub


def test_device_tensors_are_handed_back_without_a_host_copy(_stub_device_sync):
    model = _FakeModel()
    stub = _make_stub(model)
    out = [None, None]

    stub._extract_batched_prefill_full_hidden(
        _FakeLogits(),
        model_id=0,
        empty_slots=[0, 1],
        prompt_lens=[7, 9],
        output_full_hidden=out,
        keep_on_device=True,
    )

    assert all(isinstance(t, _FakeDeviceTensor) for t in out), out
    assert [t.name for t in out] == ["slot0", "slot1"]
    assert not any(t.copied_to_host for t in out), "the whole sequence still crossed to host"
    assert model.host_compositions == [], "host composition ran despite keep_on_device"
    # Nothing was queued for host, so there is nothing to wait on either.
    assert _stub_device_sync == [], "device sync ran with no host copy outstanding"


def test_the_default_still_copies_to_host_and_composes(_stub_device_sync):
    """The existing contract must be untouched when the flag is not passed."""
    model = _FakeModel()
    stub = _make_stub(model)
    out = [None, None]

    stub._extract_batched_prefill_full_hidden(
        _FakeLogits(),
        model_id=0,
        empty_slots=[0, 1],
        prompt_lens=[7, 9],
        output_full_hidden=out,
    )

    assert out == [
        ("host-composed", "slot0", 7),
        ("host-composed", "slot1", 9),
    ]
    # Each user's real length is what trims the right-padding.
    assert [seq_len for _, seq_len in model.host_compositions] == [7, 9]
    # The copies are non-blocking, so the host path has to wait before reading.
    assert _stub_device_sync == [model.mesh_device]


def test_non_contiguous_slots_keep_their_request_order():
    """Slot ids are device slots; the output list is indexed by request order."""
    model = _FakeModel()
    stub = _make_stub(model)
    out = [None, None]

    stub._extract_batched_prefill_full_hidden(
        _FakeLogits(),
        model_id=0,
        empty_slots=[3, 7],
        prompt_lens=[5, 11],
        output_full_hidden=out,
        keep_on_device=True,
    )

    assert [t.name for t in out] == ["slot3", "slot7"]


@pytest.mark.parametrize("keep_on_device", [False, True])
def test_no_scheduled_requests_is_not_an_error(keep_on_device):
    model = _FakeModel()
    stub = _make_stub(model)
    out = []

    stub._extract_batched_prefill_full_hidden(
        _FakeLogits(),
        model_id=0,
        empty_slots=[],
        prompt_lens=[],
        output_full_hidden=out,
        keep_on_device=keep_on_device,
    )

    assert out == []
