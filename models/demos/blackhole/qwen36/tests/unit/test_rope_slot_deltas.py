# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-decode-slot mrope_position_delta bookkeeping in ``tt/rope.py``.

A multimodal prompt compresses the RoPE position space by an amount derived from its own image
grid, so under batched serving each decode slot needs the delta of the request occupying it.
These tests pin the slot vector's install / pad / fallback / remap semantics, which decode reads
via ``decode_delta_vec`` to build ``rope_pos = kv_pos + delta``.

Host-only: the delta vector is assembled in torch before any device transfer.
Run: pytest models/demos/blackhole/qwen36/tests/unit/test_rope_slot_deltas.py -v
"""
import pytest
import torch

from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

BATCH = 4


@pytest.fixture
def rope():
    """Only the host-side delta state is under test, so skip the device table construction."""
    setup = Qwen36RoPESetup.__new__(Qwen36RoPESetup)
    setup.rope_delta = 0
    setup._slot_deltas = None
    return setup


def test_defaults_to_scalar_delta(rope):
    # No vector installed: every slot uses the single-sequence scalar, so B=1 behaviour is unchanged.
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.zeros(BATCH, dtype=torch.int32))

    rope.rope_delta = -7
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.full((BATCH,), -7, dtype=torch.int32))


def test_installs_one_delta_per_slot(rope):
    rope.set_slot_deltas([-3, 0, -11], BATCH)
    # Slots beyond the supplied list hold no request and take the text delta (0).
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.tensor([-3, 0, -11, 0], dtype=torch.int32))


def test_scalar_delta_ignored_once_vector_installed(rope):
    # The scalar is the single-sequence fallback; an installed vector must win for every slot.
    rope.rope_delta = -5
    rope.set_slot_deltas([-2, -9], BATCH)
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.tensor([-2, -9, 0, 0], dtype=torch.int32))


def test_none_restores_scalar_fallback(rope):
    rope.rope_delta = -4
    rope.set_slot_deltas([-1, -2], BATCH)
    rope.set_slot_deltas(None, BATCH)
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.full((BATCH,), -4, dtype=torch.int32))


def test_none_entries_are_text_deltas(rope):
    # vLLM reports no mrope delta for a text-only request; that slot must rotate at the KV position.
    rope.set_slot_deltas([-6, None, -8, None], BATCH)
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.tensor([-6, 0, -8, 0], dtype=torch.int32))


def test_widening_batch_pads_with_text_delta(rope):
    # Decode bucketing can ask for a wider batch than the vector was built for.
    rope.set_slot_deltas([-3, -4], 2)
    assert torch.equal(rope.decode_delta_vec(4), torch.tensor([-3, -4, 0, 0], dtype=torch.int32))


def test_narrowing_batch_keeps_leading_slots(rope):
    rope.set_slot_deltas([-3, -4, -5, -6], BATCH)
    assert torch.equal(rope.decode_delta_vec(2), torch.tensor([-3, -4], dtype=torch.int32))


def test_remap_follows_condensed_slots(rope):
    # vLLM condenses the decode batch: slot i takes the request that was at remap[i]. The per-slot
    # deltas must move with it, exactly as the GDN recurrent state does.
    rope.set_slot_deltas([-1, -2, -3, -4], BATCH)
    rope.remap_slot_deltas([2, 0, 1, 3])
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.tensor([-3, -1, -2, -4], dtype=torch.int32))


def test_remap_is_a_gather_not_a_swap_chain(rope):
    # Every destination reads the ORIGINAL vector; an in-place shift would corrupt later slots.
    rope.set_slot_deltas([-1, -2, -3, -4], BATCH)
    rope.remap_slot_deltas([1, 1, 0, 0])
    assert torch.equal(rope.decode_delta_vec(BATCH), torch.tensor([-2, -2, -1, -1], dtype=torch.int32))


def test_remap_without_vector_is_noop(rope):
    rope.rope_delta = -2
    rope.remap_slot_deltas([1, 0])
    assert torch.equal(rope.decode_delta_vec(2), torch.full((2,), -2, dtype=torch.int32))
