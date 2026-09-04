# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-slot mRoPE offsets through the on-device rope tables.

The host-side bookkeeping is covered by ``unit/test_rope_slot_deltas.py``. This checks the part
that reaches the device: the rotation each decode slot receives must be the one for
``kv_pos + delta[slot]``, so a batch mixing multimodal and text requests rotates every slot by its
own offset. A regression to the previous model-wide scalar makes the per-slot rows identical, which
the equality against the reference batch catches.

Weights are not involved — only the rope module and its device tables.
Run: pytest models/demos/blackhole/qwen36/tests/test_rope_slot_deltas_device.py -v
"""
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

pytestmark = run_for_blackhole()

ROPE_HEAD_DIM = 64
ROPE_THETA = 10_000_000.0
MAX_SEQ_LEN = 4096
BATCH = 4


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture
def rope(device):
    args = SimpleNamespace(
        rope_head_dim=ROPE_HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        rope_theta=ROPE_THETA,
        mrope_section=[16, 24, 24],
        rope_attention_scaling=1.0,
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
    )
    return Qwen36RoPESetup(device, args)


def _rows_for(rope, kv_positions):
    """Host cos/sin rows the device tables return for the given rope positions, one row per slot."""
    position_ids = torch.tensor(kv_positions, dtype=torch.long).reshape(len(kv_positions), 1)
    cos, sin = rope.get_rot_mats(position_ids)
    return ttnn.to_torch(cos).float().reshape(len(kv_positions), -1), ttnn.to_torch(sin).float().reshape(
        len(kv_positions), -1
    )


def test_each_slot_rotates_at_its_own_offset(rope):
    kv_pos = 40
    deltas = [0, -7, -11, -3]
    rope.set_slot_deltas(deltas, BATCH)

    got_cos, got_sin = _rows_for(rope, (kv_pos + rope.decode_delta_vec(BATCH)).tolist())
    # Reference: each slot's rotation computed independently at its own offset position.
    want_cos, want_sin = _rows_for(rope, [kv_pos + d for d in deltas])

    assert torch.equal(got_cos, want_cos)
    assert torch.equal(got_sin, want_sin)
    # Distinct offsets must give distinct rotations, or the check above would pass trivially.
    assert not torch.equal(got_cos[0], got_cos[1])
    logger.info(f"PASSED: per-slot rope rows distinct for deltas {deltas} at kv_pos {kv_pos}")


def test_text_batch_matches_the_unoffset_rotation(rope):
    # Every text request reports delta 0, so the batch must rotate exactly at the KV positions.
    kv_positions = [10, 11, 12, 13]
    rope.set_slot_deltas([0] * BATCH, BATCH)

    offset = rope.decode_delta_vec(BATCH)
    assert torch.equal(offset, torch.zeros(BATCH, dtype=torch.int32))

    got_cos, got_sin = _rows_for(rope, (torch.tensor(kv_positions) + offset).tolist())
    want_cos, want_sin = _rows_for(rope, kv_positions)
    assert torch.equal(got_cos, want_cos)
    assert torch.equal(got_sin, want_sin)


def test_single_sequence_path_is_unchanged(rope):
    # With no per-slot vector installed the scalar applies, as it did before per-slot tracking.
    kv_pos, delta = 25, -9
    rope.rope_delta = delta

    got_cos, got_sin = _rows_for(rope, (kv_pos + rope.decode_delta_vec(1)).tolist())
    want_cos, want_sin = _rows_for(rope, [kv_pos + delta])
    assert torch.equal(got_cos, want_cos)
    assert torch.equal(got_sin, want_sin)


def test_remapped_slots_keep_their_rotations(rope):
    kv_pos = 60
    deltas = [-1, -5, -9, -13]
    rope.set_slot_deltas(deltas, BATCH)
    remap = [2, 0, 1, 3]
    rope.remap_slot_deltas(remap)

    got_cos, _ = _rows_for(rope, (kv_pos + rope.decode_delta_vec(BATCH)).tolist())
    # After a condense, slot i must rotate at the offset of the request that moved into it.
    want_cos, _ = _rows_for(rope, [kv_pos + deltas[src] for src in remap])
    assert torch.equal(got_cos, want_cos)
