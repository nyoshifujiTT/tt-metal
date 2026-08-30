# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Batched prefill must also honour the flat/per-token pooling contract.

``prefill_forward_text(return_full_hidden_states=True)`` promises a per-user
list of ``[seq_i, hidden]`` tensors. The sequential per-user path filled that
list; the batched path -- taken whenever more than one request is scheduled with
the same padded length, i.e. the common serving case -- only implemented the
last-token extraction and left the list all-None while reporting success. The
caller then concatenated nothing and serving Qwen3-Embedding-0.6B on p150 died
with

  ValueError: torch.cat(): expected a non-empty list of Tensors

These tests pin the extraction helper directly, so they need no device: the
model's device-side ops are stubbed and only the per-slot bookkeeping (which
slot maps to which output row, and that each row is trimmed to that request's
real length) is asserted.
"""

import types

import torch

from models.tt_transformers.tt.generator import Generator


class _FakeHidden:
    """Stands in for the ttnn tensor a trace produces."""

    def __init__(self, slot):
        self.slot = slot

    def cpu(self, blocking=False):
        return self


class _FakeTraceOutput:
    """``logits[slot:slot+1, :, :, :]`` on the batched trace output."""

    def __getitem__(self, key):
        slot_slice = key[0]
        return _FakeHidden(slot_slice.start)


class _FakeModel:
    def __init__(self, hidden=4):
        self.mesh_device = object()
        self.hidden = hidden
        self.normed_slots = []

    def process_full_hidden_states_after_prefill_trace(self, hidden):
        # Whole-sequence final norm, no last-token slice: that selection belongs
        # to the Pooler on this path.
        self.normed_slots.append(hidden.slot)
        return hidden

    def process_output_prefill_full_hidden_states(self, tt_out, seq_len):
        # One distinguishable row per token, tagged with the originating slot.
        return torch.full((seq_len, self.hidden), float(tt_out.slot))


def _generator_with(model, monkeypatch):
    generator = Generator.__new__(Generator)
    generator.model = [model]
    # The helper only synchronizes; no device is involved in these tests.
    monkeypatch.setattr(
        "models.tt_transformers.tt.generator.ttnn",
        types.SimpleNamespace(synchronize_device=lambda device: None),
        raising=False,
    )
    return generator


def test_every_scheduled_slot_produces_a_hidden_tensor(monkeypatch):
    model = _FakeModel()
    generator = _generator_with(model, monkeypatch)
    output_full_hidden = [None, None]

    generator._extract_batched_prefill_full_hidden(
        _FakeTraceOutput(),
        model_id=0,
        empty_slots=[0, 1],
        prompt_lens=[7, 3],
        output_full_hidden=output_full_hidden,
    )

    assert all(h is not None for h in output_full_hidden), (
        "an unfilled entry makes the caller concatenate an empty list"
    )
    # Trimmed to each request's real token count, not the shared padded length.
    assert [h.shape[0] for h in output_full_hidden] == [7, 3]


def test_slots_map_to_their_own_output_row(monkeypatch):
    """Requests may occupy non-contiguous slots; the output is indexed by
    request order, so the mapping must follow ``empty_slots``."""
    model = _FakeModel()
    generator = _generator_with(model, monkeypatch)
    output_full_hidden = [None, None]

    generator._extract_batched_prefill_full_hidden(
        _FakeTraceOutput(),
        model_id=0,
        empty_slots=[3, 5],
        prompt_lens=[2, 4],
        output_full_hidden=output_full_hidden,
    )

    assert model.normed_slots == [3, 5]
    assert torch.equal(output_full_hidden[0], torch.full((2, 4), 3.0))
    assert torch.equal(output_full_hidden[1], torch.full((4, 4), 5.0))
