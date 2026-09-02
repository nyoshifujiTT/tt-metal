# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched multimodal prefill routing in ``Qwen36ForCausalLM.prefill_forward``.

Under batched serving vLLM hands prefill a batch of requests whose pixel values arrive as
per-request lists. These tests pin that each row's vision tower runs on its own entry, that image
and text requests may share a batch, and that the resulting M-RoPE deltas travel back to vLLM and
into the decode slot each request was prefilled into.

The routing runs against a stub model, so no weights or device allocation are involved, but
importing the wrapper pulls in ttnn and vllm — hence the serving-image requirement.
Run: pytest models/demos/blackhole/qwen36/tests/unit/test_vllm_batched_multimodal.py -v
"""
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup

pytest.importorskip("vllm", reason="the vLLM wrapper under test imports vllm and ttnn")

MAX_BATCH = 4
VOCAB = 8


class _StubModel:
    """Records which pixel rows reached the vision tower and what prefill was asked to splice."""

    def __init__(self, num_devices=4, max_batch_size=MAX_BATCH, deltas=None):
        self.num_devices = num_devices
        self.args = SimpleNamespace(max_batch_size=max_batch_size, vocab_size=VOCAB)
        self.rope = Qwen36RoPESetup.__new__(Qwen36RoPESetup)
        self.rope.rope_delta = 0
        self.rope._slot_deltas = None
        self.image_calls = []
        self.video_calls = []
        self.prefill_call = None
        self._deltas = deltas or {}

    def get_image_features(self, pixel_values, grid_thw):
        self.image_calls.append((pixel_values, grid_thw))
        return f"image-{pixel_values}"

    def get_video_features(self, pixel_values, grid_thw):
        self.video_calls.append((pixel_values, grid_thw))
        return f"video-{pixel_values}"

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None, vision_tokens_list=None):
        self.prefill_call = SimpleNamespace(
            token_ids_list=token_ids_list,
            page_table=page_table,
            empty_slots=empty_slots,
            valid_lens=valid_lens,
            vision_tokens_list=vision_tokens_list,
        )
        n = len(token_ids_list)
        logits = [torch.zeros(1, 1, VOCAB) for _ in range(n)]
        deltas = [self._deltas.get(u, 0) for u in range(n)]
        for u in range(n):
            self.rope.set_slot_delta(int(empty_slots[u]), deltas[u], self.args.max_batch_size)
        return logits, deltas


def _wrapper(model):
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

    wrapper = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    wrapper.model = [model]
    return wrapper


def _prefill(model, n_requests, **kwargs):
    tokens = torch.zeros(n_requests, 16, dtype=torch.int32)
    page_table = torch.zeros(n_requests, 4, dtype=torch.int32)
    prompt_lens = torch.full((n_requests,), 16, dtype=torch.int32)
    return _wrapper(model).prefill_forward(tokens, page_table, None, prompt_lens, **kwargs)


def test_image_request_is_accepted_in_a_batch():
    # This used to assert and take the whole engine down; the batch must now prefill normally.
    model = _StubModel()
    _prefill(model, 2, pixel_values=[torch.ones(4, 8), None], image_grid_thw=[torch.ones(1, 3), None])
    assert len(model.image_calls) == 1


def test_each_row_runs_its_own_vision_tower():
    model = _StubModel()
    first, second = torch.ones(4, 8), torch.ones(4, 8) * 2
    _prefill(
        model,
        2,
        pixel_values=[first, second],
        image_grid_thw=[torch.ones(1, 3), torch.ones(1, 3)],
    )
    assert [call[0] is first for call in model.image_calls][0]
    assert model.image_calls[1][0] is second


def test_text_rows_splice_nothing():
    # A text request inside a mixed batch must not pick up its neighbour's image rows.
    model = _StubModel()
    _prefill(model, 3, pixel_values=[None, torch.ones(4, 8), None], image_grid_thw=[None, torch.ones(1, 3), None])
    spliced = model.prefill_call.vision_tokens_list
    assert spliced[0] is None and spliced[2] is None
    assert spliced[1] is not None


def test_video_rows_take_the_video_tower():
    model = _StubModel()
    _prefill(
        model,
        2,
        pixel_values_videos=[None, torch.ones(4, 8)],
        video_grid_thw=[None, torch.ones(1, 3)],
    )
    assert len(model.video_calls) == 1
    assert model.image_calls == []


def test_text_only_batch_splices_nothing():
    model = _StubModel()
    _prefill(model, 2)
    assert model.prefill_call.vision_tokens_list == [None, None]


def test_empty_pixel_placeholder_is_not_multimodal():
    # vLLM attaches an empty placeholder to text requests of a multimodal-registered model.
    model = _StubModel()
    _prefill(model, 2, pixel_values=[None, None], image_grid_thw=[None, None])
    assert model.image_calls == []
    assert model.prefill_call.vision_tokens_list == [None, None]


def test_deltas_are_reported_to_vllm_in_request_order():
    # The plugin stores rope_deltas[i] against req_ids[i], so the row order is the contract.
    model = _StubModel(deltas={0: -11, 1: 0})
    _, rope_deltas = _prefill(model, 2, pixel_values=[torch.ones(4, 8), None], image_grid_thw=[torch.ones(1, 3), None])
    assert torch.equal(rope_deltas, torch.tensor([-11, 0], dtype=torch.long))


def test_deltas_land_in_the_slot_each_request_was_prefilled_into():
    model = _StubModel(deltas={0: -11, 1: -3})
    _prefill(
        model,
        2,
        empty_slots=[2, 0],
        pixel_values=[torch.ones(4, 8), torch.ones(4, 8)],
        image_grid_thw=[torch.ones(1, 3), torch.ones(1, 3)],
    )
    # Request 0 went to slot 2 and request 1 to slot 0, so the offsets must follow the slots.
    assert torch.equal(model.rope.decode_delta_vec(MAX_BATCH), torch.tensor([-3, 0, -11, 0], dtype=torch.int32))


def test_prompt_lens_trim_each_row():
    model = _StubModel()
    tokens = torch.arange(2 * 16, dtype=torch.int32).reshape(2, 16)
    page_table = torch.zeros(2, 4, dtype=torch.int32)
    _wrapper(model).prefill_forward(tokens, page_table, None, torch.tensor([5, 16], dtype=torch.int32))
    assert [t.shape[1] for t in model.prefill_call.token_ids_list] == [5, 16]
