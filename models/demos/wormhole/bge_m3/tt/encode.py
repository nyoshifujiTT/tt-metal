# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared encoder execution for the BGE-M3 XLM-RoBERTa backbone.

``encode_to_last_hidden`` runs the TT encoder for an arbitrary
(batch, seq_len) request by applying the sequence-length padding and the
batch padding / chunking required by the device kernels, then concatenates
the per-chunk results and returns the last hidden state as a host tensor
``[batch, seq_len_padded, hidden]``.

This is the single entry point that both the bge-m3 embedding wrapper and
downstream heads (e.g. the bge-reranker-v2-m3 cross-encoder) can build on:
callers pass tokenized input and receive the encoder output, without having
to re-implement the device padding / chunking contract.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.demo.generator_vllm import (
    _pad_batch_tensor,
    _pad_tensor,
    get_target_padded_batch_size,
    iter_execution_ranges,
    to_ttnn_ids,
)
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length


def _encode_chunk(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    chunk_batch_size: int,
) -> torch.Tensor:
    output = model(
        input_ids=to_ttnn_ids(input_ids, device=model.device),
        attention_mask=to_ttnn_ids(attention_mask, device=model.device),
    )
    if output.layout != ttnn.TILE_LAYOUT:
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
    hidden = to_torch_auto_compose(output, device=model.device).to(torch.float32)
    if hidden.dim() == 4 and hidden.shape[1] == 1:
        hidden = hidden.squeeze(1)  # [B,1,S,D] -> [B,S,D]
    return hidden[:chunk_batch_size]


def encode_to_last_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Run the TT encoder and return last hidden state [B, S_padded, D] on host.

    Handles sequence-length padding (128/1024/2048 alignment) and batch
    padding/chunking (short-seq 32-row pad, long-seq 16-row chunks) required by
    the device. ``model`` is a BgeM3Model instance (has ``.device`` and is
    callable with ttnn input_ids/attention_mask).
    """
    batch_size, seq_len = input_ids.shape
    padded_seq_len = get_padded_sequence_length(seq_len)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    target_padded_batch = get_target_padded_batch_size(batch_size, padded_seq_len)
    chunks = []
    for start, end in iter_execution_ranges(batch_size, padded_seq_len):
        ids = _pad_batch_tensor(
            _pad_tensor(input_ids[start:end], padded_seq_len, pad_value=pad_token_id),
            target_padded_batch,
            pad_value=pad_token_id,
        )
        mask = _pad_batch_tensor(
            _pad_tensor(attention_mask[start:end], padded_seq_len, pad_value=0),
            target_padded_batch,
            pad_value=0,
        )
        chunks.append(_encode_chunk(model, ids, mask, chunk_batch_size=end - start))

    return torch.cat(chunks, dim=0)
