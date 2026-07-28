# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM wrapper base for XLM-RoBERTa encoder models.

Both the bge-m3 embedding wrapper and the bge-reranker-v2-m3 cross-encoder run
the same XLM-RoBERTa encoder backbone (``create_tt_model``) and expose the same
vLLM plumbing: device/vllm_config resolution, ``initialize_vllm_model``, lazy
model construction, request validation and the ``get_max_*`` / pooler helpers.

That plumbing lives here so each model only implements what actually differs:
its ``forward`` (pooling vs classification), ``get_embedding_dim`` and any extra
per-model state. Subclasses customise weight loading and post-init via the
``_load_state_dict`` / ``_post_initialize`` hooks.

Placed in the bge-m3 package because bge-m3 and the reranker are the only two
users today (see the reranker classifier-head note); promote to a neutral
location if a third XLM-RoBERTa encoder model appears.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length


class XlmRobertaEncoder:
    """Common vLLM plumbing for TT XLM-RoBERTa encoder models (single device)."""

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "",
        vllm_config=None,
        prefix: str = "",
        tt_data_parallel: int = 1,
        **kwargs,
    ):
        del prefix, kwargs

        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device
        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # Accepted for API compatibility; execution stays single-device.
        self.tt_data_parallel = tt_data_parallel
        self.dtype = dtype
        self.model_name = model_name

        if vllm_config is not None:
            self.vllm_config = vllm_config

        self.pooler = None
        self._is_initialized = False
        self.model_args = None
        self.model = None
        self.state_dict = None
        self.tokenizer = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = 8192,
        model_location_generator=None,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
        vllm_config=None,
        dtype=ttnn.bfloat16,
        **kwargs,
    ):
        if optimizations is not None:
            raise ValueError(f"Optimizations are not supported for {cls.__name__}")

        if vllm_config is not None:
            return cls(
                device=mesh_device,
                model_location_generator=model_location_generator,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
                tt_data_parallel=tt_data_parallel,
                dtype=dtype,
                **kwargs,
            )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            dtype=dtype,
            **kwargs,
        )

    # ---- hooks for subclasses ----
    def _load_state_dict(self):
        """Return a state_dict to hand to create_tt_model, or None to let the
        backbone load its own weights. Overridden by models that build the
        state_dict themselves (e.g. the reranker seq-classification loader)."""
        return None

    def _post_initialize(self) -> None:
        """Hook run after the encoder is built (e.g. to build a classifier head)."""

    def _initialize_model(self) -> None:
        if self._is_initialized and self.model is not None:
            return

        if self.state_dict is None:
            self.state_dict = self._load_state_dict()

        self.model_args, self.model, self.state_dict = create_tt_model(
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            state_dict=self.state_dict,
            hf_model_name=self.model_name,
        )
        self.tokenizer = self.model_args.tokenizer
        self._post_initialize()
        self._is_initialized = True

    def _validate_request(self, batch_size: int, padded_seq_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if padded_seq_len > self.max_seq_len:
            raise ValueError(f"Padded sequence length {padded_seq_len} exceeds max_seq_len {self.max_seq_len}")

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def _init_pooler(self, vllm_config, prefix: str = "") -> None:
        del vllm_config, prefix
        self.pooler = None


########################################################
# ENCODER PAD / CHUNK / EXECUTION HELPERS
########################################################
# Shared device pad/chunk contract for the XLM-RoBERTa encoder, used by both the
# bge-m3 embedding wrapper and the bge-reranker-v2-m3 cross-encoder.

# Long-sequence path uses fixed 16-wide device execution regardless of max_batch_size.
BGE_M3_LONG_SEQ_LEN = 8192
BGE_M3_LONG_SEQ_CHUNK = 16
# Short-sequence multi-request path pads to 32 rows for device execution.
BGE_M3_SHORT_SEQ_PADDED_BATCH = 32


def is_long_seq_8192(padded_seq_len: int) -> bool:
    return padded_seq_len == BGE_M3_LONG_SEQ_LEN


def get_target_padded_batch_size(original_batch_size: int, padded_seq_len: int) -> int:
    """
    Device padding width for TT execution. Derived from the original request only
    (same value for every chunk, including tail chunks that pad dummy rows).
    """
    if is_long_seq_8192(padded_seq_len):
        return BGE_M3_LONG_SEQ_CHUNK
    if original_batch_size == 1:
        return 1
    return BGE_M3_SHORT_SEQ_PADDED_BATCH


def get_execution_chunk_size(original_batch_size: int, padded_seq_len: int) -> int:
    """
    Number of real batch rows per forward. For long sequences, fixed at 16; tail
    chunks still pad to get_target_padded_batch_size (16).
    """
    if is_long_seq_8192(padded_seq_len):
        return BGE_M3_LONG_SEQ_CHUNK
    if original_batch_size == 1:
        return 1
    return BGE_M3_SHORT_SEQ_PADDED_BATCH


def iter_execution_ranges(
    original_batch_size: int,
    padded_seq_len: int,
) -> Iterator[tuple[int, int]]:
    """Yields (start, end) batch slices for the original request."""
    chunk = get_execution_chunk_size(original_batch_size, padded_seq_len)
    for start in range(0, original_batch_size, chunk):
        yield (start, min(start + chunk, original_batch_size))


def _pad_tensor(tensor: torch.Tensor, padded_seq_len: int, pad_value: int = 0) -> torch.Tensor:
    batch_size, seq_len = tensor.shape
    if seq_len == padded_seq_len:
        return tensor

    padded = torch.full(
        (batch_size, padded_seq_len),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :seq_len] = tensor
    return padded


def _pad_batch_tensor(tensor: torch.Tensor, padded_batch_size: int, pad_value: int = 0) -> torch.Tensor:
    batch_size = tensor.shape[0]
    if batch_size == padded_batch_size:
        return tensor

    padded = torch.full(
        (padded_batch_size, *tensor.shape[1:]),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:batch_size] = tensor
    return padded


def _slice_optional_batch_tensor(
    tensor: Optional[torch.Tensor],
    start: int,
    end: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[0] == 1:
        return tensor
    return tensor[start:end]


def to_ttnn_ids(ids: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _pad_chunk_inputs(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    *,
    padded_seq_len: int,
    padded_batch_size: int,
    pad_token_id: int,
) -> dict[str, Optional[torch.Tensor]]:
    """Pads one batch slice to (padded_batch_size, padded_seq_len) for the device.

    Shared by every caller of the encoder: applies sequence-length padding
    (128/1024/2048/8192 alignment) and batch padding to the fixed device width.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": _pad_batch_tensor(
            _pad_tensor(input_ids, padded_seq_len, pad_value=pad_token_id),
            padded_batch_size,
            pad_value=pad_token_id,
        ),
        "attention_mask": _pad_batch_tensor(
            _pad_tensor(attention_mask, padded_seq_len, pad_value=0),
            padded_batch_size,
            pad_value=0,
        ),
        "token_type_ids": _pad_batch_tensor(
            _pad_tensor(token_type_ids, padded_seq_len, pad_value=0),
            padded_batch_size,
            pad_value=0,
        )
        if token_type_ids is not None
        else None,
        "position_ids": _pad_batch_tensor(
            _pad_tensor(position_ids, padded_seq_len, pad_value=pad_token_id),
            padded_batch_size,
            pad_value=pad_token_id,
        )
        if position_ids is not None
        else None,
    }


def encode_in_chunks(
    input_ids: torch.Tensor,
    process_chunk,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
):
    """Drives the device pad/chunk contract, delegating each chunk to a callback.

    This is the single orchestration point shared by every consumer of the
    XLM-RoBERTa encoder (bge-m3 embedding pooling and the bge-reranker
    cross-encoder). It slices the request into device-sized chunks, pads each to
    the fixed (padded_batch_size, padded_seq_len) device shape, and calls
    ``process_chunk(padded_inputs, chunk_batch_size)`` for every chunk. The
    per-chunk results are returned as a list in request order; the caller decides
    how to combine them (dict concat for bge-m3, tensor concat for the reranker).
    """
    batch_size, seq_len = input_ids.shape
    padded_seq_len = get_padded_sequence_length(seq_len)
    target_padded_batch_size = get_target_padded_batch_size(batch_size, padded_seq_len)

    chunk_outputs = []
    for start, end in iter_execution_ranges(batch_size, padded_seq_len):
        padded_inputs = _pad_chunk_inputs(
            input_ids[start:end],
            _slice_optional_batch_tensor(attention_mask, start, end),
            _slice_optional_batch_tensor(token_type_ids, start, end),
            _slice_optional_batch_tensor(position_ids, start, end),
            padded_seq_len=padded_seq_len,
            padded_batch_size=target_padded_batch_size,
            pad_token_id=pad_token_id,
        )
        chunk_outputs.append(process_chunk(padded_inputs, end - start))
    return chunk_outputs


def run_encoder_chunk(model, device, padded_inputs: dict[str, Optional[torch.Tensor]]) -> ttnn.Tensor:
    """Runs the TT encoder on one already-padded chunk, returning the ttnn output.

    ``model`` is a BgeM3Model instance callable with ttnn tensors; ``device`` is
    the mesh device the caller created the model on (BgeM3Model does not store
    the device, so the owning wrapper passes it in explicitly). The output is
    forced to TILE_LAYOUT so downstream heads can consume it directly.
    """
    token_type_ids = padded_inputs.get("token_type_ids")
    position_ids = padded_inputs.get("position_ids")
    output = model(
        input_ids=to_ttnn_ids(padded_inputs["input_ids"], device=device),
        attention_mask=to_ttnn_ids(padded_inputs["attention_mask"], device=device),
        token_type_ids=(to_ttnn_ids(token_type_ids, device=device) if token_type_ids is not None else None),
        position_ids=(to_ttnn_ids(position_ids, device=device) if position_ids is not None else None),
    )
    if output.layout != ttnn.TILE_LAYOUT:
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
    return output


def encode_to_last_hidden(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    device,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Runs the encoder and returns the last hidden state [B, S_padded, D] on host.

    Thin wrapper over ``encode_in_chunks`` for consumers (e.g. the bge-reranker
    cross-encoder) that want the raw encoder output on host rather than a pooled
    embedding. ``device`` is the mesh device the model was created on.
    """

    def _chunk_to_host(padded_inputs, chunk_batch_size):
        output = run_encoder_chunk(model, device, padded_inputs)
        hidden = to_torch_auto_compose(output, device=device).to(torch.float32)
        if hidden.dim() == 4 and hidden.shape[1] == 1:
            hidden = hidden.squeeze(1)  # [B,1,S,D] -> [B,S,D]
        return hidden[:chunk_batch_size]

    chunks = encode_in_chunks(
        input_ids,
        _chunk_to_host,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )
    return torch.cat(chunks, dim=0)
