# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator, Optional

import torch
import transformers

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.demo.m3_scores import _get_special_token_ids, _sparse_embedding_scatter_ttnn
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import XlmRobertaEncoder


class BgeM3ForEmbedding(XlmRobertaEncoder):
    """
    vLLM-facing embedding wrapper for direct BGE-M3 encoder execution.

    This implementation intentionally keeps a single-device execution path.
    """

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "BAAI/bge-m3",
        vllm_config=None,
        prefix: str = "",
        tt_data_parallel: int = 1,
        **kwargs,
    ):
        self.sentence_pooling_method = kwargs.pop("sentence_pooling_method", "mean")
        self.normalize_embeddings = kwargs.pop("normalize_embeddings", False)
        self.return_dense = kwargs.pop("return_dense", True)
        self.return_sparse = kwargs.pop("return_sparse", False)
        self.return_colbert = kwargs.pop("return_colbert", False)

        super().__init__(
            device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            model_name=model_name,
            vllm_config=vllm_config,
            prefix=prefix,
            tt_data_parallel=tt_data_parallel,
            **kwargs,
        )

        self.config = transformers.AutoConfig.from_pretrained(model_name)
        # Compatibility placeholders for callers that probe the newer API.
        self.model_args_list = None
        self.models = None
        self.data_parallel = None
        self.submeshes = None

    def _forward_chunk(
        self,
        padded_inputs: dict[str, Optional[torch.Tensor]],
        chunk_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        # Runs the shared encoder chunk, then applies the bge-m3 embedding pooling
        # heads on device. Used as the per-chunk callback of encode_in_chunks().
        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]
        output = run_encoder_chunk(self.model, self.device, padded_inputs)

        return_dict = {}
        if self.return_dense:
            return_dict["dense_vecs"] = self._dense_embedding(output, attention_mask)[:chunk_batch_size]
        if self.return_sparse:
            return_dict["sparse_vecs"] = self._sparse_embedding(output, input_ids)[:chunk_batch_size]
        if self.return_colbert:
            return_dict["colbert_vecs"] = self._colbert_embedding(output, attention_mask)[:chunk_batch_size]

        return return_dict

    def _dense_embedding(self, last_hidden_state: ttnn.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        tt_hidden = _crop_hidden_state_ttnn(last_hidden_state, batch_size, seq_len)
        if len(tt_hidden.shape) == 3:
            tt_hidden = ttnn.unsqueeze(tt_hidden, dim=1)

        tt_hidden = ttnn.to_memory_config(tt_hidden, ttnn.DRAM_MEMORY_CONFIG)
        B, _, S, D = tt_hidden.shape

        if self.sentence_pooling_method == "cls":
            pooled_tt = ttnn.slice(tt_hidden, [0, 0, 0, 0], [B, 1, 1, D])
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        elif self.sentence_pooling_method == "mean":
            mask_torch = attention_mask[:, :S].unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)
            mask_tt = ttnn.from_torch(
                mask_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            summed_tt = ttnn.sum(ttnn.multiply(tt_hidden, mask_tt), dim=2)
            counts_torch = attention_mask[:, :S].sum(dim=1, keepdim=True).clamp(min=1).to(torch.bfloat16).unsqueeze(1)
            counts_tt = ttnn.from_torch(
                counts_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            pooled_tt = ttnn.divide(summed_tt, counts_tt)
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        elif self.sentence_pooling_method == "last_token":
            left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
            if left_padding:
                pooled_tt = ttnn.slice(tt_hidden, [0, 0, S - 1, 0], [B, 1, S, D])
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
            else:
                selector = torch.zeros((batch_size, seq_len), dtype=torch.bfloat16)
                selector[torch.arange(batch_size), attention_mask.sum(dim=1) - 1] = 1
                selector_tt = ttnn.from_torch(
                    selector.unsqueeze(1).unsqueeze(-1),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
                pooled_tt = ttnn.sum(ttnn.multiply(tt_hidden, selector_tt), dim=2)
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

        pooled = to_torch_auto_compose(pooled_tt, device=self.device)
        if pooled.dim() == 3 and pooled.shape[1] == 1:
            pooled = pooled.squeeze(1)
        return pooled.to(torch.float32)

    def _sparse_embedding(
        self,
        hidden_state: ttnn.Tensor,
        input_ids: torch.Tensor,
        return_embedding: bool = True,
    ) -> torch.Tensor:
        self._initialize_model()
        if self.model is None or self.model.sparse_linear is None:
            raise ValueError("Sparse linear head is not initialized")

        batch_size, seq_len = input_ids.shape
        token_weights_tt = self.model.sparse_linear(hidden_state)
        token_weights_tt = _crop_hidden_state_ttnn(token_weights_tt, batch_size, seq_len)

        if not return_embedding:
            token_weights = to_torch_auto_compose(token_weights_tt, device=self.device)
            if token_weights.dim() == 4 and token_weights.shape[1] == 1:
                token_weights = token_weights.squeeze(1)
            return token_weights.to(torch.float32)

        token_weights_tt = _flatten_sparse_token_weights_ttnn(token_weights_tt)
        input_ids_tt = ttnn.from_torch(
            input_ids.long().to(torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        unused_tokens = _get_special_token_ids(self.tokenizer, self.config.vocab_size)
        sparse_embedding_tt = _sparse_embedding_scatter_ttnn(
            self.device,
            token_weights_tt,
            input_ids_tt,
            self.config.vocab_size,
            unused_tokens,
        )
        sparse_embedding = to_torch_auto_compose(sparse_embedding_tt, device=self.device)
        return sparse_embedding.to(torch.float32)

    def _colbert_embedding(self, hidden_state: ttnn.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._initialize_model()
        if self.model is None or self.model.colbert_linear is None:
            raise ValueError("ColBERT linear head is not initialized")

        batch_size, seq_len = attention_mask.shape
        tt_hidden = _crop_hidden_state_ttnn(hidden_state, batch_size, seq_len)
        colbert_tt = self.model.colbert_linear(tt_hidden)
        colbert_tt = ttnn.to_memory_config(colbert_tt, ttnn.DRAM_MEMORY_CONFIG)

        if len(colbert_tt.shape) == 4:
            B, one, S, D = colbert_tt.shape
            colbert_tt = ttnn.slice(colbert_tt, [0, 0, 1, 0], [B, one, S, D])
            mask_torch = attention_mask[:, 1:S].unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)
        else:
            B, S, D = colbert_tt.shape
            colbert_tt = ttnn.slice(colbert_tt, [0, 1, 0], [B, S, D])
            mask_torch = attention_mask[:, 1:S].unsqueeze(-1).to(torch.bfloat16)

        mask_tt = ttnn.from_torch(
            mask_torch,
            device=self.device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        colbert_tt = ttnn.multiply(colbert_tt, mask_tt)

        colbert_vecs = to_torch_auto_compose(colbert_tt, device=self.device)
        if colbert_vecs.dim() == 4 and colbert_vecs.shape[1] == 1:
            colbert_vecs = colbert_vecs.squeeze(1)
        return colbert_vecs.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_request(input_ids.shape[0], get_padded_sequence_length(input_ids.shape[1]))
        self._initialize_model()

        chunk_outputs = encode_in_chunks(
            input_ids,
            self._forward_chunk,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return _concatenate_chunk_outputs(chunk_outputs)

    def get_embedding_dim(self) -> int:
        return self.config.hidden_size


def register_model() -> None:
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        ModelRegistry.register_model(
            "BAAI/bge-m3",
            BgeM3ForEmbedding,
            architecture="RobertaModel",
        )
    except ImportError:
        return


########################################################
# HELPER FUNCTIONS
########################################################

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


def _concatenate_chunk_outputs(chunk_outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not chunk_outputs:
        return {}
    output_names = chunk_outputs[0].keys()
    return {
        output_name: torch.cat([chunk_output[output_name] for chunk_output in chunk_outputs], dim=0)
        for output_name in output_names
    }


def to_ttnn_ids(ids: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _crop_hidden_state_ttnn(hidden_state: ttnn.Tensor, batch_size: int, seq_len: int) -> ttnn.Tensor:
    shape = hidden_state.shape
    if len(shape) == 4:
        return ttnn.slice(
            hidden_state,
            [0, 0, 0, 0],
            [batch_size, shape[1], seq_len, shape[3]],
        )
    if len(shape) == 3:
        return ttnn.slice(
            hidden_state,
            [0, 0, 0],
            [batch_size, seq_len, shape[2]],
        )
    raise ValueError(f"Unsupported hidden_state rank: shape={shape}")


def _flatten_sparse_token_weights_ttnn(token_weights: ttnn.Tensor) -> ttnn.Tensor:
    token_weights = ttnn.to_memory_config(token_weights, ttnn.DRAM_MEMORY_CONFIG)
    if token_weights.layout != ttnn.TILE_LAYOUT:
        token_weights = ttnn.to_layout(token_weights, ttnn.TILE_LAYOUT)
    shape = token_weights.shape
    if len(shape) == 4:
        batch_size, heads, seq_len, width = map(int, shape)
        if heads != 1 or width != 1:
            raise ValueError(f"Unsupported sparse token weight shape: {shape}")
        return ttnn.reshape(token_weights, [batch_size, seq_len])
    if len(shape) == 3:
        batch_size, dim1, dim2 = map(int, shape)
        if dim1 == 1:
            return ttnn.reshape(token_weights, [batch_size, dim2])
        if dim2 == 1:
            return ttnn.reshape(token_weights, [batch_size, dim1])
        raise ValueError(f"Unsupported sparse token weight shape: {shape}")
    if len(shape) == 2:
        return token_weights
    raise ValueError(f"Unsupported sparse token weight rank: shape={shape}")
