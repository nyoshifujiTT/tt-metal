# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

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
        # heads on device. Used as the per-chunk callback of self._encode_in_chunks().
        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]
        output = self._run_encoder_chunk(padded_inputs)

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

        chunk_outputs = self._encode_in_chunks(
            input_ids,
            self._forward_chunk,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
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
def _concatenate_chunk_outputs(chunk_outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not chunk_outputs:
        return {}
    output_names = chunk_outputs[0].keys()
    return {
        output_name: torch.cat([chunk_output[output_name] for chunk_output in chunk_outputs], dim=0)
        for output_name in output_names
    }


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
