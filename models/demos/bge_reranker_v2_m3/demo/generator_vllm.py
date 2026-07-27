# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-facing model for BAAI/bge-reranker-v2-m3 (cross-encoder).

The model is an XLM-RoBERTa encoder (reused from
``models/demos/wormhole/bge_m3``) followed by a sequence-classification head.
For each (query, document) pair vLLM tokenizes a single concatenated
sequence; the encoder runs on device, and the classification head
(``classifier.dense`` -> tanh -> ``classifier.out_proj``) emits one relevance
logit from the ``<s>`` (CLS) position. The head runs on host in fp32.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import XLMRobertaClassificationHead
from models.demos.bge_reranker_v2_m3.tt.model_config import load_reranker_state_dict


# Short sequences pad batch to 32 rows; the 8192 long-seq path runs 16-wide.
# These match the encoder kernels' safe program-config shapes.
LONG_SEQ_LEN = 8192
LONG_SEQ_CHUNK = 16
SHORT_SEQ_PADDED_BATCH = 32


def _is_long_seq(padded_seq_len: int) -> bool:
    return padded_seq_len == LONG_SEQ_LEN


def _target_padded_batch(original_batch_size: int, padded_seq_len: int) -> int:
    if _is_long_seq(padded_seq_len):
        return LONG_SEQ_CHUNK
    if original_batch_size == 1:
        return 1
    return SHORT_SEQ_PADDED_BATCH


def _execution_chunk(original_batch_size: int, padded_seq_len: int) -> int:
    if _is_long_seq(padded_seq_len):
        return LONG_SEQ_CHUNK
    if original_batch_size == 1:
        return 1
    return SHORT_SEQ_PADDED_BATCH


def _iter_ranges(original_batch_size: int, padded_seq_len: int) -> Iterator[tuple[int, int]]:
    chunk = _execution_chunk(original_batch_size, padded_seq_len)
    for start in range(0, original_batch_size, chunk):
        yield (start, min(start + chunk, original_batch_size))


def _pad_seq(tensor: torch.Tensor, padded_seq_len: int, pad_value: int = 0) -> torch.Tensor:
    seq_len = tensor.shape[1]
    if seq_len == padded_seq_len:
        return tensor
    pad = torch.full(
        (tensor.shape[0], padded_seq_len - seq_len),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _pad_batch(tensor: torch.Tensor, padded_batch_size: int, pad_value: int = 0) -> torch.Tensor:
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


def _to_ttnn_ids(ids: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


class BgeRerankerV2M3:
    """Cross-encoder execution wrapper for bge-reranker-v2-m3."""

    # Declared so vLLM treats this as a pooling / cross-encoder model and routes
    # /score and /rerank through the cross-encoding path (query+document as one
    # concatenated sequence, sigmoid applied to the logit). See interfaces_base
    # is_vllm_model / is_pooling_model.
    is_pooling_model = True
    supports_cross_encoding = True
    default_pooling_type = "CLS"

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "BAAI/bge-reranker-v2-m3",
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
        self.classifier = None

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
    ) -> "BgeRerankerV2M3":
        if optimizations is not None:
            raise ValueError("Optimizations are not supported for bge-reranker-v2-m3")

        if vllm_config is not None:
            if (
                not hasattr(vllm_config.model_config, "override_tt_config")
                or vllm_config.model_config.override_tt_config is None
            ):
                vllm_config.model_config.override_tt_config = {}
            vllm_config.model_config.override_tt_config["is_embedding_model"] = True
            return cls(
                device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
                tt_data_parallel=tt_data_parallel,
                dtype=dtype,
                **kwargs,
            )

        return cls(
            device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            dtype=dtype,
            **kwargs,
        )

    def _initialize_model(self) -> None:
        if self._is_initialized and self.model is not None:
            return
        # Load encoder + classifier weights via the reranker loader, then hand the
        # state_dict to the shared bge-m3 backbone (which skips its own loader when
        # a state_dict is provided). This keeps the bge-m3 module untouched.
        if self.state_dict is None:
            self.state_dict = load_reranker_state_dict(self.model_name)
        self.model_args, self.model, self.state_dict = create_tt_model(
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            state_dict=self.state_dict,
            hf_model_name=self.model_name,
        )
        self.tokenizer = self.model_args.tokenizer
        self.classifier = XLMRobertaClassificationHead.from_state_dict(self.state_dict)
        self._is_initialized = True

    def _validate_request(self, batch_size: int, padded_seq_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if padded_seq_len > self.max_seq_len:
            raise ValueError(f"Padded sequence length {padded_seq_len} exceeds max_seq_len {self.max_seq_len}")

    def _encoder_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=_to_ttnn_ids(input_ids, device=self.device),
            attention_mask=_to_ttnn_ids(attention_mask, device=self.device),
        )
        if output.layout != ttnn.TILE_LAYOUT:
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        hidden = to_torch_auto_compose(output, device=self.device)
        # Normalize [B,1,S,D] -> [B,S,D].
        if hidden.dim() == 4 and hidden.shape[1] == 1:
            hidden = hidden.squeeze(1)
        return hidden.to(torch.float32)

    def _forward_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        chunk_batch_size: int,
    ) -> torch.Tensor:
        hidden = self._encoder_last_hidden(input_ids, attention_mask)  # [B,S,D]
        cls_hidden = hidden[:, 0, :]  # CLS token
        logits = self.classifier(cls_hidden)  # [B,1]
        return logits[:chunk_batch_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del positions, token_type_ids, position_ids
        batch_size, seq_len = input_ids.shape
        padded_seq_len = get_padded_sequence_length(seq_len)
        self._validate_request(batch_size, padded_seq_len)
        self._initialize_model()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        target_padded_batch = _target_padded_batch(batch_size, padded_seq_len)
        chunk_logits = []
        for start, end in _iter_ranges(batch_size, padded_seq_len):
            ids = _pad_batch(
                _pad_seq(input_ids[start:end], padded_seq_len, pad_value=self.tokenizer.pad_token_id),
                target_padded_batch,
                pad_value=self.tokenizer.pad_token_id,
            )
            mask = _pad_batch(
                _pad_seq(attention_mask[start:end], padded_seq_len, pad_value=0),
                target_padded_batch,
                pad_value=0,
            )
            chunk_logits.append(
                self._forward_chunk(ids, mask, chunk_batch_size=end - start)
            )

        return torch.cat(chunk_logits, dim=0)  # [batch, 1]

    # ---- vLLM interface helpers ----
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def get_embedding_dim(self) -> int:
        return 1

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def _init_pooler(self, vllm_config, prefix: str = "") -> None:
        del vllm_config, prefix
        self.pooler = None


def register_model() -> None:
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        ModelRegistry.register_model(
            "BAAI/bge-reranker-v2-m3",
            BgeRerankerV2M3,
            architecture="XLMRobertaForSequenceClassification",
        )
    except ImportError:
        return
