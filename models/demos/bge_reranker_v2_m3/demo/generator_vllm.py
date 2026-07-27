# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-facing model for BAAI/bge-reranker-v2-m3 (cross-encoder).

The model is an XLM-RoBERTa encoder (reused from
``models/demos/wormhole/bge_m3``) followed by a sequence-classification head.
For each (query, document) pair vLLM tokenizes a single concatenated
sequence; the encoder runs on device via the shared
``bge_m3.tt.encode.encode_to_last_hidden`` entry point (which owns the device
padding/chunking contract), and the classification head
(``classifier.dense`` -> tanh -> ``classifier.out_proj``) emits one relevance
logit from the ``<s>`` (CLS) position. The head runs on host in fp32.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.encode import encode_to_last_hidden
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import XLMRobertaClassificationHead
from models.demos.bge_reranker_v2_m3.tt.model_config import load_reranker_state_dict


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
        self._validate_request(batch_size, get_padded_sequence_length(seq_len))
        self._initialize_model()

        # Shared backbone entry point owns the device padding/chunking contract
        # and returns the encoder last hidden state [B, S_padded, D] on host.
        hidden = encode_to_last_hidden(
            self.model,
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        cls_hidden = hidden[:, 0, :]  # <s> (CLS) position
        return self.classifier(cls_hidden)  # [batch, 1] relevance logits

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
