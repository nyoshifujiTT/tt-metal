# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-facing model for BAAI/bge-reranker-v2-m3 (cross-encoder).

The model is an XLM-RoBERTa encoder (reused from
``models/demos/wormhole/bge_m3``) followed by a sequence-classification head.
For each (query, document) pair vLLM tokenizes a single concatenated
sequence; the encoder runs on device via the shared
``XlmRobertaEncoder._encode_to_last_hidden`` method (which owns the device
padding/chunking contract), and the classification head
(``classifier.dense`` -> tanh -> ``classifier.out_proj``) emits one relevance
logit from the ``<s>`` (CLS) position. The head runs on host in fp32.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import XLMRobertaClassificationHead
from models.demos.bge_reranker_v2_m3.tt.model_config import load_reranker_state_dict
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import XlmRobertaEncoder


class BgeRerankerV2M3(XlmRobertaEncoder):
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
        self.classifier = None

    # Load encoder + classifier weights via the reranker seq-classification
    # loader, then hand the state_dict to the shared bge-m3 backbone (which skips
    # its own loader when a state_dict is provided). This keeps bge-m3 untouched.
    def _load_state_dict(self):
        return load_reranker_state_dict(self.model_name)

    def _post_initialize(self) -> None:
        self.classifier = XLMRobertaClassificationHead.from_state_dict(self.state_dict)

    def _encode_to_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the encoder and returns the last hidden state [B, S_padded, D] on host.

        Thin wrapper over ``_encode_in_chunks`` for consumers (e.g. the
        bge-reranker cross-encoder) that want the raw encoder output on host
        rather than a pooled embedding. Relies on the default ``_forward_chunk``
        (raw last hidden); models that override ``_forward_chunk`` for pooling
        (bge-m3) do not use this path.
        """
        chunks = self._encode_in_chunks(input_ids, attention_mask=attention_mask)
        return torch.cat(chunks, dim=0)

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
        hidden = self._encode_to_last_hidden(input_ids, attention_mask)
        cls_hidden = hidden[:, 0, :]  # <s> (CLS) position
        return self.classifier(cls_hidden)  # [batch, 1] relevance logits

    # ---- vLLM interface helpers ----
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def get_embedding_dim(self) -> int:
        return 1


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
