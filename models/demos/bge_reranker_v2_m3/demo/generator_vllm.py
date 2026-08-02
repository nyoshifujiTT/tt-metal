# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-facing model for BAAI/bge-reranker-v2-m3 (cross-encoder).

The model is an XLM-RoBERTa encoder (reused from
``models/demos/wormhole/bge_m3``) followed by a sequence-classification head.
For each (query, document) pair vLLM tokenizes a single concatenated
sequence; the encoder runs on device via the shared
``XlmRobertaEncoder`` chunking contract, and the classification head
(``classifier.dense`` -> tanh -> ``classifier.out_proj``) emits one relevance
logit from the ``<s>`` (CLS) position. Both the CLS extraction and the head run
on device in fp32 (``XLMRobertaClassificationHeadTT``): each chunk is scored to
a ``[chunk, 1]`` logit on device and only that tiny logit crosses back to host,
so the score is computed end-to-end on device (the full encoder hidden state is
never transferred to host).
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head_tt import (
    XLMRobertaClassificationHeadTT,
)
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import score_cls_on_device
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
        # Device (ttnn) head: CLS extraction + dense->tanh->out_proj run on
        # device in fp32, so the reranker score is computed end-to-end on device.
        self.classifier = XLMRobertaClassificationHeadTT.from_state_dict(self.device, self.state_dict)

    def _score_chunk_on_device(self, output: ttnn.Tensor, attention_mask: torch.Tensor) -> ttnn.Tensor:
        """Extract the ``<s>`` (CLS) hidden and run the device head on one chunk.

        ``output`` is the encoder's ttnn last-hidden-state for this chunk. The
        CLS row (position 0) is sliced on device, then the device classification
        head produces a ``[chunk, 1]`` relevance logit -- all on device, so the
        full hidden state never leaves the device.
        """
        batch_size, seq_len = attention_mask.shape
        # CLS extraction + device head; shared with the device pooler so the
        # fork (per-chunk logit) and canonical (model.pooler) paths score
        # identically. Crops the padded chunk to its real [batch, seq] first.
        return score_cls_on_device(output, self.classifier, batch_size, seq_len)

    def _forward_chunk(self, padded_inputs: dict[str, Optional[torch.Tensor]], chunk_batch_size: int) -> torch.Tensor:
        """Per-chunk primitive for the cross-encoder: run the encoder on one
        already-padded chunk, extract CLS and run the classification head on
        device, and return the ``[chunk_batch_size, 1]`` relevance logit on host.

        Only the small per-chunk logit crosses back to host; the full encoder
        hidden state stays on device. Called by the shared ``_encode_in_chunks``
        template method.
        """
        output = self._run_encoder_chunk(padded_inputs)
        logits_tt = self._score_chunk_on_device(output, padded_inputs["attention_mask"])
        logits = to_torch_auto_compose(logits_tt, device=self.device).to(torch.float32)
        logits = logits.reshape(-1, 1)
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
        self._validate_request(batch_size, get_padded_sequence_length(seq_len))
        self._initialize_model()

        # The shared chunking template runs the encoder + device CLS/head per
        # chunk (see _forward_chunk), so each chunk returns a [chunk, 1] logit on
        # host. Concatenate them back into the request's [batch, 1] logits.
        chunk_logits = self._encode_in_chunks(input_ids, attention_mask=attention_mask)
        return torch.cat(chunk_logits, dim=0)

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
