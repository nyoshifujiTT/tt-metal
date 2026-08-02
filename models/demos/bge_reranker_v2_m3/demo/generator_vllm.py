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
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import (
    RerankerChunkedHidden,
    RerankerClassifierPooler,
)
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
        # The pooling runner queries model.pooler right after load (before the
        # first forward that builds the device head), so install the pooler now;
        # it reads self.classifier / self.device lazily at scoring time (the head
        # is built in _post_initialize). Its get_supported_tasks (classify/score)
        # is static and needs no head.
        self.pooler = RerankerClassifierPooler(self)
        # Per-call toggle for the two-track forward (see forward): False = score
        # each chunk to a logit (fork path); True = keep the chunk's un-pooled
        # ttnn hidden for model.pooler (canonical path). Reset in forward.
        self._collect_hidden = False

    # Load encoder + classifier weights via the reranker seq-classification
    # loader, then hand the state_dict to the shared bge-m3 backbone (which skips
    # its own loader when a state_dict is provided). This keeps bge-m3 untouched.
    def _load_state_dict(self):
        return load_reranker_state_dict(self.model_name)

    @classmethod
    def initialize_vllm_model(cls, hf_config, *args, **kwargs):
        """Load classifier weights from the checkpoint vLLM was pointed at.

        The shared base does not thread the served model path into the wrapper,
        so ``model_name`` would otherwise stay the class-default HF id and the
        seq-classification loader (``from_pretrained(model_name)``) would try to
        fetch from the Hub. vLLM passes the resolved checkpoint location in
        ``hf_config._name_or_path`` (a local dir or an HF id), so use it as
        ``model_name`` unless the caller set one explicitly.
        """
        name_or_path = getattr(hf_config, "_name_or_path", None)
        if name_or_path and "model_name" not in kwargs:
            kwargs["model_name"] = name_or_path
        return super().initialize_vllm_model(hf_config, *args, **kwargs)

    def _post_initialize(self) -> None:
        # Device (ttnn) head: CLS extraction + dense->tanh->out_proj run on
        # device in fp32, so the reranker score is computed end-to-end on device.
        self.classifier = XLMRobertaClassificationHeadTT.from_state_dict(self.device, self.state_dict)
        # model.pooler was installed in __init__ (the runner queries it before
        # the first forward); it reads this freshly-built classifier at scoring
        # time, so both the fork path (per-chunk logit) and the canonical path
        # (runner -> model.pooler) use the same device head and score identically.

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
        """Per-chunk primitive for the cross-encoder (fork / default path).

        Runs the encoder on one already-padded chunk, extracts CLS and runs the
        classification head on device, and returns the ``[chunk_batch_size, 1]``
        relevance logit on host. Only the small per-chunk logit crosses back to
        host; the full encoder hidden state stays on device. Called by the
        shared ``_encode_in_chunks`` template method.
        """
        output = self._run_encoder_chunk(padded_inputs)
        if getattr(self, "_collect_hidden", False):
            # Canonical path: return the un-pooled device hidden for this chunk
            # (with its mask + real row count) so model.pooler scores it.
            return (output, padded_inputs["attention_mask"], chunk_batch_size)
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
        return_full_hidden_states: bool = False,
    ) -> torch.Tensor:
        del positions, token_type_ids, position_ids
        batch_size, seq_len = input_ids.shape
        self._validate_request(batch_size, get_padded_sequence_length(seq_len))
        self._initialize_model()

        # Two-track contract (default off): the fork's pooling runner calls
        # forward without the flag and expects the already-scored [batch, 1]
        # logit, so the default path runs the encoder + device CLS/head per chunk
        # and concatenates the per-chunk logits (byte-for-byte the prior
        # behaviour). The canonical runner sets return_full_hidden_states=True
        # because it delegates scoring to model.pooler; in that case forward
        # returns the un-pooled device hidden (per chunk) and the pooler does the
        # CLS extraction + head on device.
        # Both paths reuse the shared _encode_in_chunks template (identical
        # chunk/pad contract); only the per-chunk step differs. A private toggle
        # selects it inside _forward_chunk so the base template stays untouched:
        # default = score the chunk to a host [chunk, 1] logit (fork path); when
        # returning full hidden = keep the chunk's un-pooled ttnn hidden for the
        # pooler. The toggle is scoped to this call (reset in finally).
        if not return_full_hidden_states:
            chunk_logits = self._encode_in_chunks(input_ids, attention_mask=attention_mask)
            return torch.cat(chunk_logits, dim=0)

        self._collect_hidden = True
        try:
            chunks = self._encode_in_chunks(input_ids, attention_mask=attention_mask)
        finally:
            self._collect_hidden = False
        return RerankerChunkedHidden(chunks)

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
