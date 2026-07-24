# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-classification head for BAAI/bge-reranker-v2-m3.

The reranker is an XLM-RoBERTa encoder followed by
``RobertaClassificationHead``: a dense layer, tanh, then a projection to a
single logit taken from the ``<s>`` (CLS) position. The head is tiny
(1024x1024 + 1x1024) and is evaluated on host in fp32 so the relevance
score matches the Hugging Face reference exactly.
"""

from __future__ import annotations

import torch

CLASSIFIER_KEYS = (
    "classifier.dense.weight",
    "classifier.dense.bias",
    "classifier.out_proj.weight",
    "classifier.out_proj.bias",
)


class RerankerClassifierHead:
    """Host-side RoBERTa sequence-classification head (fp32)."""

    def __init__(
        self,
        dense_weight: torch.Tensor,
        dense_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        self.dense_weight = dense_weight.to(torch.float32)
        self.dense_bias = dense_bias.to(torch.float32)
        self.out_proj_weight = out_proj_weight.to(torch.float32)
        self.out_proj_bias = out_proj_bias.to(torch.float32)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "RerankerClassifierHead":
        missing = [k for k in CLASSIFIER_KEYS if k not in state_dict]
        if missing:
            raise KeyError(
                "state_dict is missing reranker classifier tensors: "
                f"{missing}. bge-reranker-v2-m3 requires a sequence-classification head."
            )
        return cls(
            dense_weight=state_dict["classifier.dense.weight"],
            dense_bias=state_dict["classifier.dense.bias"],
            out_proj_weight=state_dict["classifier.out_proj.weight"],
            out_proj_bias=state_dict["classifier.out_proj.bias"],
        )

    def __call__(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """Map [batch, hidden] CLS hidden states to [batch, 1] logits.

        Mirrors transformers RobertaClassificationHead at inference (dropout is
        identity): dense -> tanh -> out_proj.
        """
        x = torch.nn.functional.linear(
            cls_hidden.to(torch.float32), self.dense_weight, self.dense_bias
        )
        x = torch.tanh(x)
        x = torch.nn.functional.linear(x, self.out_proj_weight, self.out_proj_bias)
        return x
