# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device (ttnn) XLM-RoBERTa sequence-classification head.

Device-native counterpart of ``xlm_roberta_classification_head.XLMRobertaClassificationHead``
(the host fp32 reference). It reproduces the transformers
``XLMRobertaClassificationHead`` / ``RobertaClassificationHead`` at inference
(dropout is identity): ``dense -> tanh -> out_proj`` applied to the ``<s>``
(CLS) hidden state, emitting one relevance logit per (query, document) pair.

Unlike the host head, this runs on the Tenstorrent device in fp32: the two
linears use ``fp32_dest_acc_en`` + ``MathFidelity.HiFi4`` so the pooled logit is
computed on-device with full fp32 accumulation and only the final ``[B, 1]``
logit crosses back to host. On Blackhole the fp32-dest-accumulation path is
bug-free (the Wormhole ``HiFi4 + fp32_acc`` corner case is fixed there), and it
matches the host fp32 reference to PCC ~= 1 (max abs diff ~1e-3, tile rounding).

Reference (transformers, pinned to tag v4.44.2 =
commit 174890280b340b89c5bfa092f6b4fb0e2dc2d7fc):
- RobertaClassificationHead:
  https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/roberta/modeling_roberta.py#L1423-L1442
- XLMRobertaClassificationHead (same implementation):
  https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1438-L1457

Like the host head, this is not specific to bge-reranker-v2-m3; it applies to
any XLM-RoBERTa / RoBERTa ``*ForSequenceClassification`` checkpoint. It lives
under the reranker demo because that is its only user; promote to a shared
location if a second such model is added.
"""

from __future__ import annotations

import torch

import ttnn


def _classification_head_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """fp32 compute config for the head's linears.

    ``WormholeComputeKernelConfig`` is the generic struct name used across this
    codebase for both Wormhole and Blackhole. ``fp32_dest_acc_en`` keeps the
    matmul accumulation in fp32 (32-bit dest registers) and ``HiFi4`` is the
    highest math fidelity, so the tiny head matches the host fp32 reference.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


class XLMRobertaClassificationHeadTT:
    """Device (ttnn) XLM-RoBERTa sequence-classification head (fp32)."""

    # The state_dict tensors this head consumes.
    CLASSIFIER_KEYS = (
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    )

    def __init__(
        self,
        device,
        dense_weight: torch.Tensor,
        dense_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        self.device = device
        self._compute_kernel_config = _classification_head_compute_kernel_config()

        def _to_device(t: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                t.to(torch.float32),
                device=device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # ttnn.linear computes ``x @ W`` with W shaped [in_features, out_features],
        # whereas torch stores nn.Linear weights as [out_features, in_features].
        # Transpose on load so the device matmul reproduces torch's ``x @ W.T``.
        self._dense_weight = _to_device(dense_weight.t().contiguous())
        self._dense_bias = _to_device(dense_bias.reshape(1, -1))
        self._out_proj_weight = _to_device(out_proj_weight.t().contiguous())
        self._out_proj_bias = _to_device(out_proj_bias.reshape(1, -1))

    @classmethod
    def from_state_dict(cls, device, state_dict: dict) -> "XLMRobertaClassificationHeadTT":
        missing = [k for k in cls.CLASSIFIER_KEYS if k not in state_dict]
        if missing:
            raise KeyError(
                "state_dict is missing reranker classifier tensors: "
                f"{missing}. bge-reranker-v2-m3 requires a sequence-classification head."
            )
        return cls(
            device,
            dense_weight=state_dict["classifier.dense.weight"],
            dense_bias=state_dict["classifier.dense.bias"],
            out_proj_weight=state_dict["classifier.out_proj.weight"],
            out_proj_bias=state_dict["classifier.out_proj.bias"],
        )

    def __call__(self, cls_hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Map a ``[batch, hidden]`` CLS hidden state to ``[batch, 1]`` logits.

        ``cls_hidden`` is a ttnn device tensor (TILE_LAYOUT). The two linears and
        the tanh run on device; the returned ttnn tensor is the ``[batch, 1]``
        relevance logit, still on device (the caller moves the small result to
        host).
        """
        x = ttnn.linear(
            cls_hidden,
            self._dense_weight,
            bias=self._dense_bias,
            compute_kernel_config=self._compute_kernel_config,
            dtype=ttnn.float32,
        )
        x = ttnn.tanh(x)
        logits = ttnn.linear(
            x,
            self._out_proj_weight,
            bias=self._out_proj_bias,
            compute_kernel_config=self._compute_kernel_config,
            dtype=ttnn.float32,
        )
        return logits
