# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TT-native XLM-RoBERTa cross-encoder ClassifierPooler.

Device-native counterpart of vLLM's standard classification pooler. Upstream
vLLM builds a two-stage ``SequencePooler`` for cross-encoders / rerankers:
``CLSPool`` (take the ``<s>`` / first token) followed by ``ClassifierPoolerHead``
(``classifier(pooled) -> activation``). Both stages are torch ops that run on
the model's device (see vLLM v0.24.0
``vllm/model_executor/layers/pooler/seqwise/methods.py::CLSPool`` and
``.../heads.py::ClassifierPoolerHead``).

On Tenstorrent the hidden state is a ``ttnn.Tensor`` on device, so the standard
torch stages (``hidden_states[first_token_indices]`` advanced-indexing +
``nn.Linear``) cannot consume it without first copying the whole
``[batch, seq, hidden]`` tensor back to host. This class is the TT-native
version of the same two stages: CLS extraction via ``ttnn.slice`` and the fp32
device classification head (:class:`XLMRobertaClassificationHeadTT`), so the
score is computed end-to-end on device and only the final ``[batch, 1]`` logit
crosses back to host. The public interface matches vLLM's ``Pooler`` ABC
(``get_supported_tasks`` / ``forward(hidden_states, pooling_metadata)``) so the
pooling runner drives it exactly like any upstream pooler.

The activation (sigmoid for /score) is applied by vLLM downstream of the raw
logit, matching how a cross-encoder ``ClassifierPooler`` keeps the un-activated
logit here; the runner / serving layer applies ``use_activation`` per request.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.demo.generator_vllm import _crop_hidden_state_ttnn

# vLLM's Pooler ABC is only importable where vLLM is installed. The pure-ttnn
# scoring helpers below (and the reranker's device-free unit tests) must import
# without vLLM, so fall back to a plain base when it is unavailable; the runtime
# pooler path always has vLLM present.
try:
    from vllm.model_executor.layers.pooler.abstract import Pooler as _PoolerBase

    _HAS_VLLM_POOLER = True
except Exception:  # pragma: no cover - exercised only without vLLM installed
    _PoolerBase = object
    _HAS_VLLM_POOLER = False


def extract_cls_hidden(
    hidden_states: ttnn.Tensor,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> ttnn.Tensor:
    """Extract the ``<s>`` (CLS, position 0) hidden row on device.

    ``hidden_states`` is the encoder's ttnn last-hidden-state, rank 3
    (``[B, S, D]``) or rank 4 (``[B, 1, S, D]``). When ``batch_size`` /
    ``seq_len`` are given (the padded-chunk path) the tensor is first cropped to
    the real rows/positions; otherwise it is used as-is. Returns a ``[B, D]``
    ttnn tensor still on device.
    """
    tt_hidden = hidden_states
    if batch_size is not None and seq_len is not None:
        tt_hidden = _crop_hidden_state_ttnn(tt_hidden, batch_size, seq_len)
    if len(tt_hidden.shape) == 3:
        tt_hidden = ttnn.unsqueeze(tt_hidden, dim=1)  # [B,S,D] -> [B,1,S,D]
    tt_hidden = ttnn.to_memory_config(tt_hidden, ttnn.DRAM_MEMORY_CONFIG)
    b, _, _, d = tt_hidden.shape
    # CLS = position 0 along the sequence axis. [B,1,S,D] -> [B,1,1,D] -> [B,D].
    cls_tt = ttnn.slice(tt_hidden, [0, 0, 0, 0], [b, 1, 1, d])
    cls_tt = ttnn.squeeze(cls_tt, dim=1)
    cls_tt = ttnn.squeeze(cls_tt, dim=1)
    return cls_tt


def score_cls_on_device(
    hidden_states: ttnn.Tensor,
    classifier,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> ttnn.Tensor:
    """CLS extraction + device classification head, returning ``[B, 1]`` logits.

    Both stages run on device (CLS via ``ttnn.slice``, the head via
    :class:`XLMRobertaClassificationHeadTT`), so the full hidden state never
    leaves the device -- only the caller moves the small ``[B, 1]`` logit to
    host.
    """
    cls_tt = extract_cls_hidden(hidden_states, batch_size, seq_len)
    return classifier(cls_tt)  # [B, 1] logits on device


class RerankerClassifierPooler(_PoolerBase):
    """TT-native cross-encoder ClassifierPooler (CLS extract + device head).

    Placed on the model as ``model.pooler`` so the pooling runner delegates
    scoring to it exactly like an upstream vLLM pooler: it advertises the
    ``classify`` / ``score`` tasks and turns encoder hidden states into one raw
    relevance logit per request, all on device.
    """

    def __init__(self, classifier, device):
        if _HAS_VLLM_POOLER:
            super().__init__()
        self.classifier = classifier
        self.device = device

    def get_supported_tasks(self):
        # Cross-encoder scoring: /score and /rerank both route through the
        # classify/score pooling tasks (an embed Pooler would report "embed").
        return {"classify", "score"}

    def forward(self, hidden_states: ttnn.Tensor, pooling_metadata):
        """Score a batch of encoder hidden states into per-request logits.

        ``hidden_states`` is the model's device (ttnn) last-hidden-state for the
        batch; ``pooling_metadata`` carries one ``pooling_params`` entry per
        request. Scoring runs on device and only the final ``[B, 1]`` logit is
        moved to host, returned as a per-request list (a valid
        ``PoolerOutput``).
        """
        logits_tt = score_cls_on_device(hidden_states, self.classifier)
        logits = to_torch_auto_compose(logits_tt, device=self.device).to(torch.float32)
        logits = logits.reshape(-1, 1)
        num_reqs = len(pooling_metadata.pooling_params)
        return [logits[i] for i in range(num_reqs)]
