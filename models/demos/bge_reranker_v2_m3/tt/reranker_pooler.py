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


def flatten_request_hidden_to_device(
    per_request_hidden: list,
) -> ttnn.Tensor:
    """Concatenate per-request device hidden rows into a flat ``[total_tokens, D]``.

    ``per_request_hidden`` is a list of ttnn tensors, one per scheduled request,
    each already cropped to that request's real tokens and shaped ``[seq_i, D]``
    (no padding). They are concatenated on the token axis, on device, into the
    flat, unpadded ``[total_tokens, D]`` layout the upstream pooling contract
    expects (``model.forward`` returns this; the pooler's cursor indexes it).

    Keeping the flatten here (device ``ttnn.concat``) means only the final pooled
    result crosses to host, not the hidden state.
    """
    if len(per_request_hidden) == 1:
        return per_request_hidden[0]
    return ttnn.concat(per_request_hidden, dim=0)


def crop_request_rows(chunk_hidden: ttnn.Tensor, real_len: int, hidden_dim: int) -> ttnn.Tensor:
    """Crop one request's real tokens from its padded chunk hidden -> ``[real_len, D]``.

    ``chunk_hidden`` is the encoder ttnn output for a single-request chunk, rank
    3 (``[1, S, D]``) or rank 4 (``[1, 1, S, D]``). Slices the first ``real_len``
    sequence positions (dropping padding) and reshapes to ``[real_len, D]`` so it
    can be concatenated into the flat layout.
    """
    tt_hidden = chunk_hidden
    if len(tt_hidden.shape) == 3:
        tt_hidden = ttnn.unsqueeze(tt_hidden, dim=1)  # [1,S,D] -> [1,1,S,D]
    tt_hidden = ttnn.to_memory_config(tt_hidden, ttnn.DRAM_MEMORY_CONFIG)
    cropped = ttnn.slice(tt_hidden, [0, 0, 0, 0], [1, 1, real_len, hidden_dim])
    return ttnn.reshape(cropped, (real_len, hidden_dim))


def gather_cls_from_flat(
    flat_hidden: ttnn.Tensor,
    first_token_indices: torch.Tensor,
    device,
) -> ttnn.Tensor:
    """Gather each request's CLS row from a flat ``[total_tokens, D]`` on device.

    ``first_token_indices`` is the pooling cursor's per-request first-token index
    into the flat token axis (the standard ``CLSPool`` selector). The gather runs
    on device via ``ttnn.embedding`` (treating the flat hidden as the embedding
    table), returning ``[num_reqs, D]`` still on device. This is the device-native
    equivalent of upstream ``CLSPool``'s ``hidden_states[first_token_indices]``.
    """
    d = int(flat_hidden.shape[-1])
    idx_tt = ttnn.from_torch(
        first_token_indices.reshape(1, -1).to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    flat_rm = ttnn.to_layout(flat_hidden, ttnn.ROW_MAJOR_LAYOUT)
    gathered = ttnn.embedding(idx_tt, flat_rm)  # [1, num_reqs, D]
    return ttnn.reshape(gathered, (first_token_indices.numel(), d))


class RerankerChunkedHidden:
    """Device (ttnn) encoder hidden states carried from forward to the pooler.

    The reranker runs the encoder over device-sized chunks; the canonical path
    (``forward(return_full_hidden_states=True)``) returns the un-pooled hidden
    for the whole batch so the pooler can do CLS extraction + scoring. Rather
    than concatenate variable-length device tensors, this holds the per-chunk
    ttnn hidden plus each chunk's padded ``attention_mask`` (for cropping) and
    real ``chunk_batch_size`` (the number of real rows in the chunk). The pooler
    scores each chunk on device and concatenates the small per-request logits.

    It deliberately exposes no torch ``.device`` so the runner treats it as a
    device-native hidden (builds the pooling cursor on CPU; see the pooling
    runner's device tolerance).
    """

    def __init__(self, chunks: list[tuple[ttnn.Tensor, torch.Tensor, int]]):
        self.chunks = chunks


class RerankerClassifierPooler(_PoolerBase):
    """TT-native cross-encoder ClassifierPooler (CLS extract + device head).

    Placed on the model as ``model.pooler`` so the pooling runner delegates
    scoring to it exactly like an upstream vLLM pooler: it advertises the
    ``classify`` / ``score`` tasks and turns encoder hidden states into one raw
    relevance logit per request, all on device.

    The device classification head is built lazily by the model (in
    ``_post_initialize``, on first forward), whereas the runner queries
    ``get_supported_tasks`` right after load. So the pooler holds the model and
    reads ``model.classifier`` / ``model.device`` at scoring time rather than
    capturing them at construction: ``get_supported_tasks`` (static) works
    before the head exists, and ``forward`` sees the head once it is built.
    """

    def __init__(self, model):
        if _HAS_VLLM_POOLER:
            super().__init__()
        self._model = model

    def get_supported_tasks(self):
        # Cross-encoder scoring: /score and /rerank both route through the
        # classify/score pooling tasks (an embed Pooler would report "embed").
        return {"classify", "score"}

    def forward(self, hidden_states: ttnn.Tensor, pooling_metadata):
        """Score a batch of encoder hidden states into per-request logits.

        ``hidden_states`` is the model's device (ttnn) last-hidden-state for the
        batch: either a single ttnn tensor (one chunk) or a
        :class:`RerankerChunkedHidden` carrying the per-chunk device hidden.
        ``pooling_metadata`` carries one ``pooling_params`` entry per request.
        Scoring runs on device (CLS extract + head per chunk) and only the final
        ``[B, 1]`` logit is moved to host, returned as a per-request list (a
        valid ``PoolerOutput``).
        """
        classifier = self._model.classifier
        device = self._model.device
        if isinstance(hidden_states, RerankerChunkedHidden):
            per_chunk = []
            for output, attention_mask, chunk_batch_size in hidden_states.chunks:
                b, s = attention_mask.shape
                logits_tt = score_cls_on_device(output, classifier, b, s)
                chunk_logits = to_torch_auto_compose(
                    logits_tt, device=device
                ).to(torch.float32)
                per_chunk.append(chunk_logits.reshape(-1, 1)[:chunk_batch_size])
            logits = torch.cat(per_chunk, dim=0)
        else:
            logits_tt = score_cls_on_device(hidden_states, classifier)
            logits = (
                to_torch_auto_compose(logits_tt, device=device)
                .to(torch.float32)
                .reshape(-1, 1)
            )
        num_reqs = len(pooling_metadata.pooling_params)
        return [logits[i] for i in range(num_reqs)]
