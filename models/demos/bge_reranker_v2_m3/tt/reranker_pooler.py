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

``ClassifierPoolerHead`` does more than run the classifier, and all of it is
part of the API contract, so this class reproduces the same stages in the same
order: the classifier, then the ``logit_mean`` / ``logit_sigma`` affine
calibration, then the activation gated by each request's ``use_activation``.

The activation is not optional decoration. vLLM resolves it from the HF config
(``resolve_classifier_act_fn`` -> ``get_act_fn`` -> ``PoolerClassify``), and for
a single-label cross-encoder like this one that is ``sigmoid``, so upstream's
``/score`` answers in 0..1. Nothing downstream of the pooler applies it: the
entrypoints only forward ``use_activation`` into ``PoolingParams``. Leaving it
out therefore does not "let vLLM do it later", it changes the API's answer.
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

# Upstream resolves the classification activation from the HF config rather than
# hardcoding one, so call the same resolver instead of restating its rules (a
# copy would drift, and it already handles problem_type and the sentence-
# transformers override). Optional for the same reason as the Pooler ABC above.
try:
    from vllm.model_executor.layers.pooler.activations import resolve_classifier_act_fn

    _HAS_VLLM_ACT_FN = True
except Exception:  # pragma: no cover - exercised only without vLLM installed
    resolve_classifier_act_fn = None
    _HAS_VLLM_ACT_FN = False

try:
    from vllm.config import get_current_vllm_config_or_none

    _HAS_VLLM_CURRENT_CONFIG = True
except Exception:  # pragma: no cover - exercised only without vLLM installed
    get_current_vllm_config_or_none = None
    _HAS_VLLM_CURRENT_CONFIG = False


def _activation_module():
    """vLLM's pooler activations module, imported on use.

    Kept out of the module-level import so this file still imports without vLLM
    (the device-free tests rely on that), while the runtime pooler path, which
    always has vLLM, can compare against the real activation classes.
    """
    from vllm.model_executor.layers.pooler import activations

    return activations


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
    # ttnn.embedding requires a BFLOAT16 weight table; the encoder hidden is
    # already bf16 in production, but cast defensively so an fp32 flat (e.g. a
    # test feeding fp32 rows) also works. The row values are unchanged (widening
    # is exact; a bf16 input is a no-op).
    flat_rm = ttnn.to_layout(flat_hidden, ttnn.ROW_MAJOR_LAYOUT)
    if flat_rm.dtype != ttnn.bfloat16:
        flat_rm = ttnn.typecast(flat_rm, ttnn.bfloat16)
    gathered = ttnn.embedding(idx_tt, flat_rm)  # [1, num_reqs, D]
    gathered = ttnn.reshape(gathered, (first_token_indices.numel(), d))
    # The device head runs matmuls, which need TILE layout; ttnn.embedding emits
    # ROW_MAJOR, so convert before returning.
    return ttnn.to_layout(gathered, ttnn.TILE_LAYOUT)


def _first_token_indices_cpu(pooling_metadata, num_reqs: int) -> torch.Tensor:
    """Per-request first-token index into the flat token axis (CLSPool selector).

    Prefers the pooling cursor's ``first_token_indices`` (the standard
    ``CLSPool`` selector built by the runner); returns it as a CPU int tensor for
    the device gather. Falls back to computing prefix sums from ``prompt_lens``
    when no cursor is present (e.g. a stubbed metadata in a unit test), which is
    the same quantity the cursor holds for single-shot prefill pooling.
    """
    cursor = getattr(pooling_metadata, "pooling_cursor", None)
    if cursor is not None and getattr(cursor, "first_token_indices_gpu", None) is not None:
        return cursor.first_token_indices_gpu.detach().to("cpu").to(torch.int32)
    prompt_lens = pooling_metadata.prompt_lens
    if not torch.is_tensor(prompt_lens):
        prompt_lens = torch.tensor(list(prompt_lens), dtype=torch.int64)
    starts = torch.zeros(num_reqs, dtype=torch.int32)
    if num_reqs > 1:
        starts[1:] = torch.cumsum(prompt_lens.to(torch.int64), 0)[:-1].to(torch.int32)
    return starts


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
        self._logit_mean = None
        self._logit_sigma = None
        # Resolve the head's configuration now, not at scoring time. vLLM's
        # ambient config is only set for the duration of the load, which is when
        # the model (and so this pooler) is constructed; reading it later would
        # find nothing. This is the same reason upstream builds its head in
        # pooler_for_classify() during model construction.
        model_config = self._model_config()
        self._act_fn = self._resolve_activation(model_config)
        pooler_config = getattr(model_config, "pooler_config", None)
        if pooler_config is not None:
            self._logit_mean = getattr(pooler_config, "logit_mean", None)
            self._logit_sigma = getattr(pooler_config, "logit_sigma", None)

    def _model_config(self):
        """The vLLM ModelConfig, or None outside the vLLM serving path.

        Prefers the config the model was built with, and falls back to vLLM's
        ambient one. The fallback is needed because ``initialize_vllm_model`` is
        called without ``vllm_config`` by the plugin's loader (most TT models do
        not accept the argument, so the loader cannot pass it), yet the load runs
        inside a ``set_current_vllm_config`` context. This is how upstream reads
        it too -- ``pooler_for_classify`` calls ``get_current_vllm_config()``
        rather than taking the config as an argument.

        Returns None on the demo path, where there is no vLLM at all. The pooler
        is constructed unconditionally there, so construction must tolerate that;
        only the stages that genuinely need the config fail, and only if reached.
        """
        model_config = getattr(getattr(self._model, "vllm_config", None), "model_config", None)
        if model_config is not None:
            return model_config
        if not _HAS_VLLM_CURRENT_CONFIG:
            return None
        return getattr(get_current_vllm_config_or_none(), "model_config", None)

    @staticmethod
    def _resolve_activation(model_config):
        """Upstream's classification activation for this model, or None off-vLLM.

        Delegates to vLLM's own ``resolve_classifier_act_fn`` so the rules
        (``problem_type``, the sentence-transformers override, single-label
        sigmoid vs multi-label softmax) come from one place instead of being
        restated here, where they would drift.
        """
        if model_config is None or not _HAS_VLLM_ACT_FN:
            return None
        return resolve_classifier_act_fn(model_config, static_num_labels=True)

    def _activation(self):
        if self._act_fn is None:
            raise RuntimeError(
                "the classification activation could not be resolved (no vLLM "
                "model config was available when the pooler was built); the "
                "pooler is only driven from the vLLM serving path"
            )
        return self._act_fn

    def get_supported_tasks(self):
        # "classify" is the cross-encoder pooling task: vLLM maps it to the
        # "cross-encoder" score type (tasks.py::SCORE_TYPE_MAP), which is what
        # routes /score and /rerank here. An embed Pooler would report "embed".
        #
        # Only names in vllm.tasks.PoolingTask are valid. "score" is not one of
        # them -- it is the endpoint's name, not a pooling task -- so reporting
        # it advertised a task that cannot be selected. It happened to be
        # harmless because get_pooling_task() picks from a fixed priority list
        # and ignores anything else, but it would surface verbatim in the
        # supported-task list, and asking for it explicitly (pooler_config.task)
        # would fail the membership check with a confusing error.
        return {"classify"}

    def forward(self, hidden_states: ttnn.Tensor, pooling_metadata):
        """Score a flat ``[total_tokens, hidden]`` batch into per-request logits.

        Upstream-standard cross-encoder pooling (``CLSPool`` + classifier head),
        done device-native:

        - ``hidden_states`` is the model's flat, unpadded ``[total_tokens, D]``
          device (ttnn) hidden -- every scheduled request's real tokens
          concatenated in request order (``model.forward`` returns this).
        - CLS selection uses ``pooling_metadata``'s cursor
          ``first_token_indices`` (the standard ``CLSPool`` selector), gathered
          on device.
        - the fp32 device head maps the ``[num_reqs, D]`` CLS rows to
          ``[num_reqs, 1]`` logits.
        - the head's remaining stages follow, in upstream's order: the
          ``logit_mean`` / ``logit_sigma`` affine calibration, then the
          activation, applied per request according to its ``use_activation``.

        Scoring stays on device -- the calibration and the activation are ttnn
        ops too, so only the final ``[num_reqs, 1]`` score crosses to host,
        returned as a per-request list (a valid ``PoolerOutput``).
        """
        classifier = self._model.classifier
        device = self._model.device
        pooling_params = pooling_metadata.pooling_params
        num_reqs = len(pooling_params)

        first_indices = _first_token_indices_cpu(pooling_metadata, num_reqs)
        cls_tt = gather_cls_from_flat(hidden_states, first_indices, device)
        logits_tt = classifier(cls_tt)  # [num_reqs, 1] on device
        logits_tt = self._calibrate_on_device(logits_tt)
        logits_tt = self._activate_on_device(logits_tt, pooling_params)
        logits = to_torch_auto_compose(logits_tt, device=device).to(torch.float32).reshape(-1, 1)
        return [logits[i] for i in range(num_reqs)]

    def _calibrate_on_device(self, logits_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Upstream's affine score calibration, on device.

        ``(logit - logit_mean) / logit_sigma``, each step applied only when its
        value is configured, matching ``ClassifierPoolerHead``. Both are unset
        for this checkpoint, so this is normally a no-op; it exists because the
        pooler config can carry them and dropping them would silently change the
        score for a checkpoint that does.
        """
        if self._logit_mean is not None:
            logits_tt = ttnn.subtract(logits_tt, self._logit_mean)
        if self._logit_sigma is not None:
            logits_tt = ttnn.multiply(logits_tt, 1.0 / self._logit_sigma)
        return logits_tt

    def _activate_on_device(self, logits_tt: ttnn.Tensor, pooling_params) -> ttnn.Tensor:
        """Apply the classification activation, gated per request, on device.

        For this single-label cross-encoder upstream resolves the activation to
        sigmoid, so ``/score`` answers in 0..1; ``ttnn.sigmoid`` keeps that on
        device. Each request may opt out via ``use_activation``, exactly as in
        ``ClassifierPoolerHead``.

        A batch that mixes the flag cannot be served by one device op. Upstream
        handles it per row; rather than fall back to host arithmetic, this
        rejects the mixed batch, because silently returning activated scores for
        a request that asked for raw logits (or the reverse) is worse than a
        clear error. Requests are independent, so a client can split them.
        """
        flags = {bool(getattr(param, "use_activation", None) is not False) for param in pooling_params}
        if len(flags) > 1:
            raise NotImplementedError(
                "use_activation must be the same for every request in a batch on TT: "
                "the activation is a single device op over the batch. Split the "
                "requests to mix activated and raw scores."
            )
        if not flags or not flags.pop():
            return logits_tt
        self._assert_activation_is_sigmoid(self._activation(), int(logits_tt.shape[-1]))
        return ttnn.sigmoid(logits_tt)

    @staticmethod
    def _assert_activation_is_sigmoid(activation, num_labels: int) -> None:
        """Refuse to substitute ttnn.sigmoid for something else.

        Only sigmoid has a device equivalent here, and upstream's resolver can
        legitimately return other activations: ``PoolerIdentity`` for a
        regression head, ``PoolerMultiLabelClassify``, a sentence-transformers
        override, or ``PoolerClassify`` in its softmax branch when the head has
        two or more labels. Running sigmoid in any of those cases would return a
        wrong score silently, so check rather than assume. This model is
        single-label, so the check passes for it.
        """
        classify_cls = getattr(_activation_module(), "PoolerClassify", None)
        if classify_cls is None or not isinstance(activation, classify_cls):
            raise NotImplementedError(
                f"only the sigmoid classification activation has a device "
                f"implementation; vLLM resolved {type(activation).__name__}"
            )
        resolved_labels = activation.num_labels
        if resolved_labels is None:
            resolved_labels = num_labels
        if resolved_labels >= 2:
            raise NotImplementedError(
                "PoolerClassify uses softmax for a multi-label head "
                f"(num_labels={resolved_labels}), which has no device "
                "implementation here"
            )
