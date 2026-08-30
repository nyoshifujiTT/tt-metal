# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A vLLM Pooler for Qwen3-Embedding that pools on the Tenstorrent device.

vLLM splits an embedding model in two: the model produces per-token hidden
states, and a ``Pooler`` turns them into the served vector -- selecting the
token the pooling type calls for and applying whatever the resolved
``PoolerConfig`` asks for. vLLM's own Pooler is a torch module, so on a TT
backend that split would send every token's hidden state to the host and then
reduce it there, discarding all but one row per request after paying to move
them all.

This Pooler keeps that same split but performs it with ttnn ops, so only the
finished vector crosses to the host. The reduction is the model's own: the last
token is selected and normalized by the very methods the standalone demo runs
inside its prefill trace, so the served numbers are the ones the device path
already produces rather than a second implementation of them.

What stays on the host is the boundary vLLM defines: ``PoolerOutput`` is a torch
tensor, so the finished vector is converted at the end.
"""

from typing import Optional

import torch


class Qwen3EmbeddingDevicePooler:
    """Pools Qwen3-Embedding hidden states on device.

    Implements the calling convention vLLM's pooling runner expects --
    ``pooler(hidden_states=..., pooling_metadata=...)`` plus
    ``get_supported_tasks()`` -- while doing the work in ttnn.

    ``hidden_states`` is whatever the model's forward returned for this batch.
    A device (ttnn) tensor is pooled on device; a torch tensor is pooled with
    torch, so the same Pooler serves a run whose model handed back host tensors
    (the trace-free paths do) without the caller having to know which it got.
    """

    def __init__(self, owner, pooler_config=None):
        # The wrapper, not the transformer it builds: the wrapper constructs the
        # model lazily on its first forward, which is after the Pooler is built
        # (that happens during model construction, the only window in which
        # vLLM's current-config context is set). Holding the wrapper lets the
        # device ops be looked up when they are actually used.
        self._owner = owner
        self._pooler_config = pooler_config

    @property
    def _model(self):
        """The transformer that owns the device ops, once the wrapper has built it."""
        model = getattr(self._owner, "model", None)
        if model is None:
            raise RuntimeError(
                "Qwen3EmbeddingDevicePooler was asked to pool before the model "
                "was built; the wrapper builds it on its first forward, which "
                "the pooling runner always calls first."
            )
        # The wrapper keeps a list of per-device models; pooling ops live on the
        # first, which is where the hidden states come from.
        return model[0] if isinstance(model, (list, tuple)) else model

    def get_supported_tasks(self):
        return {"embed"}

    @property
    def _normalize(self) -> bool:
        """Whether to L2-normalize, per the resolved PoolerConfig.

        vLLM documents ``PoolerConfig.normalize`` as defaulting to True and
        leaves it None when nothing overrode it, so None means normalize. Only
        an explicit False turns it off -- which is a request a client can make
        per call, and which Qwen3-Embedding's own definition would not.
        """
        configured = getattr(self._pooler_config, "normalize", None)
        return configured is not False

    def _last_token_indices(self, hidden_states, pooling_metadata):
        """The flat-token index of each request's final token.

        The runner's pooling cursor carries these already; fall back to the
        prompt lengths, whose running total ends one past each request's last
        token, when the cursor is absent.
        """
        cursor = getattr(pooling_metadata, "pooling_cursor", None)
        if cursor is not None:
            for name in ("last_token_indices_gpu", "last_token_indices"):
                indices = getattr(cursor, name, None)
                if indices is not None:
                    return [int(i) for i in indices]

        prompt_lens = pooling_metadata.prompt_lens
        ends = torch.cumsum(torch.as_tensor(prompt_lens, dtype=torch.int64), dim=0)
        return [int(end) - 1 for end in ends]

    def __call__(self, hidden_states, pooling_metadata):
        indices = self._last_token_indices(hidden_states, pooling_metadata)

        if isinstance(hidden_states, torch.Tensor):
            pooled = hidden_states[indices, :]
            if self._normalize:
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
            return [row for row in pooled]

        return [self._pool_one_on_device(hidden_states, index) for index in indices]

    def _pool_one_on_device(self, hidden_states, index: int) -> torch.Tensor:
        """Select and normalize one request's vector without leaving the device."""
        # Resolved first: it reports a model that has not been built yet, which
        # is a clearer failure than whatever the device ops would make of it.
        model = self._model
        ops = self._device_ops()

        row = ops.slice(
            hidden_states,
            (0, 0, index, 0),
            (1, 1, index + 1, hidden_states.shape[-1]),
        )
        if self._normalize:
            row = model.l2_normalize_hidden(row)
        host = ops.to_torch(ops.get_device_tensors(row)[0]).float()
        return host.reshape(-1)[: self._embedding_dim(model, host)]

    def _device_ops(self):
        """The ttnn ops this Pooler uses, looked up as one object.

        Indirected through an attribute so a test can substitute the three
        functions: ttnn's own are native bindings and cannot be reassigned on
        the module.
        """
        import ttnn

        return ttnn

    def _embedding_dim(self, model, host: torch.Tensor) -> Optional[int]:
        """The checkpoint's hidden size, if the model knows it.

        Output replicated across devices is wider than the model's hidden size;
        trimming to it keeps the served vector the checkpoint's own width, the
        same de-duplication the host-composition helpers do.
        """
        args = getattr(model, "args", None)
        dim = getattr(args, "dim", None)
        return dim if dim is not None else host.numel()
