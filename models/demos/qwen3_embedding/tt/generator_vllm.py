# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared vLLM pooling adapter for the Qwen3-Embedding family.

The upstream embedding wrapper (PR #35941) lives at
``models/demos/wormhole/qwen3_embedding_8b/demo/model.py``. It targets
plain vLLM. This adapter subclasses it and adds only what the TT vLLM plugin's
pooling path needs, for both 0.6B and 8B.

Pooling contract (identical to upstream vLLM's). The plugin's pooling runner
mirrors ``GPUModelRunner._pool``: it takes the flat per-token hidden states from
``forward`` and hands them to ``model.pooler`` together with a
``PoolingMetadata`` describing each request's token span. The Pooler owns every
pooling directive -- pooling type (LAST for Qwen3-Embedding), normalization,
activation -- exactly as it does on GPU. So this adapter must

  * return the flat ``[total_tokens, hidden]`` layout from ``forward``
    (``return_full_hidden_states``), not an already-pooled tensor, and
  * expose a ``pooler``.

Returning a pooled tensor instead would be silently wrong: the runner would treat
the ``[batch, hidden]`` result as if its first axis were the token axis.

``forward`` also accepts ``positions`` so vLLM's ``_check_vllm_model_forward``
signature check passes; the runner never passes it and the base forward does not
use it.

Model resolution (0.6B vs 8B) and the KV-cache context limit are handled outside
this adapter: the base wrapper uses the caller-provided ``hf_config`` as-is, and
the on-device context cap lives in the tt-inference-server model spec
(``max_context`` -> ``max_model_len``). So this adapter carries only the pooling
contract and nothing size- or environment-specific.
"""

from typing import Optional

import torch

from models.demos.wormhole.qwen3_embedding_8b.demo.model import (
    Qwen3ForEmbedding,
)


class Qwen3EmbeddingForTTvLLM(Qwen3ForEmbedding):
    """Qwen3-Embedding wrapper wired for the TT vLLM plugin's pooling path
    (0.6B / 8B, WH / BH). Adds only the pooling contract; all numerics and
    model/config resolution stay in the base wrapper."""

    # is_pooling_model(model) reads this via getattr; inspect_model_cls
    # enumerates the "embed" task only when it is truthy.
    is_pooling_model = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The Pooler is built here, during model construction, because that is
        # the only window in which vLLM guarantees the current-config context:
        # the loader wraps construction in set_current_vllm_config(), and the
        # pooling methods underneath DispatchPooler read it via
        # get_current_vllm_config() in their __init__. Building it lazily on
        # first access instead put construction inside get_supported_tasks(),
        # outside that context, and serving died with "Current vLLM config is
        # not set".
        #
        # Not every instantiation path has a vLLM config -- the metal-only demo
        # builds this class with a bare device plus model name, and never pools
        # through vLLM -- so it stays None there and only an actual pooler
        # access reports the problem.
        self._pooler = (
            self._build_pooler()
            if getattr(self, "vllm_config", None) is not None
            else None
        )

    def _build_pooler(self):
        from vllm.model_executor.layers.pooler import DispatchPooler

        return DispatchPooler.for_embedding(self._resolve_pooler_config())

    @property
    def pooler(self):
        """The vLLM Pooler that turns flat hidden states into embeddings.

        ``DispatchPooler.for_embedding`` builds the standard embedding pooler for
        the resolved ``PoolerConfig`` -- the configured pooling type plus the
        activation/normalization the task implies. It is the same constructor
        vLLM's own embedding adapters use (see
        ``vllm/model_executor/models/adapters.py``), so normalization stays where
        vLLM puts it on every other backend instead of being re-implemented in
        the runner or in this adapter.
        """
        if self._pooler is None:
            # Only reachable when the instance was built without a vLLM config;
            # _resolve_pooler_config explains that case and raises.
            self._pooler = self._build_pooler()
        return self._pooler

    @pooler.setter
    def pooler(self, value):
        # The base wrapper assigns ``self.pooler = None`` during __init__ to
        # satisfy older vLLM wrappers; absorb that without clobbering the property.
        self._pooler = value

    def _resolve_pooler_config(self):
        """The resolved ``PoolerConfig`` for this model, from the vLLM config.

        Taken as-is from ``vllm_config.model_config``: vLLM derives it from the
        checkpoint plus any ``--override-pooler-config``, and its field set has
        changed across releases, so reconstructing one here would both ignore the
        user's configuration and break whenever those fields move. A missing
        config is an error rather than a guess -- the serving path always has one,
        and the metal-only demo does not go through the Pooler at all.
        """
        vllm_config = getattr(self, "vllm_config", None)
        model_config = getattr(vllm_config, "model_config", None) if vllm_config else None
        pooler_config = getattr(model_config, "pooler_config", None) if model_config else None
        if pooler_config is None:
            raise RuntimeError(
                "Qwen3EmbeddingForTTvLLM.pooler needs vllm_config.model_config."
                "pooler_config, which vLLM populates for pooling models. It is "
                "absent here, so this instance was not built by the vLLM/"
                "tt-inference-server path; use the base wrapper's forward "
                "directly for metal-only runs."
            )
        return pooler_config

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # vLLM is_vllm_model introspection hook; the real embedding is produced
        # on the prefill path inside forward().
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward the pooling runner's call to the model's pre-pooling stage.

        The runner indexes the flat ``[total_tokens, hidden]`` token axis with a
        ``PoolingMetadata`` cursor before handing the tensor to ``pooler``, so
        this hands back exactly what
        :meth:`Qwen3ForEmbedding.encode_token_hidden_states` produces -- no
        keyword juggling of its own.

        It used to reach this layout by setting ``return_full_hidden_states`` on
        the base's ``forward`` with ``setdefault``, i.e. by flipping a flag on
        the shared entry point rather than by naming the stage it wants. Calling
        the named accessor states the requirement instead of configuring it, and
        leaves ``forward``'s flags to the base.

        Note that this layout is specific to the vLLM pooling contract: a caller
        without a Pooler must not come through this class. See
        :meth:`Qwen3ForEmbedding.encode` for that case.
        """
        return self.encode_token_hidden_states(input_ids, kwargs.pop("attention_mask", None), **kwargs)
