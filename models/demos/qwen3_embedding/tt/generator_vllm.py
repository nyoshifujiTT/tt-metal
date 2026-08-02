# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared vLLM pooling adapter for the Qwen3-Embedding family.

The upstream embedding wrapper (PR #35941) lives at
``models/demos/wormhole/qwen3_embedding_8b/demo/generator_vllm.py``. It targets
plain vLLM. This adapter subclasses it and adds only what the TT vLLM plugin's
pooling path needs, for both 0.6B and 8B:

  Plugin pooling contract. vLLM's ``is_vllm_model`` check
     (vllm/model_executor/models/interfaces_base.py) needs ``embed_input_ids`` to
     exist and ``forward`` to accept ``input_ids`` and ``positions`` kwargs, and the
     pooling task is only enumerated when ``is_pooling_model`` is truthy. The pooling
     runner (TTModelRunnerPooling) calls ``forward(input_ids=..., attention_mask=...)``
     and never passes ``positions``; it only needs to be an accepted keyword for the
     signature check, so ``forward`` just delegates to the base implementation (all
     embedding numerics stay in the base wrapper, which drives the real prefill length
     from ``prompt_lens`` via ``get_padded_prefill_len``).

Model resolution (offline / 0.6B vs 8B) and the KV-cache context limit are handled
outside this adapter: the base wrapper now uses the caller-provided ``hf_config``
as-is, and the on-device context cap lives in the tt-inference-server model spec
(``max_context`` -> ``max_model_len``). So this adapter carries only the pooling
contract and nothing size- or environment-specific.
"""

from typing import Optional

import torch

from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import (
    Qwen3ForEmbedding,
)


class Qwen3EmbeddingForTTvLLM(Qwen3ForEmbedding):
    """Qwen3-Embedding wrapper wired for the TT vLLM plugin's pooling path
    (0.6B / 8B, WH / BH). Adds only the plugin pooling contract; all numerics and
    model/config resolution stay in the base wrapper."""

    # is_pooling_model(model) reads this via getattr; inspect_model_cls
    # enumerates the "embed" task only when it is truthy.
    is_pooling_model = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Give the model a real vLLM Pooler so the upstream-conforming pooling
        # runner can delegate to ``model.pooler(hidden_states, pooling_metadata)``
        # exactly as it does for every other vLLM pooling model
        # (``VllmModelForPooling.pooler`` is a required, non-Optional member).
        # The base wrapper leaves ``self.pooler = None`` because the *legacy* TT
        # runner did the last-token slice + L2 normalize itself; the standard
        # runner instead hands the flat per-token hidden to ``model.pooler``,
        # which resolves LAST pooling + normalize from the served ``PoolerConfig``.
        # All pooling directives therefore live in the Pooler layer, not here.
        self.pooler = self._build_embed_pooler()

    def _build_embed_pooler(self):
        """Standard vLLM embed Pooler from the served config, or None off-vLLM.

        Imported lazily so this adapter still imports in a plain-vLLM / metal-only
        environment without ``vllm.model_executor.layers.pooler``; there the
        pooler stays None (only the plugin pooling runner needs it).
        """
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return None
        try:
            from vllm.model_executor.layers.pooler import Pooler
        except Exception:
            return None
        return Pooler.for_embed(vllm_config.model_config.pooler_config)

    # vLLM is_vllm_model introspection hook. The real embedding is produced on
    # the prefill path inside forward(); this only needs to exist.
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # positions is accepted only so vLLM's _check_vllm_model_forward
        # signature check passes; the pooling runner does not pass it and the base
        # forward does not use it.
        #
        # Return the FLAT per-token hidden ``[total_tokens, hidden]`` (not the
        # pooled last-token vector): the upstream-conforming pooling runner hands
        # this whole tensor to ``model.pooler``, whose LastPool selects each
        # request's final token and whose EmbeddingPoolerHead L2-normalizes. All
        # embedding numerics stay in the base implementation; we only switch it to
        # the flat layout the standard Pooler contract requires.
        return super().forward(input_ids, return_full_hidden_states=True, **kwargs)
