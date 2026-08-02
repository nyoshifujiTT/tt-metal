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
        # ``pooler`` is exposed as a lazily-built property (see below). The base
        # wrapper sets ``self.pooler = None`` in its __init__; we shadow that with a
        # property so the real embed Pooler is constructed on first access, when
        # vLLM's config context is reliably installed (the TT model loader builds
        # the model OUTSIDE ``set_current_vllm_config`` scope, so building it here
        # in __init__ would miss the config).
        self._embed_pooler = None
        self._embed_pooler_built = False

    @property
    def pooler(self):
        """The model's vLLM Pooler (built lazily, cached).

        The upstream-conforming pooling runner delegates to
        ``model.pooler(hidden_states, pooling_metadata)`` and reads
        ``model.pooler.get_supported_tasks()``; both run inside vLLM's engine
        where the config context is set, so we build the standard embed Pooler on
        first access rather than at construction time.
        """
        if not self._embed_pooler_built:
            self._embed_pooler = self._build_embed_pooler()
            self._embed_pooler_built = True
        return self._embed_pooler

    @pooler.setter
    def pooler(self, value):
        # The base wrapper assigns ``self.pooler = None`` in its __init__; absorb
        # that (and any explicit override) into the cached slot.
        self._embed_pooler = value
        self._embed_pooler_built = value is not None

    def _build_embed_pooler(self):
        """Standard vLLM embed Pooler from the served config, or None off-vLLM.

        Imported lazily so this adapter still imports in a plain-vLLM / metal-only
        environment without ``vllm.model_executor.layers.pooler``; there the
        pooler stays None (only the plugin pooling runner needs it).
        """
        # Prefer the vllm_config handed to the wrapper; fall back to vLLM's
        # current-config context. The TT model loader calls initialize_vllm_model
        # WITHOUT forwarding vllm_config, so self.vllm_config is usually unset even
        # under vLLM -- but the engine has already installed the config in context
        # (this is exactly how the standard Pooler heads read pooler_config), so
        # get_current_vllm_config() resolves it. Only a true metal-only run (no
        # vLLM in the process) leaves the pooler None.
        try:
            from vllm.model_executor.layers.pooler import Pooler
        except Exception:
            import loguru

            loguru.logger.warning("Qwen3EmbeddingForTTvLLM: vLLM pooler import failed")
            return None
        # Prefer the stock factory when the running vLLM exposes it.
        for_embed = getattr(Pooler, "for_embed", None)
        if for_embed is not None:
            try:
                from vllm.config import get_current_vllm_config

                return for_embed(self._resolve_pooler_config(get_current_vllm_config))
            except Exception as exc:
                import loguru

                loguru.logger.warning(
                    f"Qwen3EmbeddingForTTvLLM: Pooler.for_embed failed ({exc!r}); "
                    "falling back to a directly-built embed SimplePooler"
                )
        # vLLM builds whose Pooler factory / component layout has drifted (the
        # layers.pooler package has been reshuffled across releases) fall back to a
        # small self-contained embed Pooler that depends only on the stable Pooler
        # base class and PoolingMetadata: LAST-token selection over the flat
        # [total_tokens, hidden] tensor + L2 normalization -- exactly the embed
        # contract, matching Pooler.for_embed's numerics.
        return _TTEmbedPooler(Pooler)

    @staticmethod
    def _resolve_pooler_config(get_current_vllm_config):
        """Return the served ``PoolerConfig`` (or the embed default LAST+normalize).

        Prefer the engine's config; the TT model loader may build the model
        outside ``set_current_vllm_config`` scope, so if the context is not
        installed yet fall back to the documented embed default (``LAST`` pooling
        with normalization), which is exactly what vLLM resolves for an embed
        model that does not override the pooler.
        """
        from vllm.config import PoolerConfig

        try:
            vllm_config = get_current_vllm_config()
        except Exception:
            vllm_config = None
        if vllm_config is not None and getattr(vllm_config, "model_config", None) is not None:
            cfg = vllm_config.model_config.pooler_config
            if cfg is not None:
                return cfg
        # No engine config in context (the TT loader builds the model outside
        # ``set_current_vllm_config``). Use PoolerConfig defaults, which resolve to
        # the embed default (LAST pooling + normalize) without hard-coding
        # version-specific field names.
        return PoolerConfig()

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


def _make_tt_embed_pooler_cls(PoolerBase):
    """Define a LAST + L2-normalize embed Pooler on the running vLLM's Pooler base."""

    import torch.nn.functional as F

    class _TTEmbedPoolerImpl(PoolerBase):
        """Minimal, version-robust embed Pooler (LAST pooling + L2 normalize).

        Used when the running vLLM's own embed-Pooler factory/components are not
        importable at the paths this adapter knows. Consumes the standard flat
        ``[total_tokens, hidden]`` hidden states and the ``PoolingMetadata`` the
        pooling runner builds, selects each request's last token via the pooling
        cursor (falling back to prompt-length arithmetic) and L2-normalizes -- the
        same result as the standard embed Pooler.
        """

        def get_supported_tasks(self):
            return {"embed"}

        def forward(self, hidden_states, pooling_metadata):
            cursor = getattr(pooling_metadata, "pooling_cursor", None)
            last_idx = None
            if cursor is not None:
                last_idx = getattr(cursor, "last_token_indices_gpu", None)
                if last_idx is None:
                    last_idx = getattr(cursor, "last_token_indices", None)
            if last_idx is None:
                # Derive last-token positions from prompt lengths: cumulative end
                # of each request minus one on the flat token axis.
                prompt_lens = pooling_metadata.prompt_lens
                ends = torch.cumsum(prompt_lens, dim=0)
                last_idx = (ends - 1).to(hidden_states.device)
            pooled = hidden_states[last_idx]
            return F.normalize(pooled, p=2, dim=-1)

    return _TTEmbedPoolerImpl


_TT_EMBED_POOLER_CLS = None


def _TTEmbedPooler(PoolerBase):
    """Instantiate the self-contained embed Pooler (class cached per process)."""
    global _TT_EMBED_POOLER_CLS
    if _TT_EMBED_POOLER_CLS is None:
        _TT_EMBED_POOLER_CLS = _make_tt_embed_pooler_cls(PoolerBase)
    return _TT_EMBED_POOLER_CLS()
