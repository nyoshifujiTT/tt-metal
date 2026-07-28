# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared vLLM pooling adapter for the Qwen3-Embedding family.

The upstream embedding wrapper (PR #35941) lives at
``models/demos/wormhole/qwen3_embedding_8b/demo/generator_vllm.py``. It targets
plain vLLM. This adapter subclasses it and adds only what the tenstorrent vLLM
fork's pooling path needs, for both 0.6B and 8B:

  Fork pooling contract. vLLM's ``is_vllm_model`` check
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
    """Qwen3-Embedding wrapper wired for the tenstorrent vLLM fork's pooling path
    (0.6B / 8B, WH / BH). Adds only the fork pooling contract; all numerics and
    model/config resolution stay in the base wrapper."""

    # is_pooling_model(model) reads this via getattr; inspect_model_cls
    # enumerates the "embed" task only when it is truthy.
    is_pooling_model = True

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
        # forward does not use it. All numerics stay in the base implementation.
        return super().forward(input_ids, **kwargs)
