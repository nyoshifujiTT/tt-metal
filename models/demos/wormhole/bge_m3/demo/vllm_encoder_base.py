# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM wrapper base for XLM-RoBERTa encoder models.

Both the bge-m3 embedding wrapper and the bge-reranker-v2-m3 cross-encoder run
the same XLM-RoBERTa encoder backbone (``create_tt_model``) and expose the same
vLLM plumbing: device/vllm_config resolution, ``initialize_vllm_model``, lazy
model construction, request validation and the ``get_max_*`` / pooler helpers.

That plumbing lives here so each model only implements what actually differs:
its ``forward`` (pooling vs classification), ``get_embedding_dim`` and any extra
per-model state. Subclasses customise weight loading and post-init via the
``_load_state_dict`` / ``_post_initialize`` hooks.

Placed in the bge-m3 package because bge-m3 and the reranker are the only two
users today (see the reranker classifier-head note); promote to a neutral
location if a third XLM-RoBERTa encoder model appears.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model


class XlmRobertaEncoderVllmModel:
    """Common vLLM plumbing for TT XLM-RoBERTa encoder models (single device)."""

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "",
        vllm_config=None,
        prefix: str = "",
        tt_data_parallel: int = 1,
        **kwargs,
    ):
        del prefix, kwargs

        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device
        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # Accepted for API compatibility; execution stays single-device.
        self.tt_data_parallel = tt_data_parallel
        self.dtype = dtype
        self.model_name = model_name

        if vllm_config is not None:
            self.vllm_config = vllm_config

        self.pooler = None
        self._is_initialized = False
        self.model_args = None
        self.model = None
        self.state_dict = None
        self.tokenizer = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = 8192,
        model_location_generator=None,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
        vllm_config=None,
        dtype=ttnn.bfloat16,
        **kwargs,
    ):
        if optimizations is not None:
            raise ValueError(f"Optimizations are not supported for {cls.__name__}")

        if vllm_config is not None:
            if (
                not hasattr(vllm_config.model_config, "override_tt_config")
                or vllm_config.model_config.override_tt_config is None
            ):
                vllm_config.model_config.override_tt_config = {}
            vllm_config.model_config.override_tt_config["is_embedding_model"] = True

            return cls(
                device=mesh_device,
                model_location_generator=model_location_generator,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
                tt_data_parallel=tt_data_parallel,
                dtype=dtype,
                **kwargs,
            )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            dtype=dtype,
            **kwargs,
        )

    # ---- hooks for subclasses ----
    def _load_state_dict(self):
        """Return a state_dict to hand to create_tt_model, or None to let the
        backbone load its own weights. Overridden by models that build the
        state_dict themselves (e.g. the reranker seq-classification loader)."""
        return None

    def _post_initialize(self) -> None:
        """Hook run after the encoder is built (e.g. to build a classifier head)."""

    def _initialize_model(self) -> None:
        if self._is_initialized and self.model is not None:
            return

        if self.state_dict is None:
            self.state_dict = self._load_state_dict()

        self.model_args, self.model, self.state_dict = create_tt_model(
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            state_dict=self.state_dict,
            hf_model_name=self.model_name,
        )
        self.tokenizer = self.model_args.tokenizer
        self._post_initialize()
        self._is_initialized = True

    def _validate_request(self, batch_size: int, padded_seq_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if padded_seq_len > self.max_seq_len:
            raise ValueError(f"Padded sequence length {padded_seq_len} exceeds max_seq_len {self.max_seq_len}")

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def _init_pooler(self, vllm_config, prefix: str = "") -> None:
        del vllm_config, prefix
        self.pooler = None
