# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the reranker vLLM generator wiring (no device required).

Checks the class-level declarations vLLM relies on to route bge-reranker-v2-m3
through the cross-encoding score/rerank path, and that forward exposes the
vLLM-required keyword arguments.
"""

import inspect

from models.demos.wormhole.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3


def test_cross_encoder_class_flags():
    # vLLM is_pooling_model() requires the attribute AND is_vllm_model(); the
    # cross-encoder routing additionally needs supports_cross_encoding.
    assert BgeRerankerV2M3.is_pooling_model is True
    assert BgeRerankerV2M3.supports_cross_encoding is True
    assert BgeRerankerV2M3.default_pooling_type == "CLS"


def test_forward_exposes_vllm_kwargs():
    # is_vllm_model() checks forward accepts input_ids and positions.
    params = inspect.signature(BgeRerankerV2M3.forward).parameters
    assert "input_ids" in params
    assert "positions" in params


def test_vllm_interface_methods_present():
    for name in ("embed_input_ids", "initialize_vllm_model", "get_embedding_dim"):
        assert hasattr(BgeRerankerV2M3, name)
