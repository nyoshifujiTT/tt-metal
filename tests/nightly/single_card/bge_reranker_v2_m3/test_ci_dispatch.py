# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# Runs the bge-reranker-v2-m3 fast-dispatch test suite in CI. Includes the
# device-free unit tests (classifier head, weight loader, vLLM wiring, encoder
# padding/chunking) and the single-device end-to-end logit test.
@pytest.mark.parametrize(
    "model_weights",
    [
        "BAAI/bge-reranker-v2-m3",
    ],
    ids=[
        "bge_reranker_v2_m3",
    ],
)
def test_ci_dispatch(model_weights):
    logger.info(f"Running fast dispatch tests for {model_weights}")
    os.environ["HF_MODEL"] = model_weights
    os.environ["TT_CACHE_PATH"] = get_hf_tt_cache_path(model_weights)

    exit_code = pytest.main(
        [
            # device-free unit tests
            "models/demos/bge_reranker_v2_m3/tests/test_xlm_roberta_classification_head.py",
            "models/demos/bge_reranker_v2_m3/tests/test_model_config.py",
            "models/demos/bge_reranker_v2_m3/tests/test_generator_vllm.py",
            "models/demos/wormhole/bge_m3/tests/unit/test_encode_in_chunks.py",
            # single-device end-to-end logit vs HF reference
            "models/demos/bge_reranker_v2_m3/tests/test_model.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
