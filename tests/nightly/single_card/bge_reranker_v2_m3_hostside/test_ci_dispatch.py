# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger


# Host-side (device-free) fast-dispatch tests for bge-reranker-v2-m3.
#
# These exercise the classifier head, the seq-classification weight loader, the
# vLLM wiring / shared encoder base and the encoder pad/chunk contract. They
# import ttnn but never open a device, so they run on a CPU runner (no
# Tenstorrent hardware) -- kept separate from the single-device suite in
# ``bge_reranker_v2_m3`` so they do not consume Blackhole nightly time. Wired
# into CI by the fast-dispatch-hostside-models job.
def test_ci_dispatch():
    logger.info("Running host-side (device-free) fast dispatch tests for bge-reranker-v2-m3")

    exit_code = pytest.main(
        [
            "models/demos/bge_reranker_v2_m3/tests/test_xlm_roberta_classification_head.py",
            "models/demos/bge_reranker_v2_m3/tests/test_model_config.py",
            "models/demos/bge_reranker_v2_m3/tests/test_generator_vllm.py",
            "models/demos/wormhole/bge_m3/tests/unit/test_encode_in_chunks.py",
            "models/demos/wormhole/bge_m3/tests/unit/test_xlm_roberta_encoder.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            "One or more host-side CI dispatch tests failed for bge-reranker-v2-m3. "
            "Please check the log above for more info",
            pytrace=False,
        )
