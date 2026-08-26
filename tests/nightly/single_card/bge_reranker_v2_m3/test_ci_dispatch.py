# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# Runs the bge-reranker-v2-m3 single-device fast-dispatch tests in CI (Blackhole
# nightly-bh-models job). This entry is device-only: the end-to-end logit-vs-HF
# test plus the two component device tests it cannot localise a failure to (the
# fp32 classification head against its host reference, and the pooler's on-device
# CLS extraction). The device-free unit tests are registered separately in
# tests/pipeline_reorg/models_cpu_only_unit_tests.yaml, which runs on a CPU
# runner (no Tenstorrent device) so it does not spend scarce Blackhole time.
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
            # single-device end-to-end logit vs HF reference
            "models/demos/bge_reranker_v2_m3/tests/test_model.py",
            # device classification head vs host fp32 reference
            "models/demos/bge_reranker_v2_m3/tests/test_xlm_roberta_classification_head_tt.py",
            # device CLS extraction + head, the path the pooler drives
            "models/demos/bge_reranker_v2_m3/tests/test_reranker_pooler.py",
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
