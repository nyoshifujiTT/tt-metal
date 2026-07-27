# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from loguru import logger

from models.tt_transformers.tt.common import get_hf_tt_cache_path


# Runs the bge-reranker-v2-m3 single-device fast-dispatch tests in CI (Blackhole
# nightly-bh-models job). This entry is device-only: the end-to-end logit-vs-HF
# test. The device-free unit tests live in the sibling ``bge_reranker_v2_m3_hostside``
# suite, which runs on a CPU runner (no Tenstorrent device) so it does not spend
# scarce Blackhole time -- see fast-dispatch-hostside-models.
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
        ]
        + ["-x"]  # Fail if one of the tests fails
    )
    if exit_code == pytest.ExitCode.TESTS_FAILED:
        pytest.fail(
            f"One or more CI dispatch tests failed for {model_weights}. Please check the log above for more info",
            pytrace=False,
        )
