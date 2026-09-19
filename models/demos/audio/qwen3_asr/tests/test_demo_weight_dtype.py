# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demos must build the decoder at the same weight dtype a server uses.

Running a demo at bfloat16 while the serving stack builds the decoder at
bfloat8_b decodes the same clip differently ("コース" instead of "構成"), which
reads as a model bug but is just a different quantisation. Route both through
decoder_weight_dtype() so they cannot drift.
"""

import os

HERE = os.path.dirname(__file__)
DEMOS = [
    os.path.join(HERE, "..", "demo", "demo.py"),
    os.path.join(HERE, "..", "demo", "demo_wav.py"),
]


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_demos_use_the_shared_dtype_helper():
    for path in DEMOS:
        src = _read(path)
        assert "decoder_weight_dtype" in src, f"{path} must import the helper"
        assert "dtype = decoder_weight_dtype()" in src
        assert "args.weight_cache_path(dtype)" in src


def test_demos_do_not_hardcode_bfloat16():
    for path in DEMOS:
        src = _read(path)
        assert "ttnn.bfloat16, dev" not in src, f"{path} must not pin bfloat16"


ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def test_the_served_adapter_uses_the_same_helper():
    """The docstring's premise only holds if BOTH sides route through it.

    This file was written to stop a demo drifting from the server, but only
    checked the demos. The adapter hardcoded ttnn.bfloat8_b, so
    QWEN3ASR_DECODER_DTYPE moved the demo and left the server where it was --
    turning any demo-vs-served comparison run with the override set into a
    bfloat16-vs-bfloat8_b comparison, which is the exact confusion the helper
    exists to prevent.
    """
    src = _read(ADAPTER)

    # Require the import statement, not just the name: the call site alone
    # satisfies a bare substring check while the module fails at import time.
    assert "from .qwen3_asr_decoder import decoder_weight_dtype" in src, (
        "the adapter must import the helper, not merely reference the name"
    )
    assert "decoder_dtype = decoder_weight_dtype()" in src
    assert "dtype=decoder_dtype," in src, "the decoder must be built at that dtype"
    assert "weight_cache_path(decoder_dtype)" in src, (
        "the cache path must match the dtype, or a rebuild reuses the wrong cache"
    )


def test_the_served_adapter_does_not_hardcode_the_decoder_dtype():
    """A literal here re-opens the drift even with the helper imported."""
    src = _read(ADAPTER)
    start = src.index("decoder = Qwen3ASRDecoder(")
    call = src[start : src.index(")", src.index("use_paged_kv_cache", start))]
    assert "ttnn.bfloat8_b" not in call, (
        "the decoder construction must take the dtype from the helper, not a literal"
    )
