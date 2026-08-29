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
