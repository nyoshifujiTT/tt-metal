# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decoder must be drivable the way a paged, traced serving stack drives it.

Paged and non-paged decode dispatch to DIFFERENT SDPA kernels
(paged_scaled_dot_product_attention_decode vs scaled_dot_product_attention_decode)
and do not produce the same output, so a front-end that cannot be handed a page
table cannot be compared against a serving path that always runs paged KV.
Likewise an untraced decode costs 489 ms/token on p150 versus 113 ms traced.
"""

import ast
import os

HERE = os.path.dirname(__file__)
DECODER = os.path.join(HERE, "..", "tt", "qwen3_asr_decoder.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _sig(name):
    tree = ast.parse(_read(DECODER))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    raise AssertionError(f"{name} not found")


def test_prefill_logits_accepts_paged_kv():
    args = _sig("prefill_logits")
    assert "page_table" in args and "kv_cache" in args


def test_generate_threads_paged_kv():
    args = _sig("generate")
    assert "page_table" in args and "kv_cache" in args
    src = _read(DECODER)
    assert "self.prefill_logits(inputs_embeds, page_table=page_table, kv_cache=kv_cache)" in src
    assert "page_table=page_table," in src and "kv_cache=kv_cache," in src


def test_paged_prefill_trims_the_page_table():
    # Generator._get_prefill_user_page_table trims to the blocks the PADDED
    # length covers; handing over the whole range attends past the prompt.
    src = _read(DECODER)
    assert "num_blocks_in_seq(S_pad, get_block_size(kv_cache))" in src
    assert "prefill_page_table[:, :num_blocks]" in src
    assert "from models.tt_transformers.tt.common import get_block_size, num_blocks_in_seq" in src


def test_non_paged_stays_the_default():
    args_prefill = _sig("prefill_logits")
    src = _read(DECODER)
    assert "page_table=None" in src, "paged KV must be opt-in"
    assert args_prefill[0] == "self"


def test_decode_trace_is_on_by_default_and_overridable():
    src = _read(DECODER)
    assert 'os.environ.get("QWEN3ASR_DECODE_TRACE", "1")' in src
    assert "enable_trace=DECODE_TRACE" in src


def test_weight_dtype_defaults_to_bfloat8_b():
    src = _read(DECODER)
    assert 'os.environ.get("QWEN3ASR_DECODER_DTYPE", "bfloat8_b")' in src
    assert "def decoder_weight_dtype()" in src
