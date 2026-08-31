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
    assert "num_blocks_in_seq(S_pad, get_block_size(model_kv_cache))" in src
    assert "prefill_page_table[:, :num_blocks]" in src
    assert "from models.tt_transformers.tt.common import get_block_size, num_blocks_in_seq" in src


def test_paged_prefill_passes_only_the_single_user_row():
    # paged_fill_cache indexes the row it is given, so the prefill must hand
    # over exactly one row rather than the whole batch's table.
    assert "page_table[0:1]" in _read(DECODER)


def test_paged_prefill_pads_short_rows_with_minus_one():
    # A row shorter than the padded length must be extended with the upstream
    # "unmapped" sentinel, not with block 0, which is a live page.
    assert "dtype=torch.int32) * -1" in _read(DECODER)


def test_non_paged_stays_the_default():
    args_prefill = _sig("prefill_logits")
    src = _read(DECODER)
    assert "page_table=None" in src, "paged KV must be opt-in"
    assert args_prefill[0] == "self"


def test_decode_trace_is_on_by_default_and_overridable():
    src = _read(DECODER)
    assert 'os.environ.get("QWEN3ASR_DECODE_TRACE", "1")' in src
    assert "enable_trace=DECODE_TRACE" in src


def _decode_trace(env_value):
    """Mirror of the shipped parse, so the accepted spellings stay pinned.

    The literal default is asserted above; this pins the *behaviour* of the
    surrounding ``.strip().lower() in (...)`` that turns the raw environment
    string into the flag, so an operator writing ``QWEN3ASR_DECODE_TRACE=off``
    keeps getting an untraced decode.
    """
    raw = "1" if env_value is None else env_value
    return raw.strip().lower() in ("1", "true", "yes", "on")


def test_decode_trace_parse_matches_the_shipped_expression():
    """The mirror is only meaningful if it is the same expression."""
    assert '.strip().lower() in ("1", "true", "yes", "on")' in _read(DECODER)


def test_decode_trace_defaults_on_when_unset():
    assert _decode_trace(None) is True


def test_decode_trace_accepts_the_documented_spellings():
    for value in ("1", "true", "TRUE", "Yes", " on "):
        assert _decode_trace(value) is True, value


def test_decode_trace_treats_anything_else_as_off():
    for value in ("0", "false", "no", "off", "", "  ", "nope"):
        assert _decode_trace(value) is False, value


def test_weight_dtype_defaults_to_bfloat8_b():
    src = _read(DECODER)
    assert 'os.environ.get("QWEN3ASR_DECODER_DTYPE", "bfloat8_b")' in src
    assert "def decoder_weight_dtype()" in src


def test_decode_feeds_the_full_configured_batch():
    # tt_transformers builds the decode graph for args.max_batch_size and
    # prepare_decode_inputs_host asserts the token batch matches it, so a
    # single-user decode still has to submit a B-wide step and read slot 0.
    # Submitting a 1-wide step raised
    # "Batch size 1 must be equal to max_batch_size 4" for every clip.
    src = _read(DECODER)
    assert 'batch = int(getattr(self.args, "max_batch_size", 1) or 1)' in src
    assert "tokens = torch.zeros(batch, 1, dtype=torch.long)" in src
    assert "positions = torch.zeros(batch, dtype=torch.int64)" in src
    assert ".reshape(batch, -1)[0]" in src, "only slot 0 belongs to this user"


def test_idle_decode_slots_are_parked_at_zero():
    # Idle slots must not index a live KV page.
    src = _read(DECODER)
    assert "positions[0] = pos" in src


def test_kv_cache_submesh_dimension_is_handled_in_one_place():
    # allocate_vllm_kv_cache returns list[submesh][layer][k, v]. The two shared
    # entry points disagree on what they want: prefill_forward_single_user_text
    # passes kv_cache straight to the model (so it needs THIS replica's layer
    # list) while decode_forward indexes kv_cache[model_id]. Unwrapping at the
    # caller made the paged ops read a layer's [k, v] pair as the submesh list and
    # report n_kv_heads as the block count ("max_num_blocks=8").
    src = _read(DECODER)
    assert "model_kv_cache = kv_cache[0] if kv_cache is not None else None" in src
    assert "kv_cache=model_kv_cache," in src, "prefill takes the unwrapped list"
    assert "get_block_size(model_kv_cache)" in src


def test_eval_keeps_the_submesh_dimension():
    eval_src = _read(os.path.join(HERE, "..", "eval", "corpus_eval.py"))
    body = eval_src[eval_src.index("def build_paged_kv") : eval_src.index("def feat_len")]
    assert "tt_cache_path=args.model_cache_path,\n    )\n" in body, "must not strip [0] at the caller"


def test_prefill_delegates_the_page_table_conversion():
    # An earlier bring-up drove ttnn_prefill_forward directly and had to hand it
    # the ttnn-converted page table itself (passing the host copy raises a pybind
    # TypeError in paged_fill_cache). Going through
    # Generator.prefill_forward_single_user_text means the conversion happens
    # inside prepare_inputs_prefill, so the decoder must NOT re-implement it.
    src = _read(DECODER)
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "prefill_forward_single_user_text" in code, "prefill must go through the Generator"
    assert "ttnn_prefill_forward(" not in code, "do not re-drive the low-level prefill"
    assert "tt_page_table" not in code, "the Generator owns the host->ttnn conversion"
