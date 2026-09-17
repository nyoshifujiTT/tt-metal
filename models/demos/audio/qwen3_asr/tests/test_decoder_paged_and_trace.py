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


def _defaults(name):
    """Map every argument of ``name`` that has a default to that default."""
    tree = ast.parse(_read(DECODER))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args.args + node.args.kwonlyargs
            defaults = list(node.args.defaults)
            padded = [None] * (len(node.args.args) - len(node.args.defaults)) + defaults
            padded += list(node.args.kw_defaults)
            return {a.arg: d for a, d in zip(args, padded) if d is not None}
    raise AssertionError(f"{name} not found")


def test_prefill_logits_accepts_paged_kv():
    args = _sig("prefill_logits")
    assert "page_table" in args and "kv_cache" in args


def _call_kwargs(src, name):
    """Keyword names passed to every call of ``name``, via the AST.

    Matching `"page_table=page_table," in src` cannot say *which* call
    carries it: the prefill call on the line above also matches, so decode
    could stop forwarding the page table with the assertion still satisfied.
    """
    import ast

    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and (
            getattr(node.func, "attr", None) == name
            or getattr(node.func, "id", None) == name
        ):
            out.append({kw.arg for kw in node.keywords if kw.arg})
    return out


def test_generate_threads_paged_kv():
    args = _sig("generate")
    assert "page_table" in args and "kv_cache" in args
    src = _read(DECODER)
    assert "self.prefill_logits(inputs_embeds, page_table=page_table, kv_cache=kv_cache)" in src

    # Both halves of the loop must be threaded, checked per call site rather
    # than by a text match one of them can satisfy for the other.
    decode_calls = _call_kwargs(src, "decode_forward")
    assert decode_calls, "generate() must drive decode_forward"
    for kwargs in decode_calls:
        assert "page_table" in kwargs, (
            "decode_forward must be given the page table, or decode silently "
            "runs the non-paged SDPA kernel while prefill ran the paged one"
        )
        assert "kv_cache" in kwargs, "decode_forward must be given the kv cache"

    prefill_calls = _call_kwargs(src, "prefill_logits")
    assert prefill_calls, "generate() must drive prefill_logits"
    for kwargs in prefill_calls:
        assert {"page_table", "kv_cache"} <= kwargs, kwargs


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


def test_paged_kv_arguments_default_to_none():
    # Non-paged must stay the default on EVERY entry point: a positional-only
    # caller (the ttnn demo) must not have to know about paging at all.
    for name in ("prefill_logits", "generate"):
        defaults = _defaults(name)
        for arg in ("page_table", "kv_cache"):
            assert arg in defaults, f"{name}: {arg} must be optional"
            node = defaults[arg]
            assert isinstance(node, ast.Constant) and node.value is None, (
                f"{name}: {arg} must default to None, got {ast.dump(node)}"
            )


def test_non_paged_stays_the_default():
    """Both entry points default to non-paged, checked as defaults.

    `"page_table=None" in src` matched two different lines -- the signature
    of prefill_logits and the signature of generate -- so either one could
    stop defaulting to None with the assertion still satisfied. Read the
    default off each signature instead.
    """
    import ast

    assert _sig("prefill_logits")[0] == "self"

    tree = ast.parse(_read(DECODER))
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in (
            "prefill_logits",
            "generate",
        ):
            continue
        args = node.args.args + node.args.kwonlyargs
        padded = [None] * (len(node.args.args) - len(node.args.defaults))
        padded += list(node.args.defaults) + list(node.args.kw_defaults)
        defaults = {a.arg: d for a, d in zip(args, padded)}
        assert "page_table" in defaults, f"{node.name} must take a page_table"
        default = defaults["page_table"]
        assert isinstance(default, ast.Constant) and default.value is None, (
            f"{node.name}'s page_table must default to None; paged KV is opt-in"
        )
        seen[node.name] = True

    assert seen == {"prefill_logits": True, "generate": True}, seen


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


GENERATOR = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def test_the_prefill_deallocate_is_not_justified_by_a_trace_mode():
    """Freeing the prefill input is safe for a structural reason, not a setting.

    The comment beside ttnn.deallocate(prefill_input) read "Safe because
    prefill is never traced (trace_mode=none ...)", naming a value the shipped
    spec no longer uses -- it runs trace_mode=decode_only. The conclusion was
    right and the reason was not: prefill is untraced because it *cannot* be
    traced (the encoder allocates device buffers dynamically, so
    warmup_model_prefill is a no-op), which holds under every trace_mode.

    That matters because this comment supports a safety judgement -- whether a
    device tensor may be freed while a trace could hold it. A reader who
    changes trace_mode should not be left thinking the premise has moved.
    """
    src = _read(GENERATOR)
    start = src.index("ttnn.deallocate(prefill_input)")
    # the justification sits in the comment block immediately above the call
    reason = src[max(0, start - 700) : start]

    assert "trace_mode=none" not in reason, (
        "the deallocate is justified by a trace_mode the spec does not use"
    )
    assert "warmup_model_prefill" in reason, (
        "name the structural reason: prefill warmup never captures a prefill "
        "trace, so no prefill trace exists"
    )


def test_prefill_warmup_never_captures_the_trace_the_comment_relies_on():
    """Guard the premise rather than trusting the prose.

    If warmup_model_prefill ever starts capturing a trace, the deallocate above
    stops being safe and this test says so.

    The premise is "no prefill trace is ever captured", not "this method does
    nothing". It used to be stated as the latter -- a body of at most one
    logger call -- which was true while the method was a bare no-op and turned
    into an obstacle when it had to run the encoder once during warmup so those
    allocations happen before the decode trace is captured. Running the encoder
    eagerly is not capturing a trace and does not retain the tensor, so it
    cannot invalidate the deallocate. Check the property that actually matters:
    the trace pass returns without doing work.
    """
    tree = ast.parse(_read(GENERATOR))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_model_prefill":
            calls = [
                n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and "trace" in ast.unparse(n).lower()
                and "logger" not in ast.unparse(n).lower()
            ]
            assert not calls, (
                f"warmup_model_prefill now does trace work: {[ast.unparse(c) for c in calls]}"
            )
            # The trace pass must bail out before any work: that is what keeps
            # "no prefill trace is ever captured" true regardless of what the
            # eager pass does.
            guards = [
                s for s in node.body
                if isinstance(s, ast.If) and "enable_trace" in ast.unparse(s.test)
            ]
            assert len(guards) == 1, (
                "prefill warmup must branch on enable_trace exactly once, so the "
                "trace pass has a single, obvious exit"
            )
            assert any(isinstance(s, ast.Return) for s in guards[0].body), (
                "the enable_trace branch must return; falling through would let "
                "the trace pass allocate behind its own capture"
            )
            assert node.body.index(guards[0]) == 0, (
                "the guard must be the first statement, or work runs before the "
                "trace pass can decline it"
            )
            return
    raise AssertionError("warmup_model_prefill not found")


def test_a_trace_mode_value_here_is_a_quotation_not_a_claim():
    """tt/ must not assert what the spec ships; the spec is the source.

    trace_mode lives in workflows/model_specs in tt-inference-server, so a
    bare mention here is a second source of truth -- and one did drift: the
    deallocate above was justified by "trace_mode=none" long after the spec
    moved to decode_only.

    Reporting a past measurement is different, and the file legitimately does
    it once ("trace_mode=none + warmup -> 50/50, 30/30 no wedge", quoted from
    the worklog). Require every occurrence to be inside quotes, so a
    measurement can be cited but a live claim cannot be made.
    """
    src = _read(GENERATOR)

    for line in src.splitlines():
        if "trace_mode=none" not in line and "trace_mode = none" not in line:
            continue
        quoted = line.split('"')[1::2]
        assert any("trace_mode" in part for part in quoted), (
            "a trace_mode value here must be a quoted measurement, not a "
            f"statement about the shipped configuration: {line.strip()}"
        )
