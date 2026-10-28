# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Prefill warmup must actually run the encoder, not just decline to trace it.

Prefill is untraced here because the audio encoder allocates device buffers
dynamically. The tempting conclusion -- that prefill warmup has nothing to do --
is wrong, and was the live defect: those allocations still happen per request,
and once the decode trace is live they land beside its scratch.

Measured on the server with a byte-identical clip, the merged-embeds checksum
changed on every request after the first (sum 190.63 -> -78.13 -> -47.84), the
first generated token went 136518 -> 15322, and the transcript collapsed to
"はい。". Running the encoder once during the eager warmup pass -- before the
decode trace is captured -- made the output bit-identical across requests.

So the rule is: the eager pass does real work, the trace pass does not.
"""

import ast
import os

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _method(name):
    for node in ast.walk(ast.parse(_read(ADAPTER))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _module_constant(name):
    for node in ast.parse(_read(ADAPTER)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found at module level")


def test_the_eager_pass_runs_the_encoder():
    """The whole point: the allocations must happen during warmup.

    Asserted against the call, not against the log line: a warmup that only
    announces itself leaves the defect in place.
    """
    body = ast.unparse(_method("warmup_model_prefill"))
    assert "self._merge_embeds(" in body, (
        "prefill warmup must run the encoder (via _merge_embeds) so its device "
        "buffers are allocated before the decode trace is captured"
    )


def test_the_trace_pass_returns_without_capturing():
    """Prefill still cannot be traced; only the eager pass does work."""
    node = _method("warmup_model_prefill")
    body = ast.unparse(node)
    assert "if enable_trace:" in body and "return" in body, (
        "the trace pass must return early; prefill is not traceable here"
    )
    # The early return has to come before the encoder run, or the trace pass
    # would allocate behind its own capture -- the exact ordering being fixed.
    guard = next(
        s
        for s in node.body
        if isinstance(s, ast.If) and "enable_trace" in ast.unparse(s.test)
    )
    assert any(isinstance(s, ast.Return) for s in guard.body), (
        "the enable_trace branch must return, not fall through to the warmup"
    )
    guard_index = node.body.index(guard)
    merge_index = next(
        i for i, s in enumerate(node.body) if "_merge_embeds" in ast.unparse(s)
    )
    assert guard_index < merge_index, "the guard must precede the encoder run"


def test_the_warmup_uses_the_pinned_shapes_the_server_uses():
    """A different shape would warm an allocation no request ever repeats.

    Every served request pins mel to PIN_MEL_FRAMES and the prompt to
    PREFILL_PIN_LEN, which is the only reason one warmup can cover them all.
    """
    body = ast.unparse(_method("warmup_model_prefill"))
    assert "PIN_MEL_FRAMES" in body, "the warmup mel must use the served pin"
    assert "PREFILL_PIN_LEN" in body, "the warmup prompt must use the served pin"
    assert "N_MEL_BINS" in body, "the mel must have the encoder's bin count"


def test_the_warmup_prompt_actually_contains_audio_positions():
    """Without audio ids the splice is skipped and the encoder never runs.

    _merge_embeds only reaches the encoder for the audio-token positions, so a
    prompt of plain text would satisfy the call-site assertion above while
    warming nothing.
    """
    body = ast.unparse(_method("warmup_model_prefill"))
    assert "AUDIO_TOKEN_ID" in body, "the warmup prompt must mark audio positions"


def test_the_audio_span_leaves_text_on_both_sides():
    """Mirrors a real prompt rather than a full-width special case."""
    start = _module_constant("_WARMUP_AUDIO_START")
    tail = _module_constant("_WARMUP_AUDIO_TAIL")
    assert start > 0 and tail > 0, "keep text either side of the audio span"
    # PREFILL_PIN_LEN is re-exported from the decoder module, so read the
    # default from there rather than literal-evaluating the import.
    decoder = _read(os.path.join(HERE, "..", "tt", "qwen3_asr_decoder.py"))
    pin = int(
        decoder.split('os.environ.get("QWEN3ASR_PREFILL_PIN", "')[1].split('"')[0]
    )
    assert start + tail < pin, "the span must be non-empty at the shipped pin"


def test_the_mel_bin_count_matches_the_encoder_frontend():
    """128 is the conv frontend's in_channels; a mismatch warms the wrong shape."""
    assert _module_constant("N_MEL_BINS") == 128
    encoder = _read(os.path.join(HERE, "..", "tt", "audio_encoder.py"))
    assert "num_mel=128" in encoder, (
        "the encoder no longer documents 128 mel bins; re-derive N_MEL_BINS"
    )
