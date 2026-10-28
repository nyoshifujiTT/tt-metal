# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Where the prompt embedding gather runs, and why the default is measured.

The CUDA reference embeds on the accelerator, and in isolation the device gather
is ~10x faster here (10.5 ms host vs 1.1 ms device for a 149-token prompt,
bit-identical output). End-to-end on real traffic it is not: prompt lengths vary,
so an unpinned device gather compiles a program per length and churns the program
cache (TED 6.08 -> 3.48 audio-s/s), and pinning it to the prefill bucket makes it
embed ~1024 rows for a ~149-token prompt (5.79 a/s, still under the host path).

Both paths therefore have to stay available and correct, the default has to be
the measured winner (host), and the device path has to stay pinned so nobody
re-introduces the cache churn while experimenting.
"""

import os
import re

TT_DIR = os.path.join(os.path.dirname(__file__), "..", "tt")


def _source():
    with open(os.path.join(TT_DIR, "generator_vllm.py")) as fh:
        return fh.read()


def _method_body(src, name):
    """Source of one method, from its def to the next def at the same indent."""
    lines = src.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"    def {name}("):
            start = i
            break
    assert start is not None, f"{name} not found"
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("    def "):
            return "".join(lines[start:j])
    return "".join(lines[start:])


def test_device_embed_defaults_off_because_host_measured_faster():
    src = _source()
    assert (
        'os.environ.get("QWEN3ASR_DEVICE_EMBED", "0").strip().lower() in ("1", "true", "yes", "on")' in src
    ), (
        "the default must be the host gather (measured 6.08 vs 5.79 audio-s/s on "
        "TED); the device path stays opt-in via QWEN3ASR_DEVICE_EMBED=1"
    )


def test_merge_embeds_goes_through_the_shared_helper():
    src = _source()
    body = _method_body(src, "_merge_embeds")
    assert "self._embed_prompt(input_ids)" in body, (
        "_merge_embeds must route the gather through _embed_prompt so the device "
        "path is actually used"
    )
    assert "self.text_embed[input_ids]" not in body, (
        "the host gather belongs in the fallback, not inline in _merge_embeds"
    )


def test_host_fallback_is_still_reachable():
    src = _source()
    body = _method_body(src, "_embed_prompt")
    assert "self.text_embed[input_ids]" in body, "host fallback must remain"


def test_device_embed_is_shape_pinned():
    """An unpinned device embed compiles a program per prompt length.

    Everything else the adapter feeds the device is pinned for exactly this
    reason (PIN_MEL_FRAMES for the encoder, PREFILL_PIN_LEN for prefill). Without
    pinning here the program cache churns and throughput regresses: measured
    6.08 -> 3.48 audio-s/s on TED, recovered by pinning.
    """
    body = _method_body(_source(), "_embed_prompt")
    assert "PREFILL_PIN_LEN" in body, "the device embed must use the prefill pin"
    assert "get_padded_prefill_len" in body, "pad up to the shared bucket"
    assert "input_ids.reshape(1, 1, 1, S)" not in body, (
        "feeding the raw per-request length defeats the pin"
    )


def test_no_inline_host_gather_anywhere():
    """Every embedding gather must go through _embed_prompt.

    The text-only prefill branch kept its own `self.text_embed[input_ids].clone()`,
    so it neither used the device path nor benefited from dropping the redundant
    copy. One helper, one place to fix.
    """
    src = _source()
    outside = src.replace(_method_body(src, "_embed_prompt"), "")
    assert "self.text_embed[input_ids]" not in outside, (
        "the only host gather may live inside _embed_prompt"
    )
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if "self.text_embed[" in ln and ".clone()" in ln]
    assert not offenders, (
        "advanced indexing already returns a fresh tensor; a clone doubles the "
        f"host copy: {offenders}"
    )
