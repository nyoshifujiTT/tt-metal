# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Every front-end must land on the SAME prefill bucket.

tt-metal has a length-keyed program-cache collision (tenstorrent/tt-metal#49451):
the prefill matmul hash does not cover the reshaped dim -3, so mixing a
512-padded and a 1024-padded prefill in one process reuses the wrong program.
Even where it survives, the same audio transcribes differently ("構成" vs
"コース") depending on which bucket it landed in, so two front-ends that pad
differently cannot be compared at all.
"""

import ast
import math
import os

HERE = os.path.dirname(__file__)
TT = os.path.join(HERE, "..", "tt")
DECODER = os.path.join(TT, "qwen3_asr_decoder.py")
ADAPTER = os.path.join(TT, "generator_vllm.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _pin():
    tree = ast.parse(_read(DECODER))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PREFILL_PIN_LEN" for t in node.targets
        ):
            return int(os.environ.get("QWEN3ASR_PREFILL_PIN", "512"))
    raise AssertionError("PREFILL_PIN_LEN must be defined in the decoder")


def _audio_tokens(mel_frames):
    leave = mel_frames % 100
    f = (leave - 1) // 2 + 1
    return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13


def test_single_source_of_truth_for_the_bucket():
    # The adapter must not define its own pin; it has to import the decoder's.
    src = _read(ADAPTER)
    assert "PREFILL_PIN_LEN as _DECODER_PREFILL_PIN_LEN" in src
    assert "PREFILL_PIN_LEN = _DECODER_PREFILL_PIN_LEN" in src


def test_decoder_pads_to_at_least_the_pin():
    src = _read(DECODER)
    assert "S_pad = max(((S + 511) // 512) * 512, PREFILL_PIN_LEN)" in src


def test_the_whole_supported_audio_range_fits_one_bucket():
    # A 30s clip (the WhisperFeatureExtractor cap) is ~390 audio tokens plus a
    # ~13-token prompt, so every real request lands in the first bucket. If this
    # ever stops holding, the two front-ends can diverge and must be re-checked.
    pin = _pin()
    for seconds in (1, 5, 10, 15, 30):
        prompt_len = _audio_tokens(seconds * 100) + 13
        assert prompt_len <= pin, f"{seconds}s prompt ({prompt_len}) exceeds the pin ({pin})"


def test_decoder_and_adapter_agree_over_that_range():
    pin = _pin()

    def padded_prefill_len(seq_len):
        return 128 if seq_len <= 128 else 2 ** math.ceil(math.log2(seq_len))

    def decoder_pad(seq_len):
        return max(((seq_len + 511) // 512) * 512, pin)

    def adapter_pad(seq_len):
        natural = padded_prefill_len(seq_len)
        return max(natural, pin) if natural <= pin else natural

    for seconds in (1, 5, 10, 15, 30):
        prompt_len = _audio_tokens(seconds * 100) + 13
        assert decoder_pad(prompt_len) == adapter_pad(prompt_len), (
            f"{seconds}s: decoder {decoder_pad(prompt_len)} vs adapter {adapter_pad(prompt_len)}"
        )
