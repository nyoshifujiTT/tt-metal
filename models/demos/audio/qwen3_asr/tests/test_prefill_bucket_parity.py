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
import re

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
    #
    # The range here stops at 30 s deliberately: past that the pin IS crossed
    # (see test_the_bucket_is_crossed_past_the_documented_length), so extending
    # this loop would assert something false rather than something protective.
    pin = _pin()
    for seconds in (1, 5, 10, 15, 30):
        prompt_len = _audio_tokens(seconds * 100) + 13
        assert prompt_len <= pin, f"{seconds}s prompt ({prompt_len}) exceeds the pin ({pin})"


def test_the_token_model_matches_the_server():
    """The formula is only useful if it predicts what vLLM actually reports.

    Measured by reading vllm:request_prompt_tokens_sum per request against the
    running server, clips built from the FLEURS golden clip:

        5s->78  11s->156  15s->208  25s->338  30s->403

    All exact. Beyond 30 s the model runs 13 low (38s->507 predicted vs 520
    measured) because WhisperFeatureExtractor caps a window at 30 s and the
    extra window carries its own 13 tokens, so the formula is only claimed
    within the cap.
    """
    for seconds, expected in ((5, 78), (11, 156), (15, 208), (25, 338), (30, 403)):
        got = _audio_tokens(seconds * 100) + 13
        assert got == expected, f"{seconds}s: model {got} != measured {expected}"


def test_the_bucket_is_crossed_past_the_documented_length():
    """"real prompts are always <=512" is false on the vLLM path.

    That held for the standalone server, which pinned every request to 14 s.
    vLLM accepts whatever it is given up to max_model_len, and measured prompt
    growth is 13.0*seconds + 13, so the 512 pin is crossed at ~38 s:
    38 s measured 520 tokens and 45 s measured 611, both HTTP 200 with /health
    still 200 after.

    Encode the crossover so the docs cannot drift back to "never": a process
    that mixes sub-38 s and over-38 s clips is exactly the case tt-metal#49451
    still needs fixing for.
    """
    pin = _pin()

    def measured_tokens(seconds):
        # 13 tokens per 100 mel frames, plus 13 for the prompt template, plus
        # 13 per additional 30 s feature-extractor window.
        windows = max(1, -(-seconds // 30))
        return seconds * 13 + 13 * windows

    assert measured_tokens(30) <= pin, "30s must still be inside the pin"
    assert measured_tokens(38) > pin, (
        "38s crosses the pin; the README must not claim prompts are always under it"
    )
    # and the two points that were actually measured
    assert measured_tokens(38) == 520
    assert measured_tokens(45) == 611


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


README = os.path.join(HERE, "..", "README.md")


def test_the_docs_do_not_claim_prompts_are_always_under_the_pin():
    """Both the README and the decoder comment said "always <=512".

    Measured on the vLLM server, 38 s clips produce 520 prompt tokens. The
    claim was inherited from the standalone server's fixed 14 s pin and does
    not hold for a path that accepts arbitrary clip lengths, so it has to be
    stated as a length bound rather than as an invariant.
    """
    for path in (README, DECODER):
        flat = " ".join(_read(path).split())
        assert "always <=512 tokens" not in flat and "always ≤512 tokens" not in flat, (
            f"{os.path.basename(path)}: prompts exceed 512 past ~38 s; state the bound"
        )
        # Not a bare `"38" in flat`: the README's e2e command names the clip
        # models/demos/audio/whisper/.../17646385371758249908.wav, whose file
        # name contains "38", so every statement about the crossover could be
        # deleted and this still passed. Require the crossover as a phrase.
        assert re.search(r"(?:≈|~)38 s", flat), (
            f"{os.path.basename(path)}: name the length at which the bucket "
            f"changes, as a duration"
        )


def test_the_readme_shows_the_measured_prompt_lengths():
    """The bound has to be evidenced, not asserted, or it drifts again."""
    flat = " ".join(_read(README).split())
    # the measured pairs, including the two that cross the pin
    for tokens in ("78", "156", "208", "403", "520", "611"):
        assert tokens in flat, f"record the measured {tokens}-token point"
    # The rate must appear where the crossover is derived, not merely somewhere
    # in the file: the Prefill-seqlen section quotes it too, so a bare "in flat"
    # check still passed after the Known-limitations derivation was reworded.
    body = flat[flat.index("Why the shipped model works despite this") :]
    body = body[: body.index("The workaround is therefore")]
    assert "13.0 * seconds + 13" in body, (
        "state the growth rate where the ~38 s crossover is derived"
    )
    assert "512 is crossed at" in body, "name the crossover explicitly"
    # and why crossing it is still safe here, so the bound is not read as a bug
    assert "PIN_MEL_FRAMES" in flat, (
        "explain that the encoder is pinned separately, or 520 tokens reads as broken"
    )
