# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How the pinned mel is padded, pinned down by measurement.

Pinning the mel to a fixed frame count is our vLLM-path addition (upstream pins
the AUDIO instead: its standalone server zero-extends the waveform and lets the
feature extractor emit the tail). Three options were measured against the
unpadded CPU reference on TED clips, max-abs on the encoder output:

    extractor's own tail   0.0904   <- best, but only available before slicing
    zero padding           0.0971
    constant silence col   0.0988   <- WORST, and it was tried and rejected

So: never synthesise a constant, and where the extractor's full-width mel is
still available (the corpus eval) keep it instead of re-padding.
"""

import os

import torch

HERE = os.path.dirname(__file__)
TT = os.path.join(HERE, "..", "tt")
EVAL = os.path.join(HERE, "..", "eval", "corpus_eval.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _fn(path, name):
    """Compile one function out of a module that needs a device build to import.

    Locate the definition through the AST rather than by string slicing, so the
    extraction cannot silently pick up neighbouring module-level code.
    """
    import ast

    src = _read(path)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"torch": torch}
            exec(compile(module, path, "exec"), namespace)  # noqa: S102 - our own source
            return namespace[name]
    raise AssertionError(f"{name} not found in {path}")


def test_no_constant_silence_column_anywhere():
    # Measured worse than plain zero padding; keep it out of both front-ends.
    for path in (os.path.join(TT, "generator_vllm.py"), EVAL):
        src = _read(path)
        assert "silence_col" not in src
        assert "_silence_mel_value" not in src


def test_served_path_pins_without_zeros():
    src = _read(os.path.join(TT, "generator_vllm.py"))
    assert "mel = _pin_mel(mel, PIN_MEL_FRAMES)" in src
    assert "torch.nn.functional.pad(mel, (0, PIN_MEL_FRAMES" not in src


def test_pin_truncates_when_longer():
    for path, name in ((EVAL, "pin_mel"), (os.path.join(TT, "generator_vllm.py"), "_pin_mel")):
        pin = _fn(path, name)
        mel = torch.arange(4 * 10, dtype=torch.float32).reshape(4, 10)
        out = pin(mel, 6)
        assert out.shape == (4, 6)
        assert torch.equal(out, mel[:, :6])


def test_pin_is_a_noop_at_exact_width():
    for path, name in ((EVAL, "pin_mel"), (os.path.join(TT, "generator_vllm.py"), "_pin_mel")):
        pin = _fn(path, name)
        mel = torch.ones(4, 8)
        assert torch.equal(pin(mel, 8), mel)


def test_pin_replicates_the_last_real_frame_when_shorter():
    # A frame from this recording, not a synthetic constant.
    for path, name in ((EVAL, "pin_mel"), (os.path.join(TT, "generator_vllm.py"), "_pin_mel")):
        pin = _fn(path, name)
        mel = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = pin(mel, 5)
        assert out.shape == (2, 5)
        assert torch.equal(out[:, :3], mel)
        for col in range(3, 5):
            assert torch.equal(out[:, col], mel[:, -1])


def test_encoder_is_told_the_real_frame_count():
    # Pinning the mel is only safe because the padded tail is masked out of
    # attention; passing valid_frames is what turns the mask on.
    src = _read(os.path.join(TT, "generator_vllm.py"))
    assert "valid_frames=n_frames" in src


def test_adapter_reuses_the_upstream_decoder():
    # The upstream Qwen3ASRDecoder already implements the qwen3_vl
    # "tokens is actually embeddings" prepare_inputs_prefill, so the adapter must
    # not carry a second copy of it.
    src = _read(os.path.join(TT, "generator_vllm.py"))
    assert "from .qwen3_asr_decoder import Qwen3ASRDecoder" in src
    assert "class Qwen3ASRPagedDecoder" not in src
    assert "decoder.prepare_inputs_prefill(\n                merged.unsqueeze(0)," in src


def test_eval_keeps_the_extractor_width():
    # WhisperFeatureExtractor already returns a fixed 3000-frame mel whose tail is
    # genuine silence (it zero-extends the WAVEFORM and runs the filterbank over
    # the whole 30s window). Slicing that to the real frames and re-padding is
    # strictly worse - measured against the unpadded CPU reference: extractor tail
    # 0.0904, zero padding 0.0971, constant silence column 0.0988.
    src = _read(EVAL)
    assert "mel = pin_mel(mel, MEL_PIN)" in src
    assert "mel = mel[:, :nf]" not in src, "do not slice away the extractor's own tail"
