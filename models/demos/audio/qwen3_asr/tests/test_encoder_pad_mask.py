# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pinned mel padding must be masked out of the encoder's attention.

The served path pads every clip to a fixed mel length so the encoder keeps one
program shape. The encoder runs FULL bidirectional attention, so without a mask
those padded positions become ordinary key/value entries that every real token
attends to. Measured against the CPU reference, padding a 190-frame clip to 3000
frames moves the encoder output by **0.70 relative** (‖a-b‖ / ‖a‖ over the real
rows), i.e. the padding is not transparent to the transcript.

Quote the relative figure, not an absolute one. Reproduced here: 0.6215 on the
golden clip's mel and 0.7648 on a randn mel -- same conclusion either way. The
mean-|diff| behind it is input-scaled (0.0068 and 0.0126 respectively), so an
absolute number cannot be checked without also fixing the input. An earlier
version of this docstring paired the relative figure with a borrowed absolute
one, which is not an encoder-output difference at all: 0.0971 is the
mel-padding comparison in test_mel_pin.py (extractor tail 0.0904 < zero pad
0.0971 < constant column 0.0988).
"""

import os
import types

import torch

HERE = os.path.dirname(__file__)
TT = os.path.join(HERE, "..", "tt")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_sdpa_receives_the_mask():
    src = _read(os.path.join(TT, "audio_encoder.py"))
    assert "attn_mask=attn_mask" in src, "the encoder SDPA must take the pad mask"
    assert "def _layer(x, lp, device, attn_mask=None)" in src


def test_encode_mel_accepts_the_real_frame_count():
    src = _read(os.path.join(TT, "audio_encoder.py"))
    assert "def encode_mel(mel, params, device, valid_frames=None)" in src
    assert "valid_len=valid_len" in src


def _recorder(capture):
    def from_torch(tensor, **kwargs):
        capture["mask"] = tensor
        return "TT_MASK"

    return from_torch


def _build_pad_mask(capture):
    """Compile build_pad_mask alone, with a recording ttnn stub.

    Importing audio_encoder pulls in the whole ttnn extension, which needs a
    matching device build; the mask itself is pure torch, so extract just that
    function via the AST and give it a stub. That keeps the behavioural check
    runnable anywhere and still fails if the real logic drifts.
    """
    import ast

    src = _read(os.path.join(TT, "audio_encoder.py"))
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_pad_mask":
            stub = types.SimpleNamespace(
                bfloat16="bf16",
                TILE_LAYOUT="tile",
                DRAM_MEMORY_CONFIG="dram",
                from_torch=_recorder(capture),
            )
            namespace = {"torch": torch, "ttnn": stub}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, "audio_encoder.py", "exec"), namespace)  # noqa: S102 - our own source
            return namespace["build_pad_mask"]
    raise AssertionError("build_pad_mask not found")


def test_mask_is_none_when_nothing_is_padded():
    build_pad_mask = _build_pad_mask({})
    assert build_pad_mask(100, None, None) is None
    assert build_pad_mask(100, 100, None) is None
    assert build_pad_mask(100, 200, None) is None


def test_mask_hides_exactly_the_padded_tail():
    captured = {}
    build_pad_mask = _build_pad_mask(captured)
    assert build_pad_mask(8, 3, device=None) == "TT_MASK"
    mask = captured["mask"]
    assert mask.shape == (1, 1, 8, 8)
    assert torch.isfinite(mask[..., :3]).all(), "real positions stay visible"
    assert torch.isneginf(mask[..., 3:]).all(), "padded positions are masked out"


def test_projector_frees_its_intermediates():
    # ttnn ops return NEW device tensors. Reassigning `x` through ln_post ->
    # proj1 -> proj2 without freeing the consumed input leaks one tensor per
    # stage per request, which accumulates in a long-lived server until the
    # device wedges.
    src = _read(os.path.join(TT, "audio_encoder.py"))
    tail = src[src.index("def encode(x_host") :]
    for consumed in ("ttnn.deallocate(x)", "ttnn.deallocate(x_ln)", "ttnn.deallocate(x_p1)"):
        assert consumed in tail, f"{consumed} must free the consumed intermediate"


def test_mask_is_freed_even_if_a_layer_raises():
    src = _read(os.path.join(TT, "audio_encoder.py"))
    tail = src[src.index("def encode(x_host") :]
    assert "finally:" in tail, "the mask must be freed on the error path too"


def test_this_docstring_does_not_borrow_the_mel_pin_number():
    """0.0971 belongs to test_mel_pin.py, not to the encoder-output delta.

    This file once claimed "0.097 absolute / 0.70 relative" for the effect of
    unmasked padding. The relative half reproduces (0.6215 on the golden clip's
    mel, 0.7648 on a randn mel); the absolute half does not -- the measured
    mean-|diff| is 0.0068 and 0.0126 for those two inputs, an order out.

    0.0971 is the zero-padding entry in test_mel_pin.py's mel comparison
    (extractor tail 0.0904 < zero pad 0.0971 < constant column 0.0988). Two
    different measurements had collided, and the borrowed one made the claim
    look checkable when it was not.
    """
    src = _read(os.path.join(HERE, "test_encoder_pad_mask.py"))
    head = src[: src.index('"""', src.index('"""') + 3)]
    flat = " ".join(head.split())

    assert "0.097 absolute" not in flat, (
        "0.0971 is the mel-padding number from test_mel_pin.py, not an "
        "encoder-output difference"
    )
    # the figure that does reproduce, with the definition that makes it checkable
    assert "0.70 relative" in flat
    assert "‖a-b‖ / ‖a‖" in flat, "define the ratio, or 'relative' is ambiguous"
    # both reproductions, so the range is visible rather than a single point
    assert "0.6215" in flat and "0.7648" in flat
    # and why an absolute figure is the wrong thing to quote here
    assert "input-scaled" in flat


def test_the_mel_pin_numbers_stay_where_they_belong():
    """Guard the other side: those three values must remain in test_mel_pin.py.

    If they move or change, this file's explanation of the old mix-up goes
    stale and the next reader cannot tell which measurement was which.
    """
    src = _read(os.path.join(HERE, "test_mel_pin.py"))
    for value in ("0.0904", "0.0971", "0.0988"):
        assert value in src, (
            f"{value} left test_mel_pin.py; update the note in "
            f"test_encoder_pad_mask.py that points at it"
        )
