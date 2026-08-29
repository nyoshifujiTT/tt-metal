# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pinned mel padding must be masked out of the encoder's attention.

The served path pads every clip to a fixed mel length so the encoder keeps one
program shape. The encoder runs FULL bidirectional attention, so without a mask
those padded positions become ordinary key/value entries that every real token
attends to. Measured against the CPU reference, padding a 190-frame clip to 3000
frames moves the encoder output by 0.097 absolute / 0.70 relative, i.e. the
padding is not transparent to the transcript.
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
