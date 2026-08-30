# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demos must resolve the HF snapshot through the shared helper.

An earlier bring-up carried its own QWEN3ASR_SNAP env lookup plus a hardcoded
in-container cache path, which meant the demos could only run where that exact
path existed. reference.resolve_snap_dir already handles this (explicit dir ->
$QWEN3ASR_SNAP_DIR -> huggingface_hub snapshot_download), so the demos must go
through it rather than re-deriving the location.
"""

import os

HERE = os.path.dirname(__file__)
DEMOS = [
    os.path.join(HERE, "..", "demo", "demo.py"),
    os.path.join(HERE, "..", "demo", "demo_wav.py"),
]
REF = os.path.join(HERE, "..", "reference", "audio_encoder_ref.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_helper_exists_and_honours_the_env_override():
    src = _read(REF)
    assert "def resolve_snap_dir(" in src, "the shared resolver must exist"
    assert "QWEN3ASR_SNAP_DIR" in src, "it must honour the documented env override"


def test_demos_do_not_hardcode_a_snapshot_path():
    for path in DEMOS:
        src = _read(path)
        assert "/root/.cache/huggingface" not in src, f"{path}: hardcoded container path"
        assert "os.listdir(SNAP)" not in src, f"{path}: must not guess the snapshot dir"


def test_demos_load_weights_through_the_resolver():
    # Passing no snap_dir lets load_audio_tower_weights call resolve_snap_dir.
    for path in DEMOS:
        src = _read(path)
        assert "load_audio_tower_weights(" in src, f"{path}: must load the audio tower"
        assert "snap_dir=snap" not in src, f"{path}: must not pre-resolve the snapshot itself"
