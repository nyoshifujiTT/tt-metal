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
CONFTEST = os.path.join(HERE, "conftest.py")
PREP = os.path.join(HERE, "..", "reference", "prep_wav.py")


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


def _env_lookup(src, var):
    """The full os.environ.get(...) expression that resolves ``var``, if any."""
    for line in src.splitlines():
        if f'"{var}"' in line and "os.environ.get(" in line:
            return line.strip()
    return None


def test_the_demos_read_the_prefixed_names_the_readme_exports():
    """The docs say to export QWEN3ASR_*; the demos only read the old names.

    tests/conftest.py already fixed the resolution order -- prefixed name first,
    unprefixed as fallback -- and the README's setup blocks export the prefixed
    ones. demo.py read only GOLDEN_DIR and demo_wav.py only WAV_DIR, so
    following the documented setup left both demos looking somewhere else
    entirely while appearing configured.
    """
    for path, first, fallback in (
        (DEMOS[0], "QWEN3ASR_GOLDEN_DIR", "GOLDEN_DIR"),
        (DEMOS[1], "QWEN3ASR_WAV_DIR", "WAV_DIR"),
    ):
        expr = _env_lookup(_read(path), first)
        assert expr is not None, f"{os.path.basename(path)}: must read {first}"
        assert expr.index(f'"{first}"') < expr.index(f'"{fallback}"'), (
            f"{os.path.basename(path)}: {first} must take precedence over {fallback}"
        )


def test_the_checkpoint_env_follows_the_same_order_in_both_demos():
    for path in DEMOS:
        expr = _env_lookup(_read(path), "QWEN3ASR_TEXT_DECODER")
        assert expr is not None, f"{os.path.basename(path)}: must read QWEN3ASR_TEXT_DECODER"
        assert expr.index('"QWEN3ASR_TEXT_DECODER"') < expr.index('"HF_MODEL"'), (
            f"{os.path.basename(path)}: the prefixed name must win, as in conftest"
        )


def test_the_order_matches_the_one_conftest_established():
    """The demos are following a convention, so it has to still be there."""
    src = _read(CONFTEST)
    for first, fallback in (
        ("QWEN3ASR_GOLDEN_DIR", "GOLDEN_DIR"),
        ("QWEN3ASR_TEXT_DECODER", "HF_MODEL"),
    ):
        expr = _env_lookup(src, first)
        assert expr is not None and expr.index(f'"{first}"') < expr.index(f'"{fallback}"'), (
            f"conftest no longer prefers {first}; the demos copied that order"
        )


def test_the_wav_demo_defaults_where_its_producer_writes():
    """prep_wav.py writes the npz this demo globs; the defaults must agree.

    demo_wav.py defaulted to /ttwork/qwen3_asr_wav, a path that exists only
    inside the bring-up container, while prep_wav.py defaults to
    /tmp/qwen3_asr_wav. Run with no environment set, the producer and the
    consumer used different directories and the demo found no clips.
    """
    consumer = _read(DEMOS[1])
    producer = _read(PREP)
    assert '"/tmp/qwen3_asr_wav"' in consumer, "the consumer must default where the producer writes"
    assert '"/tmp/qwen3_asr_wav"' in producer, "the producer's default moved; re-check the consumer"


def _recipe_lines(src):
    """Lines of the module docstring that read as something to run.

    Prose may name a path while explaining that it was removed; an indented
    command line or an ``export``/``docker``/``python3`` invocation is what a
    reader will actually copy.
    """
    body = src[src.index('"""') + 3 : src.index('"""', src.index('"""') + 3)]
    out = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("export ", "docker ", "python3 ", "pytest ", "pip ", "$ ")):
            out.append(stripped)
        elif line.startswith("  ") and stripped.endswith("\\"):
            out.append(stripped)
    return out


def test_no_demo_defaults_to_a_container_only_path():
    """/ttwork existed only in a bring-up container that is gone.

    It may still be named where a doc comment explains that it was removed, but
    it may not appear in code, nor in a command a reader is told to run --
    demo.py's docstring carried a `docker exec ... -e HF_MODEL=/ttwork/...`
    recipe that could not work anywhere, alongside a README that documents a
    different invocation entirely.
    """
    for path in DEMOS:
        src = _read(path)
        doc_lines = src[src.index('"""') + 3 : src.index('"""', src.index('"""') + 3)].splitlines()
        code = [
            ln
            for ln in src.splitlines()
            if not ln.lstrip().startswith("#") and ln not in doc_lines
        ]
        runnable = [ln for ln in _recipe_lines(src) if "/ttwork" in ln]
        offenders = [ln.strip() for ln in code if "/ttwork" in ln]
        assert not runnable, f"{os.path.basename(path)}: docstring recipe uses /ttwork: {runnable}"
        assert not offenders, f"{os.path.basename(path)}: {offenders}"


def test_the_demo_docstring_points_at_the_documented_artifacts():
    """The recipe in the docstring must be the one the README sets up."""
    src = _read(DEMOS[0])
    recipe = "\n".join(_recipe_lines(src))
    for var in ("QWEN3ASR_GOLDEN_DIR", "QWEN3ASR_TEXT_DECODER"):
        assert var in recipe, f"the recipe must export {var}, which the README also exports"
    assert "qwen3asr-dev" not in recipe, "that container image is not part of this tree"
