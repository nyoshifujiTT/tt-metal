# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The wav prep must not ask for more audio than its default clip holds.

``--dur`` decides how many mel frames and audio tokens the processor emits, and
the TT demo is compared against the CPU baseline clip for clip, so the duration
is part of what the run means. ``prep_wav.py`` defaulted to 20.0 s on an in-repo
clip that is 7.62 s long: ``load_slice`` clamped with ``min(len(w), ...)``, so
nothing raised, and ``summary.json`` recorded ``"dur": 20.0`` next to shapes
derived from 7.62 s of audio.

``reference/dump_reference.py`` already documents the correct choice for the
same file -- ``DEFAULT_DUR = 7.0  # <= the in-repo sample clip (7.62 s)`` -- so
this is the two generators disagreeing about one clip, not a judgement call.
"""

import ast
import os
import struct

import pytest

HERE = os.path.dirname(__file__)
PREP = os.path.join(HERE, "..", "reference", "prep_wav.py")
DUMP = os.path.join(HERE, "..", "reference", "dump_reference.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _const(path, name):
    for node in ast.parse(_read(path)).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {os.path.basename(path)}")


def _default_clip():
    """(path, start, dur) of the single default clip, read out of the source."""
    tree = ast.parse(_read(PREP))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_CLIPS" for t in node.targets
        ):
            (entry,) = node.value.elts
            _name, path_expr, start, dur = entry.elts
            # the path is os.path.join(REPO_ROOT, *parts); rebuild it here
            parts = [ast.literal_eval(a) for a in path_expr.args[1:]]
            repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
            resolved = os.path.join(repo_root, *parts)
            dur_value = dur.id if isinstance(dur, ast.Name) else ast.literal_eval(dur)
            if isinstance(dur_value, str):
                dur_value = _const(PREP, dur_value)
            return resolved, ast.literal_eval(start), dur_value
    raise AssertionError("DEFAULT_CLIPS not found")


def _wav_seconds(path):
    """Duration from the RIFF header, without needing soundfile installed."""
    data = open(path, "rb").read(4096)
    assert data[0:4] == b"RIFF" and data[8:12] == b"WAVE", path
    rate = channels = width = None
    i = 12
    while i + 8 <= len(data):
        cid = data[i : i + 4]
        size = struct.unpack("<I", data[i + 4 : i + 8])[0]
        if cid == b"fmt ":
            _fmt, channels, rate, _br, _al, bits = struct.unpack("<HHIIHH", data[i + 8 : i + 24])
            width = bits // 8
        elif cid == b"data":
            assert rate, "fmt chunk must precede data"
            return size / float(rate * channels * width), rate, channels
        i += 8 + size
    raise AssertionError(f"no data chunk in the first 4 KiB of {path}")


def test_the_default_duration_fits_the_default_clip():
    path, start, dur = _default_clip()
    if not os.path.exists(path):
        pytest.skip(f"in-repo clip not present: {path}")
    seconds, _rate, _ch = _wav_seconds(path)
    assert start + dur <= seconds, (
        f"default asks for {start}+{dur}s of a {seconds:.2f}s clip; "
        f"the recorded duration would not match the audio actually used"
    )


def test_the_default_clip_is_the_16k_mono_file_the_prompt_expects():
    path, _start, _dur = _default_clip()
    if not os.path.exists(path):
        pytest.skip(f"in-repo clip not present: {path}")
    _seconds, rate, channels = _wav_seconds(path)
    # load_slice asserts the file rate is 16 kHz and downmixes; the assert is
    # only ever exercised if the shipped default actually satisfies it.
    assert rate == 16000 and channels == 1, f"{rate} Hz, {channels} ch"


def test_both_generators_agree_on_the_duration_for_that_clip():
    """dump_reference.py and prep_wav.py slice the same file; one answer only."""
    assert _const(PREP, "DEFAULT_DUR") == _const(DUMP, "DEFAULT_DUR")


def test_the_repeatable_clip_flag_uses_the_same_default():
    """--clip name=path omits the duration, so its fallback is a default too."""
    src = _read(PREP)
    body = src[src.index("def parse_clip(") : src.index("def load_slice(")]
    assert "else DEFAULT_DUR" in body, (
        "parse_clip must fall back to the same constant, not a second literal"
    )
    assert "else 20.0" not in body


def test_a_short_clip_is_refused_rather_than_silently_truncated():
    """Clamping produced a summary.json that described audio it did not use."""
    body = _read(PREP)
    body = body[body.index("def load_slice(") : body.index("def main(")]
    assert "raise SystemExit(" in body, "a too-short slice must fail, not clamp"
    assert "is available" in body, "and the message must say what was found"
    assert "return got.copy()" in body


def test_the_refusal_triggers_on_the_case_that_was_shipped(tmp_path):
    """Run the shipped load_slice against a clip shorter than the request."""
    sf = pytest.importorskip("soundfile")
    import numpy as np

    clip = tmp_path / "short.wav"
    sf.write(str(clip), np.zeros(int(7.62 * 16000), dtype="float32"), 16000)

    namespace = {"sf": sf}
    for node in ast.parse(_read(PREP)).body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_slice":
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), PREP, "exec"), namespace)  # noqa: S102
    load_slice = namespace["load_slice"]

    with pytest.raises(SystemExit):
        load_slice(str(clip), 0.0, 20.0)
    # and the corrected default still works on the same file
    assert len(load_slice(str(clip), 0.0, _const(PREP, "DEFAULT_DUR"))) == int(7.0 * 16000)


def _load_slice_of(path):
    """Exec just the shipped load_slice, so the import side of the file is free."""
    sf = pytest.importorskip("soundfile")
    namespace = {"sf": sf}
    for node in ast.parse(_read(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_slice":
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)  # noqa: S102
    assert "load_slice" in namespace, f"no load_slice in {os.path.basename(path)}"
    return namespace["load_slice"]


def test_the_golden_dump_refuses_a_short_clip_too(tmp_path):
    """The same clamp was fixed in prep_wav.py and left in dump_reference.py.

    Measured on the shipped functions with a 3 s wav and ``dur=7.0``:
    prep_wav raised, dump_reference returned 3 s of audio. dump_reference then
    writes ``"dur": args.dur`` into the manifest, so the golden data would be
    described by a duration it was not produced from -- the exact failure the
    prep_wav refusal exists to prevent, and worse here because the PCC tests
    derive their expected shapes from that manifest.

    Both generators are asserted through one helper so a future fix to one of
    them cannot drift from the other.
    """
    import numpy as np

    sf = pytest.importorskip("soundfile")
    clip = tmp_path / "three.wav"
    sf.write(str(clip), np.zeros(3 * 16000, dtype="float32"), 16000)

    for path in (PREP, DUMP):
        load_slice = _load_slice_of(path)
        with pytest.raises(SystemExit) as excinfo:
            load_slice(str(clip), 0.0, 7.0)
        message = str(excinfo.value)
        assert "3.00s is available" in message, (
            f"{os.path.basename(path)} must report what it found: {message}"
        )


def test_the_golden_dump_still_accepts_the_duration_it_ships_with(tmp_path):
    """The refusal must not fire on the shipped default, or nothing can run.

    dump_reference's default clip is 7.62 s and DEFAULT_DUR is 7.0, so the
    request is satisfiable exactly; a refusal written with >= instead of > (or
    against the file length rather than the requested slice) would break the
    documented command.
    """
    import numpy as np

    sf = pytest.importorskip("soundfile")
    clip = tmp_path / "seven62.wav"
    sf.write(str(clip), np.zeros(int(7.62 * 16000), dtype="float32"), 16000)

    load_slice = _load_slice_of(DUMP)
    dur = _const(DUMP, "DEFAULT_DUR")
    assert len(load_slice(str(clip), 0.0, dur)) == int(dur * 16000)


def test_the_golden_dump_still_allows_the_whole_file(tmp_path):
    """``dur=None`` means "to the end" and must stay exempt from the refusal."""
    import numpy as np

    sf = pytest.importorskip("soundfile")
    clip = tmp_path / "two.wav"
    sf.write(str(clip), np.zeros(2 * 16000, dtype="float32"), 16000)

    load_slice = _load_slice_of(DUMP)
    assert len(load_slice(str(clip), 0.0, None)) == 2 * 16000
