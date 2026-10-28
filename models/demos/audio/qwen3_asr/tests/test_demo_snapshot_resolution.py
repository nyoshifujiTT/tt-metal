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


def _compile_fn(path, name, namespace=None):
    """Compile one function out of a demo that cannot be imported here.

    The demos import torch, transformers and the ttnn model modules at module
    level, so extract the single definition through the AST -- the same
    technique test_mel_pin.py uses for pin_mel.
    """
    import ast

    for node in ast.parse(_read(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ns = dict(namespace or {})
            exec(compile(module, path, "exec"), ns)  # noqa: S102 - our own source
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


def _demo_wav_parse_asr():
    import re

    return _compile_fn(
        os.path.join(HERE, "..", "demo", "demo_wav.py"), "parse_asr", {"re": re}
    )


def test_the_wav_demo_splits_the_language_off_the_transcript():
    """demo_wav.py is the one-clip entry point the README sends readers to.

    Its parse_asr had no test, while the identical regex in
    eval/corpus_eval.py was covered. If this stopped matching, the demo would
    print the prompt echo as the transcript and look like a model failure.
    """
    parse = _demo_wav_parse_asr()

    assert parse("<|im_start|>assistant\nlanguage ja<asr_text>こんにちは") == (
        "ja",
        "こんにちは",
    )


def test_the_wav_demo_keeps_a_multi_line_transcript(monkeypatch=None):
    """DOTALL again: without it everything after the first newline is lost."""
    parse = _demo_wav_parse_asr()

    assert parse("language ja<asr_text>一行目\n二行目") == ("ja", "一行目\n二行目")


def test_the_wav_demo_falls_back_without_losing_the_text():
    """No tag means nothing to split; the text must survive with an empty language.

    Returning ("", "") here would print an empty transcript for output the
    model did produce.
    """
    parse = _demo_wav_parse_asr()

    assert parse("  素の転写だけ  ") == ("", "素の転写だけ")
    assert parse("") == ("", "")


def test_both_parsers_agree_on_the_transcript():
    """Two copies of one protocol; a fix to either must not split them.

    corpus_eval.py returns the transcript alone and demo_wav.py returns it
    beside the language, but the transcript itself has to be identical -- the
    demo and the corpus eval are compared against each other on the same
    clips.
    """
    import re

    eval_parse = _compile_fn(
        os.path.join(HERE, "..", "eval", "corpus_eval.py"), "parse_asr", {"re": re}
    )
    demo_parse = _demo_wav_parse_asr()

    for decoded in (
        "<|im_start|>assistant\nlanguage ja<asr_text>こんにちは",
        "language ja<asr_text>一行目\n二行目",
        "language ja<asr_text>まえ<asr_text>あと",
        "  素の転写だけ  ",
        "",
    ):
        assert demo_parse(decoded)[1] == eval_parse(decoded), (
            f"the two parsers disagree on {decoded!r}: "
            f"{demo_parse(decoded)[1]!r} vs {eval_parse(decoded)!r}"
        )


def test_the_two_parsers_use_the_same_pattern():
    """Compare the source too: agreeing on five cases is not agreeing always."""
    import re

    pattern = re.compile(r'm = re\.search\(\s*r"([^"]+)"')
    found = {}
    for rel in (("demo", "demo_wav.py"), ("eval", "corpus_eval.py")):
        src = _read(os.path.join(HERE, "..", *rel))
        match = pattern.search(src)
        assert match, f"{rel[-1]}: the parse pattern must stay greppable"
        found[rel[-1]] = match.group(1)

    assert len(set(found.values())) == 1, (
        f"the two parsers no longer share a pattern: {found}"
    )


GENERATOR = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def _adapter_resolvers(tmp_env):
    """The adapter's two snapshot helpers, compiled with a shared namespace.

    _resolve_audio_snapshot calls _is_full_asr_snapshot, so they have to be
    compiled into one namespace rather than extracted separately.
    """
    import ast
    import os as _os

    wanted = {"_is_full_asr_snapshot", "_resolve_audio_snapshot"}
    ns = {"os": _os}
    found = {}
    for node in ast.parse(_read(GENERATOR)).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, GENERATOR, "exec"), ns)  # noqa: S102 - our own source
            found[node.name] = ns[node.name]
    assert wanted <= set(found), f"missing {wanted - set(found)} in generator_vllm.py"
    return found["_is_full_asr_snapshot"], found["_resolve_audio_snapshot"]


def _write_config(tmp_path, model_type):
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(str(tmp_path), "config.json"), "w") as fh:
        json.dump({"model_type": model_type}, fh)
    return str(tmp_path)


def test_a_full_snapshot_is_told_apart_from_an_extracted_decoder(tmp_path):
    """This one boolean picks the branch that decides whether to extract.

    _ensure_text_decoder returns the directory untouched when this is False
    and extracts the thinker decoder when it is True, so getting it wrong
    either feeds the audio tower's checkpoint to a plain-Qwen3 loader or
    re-extracts something that is already a decoder.
    """
    is_full, _ = _adapter_resolvers(tmp_path)

    assert is_full(_write_config(tmp_path / "asr", "qwen3_asr")) is True
    assert is_full(_write_config(tmp_path / "plain", "qwen3")) is False


def test_an_unreadable_directory_is_not_a_full_snapshot(tmp_path):
    """Absent or malformed config.json must answer False, not raise.

    The caller uses this to decide whether HF_MODEL is usable at all; an
    exception here would abort start-up instead of falling through to the
    model path.
    """
    is_full, _ = _adapter_resolvers(tmp_path)

    assert is_full(str(tmp_path / "does-not-exist")) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_full(str(empty)) is False

    broken = tmp_path / "broken"
    broken.mkdir()
    with open(os.path.join(str(broken), "config.json"), "w") as fh:
        fh.write("{not json")
    assert is_full(str(broken)) is False


def test_the_explicit_override_wins_over_everything(monkeypatch, tmp_path):
    """QWEN3ASR_AUDIO_SNAPSHOT is the documented escape hatch."""
    _, resolve = _adapter_resolvers(tmp_path)

    monkeypatch.setenv("QWEN3ASR_AUDIO_SNAPSHOT", "/explicit")
    monkeypatch.setenv("HF_MODEL", _write_config(tmp_path / "asr", "qwen3_asr"))

    assert resolve(_config_stub("/from-vllm")) == "/explicit"


def test_hf_model_is_used_when_it_holds_a_full_snapshot(monkeypatch, tmp_path):
    """The spec states this: the served snapshot is reused for the audio tower.

    workflows/model_specs/dev/audio_tts.yaml says QWEN3ASR_AUDIO_SNAPSHOT
    "falls back to HF_MODEL", which is why a deployment sets neither.
    """
    _, resolve = _adapter_resolvers(tmp_path)
    snapshot = _write_config(tmp_path / "asr", "qwen3_asr")

    monkeypatch.delenv("QWEN3ASR_AUDIO_SNAPSHOT", raising=False)
    monkeypatch.setenv("HF_MODEL", snapshot)

    assert resolve(_config_stub("/from-vllm")) == snapshot


def test_an_extracted_hf_model_is_skipped_for_the_model_path(monkeypatch, tmp_path):
    """HF_MODEL pointing at a plain decoder has no audio tower in it.

    Taking it anyway would send the audio encoder at a checkpoint with no
    audio_tower.* weights -- a load failure at best, and the reason the
    is-full check exists rather than "HF_MODEL if set".
    """
    _, resolve = _adapter_resolvers(tmp_path)

    monkeypatch.delenv("QWEN3ASR_AUDIO_SNAPSHOT", raising=False)
    monkeypatch.setenv("HF_MODEL", _write_config(tmp_path / "plain", "qwen3"))

    assert resolve(_config_stub("/from-vllm")) == "/from-vllm"


def test_the_model_path_is_read_under_either_attribute_name(monkeypatch, tmp_path):
    """HF configs expose the path as _name_or_path or name_or_path."""
    _, resolve = _adapter_resolvers(tmp_path)

    monkeypatch.delenv("QWEN3ASR_AUDIO_SNAPSHOT", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)

    assert resolve(_config_stub("/underscored")) == "/underscored"

    from types import SimpleNamespace

    assert resolve(SimpleNamespace(name_or_path="/plain")) == "/plain"
    assert resolve(SimpleNamespace()) == ""


def _config_stub(path):
    from types import SimpleNamespace

    return SimpleNamespace(_name_or_path=path)


def test_the_spec_still_claims_the_fallback_these_tests_pin():
    """If the spec stops promising it, these rules should be revisited.

    The deployment sets neither variable and relies on the HF_MODEL
    fallback; that promise lives in the tt-inference-server spec, so read it
    rather than restating it here.
    """
    spec = os.path.join(
        HERE, "..", "..", "..", "..", "..", "..",
        "tt-inference-server", "workflows", "model_specs", "dev", "audio_tts.yaml",
    )
    if not os.path.exists(spec):
        import pytest

        pytest.skip("tt-inference-server is not checked out beside this repo")

    text = _read(spec)
    assert "QWEN3ASR_AUDIO_SNAPSHOT falls back to HF_MODEL" in text, (
        "the spec no longer documents the fallback these tests pin"
    )
