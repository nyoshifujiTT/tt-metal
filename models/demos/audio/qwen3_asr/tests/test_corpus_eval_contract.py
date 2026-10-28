# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demo-vs-served comparison must be reproducible from the repo.

The parity claim for this model is "same corpus CER", not "same string", so the
script that measures it has to live next to the demo rather than in someone's
home directory. Pin its interface and the invariants it has to share with
generator_vllm, since a drift in either silently invalidates the comparison.
"""

import os
import re

import pytest

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "eval", "corpus_eval.py")
TT = os.path.join(HERE, "..", "tt")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_corpus_eval_script_exists_and_is_documented():
    assert os.path.isfile(EVAL), "eval/corpus_eval.py must be committed"
    readme = _read(os.path.join(HERE, "..", "README.md"))
    assert "eval/corpus_eval.py" in readme, "the README must document how to run it"


def test_corpus_eval_exposes_the_expected_cli():
    src = _read(EVAL)
    for flag in ("--manifest", "--snapshot", "--ckpt", "--output"):
        assert f'"{flag}"' in src, f"{flag} must stay part of the CLI"


def test_corpus_eval_mirrors_the_served_audio_token_count():
    # generator_vllm relies on vLLM's _get_feat_extract_output_lengths; a
    # different count here would mean the two paths see different prompts and
    # the CER comparison would be meaningless.
    src = _read(EVAL)
    assert "(T // 100) * 13" in src, "audio-token count must follow the vLLM formula"


def test_corpus_eval_pins_the_mel_frames_like_the_served_path():
    src = _read(EVAL)
    assert "QWEN3ASR_MEL_PIN" in src, "must honour the same mel pin as generator_vllm"
    assert "QWEN3ASR_MEL_PIN" in _read(os.path.join(TT, "generator_vllm.py"))


def _calls(src, name):
    """Every call to ``name`` in ``src``, found through the AST.

    Matching the bare name in the source text also matches the import line, so
    a call replaced by something else stays undetected -- which is exactly
    what happened to PagedAttentionConfig.
    """
    import ast

    return [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and (
            getattr(node.func, "id", None) == name
            or getattr(node.func, "attr", None) == name
        )
    ]


def test_corpus_eval_runs_paged_kv_by_default():
    # The served path always allocates a paged KV cache, so decode dispatches to
    # paged_scaled_dot_product_attention_decode. Running the demo non-paged uses a
    # different kernel and yields different transcripts, which would make the CER
    # comparison measure the kernel choice rather than the front-end.
    src = _read(EVAL)
    # Not `"PagedAttentionConfig" in src`: the import line matches that too, so
    # replacing the call with anything else kept the test green. Require a call.
    assert _calls(src, "PagedAttentionConfig"), (
        "the eval must CALL PagedAttentionConfig, not merely import it"
    )
    assert 'os.environ.get("QWEN3ASR_EVAL_PAGED_KV", "1")' in src, "paged KV must be the default"
    assert "use_paged_kv_cache=PAGED_KV" in src, "the decoder must be built with paged KV"
    assert "page_table=page_table" in src, "generate() must be driven with the page table"


def test_corpus_eval_page_block_matches_the_served_launch():
    # tt-inference-server launches vLLM with --block_size 64; the demo has to use
    # the same block size or the paged kernel sees a different cache geometry.
    src = _read(EVAL)
    assert 'os.environ.get("QWEN3ASR_EVAL_PAGE_BLOCK", "64")' in src


def test_corpus_eval_normalises_like_the_served_eval_client():
    # The served eval client (asr_ja_eval.py) scores CER after NFKC + punctuation
    # stripping. Scoring the demo with a whitespace-only normalisation compares
    # two different metrics, which is what made the demo look 8 CER points worse.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_corpus_eval_norm", os.path.abspath(EVAL))
    src = _read(EVAL)
    assert "unicodedata.normalize" in src, "must NFKC-normalise like the served eval"
    for ch in ("、", "。", "「", "・"):
        assert ch in src, f"punctuation {ch} must be stripped before CER"
    assert spec is not None


def test_corpus_eval_norm_matches_reference_cases():
    # Behavioural check of the normalisation itself, so a future edit to the regex
    # cannot silently drift from the served client.
    import re
    import unicodedata

    src = _read(EVAL)
    match = re.search(r"_NORM_STRIP = re\.compile\((r\"[^\n]*\")\)", src)
    assert match, "the normalisation regex must stay greppable"
    pattern = re.compile(eval(match.group(1)))  # noqa: S307 - literal from our own source

    def norm(text):
        return pattern.sub("", unicodedata.normalize("NFKC", text)).strip()

    assert norm("周りを見ると。") == norm("周りを見ると")
    assert norm("ＡＢＣ") == "ABC"
    assert norm("え、あの・そう") == "えあのそう"


def test_corpus_eval_resamples_before_the_extractor():
    """sampling_rate=16000 is a declaration, not a conversion request.

    WhisperFeatureExtractor runs a 400-sample FFT with a 160-sample hop over
    whatever array it is handed; it does not resample. Handing it a 44.1 kHz
    clip while declaring 16 kHz stretches the clip 2.76x (a 10.44 s clip yields
    2878 mel frames instead of 1044) and shifts every formant, so both the
    transcript and the RTF come out wrong.
    """
    src = _read(EVAL)
    assert "def resample_to_16k(" in src, "the eval must convert, not assume"
    assert "wav, sr = resample_to_16k(wav, sr)" in src, (
        "the conversion must happen on the read path, before the extractor"
    )
    read_at = src.index("wav, sr = sf.read(")
    resample_at = src.index("wav, sr = resample_to_16k(wav, sr)")
    extract_at = src.index("fe([wav], sampling_rate=16000")
    assert read_at < resample_at < extract_at, (
        "resampling must sit between the read and the feature extraction"
    )


def test_corpus_eval_measures_duration_after_resampling():
    """clip_seconds must not be derived from a rate the wav no longer has."""
    src = _read(EVAL)
    resample_at = src.index("wav, sr = resample_to_16k(wav, sr)")
    dur_at = src.index("clip_seconds = len(wav) / float(sr)")
    assert resample_at < dur_at, (
        "duration must be computed from the resampled wav and its new rate"
    )


def test_corpus_eval_sets_hf_model_from_ckpt():
    """ModelArgs reads HF_MODEL, so the eval has to supply it.

    The README's command passes --ckpt and does not mention HF_MODEL. Run in a
    clean shell it died in ModelArgs with "Please set HF_MODEL to a HuggingFace
    name ..." before transcribing anything -- the documented invocation simply
    did not work. --ckpt already names the extracted text decoder, which is what
    ModelArgs wants, so the script derives it rather than asking the caller for
    the same path twice.
    """
    src = _read(EVAL)
    assert 'os.environ["HF_MODEL"] = a.ckpt' in src, (
        "HF_MODEL must be set from --ckpt before ModelArgs is constructed"
    )
    # ...and only around construction: the audio tower resolves its own paths
    # and must not observe the decoder directory.
    assert 'prev_hf_model = os.environ.get("HF_MODEL")' in src
    assert 'os.environ["HF_MODEL"] = prev_hf_model' in src
    set_at = src.index('os.environ["HF_MODEL"] = a.ckpt')
    model_args_at = src.index("ModelArgs(dev,")
    restore_at = src.index('os.environ["HF_MODEL"] = prev_hf_model')
    assert set_at < model_args_at < restore_at, (
        "HF_MODEL must be set before ModelArgs and restored after it"
    )


def test_corpus_eval_matches_the_served_path_handling_of_hf_model():
    """generator_vllm.py does the same save/set/restore; keep them in step.

    If one of the two front-ends leaves HF_MODEL pointing at the decoder while
    the other does not, they no longer resolve the audio tower the same way and
    the corpus-CER comparison stops being like-for-like.
    """
    served = _read(os.path.join(TT, "generator_vllm.py"))
    for fragment in (
        'prev_hf_model = os.environ.get("HF_MODEL")',
        'os.environ["HF_MODEL"] = prev_hf_model',
    ):
        assert fragment in served, f"served path lost its HF_MODEL handling: {fragment}"


def test_readme_states_the_environment_the_eval_needs():
    """The documented command has to be runnable as written.

    corpus_eval.py reads os.environ["TT_METAL_HOME"] at import time to find its
    own reference/ and tt/ packages, and the command used $QWEN3ASR_SNAP_DIR
    without ever saying what it is. Neither was mentioned, so a reader hit a
    KeyError or an empty --snapshot.
    """
    src = _read(EVAL)
    assert 'os.environ["TT_METAL_HOME"]' in src, "guard is about this hard read"

    readme = _read(os.path.join(HERE, "..", "README.md"))
    body = readme[readme.index("## Corpus eval") :]
    assert "export TT_METAL_HOME=" in body
    assert "QWEN3ASR_SNAP_DIR=" in body, "the snapshot variable must be defined"
    # and the reader must be told which tree each path is
    assert "extract_text_decoder.py" in body


def test_readme_says_hf_model_is_handled_by_the_script():
    """Otherwise the next reader re-adds HF_MODEL to the command by hand.

    Setting it globally is not harmless: the audio tower resolves its own paths
    from the snapshot, so a stray HF_MODEL pointing at the decoder changes what
    is compared.
    """
    readme = _read(os.path.join(HERE, "..", "README.md"))
    body = readme[readme.index("## Corpus eval") :]
    assert "HF_MODEL" in body
    assert "--ckpt" in body


def test_every_golden_the_tests_load_has_a_producer():
    """A golden no generator writes makes its test unrunnable.

    test_decoder.py reads inputs_embeds.npy, which dump_reference.py cannot
    produce -- it is an INPUT to the language model, so a forward hook on that
    model never sees it. extract_text_decoder.py captures it from a pre-hook
    instead. The check therefore has to span both generators; asking only
    dump_reference.py for it led me to duplicate working code.
    """
    import re

    produced = set()
    for name in ("dump_reference.py", "extract_text_decoder.py"):
        src = _read(os.path.join(HERE, "..", "reference", name))
        produced |= set(re.findall(r'"([a-z0-9_]+)\.npy"', src))
        # dump_reference writes f"{k}.npy" for every hooked stage
        produced |= set(re.findall(r'targets\["([a-z0-9_]+)"\]\s*=', src))
        # extract_text_decoder writes via s("<name>", tensor)
        produced |= set(re.findall(r'\bs\("([a-z0-9_]+)",', src))

    loaded = set()
    for name in ("test_decoder.py", "test_audio_encoder.py"):
        src = _read(os.path.join(HERE, name))
        loaded |= set(re.findall(r'golden\("([a-z0-9_]+)\.npy"\)', src))

    missing = loaded - produced
    assert not missing, (
        f"tests load goldens no generator writes: {sorted(missing)}; "
        "they can only skip or error"
    )


def test_the_readme_says_to_run_both_generators():
    """Running only dump_reference.py yields a golden dir the decoder rejects.

    The reference-golden section described one script. Following it and then
    running the suite with QWEN3ASR_REQUIRE_ARTIFACTS=1 errors all three decoder
    tests on "golden tensor not found: inputs_embeds.npy" -- the file lives in
    the other generator's output.
    """
    readme = _read(os.path.join(HERE, "..", "README.md"))
    body = readme[readme.index("## Reference golden") :]
    # A prose mention is not a recipe: the name appears elsewhere in the file
    # too, so require it as a command inside the section's runnable block.
    block = body[body.index("```bash") : body.index("```", body.index("```bash") + 7)]
    for generator in ("dump_reference.py", "extract_text_decoder.py"):
        assert (
            f"models/demos/audio/qwen3_asr/reference/{generator}" in block
        ), f"the golden recipe must actually invoke {generator}"
    # ...and into the same directory, or the tests see half a golden set
    assert "QWEN3ASR_GOLDEN_DIR=" in block
    assert "inputs_embeds.npy" in body, "say which golden it is that comes from it"
    assert "QWEN3ASR_TEXT_DECODER" in body


def test_dump_reference_does_not_duplicate_the_inputs_embeds_capture():
    """One golden, one producer.

    Both scripts hooking the text model would give two writers for
    inputs_embeds.npy into the same dir, and whichever ran last would win --
    silently, and with no guarantee the two agree.
    """
    dumper = _read(os.path.join(HERE, "..", "reference", "dump_reference.py"))
    assert "inputs_embeds" not in dumper, (
        "inputs_embeds belongs to extract_text_decoder.py; dump_reference.py "
        "must not write it too"
    )


def test_readme_gives_a_runnable_e2e_invocation():
    """QWEN3ASR_E2E_WAV=<16k-mono.wav> is a placeholder, not a command.

    The e2e test hard-fails without it under QWEN3ASR_REQUIRE_ARTIFACTS=1, and
    nothing said which clip to use -- while dump_reference.py already defaults
    to an in-repo one whose transcription the goldens record.
    """
    readme = _read(os.path.join(HERE, "..", "README.md"))
    assert "17646385371758249908.wav" in readme, (
        "the README must name the in-repo clip the e2e test can run on"
    )
    assert 'QWEN3ASR_E2E_TEXT="driver of the vehicle"' in readme


def test_extract_text_decoder_accepts_a_snapshots_parent():
    """QWEN3ASR_SNAP_DIR is naturally spelled as the hub's snapshots/ dir.

    That directory holds a <rev>/ subdirectory, not the weights, so the globs
    matched nothing and the extractor produced an empty checkpoint while
    printing success. The failure surfaced three steps later in the decoder
    tests as "No fallback tokenizer found for base model", which points at the
    wrong thing entirely.
    """
    src = _read(os.path.join(HERE, "..", "reference", "extract_text_decoder.py"))
    body = src[src.index("def snap_dir("):]
    body = body[: body.index("\ndef ", 1)]
    assert 'glob.glob(os.path.join(env, "*.safetensors"))' in body, (
        "the env value must be probed for weights before it is trusted"
    )
    assert 'os.path.join(d, "*.safetensors")' in body, (
        "and the revision subdirectory used when it is the one holding them"
    )


def test_extract_text_decoder_refuses_to_write_an_empty_checkpoint():
    """A 0-tensor checkpoint has no valid use; failing later hides the cause."""
    src = _read(os.path.join(HERE, "..", "reference", "extract_text_decoder.py"))
    assert "if not sd:" in src and "raise SystemExit" in src, (
        "an empty extraction must fail where it happens"
    )
    # ...and before the file is written, or the bad artifact is left behind
    guard_at = src.index("if not sd:")
    save_at = src.index('save_file(sd, os.path.join(out_dir, "model.safetensors")')
    assert guard_at < save_at, "the guard must precede save_file"


def test_the_decode_trace_comment_cites_the_right_hang_issue():
    """The adapter's own comment carried the same misattribution as the runbook.

    #40592 is "[Mistral] Intermittent device hang during AllGatherAsync on
    T3K" -- a CCL hang, and this deployment runs one device with CCL
    short-circuited, so it cannot be the failure that gates decode tracing.
    Citing it as "untraced eager decode still hangs per #40592" pointed the
    next reader at an unrelated thread.

    #37543 is the one that matches: ND, SDPA decode, traced, vLLM-only.
    """
    src = _read(os.path.join(HERE, "..", "tt", "generator_vllm.py"))
    head = src[: src.index("class ") if "class " in src else 4000]

    assert "37543" in head, "cite the issue whose signature matches"
    # Whitespace-normalised: the disclaimer's line wrapping is not the point,
    # and pinning it to one break made this fail on a reflow.
    assert "#40592" not in head or "cited here in error" in " ".join(head.split()), (
        "#40592 is an AllGatherAsync hang; do not present it as this failure"
    )
    # the deterministic near-match must stay marked as such
    assert "45052" in head and "not the non-determinism" in " ".join(head.split())


def test_the_comment_does_not_blame_pr_44118_for_the_deterministic_hang():
    """"a PR #44118 regression" claims more than #45052 establishes.

    Checked against the tracker: #44118's merge (7eff69a85a0) is the *first bad
    tested version*, with 747215b last known good, and triage explicitly says
    causality is unproven and names #43682 as the better bisection target.
    Writing it as the regression's cause sends the next reader to revert or
    study the wrong PR.

    The same paragraph also asserted the issue is "P300x2-specific". The
    *report* is P300x2-only, but the underlying sparse-matmul deadlock is
    described as architecture-agnostic in #45943. What actually rules it out
    here is narrower and more durable: that issue's stuck op is GPT-OSS MoE
    SparseMatmulDeviceOperation on a (1,4) mesh, and this deployment is one
    device with no MoE.
    """
    src = _read(os.path.join(HERE, "..", "tt", "generator_vllm.py"))
    head = src[: src.index("class ") if "class " in src else 4000]
    flat = " ".join(head.split())

    assert "a PR #44118 regression" not in flat, (
        "#44118 is a bisection boundary for #45052, not a proven cause"
    )
    assert "first bad tested version" in flat, "say what #44118 actually is"
    assert "43682" in flat, "name the target triage actually points at"
    # and the reason #45052 does not apply here, stated by mechanism
    assert "MoE" in flat and "SparseMatmul" in flat, (
        "rule the issue out by its op and topology, not by board name alone"
    )


def test_readme_paths_resolve():
    """Every in-tree file the README names by path must exist.

    The sibling runbook in tt-inference-server had exactly this defect: an
    Install command referenced the systemd unit by bare filename, resolving
    from no directory the document ever cds to. That check found it; this is
    the same guard for the model-side README, which names 21 paths today
    (tests, reference scripts, the demo, the server, tt/ modules).
    """
    import re

    readme = _read(os.path.join(HERE, "..", "README.md"))
    model_dir = os.path.abspath(os.path.join(HERE, ".."))
    repo_root = os.path.abspath(os.path.join(model_dir, "..", "..", "..", ".."))

    candidates = set(
        re.findall(
            r"(?:^|\s|`)((?:models|tests|reference|eval|tt|demo|server|docs)"
            r"/[A-Za-z0-9_./-]+?\.(?:py|md|txt|json|wav))(?=[\s`)]|$)",
            readme,
            re.M,
        )
    )
    assert candidates, "the README does name in-tree files; keep this meaningful"

    missing = sorted(
        p
        for p in candidates
        # repo-relative (models/...) or model-relative (tests/..., tt/...)
        if not os.path.exists(os.path.join(repo_root, p))
        and not os.path.exists(os.path.join(model_dir, p))
    )
    assert not missing, f"named in the README but absent from the tree: {missing}"


def _readme_shape_table():
    readme = _read(os.path.join(HERE, "..", "README.md"))
    start = readme.index("Verified shapes at the defaults")
    return " ".join(readme[start : readme.index("\n## ", start)].split())


def test_the_readme_quotes_the_shapes_the_default_run_produces():
    """The section listed 12 s shapes while DEFAULT_DUR was 7.0.

    conv_out (12,13,1024) / audio embeds (156,2048) / prefill logits
    (1,174,151936) cannot be produced by the command the README documents --
    dump_reference.py defaults to a 7.0 s slice, and the goldens on disk are
    (7,13,1024) / (91,2048) / (1,109,151936). A reader checking their own dump
    against those numbers would conclude their run was wrong.
    """
    body = _readme_shape_table()

    # the default this section is claimed for, so shapes and duration travel together
    assert "7.0 s" in body, "state the duration the shapes belong to"

    for shape in ("(7, 13, 1024)", "(91, 2048)", "(109, 2048)", "(1, 109, 151936)"):
        assert shape in body, f"{shape} is what the default run produces; quote it"

    # the stale numbers must not come back
    for stale in ("(12,13,1024)", "(156,2048)", "(1,174,151936)"):
        assert stale not in body, f"{stale} belongs to a 12 s clip, not to the default"


def test_the_readme_explains_how_the_shapes_follow_from_the_clip():
    """Bare shapes cannot be checked against a different --dur.

    7 whole 1 s chunks x 13 encoder rows = 91 audio rows, and inputs_embeds is
    those plus the 18-token prompt template = 109. With the arithmetic a reader
    can predict their own dump; without it they can only compare literals.
    """
    body = _readme_shape_table()
    assert "7 whole 1 s chunks" in body and "13 encoder rows = 91" in body, (
        "show how the audio row count comes from the clip length"
    )
    assert "18-token prompt template" in body, "account for the gap up to 109"
    assert "--dur" in body, "warn that the numbers move with the duration"


def test_the_default_duration_is_still_what_the_readme_says():
    """If DEFAULT_DUR moves, every shape above is stale again."""
    src = _read(os.path.join(HERE, "..", "reference", "dump_reference.py"))
    assert "DEFAULT_DUR = 7.0" in src, (
        "dump_reference's default changed; requote the shapes in the README"
    )


def test_the_golden_tensors_match_the_readme_shapes():
    """Check the dump on disk, not just the prose. Skips when absent."""
    import numpy as np

    golden_dir = os.environ.get("QWEN3ASR_GOLDEN_DIR")
    if not golden_dir or not os.path.isdir(golden_dir):
        pytest.skip("QWEN3ASR_GOLDEN_DIR not set or absent")

    expected = {
        "conv_out.npy": (7, 13, 1024),
        "audio_tower.npy": (91, 2048),
        "proj2.npy": (91, 2048),
        "inputs_embeds.npy": (109, 2048),
        "lm_head.npy": (1, 109, 151936),
    }
    for name, shape in expected.items():
        path = os.path.join(golden_dir, name)
        if not os.path.isfile(path):
            pytest.skip(f"{name} not in the golden dir")
        got = tuple(np.load(path, mmap_mode="r").shape)
        assert got == shape, f"{name}: dump is {got}, README says {shape}"

    # and the arithmetic the README states, taken from the tensors themselves
    audio_rows = np.load(os.path.join(golden_dir, "audio_tower.npy"), mmap_mode="r").shape[0]
    embed_rows = np.load(os.path.join(golden_dir, "inputs_embeds.npy"), mmap_mode="r").shape[0]
    assert audio_rows == 7 * 13
    assert embed_rows - audio_rows == 18, "the prompt template is 18 tokens"


def test_the_readme_lists_every_manifest_key_the_reader_accepts():
    """"Each manifest line is {"wav", "ref"}" is narrower than the code.

    corpus_eval resolves the audio path as
        wav -> audio -> audio_filepath -> path
    and the reference as
        ref -> text -> reference -> ""
    so an existing corpus usually needs no rewriting. Documenting only the
    canonical pair sends the reader off to convert manifests that would have
    worked, and hides the two sharp edges: a line with none of the path keys
    raises KeyError (not a skip), and a missing reference scores as "" rather
    than being dropped.
    """
    src = _read(EVAL)
    # the resolution order, as the code actually spells it
    assert 'it.get("wav") or it.get("audio") or it.get("audio_filepath") or it["path"]' in src
    assert 'it.get("ref") or it.get("text") or it.get("reference") or ""' in src

    readme = _read(os.path.join(HERE, "..", "README.md"))
    body = readme[readme.index("Each manifest line is") :]
    body = body[: body.index("| knob |")]
    for key in ("wav", "audio", "audio_filepath", "path"):
        assert f"`{key}`" in body, f"the reader accepts {key}; document it"
    for key in ("ref", "text", "reference"):
        assert f"`{key}`" in body, f"the reader accepts {key}; document it"
    assert "KeyError" in body, (
        "a line with no path key raises rather than being skipped; say so"
    )


def _env_defaults(src):
    """Every os.environ.get("QWEN3ASR_...", "<literal>") in a source file."""
    import re

    pattern = re.compile(
        r'os\.environ\.get\(\s*"(QWEN3ASR_[A-Z0-9_]+)"\s*,\s*"([^"]*)"\s*\)'
    )
    return dict(pattern.findall(src))


def _readme_knob_table():
    readme = _read(os.path.join(HERE, "..", "README.md"))
    table = readme[readme.index("| knob | default | why |") :]
    return table[: table.index("\n\n")]


def _table_default_cell(row):
    """The `default` column of a knob row, not the whole row.

    Matching the default anywhere in the row lets the `why` column stand in
    for it: the decode-trace row ends "Set `0` only where ...", so a code
    default flipped from 1 to 0 was satisfied by that `0` and the mutation
    survived. Only the second cell states the default.
    """
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    assert len(cells) >= 2, f"not a knob row: {row}"
    return cells[1]


def test_the_readme_knob_table_lists_every_knob_the_eval_reads():
    """Discover the knobs; do not restate them here.

    This test used to check four hand-listed names, so the table could -- and
    did -- fall behind the script: QWEN3ASR_EVAL_PAGE_MAX_BLOCKS (the size of
    the paged KV pool the eval allocates) and QWEN3ASR_MEL_PIN (the mel
    padding every clip is pinned to, which the served path reads as well) were
    both absent from the README while the code read them.

    A knob nobody documents is a knob whose default nobody can check, and the
    reason this file gives for pinning the defaults -- "a default that drifts
    silently changes what the eval measures" -- applies to all of them.
    """
    defaults = _env_defaults(_read(EVAL))
    assert defaults, "the scan found no knobs at all; the pattern has gone stale"

    table = _readme_knob_table()
    missing = [env for env in sorted(defaults) if f"`{env}`" not in table]
    assert not missing, (
        f"the README knob table does not mention {missing}; the eval reads them"
    )


def test_the_readme_knob_defaults_match_the_code():
    """A default that drifts silently changes what the eval measures."""
    defaults = _env_defaults(_read(EVAL))
    table = _readme_knob_table()

    for env, default in sorted(defaults.items()):
        row = [line for line in table.splitlines() if f"`{env}`" in line]
        assert len(row) == 1, f"{env} must have exactly one row, found {len(row)}"
        assert _table_default_cell(row[0]) == f"`{default}`", (
            f"{env}'s default is {default!r} in the code; the README row says "
            f"otherwise: {row[0]}"
        )


def test_the_mel_pin_default_is_shared_with_the_served_path():
    """Both front-ends must pin mel to the same number of frames.

    corpus_eval.py exists to compare the demo against a served run on the same
    clips. The mel padding fixes the encoder's input shape, so if the two read
    the same variable but disagree on its default, an unset environment gives
    two different encoders and the comparison stops meaning anything.
    """
    eval_default = _env_defaults(_read(EVAL))["QWEN3ASR_MEL_PIN"]
    served_default = _env_defaults(
        _read(os.path.join(TT, "generator_vllm.py"))
    )["QWEN3ASR_MEL_PIN"]

    assert eval_default == served_default, (
        f"corpus_eval pins {eval_default} mel frames, generator_vllm pins "
        f"{served_default}; an unset QWEN3ASR_MEL_PIN then runs two encoders"
    )

    assert "generator_vllm" in _readme_knob_table(), (
        "the README row must say the served path reads this knob too"
    )


def test_tt_metal_home_is_required_at_import_time():
    """The README says it raises KeyError without it -- verified by running it.

        $ env -u TT_METAL_HOME python eval/corpus_eval.py --help
        KeyError: 'TT_METAL_HOME'

    It is a module-level subscript, so it fires before argparse: even --help
    fails. That is worth keeping documented, because the failure looks like a
    broken checkout rather than a missing export.
    """
    src = _read(EVAL)
    assert 'os.environ["TT_METAL_HOME"]' in src, (
        "the hard requirement must stay a subscript, or the documented KeyError changes"
    )
    readme = _read(os.path.join(HERE, "..", "README.md"))
    assert "raises `KeyError` without it" in readme


COLLISION_DOC = os.path.join(HERE, "..", "docs", "prefill_program_cache_collision_issue.md")


def test_the_collision_doc_line_references_still_resolve():
    """Every `file.py:NNN` in the issue draft must point at what it claims.

    The doc walks a reader through tt_transformers to show why two padded
    prefill lengths share a program-cache entry. Its line numbers were written
    against an older tree and every one of them had drifted: mlp.py:135-137 had
    become 180-182, model_config.py:555 -> 560, :1988 -> :1993, mlp.py:275 ->
    321, and attention.py:1156 -> 1286. A reader following them lands on
    unrelated code and cannot check the argument at all -- which is the whole
    point of an issue draft meant for upstream.

    Pinning them here means an upstream rebase fails this test instead of
    silently invalidating the filing.

    That is exactly what happened next. Rebasing onto the squash-merge of
    tt-metal #49104 moved them a second time: mlp.py:180 -> 193,
    model_config.py:560 -> 657, :1993 -> 2125, mlp.py:321 -> 368, and
    attention.py:1286 -> 1315. The test caught it (the cited mlp.py:180 had
    become a `ttnn.concat` call), which is the behaviour this pin exists for.
    Expect to update both this list and the doc on every upstream move.
    """
    doc = _read(COLLISION_DOC)

    # (cited path, cited line, a substring that line must contain)
    expected = [
        ("models/tt_transformers/tt/mlp.py", 193, "prefill_len_cutoff"),
        ("models/tt_transformers/tt/model_config.py", 657, "prefill_len_cutoff = 512"),
        ("models/tt_transformers/tt/model_config.py", 2125, "get_attn_wo_program_config"),
        ("models/tt_transformers/tt/mlp.py", 368, "minimal_matmul"),
        ("models/tt_transformers/tt/attention.py", 1315, "get_attn_wo_program_config"),
        # The grid citations: declared here, applied there.
        ("models/tt_transformers/tt/model_config.py", 894, "mlp1_3_grid"),
        ("models/tt_transformers/tt/model_config.py", 989, "mlp1_3_grid"),
    ]

    repo_root = os.path.join(HERE, "..", "..", "..", "..", "..")

    # Cover *every* `file.py:NNN` the doc cites, not just the ones listed
    # above. The doc cites mlp.py twice (the walkthrough and the reference
    # list), so reverting only one of them to a stale number left this test
    # green -- verified by mutation. Requiring the listed set to equal the
    # cited set means a half-updated doc fails here.
    cited = set()
    for base_name in ("mlp.py", "model_config.py", "attention.py"):
        for match in re.finditer(rf"{re.escape(base_name)}:(\d+)", doc):
            cited.add((base_name, int(match.group(1))))
    listed = {(os.path.basename(rel), line) for rel, line, _ in expected}
    assert cited == listed, (
        f"the doc's citations and this list disagree: only in doc "
        f"{sorted(cited - listed)}, only in list {sorted(listed - cited)}"
    )

    for rel, line_no, needle in expected:
        # the doc must actually cite it, in one of the two forms it uses
        base = os.path.basename(rel)
        assert f"{base}:{line_no}" in doc or f"{rel}:{line_no}" in doc, (
            f"the doc no longer cites {base}:{line_no}; update this list with it"
        )


        path = os.path.join(repo_root, rel)
        if not os.path.isfile(path):
            pytest.skip(f"{rel} absent from this checkout")
        lines = open(path).read().splitlines()
        assert len(lines) >= line_no, f"{rel} is shorter than the cited line {line_no}"
        assert needle in lines[line_no - 1], (
            f"{rel}:{line_no} no longer contains {needle!r} -- it moved; "
            f"the doc's line references are stale again"
        )


def _served_knob_table():
    """The `tt/` knob table, which documents what a deployment can change."""
    readme = _read(os.path.join(HERE, "..", "README.md"))
    start = readme.index("## Served-path knobs")
    table = readme[readme.index("| knob | default | why |", start) :]
    return table[: table.index("\n\n")]


def _tt_env_reads():
    """Every QWEN3ASR_* variable the ttnn modules under tt/ actually read."""
    import re

    found = {}
    for name in sorted(os.listdir(TT)):
        if not name.endswith(".py"):
            continue
        src = _read(os.path.join(TT, name))
        # with a default: os.environ.get("X", "d");  without: os.environ.get("X")
        for env, default in re.findall(
            r'os\.environ\.get\(\s*"(QWEN3ASR_[A-Z0-9_]+)"\s*,\s*"([^"]*)"\s*\)', src
        ):
            found[env] = default
        for env in re.findall(
            r'os\.environ\.get\(\s*"(QWEN3ASR_[A-Z0-9_]+)"\s*\)', src
        ):
            found.setdefault(env, None)
    return found


def test_the_readme_documents_every_knob_the_served_path_reads():
    """A knob that changes the deployment must not live only in the source.

    tt/ is what a vLLM server runs. Five variables there change what it
    computes -- the prefill bucket, the decode trace, the decoder dtype, where
    the embedding gather runs, and which snapshot the audio tower comes from --
    and none of them was mentioned in this README, so the only way to find out
    a deployment could be reconfigured was to read generator_vllm.py and
    qwen3_asr_decoder.py.
    """
    reads = _tt_env_reads()
    assert reads, "the scan found no knobs under tt/; the pattern has gone stale"

    documented = _served_knob_table() + _readme_knob_table()
    missing = [env for env in sorted(reads) if f"`{env}`" not in documented]
    assert not missing, f"the README documents none of {missing}, but tt/ reads them"


def test_the_served_knob_defaults_match_the_code():
    """The table states defaults; a drift makes it lie about the deployment."""
    table = _served_knob_table()

    for env, default in sorted(_tt_env_reads().items()):
        row = [line for line in table.splitlines() if f"`{env}`" in line]
        if not row:
            # documented with the corpus eval instead (the shared mel pin)
            continue
        assert len(row) == 1, f"{env} must have exactly one row, found {len(row)}"
        cell = _table_default_cell(row[0])
        if default is None:
            assert cell == "*(unset)*", (
                f"{env} has no default in the code; the row must say so: {row[0]}"
            )
        else:
            assert cell == f"`{default}`", (
                f"{env}'s default is {default!r} in the code; the row says "
                f"otherwise: {row[0]}"
            )


def test_the_decoder_dtype_row_names_the_only_other_accepted_value():
    """"Overridable" is useless without the accepted set.

    decoder_weight_dtype() raises on anything outside {bfloat8_b, bfloat16},
    so the row has to name the alternative -- otherwise the documented knob
    invites a value that aborts start-up.
    """
    import re

    src = _read(os.path.join(TT, "qwen3_asr_decoder.py"))
    match = re.search(r"_DTYPES\s*=\s*\{([^}]*)\}", src)
    assert match, "_DTYPES is no longer a literal dict; update this test"
    accepted = set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))
    assert accepted == {"bfloat8_b", "bfloat16"}, accepted

    row = [
        line
        for line in _served_knob_table().splitlines()
        if "`QWEN3ASR_DECODER_DTYPE`" in line
    ]
    assert len(row) == 1, row
    for value in accepted:
        assert value in row[0], f"the row must name {value}: {row[0]}"
    assert "raise" in row[0], "the row must say an unaccepted value raises"


def _compile_fn(path, name, namespace=None):
    """Compile one function out of a module that cannot be imported here.

    corpus_eval.py subscripts TT_METAL_HOME at import time and pulls in torch,
    soundfile and transformers, so it cannot simply be imported in a host-only
    test. Extract the single definition through the AST, as test_mel_pin.py
    already does for pin_mel.
    """
    import ast

    src = _read(path)
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ns = dict(namespace or {})
            exec(compile(module, path, "exec"), ns)  # noqa: S102 - our own source
            return ns[name]
    raise AssertionError(f"{name} not found in {path}")


def _parse_asr():
    import re

    return _compile_fn(EVAL, "parse_asr", {"re": re})


def test_parse_asr_returns_only_what_follows_the_tag():
    """This is the hypothesis CER is computed over, and nothing tested it.

    parse_asr strips the prompt echo off the decoded output; whatever it
    returns goes through norm_ja and becomes the hypothesis. If it stopped
    matching, the prompt itself would be scored as the transcript.
    """
    parse = _parse_asr()

    decoded = "<|im_start|>assistant\nlanguage ja<asr_text>こんにちは"
    assert parse(decoded) == "こんにちは"

    # the language token must not survive into the transcript
    assert "ja" not in parse(decoded)


def test_parse_asr_keeps_a_multi_line_transcript_whole():
    """DOTALL is why `.*` reaches past a newline; without it the tail is cut."""
    parse = _parse_asr()

    decoded = "language ja<asr_text>一行目\n二行目"
    assert parse(decoded) == "一行目\n二行目"


def test_parse_asr_falls_back_to_the_whole_string_unchanged():
    """No tag means no prompt to strip; return the text rather than nothing.

    Returning "" here would score an empty hypothesis against a real
    reference -- CER 1.0 for that clip, indistinguishable from a real failure.
    """
    parse = _parse_asr()

    assert parse("  素の転写だけ  ") == "素の転写だけ"
    assert parse("") == ""


def test_parse_asr_takes_the_last_tag_not_the_first():
    """A transcript that contains the tag text must not truncate the result.

    `language\\s*(.*?)<asr_text>(.*)` is non-greedy up to the FIRST tag, so a
    second occurrence stays in group 2 -- the transcript keeps it rather than
    losing everything before it.
    """
    parse = _parse_asr()

    assert parse("language ja<asr_text>まえ<asr_text>あと") == "まえ<asr_text>あと"


def test_the_parse_pattern_and_the_prompt_use_the_same_tag():
    """ASR_TAG builds the prompt; the parser hard-codes the same literal.

    Two spellings of one protocol: changing ASR_TAG alone would make every
    parse fall through to the whole-string branch, and the corpus CER would
    jump with nothing pointing at the cause.
    """
    import re

    src = _read(EVAL)
    tag = re.search(r'ASR_TAG\s*=\s*"([^"]+)"', src)
    assert tag, "ASR_TAG must stay a greppable literal"

    pattern = re.search(r'm = re\.search\(r"([^"]+)"', src)
    assert pattern, "the parse pattern must stay greppable"
    assert tag.group(1) in pattern.group(1), (
        f"the parser matches {pattern.group(1)!r} but the prompt is built with "
        f"{tag.group(1)!r}"
    )


def test_norm_ja_is_exercised_not_just_grepped():
    """The regex was checked; the function it feeds was not.

    test_corpus_eval_norm_matches_reference_cases rebuilds the normalisation
    from _NORM_STRIP and asserts on that reconstruction, so a change to
    norm_ja's body -- dropping NFKC or the substitution -- would leave it
    passing.

    The trailing .strip() is deliberately not asserted here: \\s is already in
    _NORM_STRIP, so removing it changes nothing. Asserting it would be
    pinning a no-op, and a mutation that deletes it is not a defect. Both
    front-ends carry it for symmetry with asr_ja_eval.py, which needs it
    because it applies the same strip set.
    """
    import re
    import unicodedata

    src = _read(EVAL)
    strip = re.search(r'_NORM_STRIP = re\.compile\((r"[^\n]*")\)', src)
    assert strip, "the strip pattern must stay greppable"

    norm = _compile_fn(
        EVAL,
        "norm_ja",
        {
            "unicodedata": unicodedata,
            "_NORM_STRIP": re.compile(eval(strip.group(1))),  # noqa: S307 - our own source
        },
    )

    assert norm("ＡＢＣ") == "ABC", "NFKC must run"
    assert norm("周りを見ると。") == "周りを見ると", "the strip must run"
    assert norm("  はい  ") == "はい", "surrounding whitespace must be gone"
    assert norm("東京") == "東京", "content characters must survive"


def test_the_trailing_strip_is_redundant_because_the_set_covers_whitespace():
    """State why the .strip() above is not worth pinning.

    If \\s ever leaves _NORM_STRIP, the .strip() stops being redundant and the
    two front-ends can disagree on leading/trailing space -- so the claim in
    the test above needs to fail rather than quietly become wrong.
    """
    import re

    src = _read(EVAL)
    strip = re.search(r'_NORM_STRIP = re\.compile\((r"[^\n]*")\)', src)
    assert strip, "the strip pattern must stay greppable"
    pattern = re.compile(eval(strip.group(1)))  # noqa: S307 - our own source

    assert pattern.sub("", " \t\n") == "", (
        "whitespace has left the strip set; the trailing .strip() is now "
        "load-bearing and must be asserted directly"
    )


EXTRACT = os.path.join(HERE, "..", "reference", "extract_text_decoder.py")


def _extract_checkpoint(snapshot):
    """Compile extract_checkpoint with snap_dir() pinned at ``snapshot``.

    The module imports torch and numpy at import time and resolves the
    snapshot through huggingface_hub, so compile the one function and hand it
    the constants it closes over. Everything it does with them -- the key
    renaming, the config, the tokenizer copy -- is plain file work.
    """
    import ast
    import glob
    import json
    import os as _os
    import shutil

    src = _read(EXTRACT)
    tree = ast.parse(src)

    text_cfg = None
    tok_files = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TEXT_CFG":
            text_cfg = eval(ast.unparse(node.value))  # noqa: S307 - our own source
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TOK_FILES":
            tok_files = ast.literal_eval(node.value)
    assert text_cfg and tok_files, "TEXT_CFG / TOK_FILES must stay module constants"

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "extract_checkpoint":
            module = ast.Module(body=[node], type_ignores=[])
            ns = {
                "os": _os,
                "glob": glob,
                "json": json,
                "shutil": shutil,
                "snap_dir": lambda: snapshot,
                "TEXT_CFG": text_cfg,
                "TOK_FILES": tok_files,
            }
            exec(compile(module, EXTRACT, "exec"), ns)  # noqa: S102 - our own source
            return ns["extract_checkpoint"], text_cfg, tok_files
    raise AssertionError("extract_checkpoint not found")


def _fake_snapshot(tmp_path):
    """A snapshot shaped like Qwen3-ASR: thinker.*, audio_tower.*, tokenizer."""
    import torch
    from safetensors.torch import save_file

    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(
        {
            "thinker.model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "thinker.model.embed_tokens.weight": torch.zeros(2, 2),
            "thinker.lm_head.weight": torch.ones(2, 2),
            "audio_tower.layers.0.weight": torch.zeros(2, 2),
            "talker.model.layers.0.weight": torch.zeros(2, 2),
        },
        str(snap / "model.safetensors"),
        metadata={"format": "pt"},
    )
    for name in ("merges.txt", "vocab.json"):
        (snap / name).write_text("{}")
    return str(snap)


def test_the_extraction_renames_the_thinker_keys_the_decoder_expects(tmp_path):
    """The rename is what makes the output loadable as a plain Qwen3.

    extract_checkpoint had no test calling it, only a text match for its
    empty-checkpoint guard. Get the prefix arithmetic wrong and every key
    comes out mangled -- the checkpoint is non-empty, so the guard passes,
    and the failure surfaces as a decoder that loads no weights.
    """
    from safetensors import safe_open

    extract, _, _ = _extract_checkpoint(_fake_snapshot(tmp_path))
    out = str(tmp_path / "out")
    extract(out)

    with safe_open(os.path.join(out, "model.safetensors"), "pt") as handle:
        keys = set(handle.keys())

    assert keys == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }, keys


def test_the_extraction_leaves_the_other_towers_behind(tmp_path):
    """Only the thinker is the text decoder.

    audio_tower.* belongs to the encoder and talker.* to a head this model
    does not serve; copying either in makes the checkpoint bigger and the
    load ambiguous.
    """
    from safetensors import safe_open

    extract, _, _ = _extract_checkpoint(_fake_snapshot(tmp_path))
    out = str(tmp_path / "out")
    extract(out)

    with safe_open(os.path.join(out, "model.safetensors"), "pt") as handle:
        keys = list(handle.keys())

    assert not [k for k in keys if "audio_tower" in k or "talker" in k], keys


def test_the_extraction_writes_the_config_the_loader_reads(tmp_path):
    """Without config.json the directory is not a checkpoint at all."""
    import json

    extract, text_cfg, _ = _extract_checkpoint(_fake_snapshot(tmp_path))
    out = str(tmp_path / "out")
    extract(out)

    with open(os.path.join(out, "config.json")) as handle:
        written = json.load(handle)

    assert written == text_cfg
    assert written["model_type"] == "qwen3", (
        "the output must declare itself a plain Qwen3, not qwen3_asr -- the "
        "adapter's is-full-snapshot check keys on exactly this"
    )


def test_the_extraction_carries_the_tokenizer_files_it_finds(tmp_path):
    """A checkpoint with no tokenizer cannot decode; absent ones are skipped."""
    extract, _, tok_files = _extract_checkpoint(_fake_snapshot(tmp_path))
    out = str(tmp_path / "out")
    extract(out)

    copied = {name for name in tok_files if os.path.exists(os.path.join(out, name))}
    assert copied == {"merges.txt", "vocab.json"}, copied


def test_the_extraction_refuses_a_snapshot_with_no_thinker_weights(tmp_path):
    """Running it proves the guard fires, where the text match only proved it exists."""
    import pytest as _pytest
    import torch
    from safetensors.torch import save_file

    snap = tmp_path / "empty"
    snap.mkdir()
    save_file(
        {"audio_tower.layers.0.weight": torch.zeros(2, 2)},
        str(snap / "model.safetensors"),
        metadata={"format": "pt"},
    )

    extract, _, _ = _extract_checkpoint(str(snap))
    out = str(tmp_path / "out")
    with _pytest.raises(SystemExit):
        extract(out)

    assert not os.path.exists(os.path.join(out, "model.safetensors")), (
        "nothing may be written when the extraction found no weights"
    )
