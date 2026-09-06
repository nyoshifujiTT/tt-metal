# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demo-vs-served comparison must be reproducible from the repo.

The parity claim for this model is "same corpus CER", not "same string", so the
script that measures it has to live next to the demo rather than in someone's
home directory. Pin its interface and the invariants it has to share with
generator_vllm, since a drift in either silently invalidates the comparison.
"""

import os

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


def test_corpus_eval_runs_paged_kv_by_default():
    # The served path always allocates a paged KV cache, so decode dispatches to
    # paged_scaled_dot_product_attention_decode. Running the demo non-paged uses a
    # different kernel and yields different transcripts, which would make the CER
    # comparison measure the kernel choice rather than the front-end.
    src = _read(EVAL)
    assert "PagedAttentionConfig" in src, "the eval must build a paged attention config"
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


def test_the_readme_knob_defaults_match_the_code():
    """A default that drifts silently changes what the eval measures."""
    src = _read(EVAL)
    readme = _read(os.path.join(HERE, "..", "README.md"))
    table = readme[readme.index("| knob | default | why |") :]
    table = table[: table.index("\n\n")]

    for env, default in (
        ("QWEN3ASR_EVAL_PAGED_KV", "1"),
        ("QWEN3ASR_EVAL_PAGE_BLOCK", "64"),
        ("QWEN3ASR_EVAL_MAX_BATCH", "4"),
        ("QWEN3ASR_EVAL_REPETITION_PENALTY", "1.1"),
    ):
        assert f'"{env}", "{default}"' in src, (
            f"{env}'s default is no longer {default}; the README table is stale"
        )
        assert f"`{env}`" in table and f"`{default}`" in table, (
            f"{env} = {default} must appear in the README table"
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
