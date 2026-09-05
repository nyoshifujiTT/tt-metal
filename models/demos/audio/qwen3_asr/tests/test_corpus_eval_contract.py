# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demo-vs-served comparison must be reproducible from the repo.

The parity claim for this model is "same corpus CER", not "same string", so the
script that measures it has to live next to the demo rather than in someone's
home directory. Pin its interface and the invariants it has to share with
generator_vllm, since a drift in either silently invalidates the comparison.
"""

import os

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
    assert "#40592" not in head or "cited\n# here in error" in head, (
        "#40592 is an AllGatherAsync hang; do not present it as this failure"
    )
    # the deterministic near-match must stay marked as such
    assert "45052" in head and "not\n# the non-determinism" in head


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
