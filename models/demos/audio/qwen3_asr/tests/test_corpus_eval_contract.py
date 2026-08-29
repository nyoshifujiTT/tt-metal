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
