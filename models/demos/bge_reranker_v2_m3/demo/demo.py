# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone bge-reranker-v2-m3 demo (single Tenstorrent device, no vLLM).

Exercises the full on-device reranker scoring path the way the deployed (fork)
pooling runner does: one ``BgeRerankerV2M3.forward(...)`` call per request runs
the shared XLM-RoBERTa encoder, extracts the ``<s>`` (CLS) hidden on device, and
runs the device classification head, returning a ``[batch, 1]`` relevance logit
-- the scoring is done end-to-end on device in a single forward, byte-for-byte
what the runtime uses (the pooling runner just passes the logit through).

The device model produces per-(query, document) relevance logits only. Ranking
(sorting documents by score, ``top_n`` selection) is a serving-layer concern
(the vLLM ``/rerank`` endpoint), not a device model feature; this demo sorts on
host purely to present the result.

Covered:
- ``run_reranker_score``: positive/negative query-document pairs scored via the
  fork-path ``forward`` and checked against the Hugging Face
  ``AutoModelForSequenceClassification`` reference logit;
- ``run_reranker_rerank``: one query against several documents, scored on device
  and ranked on host, demonstrating the end-user reranking shape;
- ``run_reranker_long_seq``: an 8192-token request exercising the chunked
  long-sequence encoder path.

Requires a single Tenstorrent device.
"""

import pytest
import torch
from loguru import logger

from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3
from models.demos.wormhole.bge_m3.tt.common import resolve_model_name

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_SEQ_LEN = 8192

# bf16 encoder vs fp32 HF: logits agree to a few percent (empirically ~3%).
LOGIT_RTOL = 0.05
LOGIT_ATOL = 0.5

QUERY = "what is tenstorrent"
PAIRS = [
    (QUERY, "Tenstorrent builds AI processors.", "positive"),
    (QUERY, "Paris is in France.", "negative"),
]

# One query against several documents (mixed relevance) for the rerank demo.
RERANK_QUERY = "what is tenstorrent"
RERANK_DOCUMENTS = [
    "Paris is the capital of France.",
    "Tenstorrent builds AI processors and RISC-V CPUs.",
    "The mitochondria is the powerhouse of the cell.",
    "Tenstorrent designs high-performance AI accelerators.",
]


def _require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        raise ValueError("bge-reranker-v2-m3 demo currently expects a single device")


def _build_reranker(device, resolved_model_name, max_seq_len):
    """Construct the reranker wrapper and build the encoder + device head.

    ``forward`` builds the model lazily, but constructing it up front lets the
    demo reuse the tokenizer and fail fast if weights are missing.
    """
    reranker = BgeRerankerV2M3(
        device=device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        model_name=resolved_model_name,
    )
    reranker._initialize_model()
    return reranker


def _score_pair(reranker, query, doc, max_length):
    """Tokenize one (query, document) cross-encoder pair and score it on device.

    Returns the scalar relevance logit from the single fork-path ``forward``.
    """
    enc = reranker.tokenizer(
        [[query, doc]],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    logit = reranker.forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return float(logit.view(-1)[0])


def _hf_logit(resolved_model_name, query, doc, max_length):
    transformers = pytest.importorskip("transformers")
    hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(resolved_model_name).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_model_name)
    enc = tokenizer([[query, doc]], padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    with torch.no_grad():
        return float(hf_model(**enc).logits.view(-1)[0])


def run_reranker_score(device, model_name, model_location_generator):
    """Score positive/negative pairs on device and check against HF logits."""
    _require_single_device(device)
    resolved_model_name = resolve_model_name(model_name, model_location_generator)
    reranker = _build_reranker(device, resolved_model_name, DEFAULT_MAX_SEQ_LEN)

    scores = {}
    for query, doc, label in PAIRS:
        tt_logit = _score_pair(reranker, query, doc, max_length=512)
        ref_logit = _hf_logit(resolved_model_name, query, doc, max_length=512)
        logger.info(f"[{label}] query={query!r} doc={doc!r}")
        logger.info(f"[{label}] TT logit: {tt_logit:.4f} | HF logit: {ref_logit:.4f}")
        torch.testing.assert_close(torch.tensor(tt_logit), torch.tensor(ref_logit), rtol=LOGIT_RTOL, atol=LOGIT_ATOL)
        scores[label] = tt_logit

    assert (
        scores["positive"] > scores["negative"]
    ), f"positive logit {scores['positive']:.4f} must exceed negative {scores['negative']:.4f}"
    logger.info(f"positive ({scores['positive']:.4f}) > negative ({scores['negative']:.4f}): OK")
    return scores


def run_reranker_rerank(device, model_name, model_location_generator):
    """Score one query against several documents and rank them (host-side sort)."""
    _require_single_device(device)
    resolved_model_name = resolve_model_name(model_name, model_location_generator)
    reranker = _build_reranker(device, resolved_model_name, DEFAULT_MAX_SEQ_LEN)

    scored = []
    for doc in RERANK_DOCUMENTS:
        tt_logit = _score_pair(reranker, RERANK_QUERY, doc, max_length=512)
        scored.append((tt_logit, doc))

    # Ranking is a serving-layer concern; sort on host only to present results.
    ranking = sorted(enumerate(scored), key=lambda x: x[1][0], reverse=True)
    logger.info(f"rerank query: {RERANK_QUERY!r}")
    for rank, (orig_index, (score, doc)) in enumerate(ranking, start=1):
        logger.info(f"  rank {rank}: score={score:.4f} index={orig_index} doc={doc!r}")

    top_index = ranking[0][1][1]
    assert "Tenstorrent" in top_index, f"expected a Tenstorrent document at the top, got {top_index!r}"
    return ranking


def run_reranker_long_seq(device, model_name, model_location_generator):
    """Exercise the chunked long-sequence (8192-token) encoder path in one forward."""
    _require_single_device(device)
    resolved_model_name = resolve_model_name(model_name, model_location_generator)
    reranker = _build_reranker(device, resolved_model_name, DEFAULT_MAX_SEQ_LEN)

    query = QUERY
    # A long document: repeat a sentence so the query+doc pair fills the context.
    long_doc = "Tenstorrent builds AI processors and RISC-V CPUs. " * 2000
    enc = reranker.tokenizer(
        [[query, long_doc]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=DEFAULT_MAX_SEQ_LEN,
    )
    seq_len = enc["input_ids"].shape[1]
    logit = reranker.forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    value = float(logit.view(-1)[0])
    logger.info(f"long-seq request: seq_len={seq_len} logit={value:.4f}")
    assert seq_len == DEFAULT_MAX_SEQ_LEN, f"expected padded seq_len {DEFAULT_MAX_SEQ_LEN}, got {seq_len}"
    return value


@pytest.mark.parametrize("model_name", [DEFAULT_MODEL_NAME])
def test_reranker_score(device, model_name, model_location_generator, reset_seeds):
    run_reranker_score(device, model_name, model_location_generator)


@pytest.mark.parametrize("model_name", [DEFAULT_MODEL_NAME])
def test_reranker_rerank(device, model_name, model_location_generator, reset_seeds):
    run_reranker_rerank(device, model_name, model_location_generator)


@pytest.mark.parametrize("model_name", [DEFAULT_MODEL_NAME])
def test_reranker_long_seq(device, model_name, model_location_generator, reset_seeds):
    run_reranker_long_seq(device, model_name, model_location_generator)
