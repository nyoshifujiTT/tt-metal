# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone (non-vLLM) demo for the Qwen3-Embedding model on TT hardware.

Every other embedding demo under ``models/demos`` ships a ``demo/demo.py`` with a
``test_*_demo(device, ...)`` entry point that runs the model on a real device via
the shared ``device`` fixture (see ``models/demos/wormhole/bge_m3/demo/demo.py``).
Qwen3-Embedding only had the vLLM adapter (``generator_vllm.py``) and no such
single-device demo, so it could not be exercised the standard way. This file
fills that gap.

Qwen3-Embedding's embedding is the last token's hidden state (after the final
norm, before the LM head), L2-normalized -- the official Hugging Face usage
example applies ``F.normalize(embeddings, p=2, dim=1)`` after last-token pooling.

The main accuracy test runs that whole tail on device: with
``embed_single_trace=True`` the last-token slice, final norm and L2 normalize are
folded into the prefill trace, so a single ``execute_trace`` replay returns the
finished embedding and nothing is post-processed on the host. A standalone metal
run should exercise that path, so it is the one the demo uses, and the result is
compared against the Hugging Face reference by cosine similarity.

The other two ways of obtaining the same embedding are kept as cross-checks
rather than as the demo's main path: the pooled last-token forward plus host
normalize (what the serving stack's Pooler is handed, verified in
``test_qwen3_embedding_single_trace``) and the flat per-token forward plus host
LAST pooling (the layout the vLLM pooling runner indexes, verified in
``test_qwen3_embedding_flat_contract``).

Each prompt is run at ``batch_size=1``. Qwen3-Embedding is a decoder model whose
embedding is the *last* token's hidden state; the TT forward derives that index
from a single ``prompt_lens`` value shared across the batch, so mixing
variable-length prompts in one batched call would pick the wrong token for the
shorter rows. Running one prompt per call sidesteps that and mirrors how the
serving path feeds one request at a time.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SEQUENCE_LENGTH = 8192

# Checkpoints the accuracy demo runs against. The whole Qwen3-Embedding family
# shares this metal implementation; both are single-device and cap the context
# at 8192 for the P150 bring-up. 0.6B is the default; 8B is opt-in via -k so a
# plain run stays fast.
EMBEDDING_MODELS = [
    (DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH),
    ("Qwen/Qwen3-Embedding-8B", DEFAULT_SEQUENCE_LENGTH),
]

prompts = [
    "Artificial intelligence is transforming how we interact with technology.",
    "AI is changing the way humans use computers and machines.",
    "Machine learning algorithms are revolutionizing data analysis.",
    "The weather is sunny today with clear blue skies.",
]


def _require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        raise ValueError("Qwen3-Embedding demo currently expects a single device")


def _resolve_model_name(model_name, model_location_generator):
    if model_location_generator is None:
        return model_name
    return str(model_location_generator(model_name))


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the last non-padding token's hidden state per row (Qwen3-Embedding pooling)."""
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0]), lengths]


def _reference_embedding(model_name: str, tokenizer, prompt: str) -> torch.Tensor:
    """HF reference: backbone last_hidden_state -> last-token pool -> L2 normalize."""
    reference_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    batch = tokenizer([prompt], padding=True, truncation=True, max_length=DEFAULT_SEQUENCE_LENGTH, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = reference_model(**batch).last_hidden_state.to(torch.float32)
    pooled = _last_token_pool(last_hidden_state, batch["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


def _tt_embedding(generator_model: Qwen3ForEmbedding, tokenizer, prompt: str) -> torch.Tensor:
    """TT device path: forward last-token hidden -> L2 normalize (host)."""
    batch = tokenizer([prompt], padding=False, truncation=True, max_length=DEFAULT_SEQUENCE_LENGTH, return_tensors="pt")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    tt_hidden = generator_model.forward(input_ids=input_ids, attention_mask=attention_mask).to(torch.float32)
    if tt_hidden.dim() == 1:
        tt_hidden = tt_hidden.unsqueeze(0)
    return F.normalize(tt_hidden, p=2, dim=1)


def _tt_embedding_from_full_hidden(generator_model: Qwen3ForEmbedding, tokenizer, prompt: str) -> torch.Tensor:
    """TT device path via the FLAT per-token contract.

    Requests ``return_full_hidden_states=True`` (the layout the vLLM pooling
    runner feeds to ``model.pooler``), then does host-side LAST pooling + L2
    normalize -- exactly what the standard vLLM embed Pooler does. Must match the
    pooled last-token path bit-for-bit, proving the flat forward preserves the
    embedding numerics.
    """
    batch = tokenizer([prompt], padding=False, truncation=True, max_length=DEFAULT_SEQUENCE_LENGTH, return_tensors="pt")
    full_hidden = generator_model.forward(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_full_hidden_states=True,
    ).to(torch.float32)
    # full_hidden: [total_tokens, hidden]; LAST pooling picks the final token.
    last_token = full_hidden[-1:, :]
    return F.normalize(last_token, p=2, dim=1)


def _tt_embedding_single_trace(generator_model: Qwen3ForEmbedding, tokenizer, prompt: str) -> torch.Tensor:
    """TT device path with the WHOLE embedding folded into one prefill trace.

    ``embed_single_trace=True`` makes forward return the finished, L2-normalized
    embedding computed entirely on device in a single execute_trace replay (last-
    token slice + final norm + L2 normalize are captured in the trace). No host
    pooling or normalization is applied here.
    """
    batch = tokenizer([prompt], padding=False, truncation=True, max_length=DEFAULT_SEQUENCE_LENGTH, return_tensors="pt")
    emb = generator_model.forward(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        embed_single_trace=True,
    ).to(torch.float32)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    return emb


def run_qwen3_embedding_demo(device, prompts, model_name, sequence_length, model_location_generator):
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")

    generator_model = Qwen3ForEmbedding(
        device=device,
        max_batch_size=1,
        max_seq_len=sequence_length,
        model_name=resolved_model_name,
    )

    reference_embeddings = []
    tt_embeddings = []
    for prompt in prompts:
        reference_embeddings.append(_reference_embedding(resolved_model_name, tokenizer, prompt))
        # Device-complete path: last-token slice, final norm and L2 normalize are
        # folded into the prefill trace, so one execute_trace replay returns the
        # finished embedding with no host post-processing. That is what a
        # standalone metal run should exercise; the host-normalized variant stays
        # as the cross-check in test_qwen3_embedding_single_trace.
        tt_embeddings.append(_tt_embedding_single_trace(generator_model, tokenizer, prompt))

    reference = torch.cat(reference_embeddings, dim=0)
    tt = torch.cat(tt_embeddings, dim=0)

    # Per-prompt alignment: TT embedding vs its own HF reference (diagonal of the
    # cross-similarity matrix). This is the accuracy signal that matters.
    cross = cosine_similarity(reference.detach().cpu().numpy(), tt.detach().cpu().numpy())
    per_prompt_alignment = np.diag(cross)
    mean_alignment = float(per_prompt_alignment.mean())

    logger.info(
        f"Qwen3-Embedding demo (single-trace): model={resolved_model_name} dim={tt.shape[1]} n={tt.shape[0]}"
    )
    logger.info(f"TT L2 norms, normalized on device (should be ~1.0): {tt.norm(dim=1).tolist()}")
    logger.info(f"per-prompt cos(TT, HF): {[round(float(x), 4) for x in per_prompt_alignment]}")
    logger.info(f"mean cos(TT, HF): {mean_alignment:.4f}")

    assert torch.allclose(
        tt.norm(dim=1), torch.ones(tt.shape[0]), atol=1e-2
    ), "TT embeddings are not L2-normalized on device"
    assert mean_alignment > 0.95, f"TT embeddings do not match HF reference (mean cos {mean_alignment:.4f})"

    return tt


@pytest.mark.parametrize(
    "model_name, sequence_length", EMBEDDING_MODELS, ids=["0.6B", "8B"]
)
def test_qwen3_embedding_demo(device, model_name, sequence_length, model_location_generator):
    run_qwen3_embedding_demo(device, prompts, model_name, sequence_length, model_location_generator)


def run_qwen3_embedding_flat_contract(device, prompts, model_name, sequence_length, model_location_generator):
    """Verify the flat per-token forward (vLLM pooling contract) preserves numerics.

    For each prompt the pooled last-token path and the flat-then-LAST-pool path
    must produce the same L2-normalized embedding, and both must match the HF
    reference. This guards the ``return_full_hidden_states`` layout the pooling
    runner relies on against any last-token / norm regression.
    """
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")
    generator_model = Qwen3ForEmbedding(
        device=device, max_batch_size=1, max_seq_len=sequence_length, model_name=resolved_model_name
    )
    for prompt in prompts:
        pooled = _tt_embedding(generator_model, tokenizer, prompt)
        flat = _tt_embedding_from_full_hidden(generator_model, tokenizer, prompt)
        reference = _reference_embedding(resolved_model_name, tokenizer, prompt)
        cos_pooled_flat = float(F.cosine_similarity(pooled, flat).mean())
        cos_flat_hf = float(F.cosine_similarity(flat, reference).mean())
        logger.info(f"flat-contract: cos(pooled, flat)={cos_pooled_flat:.6f} cos(flat, HF)={cos_flat_hf:.4f}")
        assert cos_pooled_flat > 0.9999, f"flat forward diverged from pooled path (cos {cos_pooled_flat:.6f})"
        assert cos_flat_hf > 0.95, f"flat embedding does not match HF (cos {cos_flat_hf:.4f})"


@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_qwen3_embedding_flat_contract(device, model_name, sequence_length, model_location_generator):
    run_qwen3_embedding_flat_contract(device, prompts, model_name, sequence_length, model_location_generator)


def run_qwen3_embedding_single_trace(device, prompts, model_name, sequence_length, model_location_generator):
    """Verify the single-trace embedding path (whole tail on device, 1 replay).

    ``embed_single_trace=True`` folds last-token slice + final norm + L2 normalize
    into the prefill trace, so the finished normalized embedding comes back from a
    single execute_trace. It must equal the pooled last-token path (which does
    slice+norm in a trace-external dispatch + host L2 normalize) and match the HF
    reference. This guards the on-device tail against any numerical drift.
    """
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")
    generator_model = Qwen3ForEmbedding(
        device=device, max_batch_size=1, max_seq_len=sequence_length, model_name=resolved_model_name
    )
    for prompt in prompts:
        pooled = _tt_embedding(generator_model, tokenizer, prompt)
        single = _tt_embedding_single_trace(generator_model, tokenizer, prompt)
        reference = _reference_embedding(resolved_model_name, tokenizer, prompt)
        cos_pooled_single = float(F.cosine_similarity(pooled, single).mean())
        cos_single_hf = float(F.cosine_similarity(single, reference).mean())
        logger.info(
            f"single-trace: norm={float(single.norm()):.6f} "
            f"cos(pooled, single)={cos_pooled_single:.6f} cos(single, HF)={cos_single_hf:.4f}"
        )
        assert torch.allclose(single.norm(dim=1), torch.ones(single.shape[0]), atol=1e-2), "not L2-normalized on device"
        assert cos_pooled_single > 0.999, f"single-trace diverged from pooled path (cos {cos_pooled_single:.6f})"
        assert cos_single_hf > 0.95, f"single-trace embedding does not match HF (cos {cos_single_hf:.4f})"


@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_qwen3_embedding_single_trace(device, model_name, sequence_length, model_location_generator):
    run_qwen3_embedding_single_trace(device, prompts, model_name, sequence_length, model_location_generator)


def run_qwen3_embedding_batched(device, prompts, model_name, sequence_length, model_location_generator, batch_size):
    """Verify the multi-prompt batch path (``max_batch_size > 1``).

    Runs a real ``batch_size``-wide forward and asserts every row is unit-norm,
    matches its single-prompt pooled embedding (``cos > 0.99``; batched prefill is
    not bit-exact vs B=1 under bf8 weights, but agrees to ~0.996) and matches its
    HF reference (``cos > 0.95`` -- the real accuracy contract).

    The batch is one prompt repeated ``batch_size`` times. The metal forward
    derives the last-token index from a single ``prompt_lens`` value shared across
    the batch, so every row must have the *same real length*; identical prompts
    guarantee that. (Batching distinct prompts of different real lengths -- padded
    to a common width -- would make the shared ``prompt_lens`` point past the real
    last token of the shorter rows and corrupt their embeddings; the serving
    runner avoids this by batching same-length requests.)
    """
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")

    # One prompt repeated batch_size times: every row has the same real length, so
    # the batch-shared prompt_lens picks the correct last token for every row.
    batch_prompts = [prompts[0]] * batch_size

    generator_model = Qwen3ForEmbedding(
        device=device, max_batch_size=batch_size, max_seq_len=sequence_length, model_name=resolved_model_name
    )

    batch = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    tt_hidden = generator_model.forward(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).to(torch.float32)
    if tt_hidden.dim() == 1:
        tt_hidden = tt_hidden.unsqueeze(0)
    tt_batched = F.normalize(tt_hidden, p=2, dim=1)

    assert tt_batched.shape[0] == batch_size, f"expected {batch_size} rows, got {tt_batched.shape[0]}"
    assert torch.allclose(
        tt_batched.norm(dim=1), torch.ones(batch_size), atol=1e-2
    ), "batched TT embeddings are not L2-normalized"

    # Single-device B=1 reference for the same prompts (own generator to avoid
    # cross-contaminating the batched model's state).
    single_model = Qwen3ForEmbedding(
        device=device, max_batch_size=1, max_seq_len=sequence_length, model_name=resolved_model_name
    )
    for i, prompt in enumerate(batch_prompts):
        pooled_single = _tt_embedding(single_model, tokenizer, prompt)
        reference = _reference_embedding(resolved_model_name, tokenizer, prompt)
        cos_batch_single = float(F.cosine_similarity(tt_batched[i : i + 1], pooled_single).mean())
        cos_batch_hf = float(F.cosine_similarity(tt_batched[i : i + 1], reference).mean())
        logger.info(
            f"batched[{i}] B={batch_size}: cos(batched, B1)={cos_batch_single:.6f} "
            f"cos(batched, HF)={cos_batch_hf:.4f}"
        )
        # batched (B>1) prefill uses a different padded trace than the B=1 path, so
        # with bf8 weights the two agree closely but not bit-for-bit (~0.996 on
        # P150); this is a tight sanity bound, while matching the HF reference is
        # the real accuracy contract.
        assert cos_batch_single > 0.99, f"batched row {i} diverged from B=1 (cos {cos_batch_single:.6f})"
        assert cos_batch_hf > 0.95, f"batched row {i} does not match HF (cos {cos_batch_hf:.4f})"


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_qwen3_embedding_batched(device, model_name, sequence_length, model_location_generator, batch_size):
    run_qwen3_embedding_batched(device, prompts, model_name, sequence_length, model_location_generator, batch_size)


def _build_long_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt whose tokenization is ~``target_tokens`` long by repeating text."""
    unit = "Artificial intelligence systems process large amounts of natural language data. "
    unit_tokens = len(tokenizer(unit)["input_ids"])
    repeats = max(1, target_tokens // max(1, unit_tokens))
    return unit * repeats


def run_qwen3_embedding_long_context(device, model_name, sequence_length, model_location_generator, target_tokens):
    """Exercise the long-context prefill (single chunk up to ``sequence_length``).

    The default prompts are short, so they never reach the >4096-token regime that
    metal serves as one prefill chunk (the path validated for the P150 8192 cap).
    Feed one long prompt (~``target_tokens``) and assert the TT embedding is
    unit-norm and still matches the HF reference (cos>0.95), proving the long
    single-chunk prefill preserves the embedding numerics.
    """
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")

    long_prompt = _build_long_prompt(tokenizer, target_tokens)
    n_tokens = len(tokenizer(long_prompt, truncation=True, max_length=sequence_length)["input_ids"])

    generator_model = Qwen3ForEmbedding(
        device=device, max_batch_size=1, max_seq_len=sequence_length, model_name=resolved_model_name
    )
    tt = _tt_embedding(generator_model, tokenizer, long_prompt)
    reference = _reference_embedding(resolved_model_name, tokenizer, long_prompt)
    cos_tt_hf = float(F.cosine_similarity(tt, reference).mean())
    logger.info(f"long-context: n_tokens={n_tokens} dim={tt.shape[1]} cos(TT, HF)={cos_tt_hf:.4f}")

    assert torch.allclose(tt.norm(dim=1), torch.ones(tt.shape[0]), atol=1e-2), "long-context TT embedding not L2-normalized"
    assert cos_tt_hf > 0.95, f"long-context embedding does not match HF (cos {cos_tt_hf:.4f})"


@pytest.mark.parametrize("target_tokens", [4500])
@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_qwen3_embedding_long_context(device, model_name, sequence_length, model_location_generator, target_tokens):
    run_qwen3_embedding_long_context(device, model_name, sequence_length, model_location_generator, target_tokens)


def run_qwen3_embedding_accessors(device, model_name, sequence_length, model_location_generator):
    """Check the model's public accessor methods.

    ``Qwen3ForEmbedding`` exposes ``get_embedding_dim`` / ``get_max_seq_len`` /
    ``get_max_batch_size`` (used by the serving stack to size buffers). Assert they
    report values consistent with the constructor args and the resolved HF config,
    and that a real forward returns that embedding dimension. Accessors resolve the
    config without a device forward, so this stays cheap.
    """
    _require_single_device(device)
    resolved_model_name = _resolve_model_name(model_name, model_location_generator)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, padding_side="left")

    max_batch_size = 2
    generator_model = Qwen3ForEmbedding(
        device=device, max_batch_size=max_batch_size, max_seq_len=sequence_length, model_name=resolved_model_name
    )

    dim = generator_model.get_embedding_dim()
    reported_seq_len = generator_model.get_max_seq_len()
    reported_batch = generator_model.get_max_batch_size()
    logger.info(f"accessors: dim={dim} max_seq_len={reported_seq_len} max_batch_size={reported_batch}")

    hidden_size = getattr(generator_model.config, "hidden_size", getattr(generator_model.config, "dim", None))
    assert dim == hidden_size, f"get_embedding_dim {dim} != config hidden_size {hidden_size}"
    assert reported_seq_len == sequence_length, f"get_max_seq_len {reported_seq_len} != {sequence_length}"
    assert reported_batch == max_batch_size, f"get_max_batch_size {reported_batch} != {max_batch_size}"

    # A real embedding must have exactly get_embedding_dim() columns.
    tt = _tt_embedding(generator_model, tokenizer, prompts[0])
    assert tt.shape[1] == dim, f"embedding dim {tt.shape[1]} != get_embedding_dim {dim}"


@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
def test_qwen3_embedding_accessors(device, model_name, sequence_length, model_location_generator):
    run_qwen3_embedding_accessors(device, model_name, sequence_length, model_location_generator)
