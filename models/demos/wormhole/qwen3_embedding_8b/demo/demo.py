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

The demo drives the exact same TT forward path the vLLM adapter uses
(``Qwen3ForEmbedding.forward`` -> ``generator.prefill_forward_text(
return_hidden_states=True)``), which returns the last-token hidden state after
the final norm and before the LM head. The only pooling directive Qwen3-Embedding
adds on top is L2 normalization; the official Hugging Face usage example applies
``F.normalize(embeddings, p=2, dim=1)`` after last-token pooling, so the demo does
the same on the host to obtain the finished embedding and compares it against the
Hugging Face reference by cosine similarity.

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
        tt_embeddings.append(_tt_embedding(generator_model, tokenizer, prompt))

    reference = torch.cat(reference_embeddings, dim=0)
    tt = torch.cat(tt_embeddings, dim=0)

    # Per-prompt alignment: TT embedding vs its own HF reference (diagonal of the
    # cross-similarity matrix). This is the accuracy signal that matters.
    cross = cosine_similarity(reference.detach().cpu().numpy(), tt.detach().cpu().numpy())
    per_prompt_alignment = np.diag(cross)
    mean_alignment = float(per_prompt_alignment.mean())

    logger.info(f"Qwen3-Embedding demo: model={resolved_model_name} dim={tt.shape[1]} n={tt.shape[0]}")
    logger.info(f"TT L2 norms (should be ~1.0): {tt.norm(dim=1).tolist()}")
    logger.info(f"per-prompt cos(TT, HF): {[round(float(x), 4) for x in per_prompt_alignment]}")
    logger.info(f"mean cos(TT, HF): {mean_alignment:.4f}")

    assert torch.allclose(tt.norm(dim=1), torch.ones(tt.shape[0]), atol=1e-2), "TT embeddings are not L2-normalized"
    assert mean_alignment > 0.95, f"TT embeddings do not match HF reference (mean cos {mean_alignment:.4f})"

    return tt


@pytest.mark.parametrize("model_name, sequence_length", [(DEFAULT_MODEL_NAME, DEFAULT_SEQUENCE_LENGTH)])
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
