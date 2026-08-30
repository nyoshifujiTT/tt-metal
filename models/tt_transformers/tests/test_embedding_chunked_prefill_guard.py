# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free regression test for the embedding chunked-prefill guard.

Embedding (return_hidden_states=True) last-token extraction is only numerically
correct for a single, non-chunked prefill. On the chunked / prefix-caching path
the per-chunk last-token hidden state does not match the single-shot reference
and silently produces a wrong embedding (observed cos ~0.85 on the 4096x2 path).

prefill_forward_single_user_text must therefore raise NotImplementedError -- and
only for embeddings, never for the generative (logits) path. The guard runs
before any device access, so we exercise it against the real Generator method
bound to a tiny stub instance (no ttnn device required).
"""

import types

import pytest
import torch

from models.tt_transformers.tt.generator import Generator


class _FakeModelArgs:
    def __init__(self, max_prefill_chunk_size):
        self.max_prefill_chunk_size = max_prefill_chunk_size


def _make_stub(max_prefill_chunk_size):
    """A minimal object exposing just what the guard reads."""
    stub = types.SimpleNamespace()
    stub.model_args = [_FakeModelArgs(max_prefill_chunk_size)]
    # Bind the real (unbound) method so we exercise production code, not a copy.
    stub.prefill_forward_single_user_text = types.MethodType(
        Generator.prefill_forward_single_user_text, stub
    )
    return stub


def test_embedding_multichunk_raises():
    # seq_len (8192) > max_prefill_chunk_size (4096) -> chunked -> must raise for embedding.
    stub = _make_stub(max_prefill_chunk_size=4096)
    tokens = torch.zeros(1, 8192, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        stub.prefill_forward_single_user_text(
            tokens,
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            user_id=0,
            last_token_idx=8191,
            kv_cache=[None],
            model_id=0,
            num_cached_tokens=0,
            return_hidden_states=True,
        )


def test_embedding_prefix_caching_raises():
    # num_cached_tokens > 0 (prefix caching) -> must raise for embedding.
    stub = _make_stub(max_prefill_chunk_size=8192)
    tokens = torch.zeros(1, 4096, dtype=torch.long)
    with pytest.raises(NotImplementedError):
        stub.prefill_forward_single_user_text(
            tokens,
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            user_id=0,
            last_token_idx=4095,
            kv_cache=[None],
            model_id=0,
            num_cached_tokens=128,
            return_hidden_states=True,
        )


def test_embedding_single_chunk_passes_guard():
    # seq_len (8192) <= max_prefill_chunk_size (8192), no prefix cache -> guard must
    # NOT raise. It proceeds to real device work, which fails without hardware; we
    # only assert the failure is NOT our NotImplementedError guard.
    stub = _make_stub(max_prefill_chunk_size=8192)
    tokens = torch.zeros(1, 8192, dtype=torch.long)
    try:
        stub.prefill_forward_single_user_text(
            tokens,
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            user_id=0,
            last_token_idx=8191,
            kv_cache=[None],
            model_id=0,
            num_cached_tokens=0,
            return_hidden_states=True,
        )
    except NotImplementedError as e:
        pytest.fail(f"guard wrongly rejected a single-chunk embedding prefill: {e}")
    except Exception:
        # Any non-guard failure (e.g. missing device / attribute) is expected here.
        pass


def test_generative_multichunk_does_not_hit_embedding_guard():
    # return_hidden_states=False (generative) must never hit the embedding guard,
    # even when chunked. It proceeds to real chunked-prefill device work.
    stub = _make_stub(max_prefill_chunk_size=4096)
    tokens = torch.zeros(1, 8192, dtype=torch.long)
    try:
        stub.prefill_forward_single_user_text(
            tokens,
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            user_id=0,
            last_token_idx=8191,
            kv_cache=[None],
            model_id=0,
            num_cached_tokens=0,
            return_hidden_states=False,
        )
    except NotImplementedError as e:
        if "return_hidden_states=True (embedding)" in str(e):
            pytest.fail(f"generative path wrongly hit the embedding guard: {e}")
    except Exception:
        pass
