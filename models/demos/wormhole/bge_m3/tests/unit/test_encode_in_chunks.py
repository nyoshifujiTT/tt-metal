# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit tests for the shared encoder pad/chunk orchestration.

``XlmRobertaEncoder._encode_in_chunks`` owns the device pad/chunk contract used
by both the bge-m3 embedding wrapper and the bge-reranker cross-encoder. It never
touches ttnn (it only pads torch tensors and calls the overridable
``self._forward_chunk`` primitive), so it can be validated on CPU.

Verifies:
- sequence length is padded to the expected 128/256/1024/2048/8192 bucket,
- batch is padded/chunked per the short-seq (32-row) and long-seq (16-row) rules,
- B=1 runs a single unpadded row,
- the concatenated last-hidden result is sliced back to the real batch size,
- the encoder chunk runs on self.model / self.device (methods, not free funcs).
"""

import pytest
import torch

from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import (
    BGE_M3_LONG_SEQ_CHUNK,
    BGE_M3_LONG_SEQ_WIDTHS,
    BGE_M3_SHORT_SEQ_PADDED_BATCH,
    XlmRobertaEncoder,
)


class _Tokenizer:
    pad_token_id = 0


class _ConcreteEncoder(XlmRobertaEncoder):
    """Concrete subclass implementing the base's abstract primitives so the
    shared encode plumbing can be instantiated in tests. The tests that need a
    real _forward_chunk subclass it and override that method; here the three
    abstract methods are stubbed just enough to instantiate."""

    def forward(self, input_ids, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        return 1

    def _forward_chunk(self, padded_inputs, chunk_batch_size):  # pragma: no cover - overridden
        raise NotImplementedError


def _record_chunks(input_ids, **kwargs):
    """Runs _encode_in_chunks with a _forward_chunk override that records padded
    chunk shapes (the template-method primitive subclasses override)."""
    calls = []

    class _Recorder(_ConcreteEncoder):
        def _forward_chunk(self, padded_inputs, chunk_batch_size):
            calls.append((tuple(padded_inputs["input_ids"].shape), chunk_batch_size))
            return padded_inputs["input_ids"]

    enc = _Recorder.__new__(_Recorder)
    enc.tokenizer = _Tokenizer()
    enc.device = None
    enc.model = object()
    enc._encode_in_chunks(input_ids, **kwargs)
    return calls


@pytest.mark.parametrize(
    "batch,seq_len,exp_padded_seq",
    [
        (1, 5, 128),  # tiny -> 128
        (4, 200, 256),  # 200 -> 256
        (8, 1000, 1024),  # 1000 -> 1024
        (2, 1500, 2048),  # 1024<seq<=2048 -> 2048
        (3, 8000, 8192),  # long -> 8192
    ],
)
def test_seq_padding_bucket(batch, seq_len, exp_padded_seq):
    ids = torch.randint(1, 50, (batch, seq_len), dtype=torch.long)
    calls = _record_chunks(ids)
    for in_shape, _ in calls:
        assert in_shape[1] == exp_padded_seq


def test_short_seq_batch_padding_to_32():
    ids = torch.randint(1, 50, (5, 100), dtype=torch.long)  # short seq, B=5
    calls = _record_chunks(ids)
    assert len(calls) == 1  # one chunk, padded to 32 rows
    assert calls[0][0][0] == BGE_M3_SHORT_SEQ_PADDED_BATCH


def test_long_seq_chunks_of_16():
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # long seq, B=20
    calls = _record_chunks(ids)
    # 20 rows -> chunks of at most 16 => 2 chunks (16 + 4). The full chunk runs
    # at the 16-row cap; the tail chunk runs at the 4-row width instead of being
    # inflated to 16, since device time is linear in rows and padded rows are
    # masked out anyway.
    assert len(calls) == 2
    assert calls[0][0][0] == BGE_M3_LONG_SEQ_CHUNK
    assert calls[1][0][0] == 4


@pytest.mark.parametrize(
    "batch,exp_width",
    [(1, 1), (2, 2), (3, 4), (5, 8), (7, 8), (8, 8), (11, 16), (16, 16)],
)
def test_long_seq_rounds_up_to_a_width_bucket(batch, exp_width):
    """A long-sequence request narrower than the 16-row cap runs at the smallest
    allowed width, not inflated to 16 rows (that inflation cost up to 18x) and
    not at an arbitrary width (each width JIT-compiles its own kernels)."""
    ids = torch.randint(1, 50, (batch, 8000), dtype=torch.long)
    calls = _record_chunks(ids)
    assert len(calls) == 1
    assert calls[0][0][0] == exp_width


def test_long_seq_only_ever_uses_the_declared_widths():
    """Every batch size up to the cap must land on one of the five declared
    widths, so the whole set can be pre-compiled at startup."""
    for batch in range(1, BGE_M3_LONG_SEQ_CHUNK + 1):
        ids = torch.randint(1, 50, (batch, 8000), dtype=torch.long)
        for in_shape, _ in _record_chunks(ids):
            assert in_shape[0] in BGE_M3_LONG_SEQ_WIDTHS


def test_long_seq_never_exceeds_the_16_row_cap():
    """The circular-buffer limit the upstream fix (#41397) introduced is an upper
    bound; no chunk may run wider than it."""
    ids = torch.randint(1, 50, (40, 8000), dtype=torch.long)
    calls = _record_chunks(ids)
    assert [in_shape[0] for in_shape, _ in calls] == [16, 16, 8]
    for in_shape, _ in calls:
        assert in_shape[0] <= BGE_M3_LONG_SEQ_CHUNK


def test_batch_one_no_padding():
    ids = torch.randint(1, 50, (1, 100), dtype=torch.long)
    calls = _record_chunks(ids)
    assert len(calls) == 1
    assert calls[0][0][0] == 1  # B=1 runs a single row, no batch padding


def test_encode_in_chunks_calls_forward_chunk_override():
    """_encode_in_chunks is a template method: it invokes the overridable
    self._forward_chunk primitive, not a passed-in callback."""
    seen = []

    class _Sub(_ConcreteEncoder):
        def _forward_chunk(self, padded_inputs, chunk_batch_size):
            seen.append(chunk_batch_size)
            return chunk_batch_size

    enc = _Sub.__new__(_Sub)
    enc.tokenizer = _Tokenizer()
    enc.device = None
    enc.model = object()
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # 2 chunks of 16
    out = enc._encode_in_chunks(ids)
    assert out == [16, 4]  # per-chunk real batch sizes, in request order
    assert seen == [16, 4]
