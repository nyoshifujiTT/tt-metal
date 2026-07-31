# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit tests for the shared encoder pad/chunk orchestration.

``XlmRobertaEncoder._encode_in_chunks`` owns the device pad/chunk contract used
by both the bge-m3 embedding wrapper and the bge-reranker cross-encoder. It never
touches ttnn (it only pads torch tensors and calls the overridable
``self._process_chunk`` primitive), so it can be validated on CPU.
``_encode_to_last_hidden`` is the thin last-hidden wrapper used by the reranker;
it runs the encoder chunk on self.model/self.device via the default
``_process_chunk``.

Verifies:
- sequence length is padded to the expected 128/256/1024/2048/8192 bucket,
- batch is padded/chunked per the short-seq (32-row) and long-seq (16-row) rules,
- B=1 runs a single unpadded row,
- the concatenated last-hidden result is sliced back to the real batch size,
- the encoder chunk runs on self.model / self.device (methods, not free funcs).
"""

import pytest
import torch

from models.demos.wormhole.bge_m3.demo import xlm_roberta_encoder as enc_mod
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import (
    BGE_M3_LONG_SEQ_CHUNK,
    BGE_M3_SHORT_SEQ_PADDED_BATCH,
    XlmRobertaEncoder,
)

HIDDEN = 8


class _Tokenizer:
    pad_token_id = 0


class _ConcreteEncoder(XlmRobertaEncoder):
    """Concrete subclass implementing the base's abstract primitives so the
    shared encode plumbing can be instantiated in tests. _process_chunk keeps the
    base default (raw last hidden); only forward/get_embedding_dim are stubbed."""

    def forward(self, input_ids, *args, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        return 1


def _bare_encoder(device=None):
    """A concrete-encoder instance wired just enough for the encode methods,
    without constructing a device model (only self.tokenizer/self.device/self.model
    are touched by _encode_in_chunks / _encode_to_last_hidden)."""
    enc = _ConcreteEncoder.__new__(_ConcreteEncoder)
    enc.tokenizer = _Tokenizer()
    enc.device = device
    enc.model = object()
    return enc


def _record_chunks(input_ids, **kwargs):
    """Runs _encode_in_chunks with a _process_chunk override that records padded
    chunk shapes (the template-method primitive subclasses override)."""
    calls = []

    class _Recorder(_ConcreteEncoder):
        def _process_chunk(self, padded_inputs, chunk_batch_size):
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
        (1, 5, 128),      # tiny -> 128
        (4, 200, 256),    # 200 -> 256
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
    # 20 rows -> chunks of 16 => 2 chunks (16 + 4), each padded to 16 rows
    assert len(calls) == 2
    for in_shape, _ in calls:
        assert in_shape[0] == BGE_M3_LONG_SEQ_CHUNK


def test_batch_one_no_padding():
    ids = torch.randint(1, 50, (1, 100), dtype=torch.long)
    calls = _record_chunks(ids)
    assert len(calls) == 1
    assert calls[0][0][0] == 1  # B=1 runs a single row, no batch padding


def test_encode_in_chunks_calls_process_chunk_override():
    """_encode_in_chunks is a template method: it invokes the overridable
    self._process_chunk primitive, not a passed-in callback."""
    seen = []

    class _Sub(_ConcreteEncoder):
        def _process_chunk(self, padded_inputs, chunk_batch_size):
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


def test_encode_to_last_hidden_slices_and_uses_self(monkeypatch):
    """_encode_to_last_hidden concatenates chunks, slices to real batch, and runs
    the encoder chunk on the instance (self.model/self.device), not a free func."""
    seen = []

    def fake_run_encoder_chunk(self, padded_inputs):
        # Records that the bound method saw the instance's device, and returns a
        # [B, S, HIDDEN] "hidden" carrying the padded chunk shape.
        seen.append(self.device)
        padded_batch, padded_seq = padded_inputs["input_ids"].shape
        return torch.zeros(padded_batch, padded_seq, HIDDEN, dtype=torch.float32)

    monkeypatch.setattr(XlmRobertaEncoder, "_run_encoder_chunk", fake_run_encoder_chunk)
    monkeypatch.setattr(enc_mod, "to_torch_auto_compose", lambda t, *, device: t)

    sentinel_device = object()
    enc = _bare_encoder(device=sentinel_device)
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # forces 2 chunks
    out = enc._encode_to_last_hidden(ids)

    assert out.shape[0] == 20  # sliced back to real batch
    assert out.shape[1] == 8192  # padded seq length
    assert len(seen) == 2  # 20 rows -> 2 chunks of 16
    for device in seen:
        assert device is sentinel_device
