# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit tests for the shared encoder pad/chunk orchestration.

``encode_in_chunks`` owns the device pad/chunk contract used by both the bge-m3
embedding wrapper and the bge-reranker cross-encoder. It never touches ttnn (it
only pads torch tensors and calls back a per-chunk callable), so it can be
validated on CPU. ``encode_to_last_hidden`` is the thin last-hidden wrapper used
by the reranker; its device threading is checked with a stubbed encoder chunk.

Verifies:
- sequence length is padded to the expected 128/256/1024/2048/8192 bucket,
- batch is padded/chunked per the short-seq (32-row) and long-seq (16-row) rules,
- B=1 runs a single unpadded row,
- the concatenated last-hidden result is sliced back to the real batch size,
- the caller-supplied device is threaded through to every encoder chunk (the
  encoder model must NOT be required to carry a ``.device`` attribute).
"""

import pytest
import torch

from models.demos.wormhole.bge_m3.demo import xlm_roberta_encoder as enc_mod
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import (
    BGE_M3_LONG_SEQ_CHUNK,
    BGE_M3_SHORT_SEQ_PADDED_BATCH,
    encode_in_chunks,
)

HIDDEN = 8
SENTINEL_DEVICE = object()


def _record_chunks(input_ids, **kwargs):
    """Runs encode_in_chunks with a callback that records padded chunk shapes."""
    calls = []

    def process_chunk(padded_inputs, chunk_batch_size):
        calls.append((tuple(padded_inputs["input_ids"].shape), chunk_batch_size))
        return padded_inputs["input_ids"]

    encode_in_chunks(input_ids, process_chunk, **kwargs)
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
    calls = _record_chunks(ids, pad_token_id=0)
    for in_shape, _ in calls:
        assert in_shape[1] == exp_padded_seq


def test_short_seq_batch_padding_to_32():
    ids = torch.randint(1, 50, (5, 100), dtype=torch.long)  # short seq, B=5
    calls = _record_chunks(ids, pad_token_id=0)
    assert len(calls) == 1  # one chunk, padded to 32 rows
    assert calls[0][0][0] == BGE_M3_SHORT_SEQ_PADDED_BATCH


def test_long_seq_chunks_of_16():
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # long seq, B=20
    calls = _record_chunks(ids, pad_token_id=0)
    # 20 rows -> chunks of 16 => 2 chunks (16 + 4), each padded to 16 rows
    assert len(calls) == 2
    for in_shape, _ in calls:
        assert in_shape[0] == BGE_M3_LONG_SEQ_CHUNK


def test_batch_one_no_padding():
    ids = torch.randint(1, 50, (1, 100), dtype=torch.long)
    calls = _record_chunks(ids, pad_token_id=0)
    assert len(calls) == 1
    assert calls[0][0][0] == 1  # B=1 runs a single row, no batch padding


def test_encode_to_last_hidden_slices_and_threads_device(monkeypatch):
    """encode_to_last_hidden concatenates chunks, slices to real batch, and uses
    the caller's device on every chunk (never model.device)."""
    seen_devices = []

    class _StubModel:
        # Mirrors the real BgeM3Model: NO .device attribute.
        pass

    def fake_run_encoder_chunk(model, device, padded_inputs):
        seen_devices.append(device)
        # Return a [B, S, HIDDEN] "hidden" carrying the padded chunk shape.
        padded_batch, padded_seq = padded_inputs["input_ids"].shape
        return torch.zeros(padded_batch, padded_seq, HIDDEN, dtype=torch.float32)

    # to_torch_auto_compose would move a ttnn tensor to host; here the stub
    # already returns a torch tensor, so make it an identity that ignores device.
    monkeypatch.setattr(enc_mod, "run_encoder_chunk", fake_run_encoder_chunk)
    monkeypatch.setattr(enc_mod, "to_torch_auto_compose", lambda t, *, device: t)

    model = _StubModel()
    assert not hasattr(model, "device")
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # forces 2 chunks
    out = enc_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)

    assert out.shape[0] == 20  # sliced back to real batch
    assert out.shape[1] == 8192  # padded seq length
    assert len(seen_devices) == 2  # 20 rows -> 2 chunks of 16
    for device in seen_devices:
        assert device is SENTINEL_DEVICE
