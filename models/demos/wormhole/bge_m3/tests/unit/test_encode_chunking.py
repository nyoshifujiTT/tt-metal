# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit tests for encode_to_last_hidden padding / chunking.

Uses a stub encoder that records the (batch, seq_len) of each device call and
returns a hidden tensor, so the pad/chunk contract can be validated on CPU
without a Tenstorrent device. Verifies:
- sequence length is padded to the expected 128/1024/2048-aligned bucket,
- batch is padded/chunked per the short-seq (32-row) and long-seq (16-row)
  rules,
- the concatenated result is sliced back to the real batch size,
- the caller-supplied ``device`` is threaded through to every encoder call
  (the encoder model must NOT be required to carry a ``.device`` attribute).
"""

import sys
import types

import pytest
import torch

# Stub ttnn so the module imports without a real device stack.
if "ttnn" not in sys.modules:
    ttnn_stub = types.ModuleType("ttnn")
    ttnn_stub.Device = object
    ttnn_stub.Tensor = object
    ttnn_stub.TILE_LAYOUT = object()
    ttnn_stub.ROW_MAJOR_LAYOUT = object()
    ttnn_stub.uint32 = object()
    ttnn_stub.from_torch = lambda *a, **k: None
    sys.modules["ttnn"] = ttnn_stub

from models.demos.wormhole.bge_m3.tt import encode as encode_mod
from models.demos.wormhole.bge_m3.demo.generator_vllm import (
    BGE_M3_LONG_SEQ_CHUNK,
    BGE_M3_SHORT_SEQ_PADDED_BATCH,
)

HIDDEN = 8
SENTINEL_DEVICE = object()


class _StubModel:
    """TT-model stand-in with NO ``.device`` attribute.

    Mirrors the real ``BgeM3Model``, which does not store the mesh device: the
    device must be supplied by the caller and threaded through explicitly. If
    ``encode_to_last_hidden`` regressed to reading ``model.device`` this stub
    would raise ``AttributeError`` and fail the test (which is the point).
    """

    def __init__(self):
        self.calls = []


@pytest.fixture(autouse=True)
def _bypass_device(monkeypatch):
    """Replace ``_encode_chunk`` with a host stub that records shapes + device.

    Recording the ``device`` handed to each chunk lets the tests assert that the
    caller-supplied device (not ``model.device``) is threaded through.
    """
    calls = []

    def fake_encode_chunk(model, input_ids, attention_mask, *, device, chunk_batch_size):
        calls.append(
            (tuple(input_ids.shape), tuple(attention_mask.shape), chunk_batch_size, device)
        )
        padded_batch, padded_seq = input_ids.shape
        hidden = torch.zeros(padded_batch, padded_seq, HIDDEN, dtype=torch.float32)
        return hidden[:chunk_batch_size]

    monkeypatch.setattr(encode_mod, "_encode_chunk", fake_encode_chunk)
    encode_mod._unit_test_calls = calls
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
def test_seq_padding_bucket(_bypass_device, batch, seq_len, exp_padded_seq):
    model = _StubModel()
    ids = torch.randint(1, 50, (batch, seq_len), dtype=torch.long)
    out = encode_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)
    calls = _bypass_device
    # every device call uses the padded seq length
    for in_shape, _, _, _ in calls:
        assert in_shape[1] == exp_padded_seq
    # output sliced back to real batch, seq at padded length
    assert out.shape[0] == batch
    assert out.shape[1] == exp_padded_seq


def test_short_seq_batch_padding_to_32(_bypass_device):
    model = _StubModel()
    ids = torch.randint(1, 50, (5, 100), dtype=torch.long)  # short seq, B=5
    encode_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)
    calls = _bypass_device
    # one chunk, padded to 32 rows
    assert len(calls) == 1
    assert calls[0][0][0] == BGE_M3_SHORT_SEQ_PADDED_BATCH


def test_long_seq_chunks_of_16(_bypass_device):
    model = _StubModel()
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # long seq, B=20
    out = encode_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)
    calls = _bypass_device
    # 20 rows -> chunks of 16 => 2 chunks (16 + 4), each padded to 16 rows
    assert len(calls) == 2
    for in_shape, _, _, _ in calls:
        assert in_shape[0] == BGE_M3_LONG_SEQ_CHUNK
    assert out.shape[0] == 20


def test_batch_one_no_padding(_bypass_device):
    model = _StubModel()
    ids = torch.randint(1, 50, (1, 100), dtype=torch.long)
    encode_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)
    calls = _bypass_device
    assert len(calls) == 1
    assert calls[0][0][0] == 1  # B=1 runs a single row, no batch padding


def test_caller_device_is_threaded_through(_bypass_device):
    """Regression guard: encode_to_last_hidden must use the caller's device.

    The stub model deliberately has no ``.device`` attribute, mirroring the real
    BgeM3Model. Every encoder chunk must receive exactly the device the caller
    passed in.
    """
    model = _StubModel()
    assert not hasattr(model, "device")
    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # forces 2 chunks
    encode_mod.encode_to_last_hidden(model, ids, device=SENTINEL_DEVICE, pad_token_id=0)
    calls = _bypass_device
    assert len(calls) == 2
    for _, _, _, device in calls:
        assert device is SENTINEL_DEVICE
