# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free unit test for the reranker's _encode_to_last_hidden helper.

``BgeRerankerV2M3._encode_to_last_hidden`` drives the shared
``_encode_in_chunks`` template method and concatenates the per-chunk raw last
hidden states into ``[B, S_padded, D]`` on host, sliced back to the real batch.
It never opens a device: the encoder chunk (``_run_encoder_chunk``) and the
host transfer (``to_torch_auto_compose``) are stubbed so the pad/chunk/concat
contract can be validated on CPU.
"""

import torch

from models.demos.wormhole.bge_m3.demo import xlm_roberta_encoder as enc_mod
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import XlmRobertaEncoder
from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3

HIDDEN = 8


class _Tokenizer:
    pad_token_id = 0


def test_encode_to_last_hidden_slices_and_uses_self(monkeypatch):
    """_encode_to_last_hidden concatenates chunks, slices to real batch, and runs
    the encoder chunk on the instance (self.model/self.device)."""
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
    enc = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    enc.tokenizer = _Tokenizer()
    enc.device = sentinel_device
    enc.model = object()

    ids = torch.randint(1, 50, (20, 8000), dtype=torch.long)  # forces 2 chunks
    out = enc._encode_to_last_hidden(ids)

    assert out.shape[0] == 20  # sliced back to real batch
    assert out.shape[1] == 8192  # padded seq length
    assert len(seen) == 2  # 20 rows -> 2 chunks of 16
    for device in seen:
        assert device is sentinel_device
