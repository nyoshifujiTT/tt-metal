# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device test for the TT-native reranker ClassifierPooler.

``RerankerClassifierPooler`` reproduces vLLM's two-stage cross-encoder pooling
(CLS extract + classification head) on device: it slices the ``<s>`` (position
0) hidden row with ``ttnn.slice`` and runs the fp32 device head. This test
drives the shared ``score_cls_on_device`` path on a synthetic ``[B, S, D]``
encoder hidden state and asserts the resulting ``[B, 1]`` logits match a host
fp32 reference that takes position 0 and runs ``dense -> tanh -> out_proj``.

Only the CLS row must influence the score, so the reference intentionally
ignores the non-CLS positions; a device slice that picked the wrong position
(or pooled across the sequence) would diverge here.

Requires a single Tenstorrent device.
"""

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import (
    XLMRobertaClassificationHead,
)
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head_tt import (
    XLMRobertaClassificationHeadTT,
)
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import score_cls_on_device
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import (
    crop_request_rows,
    flatten_request_hidden_to_device,
    gather_cls_from_flat,
)
from models.demos.bge_reranker_v2_m3.tt.reranker_pooler import RerankerClassifierPooler
from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3

HIDDEN_SIZE = 1024
# Same fp32-dest-acc + HiFi4 device head as test_xlm_roberta_classification_head_tt:
# logits agree with the host fp32 reference to ~1e-3 (tile rounding).
LOGIT_ATOL = 5e-3
LOGIT_RTOL = 1e-2


def _random_state_dict(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "classifier.dense.weight": torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.02,
        "classifier.dense.bias": torch.randn(HIDDEN_SIZE) * 0.01,
        "classifier.out_proj.weight": torch.randn(1, HIDDEN_SIZE) * 0.02,
        "classifier.out_proj.bias": torch.randn(1) * 0.01,
    }


@pytest.mark.parametrize("batch_size", [1, 8], ids=["B1", "B8"])
@pytest.mark.parametrize("seq_len", [16], ids=["S16"])
def test_pooler_scores_cls_on_device(device, batch_size, seq_len, reset_seeds):
    require_single_device(device)

    state_dict = _random_state_dict()
    # Full [B, S, D] hidden; only position 0 (CLS) should drive the score.
    hidden = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.float32)

    # Host fp32 reference: take the CLS row, then dense -> tanh -> out_proj.
    host_head = XLMRobertaClassificationHead.from_state_dict(state_dict)
    ref_logits = host_head(hidden[:, 0, :]).to(torch.float32)

    # Device path: same slice + head via the shared pooler helper.
    tt_head = XLMRobertaClassificationHeadTT.from_state_dict(device, state_dict)
    hidden_tt = to_ttnn_tensor(hidden, device, dtype=ttnn.float32)
    logits_tt = score_cls_on_device(hidden_tt, tt_head)
    cand_logits = to_torch(logits_tt, (batch_size, 1))

    assert cand_logits.shape == (batch_size, 1)
    torch.testing.assert_close(
        cand_logits, ref_logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL
    )


@pytest.mark.parametrize(
    "real_lens", [[5, 160, 96], [1], [32, 32]], ids=["mixed", "single", "even"]
)
def test_flatten_and_cls_gather_on_device(device, real_lens, reset_seeds):
    # H1/H2 contract: forward returns a flat [total_tokens, D] and the pooler
    # gathers each request's CLS row via the cursor's first_token_indices. This
    # exercises the device flatten (crop real tokens per request + concat) and
    # the device CLS gather, comparing against a host reference.
    require_single_device(device)

    d = HIDDEN_SIZE
    s = max(max(real_lens), 32)
    b = len(real_lens)
    hidden = torch.randn(b, s, d, dtype=torch.float32)

    # Host reference: flat = concat of each request's real tokens; CLS = the
    # first token of each request in the flat layout.
    flat_ref = torch.cat([hidden[i, : real_lens[i], :] for i in range(b)], dim=0)
    first_idx = torch.tensor(
        [0] + torch.cumsum(torch.tensor(real_lens), 0)[:-1].tolist(), dtype=torch.int32
    )
    cls_ref = flat_ref[first_idx.long()]

    # Device: crop each request's rows from its padded [1, S, D] chunk, concat
    # into flat, then gather CLS rows via first_token_indices.
    per_req = []
    for i in range(b):
        chunk = to_ttnn_tensor(
            hidden[i : i + 1], device, dtype=ttnn.bfloat16
        )  # [1, S, D]
        per_req.append(crop_request_rows(chunk, real_lens[i], d))
    flat_tt = flatten_request_hidden_to_device(per_req)
    flat_back = to_torch(flat_tt, (flat_ref.shape[0], d))
    torch.testing.assert_close(flat_back, flat_ref, atol=1e-2, rtol=1e-2)

    cls_tt = gather_cls_from_flat(flat_tt, first_idx, device)
    cls_back = to_torch(cls_tt, (b, d))
    torch.testing.assert_close(cls_back, cls_ref, atol=1e-2, rtol=1e-2)


def test_pooler_forward_scores_flat_hidden(device, reset_seeds):
    # End-to-end pooler on device: given a flat [total_tokens, D] and a metadata
    # carrying prompt_lens, RerankerClassifierPooler.forward gathers each CLS row
    # (via the cursor selector) and runs the device head, returning one logit per
    # request. Compare to a host reference (CLS row -> dense->tanh->out_proj).
    from types import SimpleNamespace

    require_single_device(device)

    d = HIDDEN_SIZE
    real_lens = [4, 7, 2]
    b = len(real_lens)
    per_req = [torch.randn(li, d, dtype=torch.float32) for li in real_lens]
    flat_ref = torch.cat(per_req, dim=0)
    # The encoder hidden is bf16 on device; round the reference the same way so
    # this test isolates the pooler's gather + head numerics from bf16 encoder
    # rounding (which the encoder PCC tests already cover).
    flat_bf16 = flat_ref.to(torch.bfloat16).to(torch.float32)
    first_idx = torch.tensor(
        [0] + torch.cumsum(torch.tensor(real_lens), 0)[:-1].tolist(), dtype=torch.int64
    )
    cls_ref = flat_bf16[first_idx]

    state_dict = _random_state_dict()
    host_head = XLMRobertaClassificationHead.from_state_dict(state_dict)
    ref_logits = host_head(cls_ref).to(torch.float32)  # [b, 1]

    # Build the model stub the pooler reads (classifier + device).
    tt_head = XLMRobertaClassificationHeadTT.from_state_dict(device, state_dict)
    model_stub = SimpleNamespace(classifier=tt_head, device=device)
    pooler = RerankerClassifierPooler(model_stub)

    flat_tt = to_ttnn_tensor(flat_ref, device, dtype=ttnn.bfloat16)
    cursor = SimpleNamespace(first_token_indices_gpu=first_idx)
    meta = SimpleNamespace(
        pooling_cursor=cursor,
        prompt_lens=torch.tensor(real_lens),
        pooling_params=[object()] * b,
    )

    out = pooler.forward(flat_tt, meta)
    assert len(out) == b
    cand = torch.stack([o.reshape(1) for o in out], dim=0)  # [b, 1]
    torch.testing.assert_close(cand, ref_logits, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)


@pytest.mark.parametrize("rank", [3, 4], ids=["rank3", "rank4"])
def test_append_flat_rows_handles_encoder_rank(device, rank, reset_seeds):
    # The real encoder chunk output is rank 4 ([B, 1, S, D]); _append_flat_rows
    # must crop per-request real tokens from either a rank-3 or rank-4 chunk into
    # [real_len, D] rows. A rank-only unit slip here is exactly what the flat
    # forward's e2e hit, so cover both ranks against a host reference.
    from types import SimpleNamespace

    require_single_device(device)

    d = HIDDEN_SIZE
    chunk_bs, s = 3, 64
    real_lens = [5, 64, 40]
    hidden = torch.randn(chunk_bs, s, d, dtype=torch.float32)
    attn = torch.zeros(chunk_bs, s, dtype=torch.float32)
    for i, li in enumerate(real_lens):
        attn[i, :li] = 1.0
    ref_rows = [hidden[i, : real_lens[i], :].to(torch.bfloat16).to(torch.float32) for i in range(chunk_bs)]

    if rank == 4:
        chunk_tt = to_ttnn_tensor(hidden.reshape(chunk_bs, 1, s, d), device, dtype=ttnn.bfloat16)
    else:
        chunk_tt = to_ttnn_tensor(hidden, device, dtype=ttnn.bfloat16)

    model = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    model.device = device
    model._flat_rows = []
    model._append_flat_rows(chunk_tt, attn, chunk_bs)

    assert len(model._flat_rows) == chunk_bs
    for i in range(chunk_bs):
        got = to_torch(model._flat_rows[i], (real_lens[i], d))
        torch.testing.assert_close(got, ref_rows[i], atol=1e-2, rtol=1e-2)
