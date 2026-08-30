# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reranker logit test: TT encoder + device classifier vs HF.

Reuses the bge-m3 XLM-RoBERTa encoder backbone on device and runs the reranker
CLS extraction + classification head on device (``XLMRobertaClassificationHeadTT``),
then compares the resulting per-pair logit against the Hugging Face
``AutoModelForSequenceClassification`` reference. This exercises the same
device-scoring path (``_forward_chunk`` -> device CLS/head -> ``[chunk, 1]``)
the model uses at runtime. Requires a single Tenstorrent device.
"""

import pytest
import torch

from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head_tt import XLMRobertaClassificationHeadTT
from models.demos.wormhole.bge_m3.tests.test_utils import require_single_device
from models.demos.wormhole.bge_m3.tt.common import create_tt_model, resolve_model_name

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 1
# bf16 encoder vs fp32 HF: logits agree to a few percent (empirically ~3%).
LOGIT_RTOL = 0.05
LOGIT_ATOL = 0.5


@pytest.fixture(scope="module")
def artifacts(model_location_generator):
    transformers = pytest.importorskip("transformers")
    model_id_or_path = resolve_model_name(
        MODEL_ID, model_location_generator, download_if_ci_v2=True, ci_v2_timeout_in_s=1800
    )
    hf_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_id_or_path).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path)
    state_dict = hf_model.state_dict()
    return hf_model, tokenizer, state_dict, model_id_or_path


PAIRS = [
    ("what is tenstorrent", "Tenstorrent builds AI processors."),  # positive
    ("what is tenstorrent", "Paris is in France."),  # negative
]


@pytest.mark.slow
@pytest.mark.parametrize("query,doc", PAIRS, ids=["positive", "negative"])
def test_reranker_logit_matches_hf(device, artifacts, query, doc, reset_seeds):
    require_single_device(device)
    hf_model, tokenizer, state_dict, model_id_or_path = artifacts

    enc = tokenizer([[query, doc]], padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        ref_logit = hf_model(**enc).logits.view(-1).to(torch.float32)

    model_args, tt_model, sd = create_tt_model(
        mesh_device=device,
        max_batch_size=BATCH_SIZE,
        max_seq_len=8192,
        state_dict=state_dict,
        hf_model_name=model_id_or_path,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Drive the shared encoder + device CLS/head via the reranker wrapper wired
    # to this tt_model (the same _encode_in_chunks -> _forward_chunk device path
    # it uses at runtime). The base XlmRobertaEncoder is abstract, so use the
    # concrete reranker subclass.
    encoder = BgeRerankerV2M3.__new__(BgeRerankerV2M3)
    encoder.model = tt_model
    encoder.device = device
    encoder.tokenizer = tokenizer
    encoder.classifier = XLMRobertaClassificationHeadTT.from_state_dict(device, sd)
    chunk_logits = encoder._encode_in_chunks(input_ids, attention_mask=attention_mask)
    tt_logit = torch.cat(chunk_logits, dim=0)[:BATCH_SIZE].view(-1)

    torch.testing.assert_close(tt_logit, ref_logit, rtol=LOGIT_RTOL, atol=LOGIT_ATOL)
