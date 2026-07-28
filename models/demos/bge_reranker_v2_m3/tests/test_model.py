# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end reranker logit test: TT encoder + host classifier vs HF.

Reuses the bge-m3 XLM-RoBERTa encoder backbone on device, applies the
reranker classifier head on host, and compares the resulting per-pair logit
against the Hugging Face AutoModelForSequenceClassification reference.
Requires a single Tenstorrent device.
"""

import pytest
import torch

from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tests.test_utils import require_single_device
from models.demos.wormhole.bge_m3.demo.xlm_roberta_encoder import encode_to_last_hidden
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import XLMRobertaClassificationHead
from models.demos.bge_reranker_v2_m3.tt.model_config import load_reranker_state_dict

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BATCH_SIZE = 1
# bf16 encoder vs fp32 HF: logits agree to a few percent (empirically ~3%).
LOGIT_RTOL = 0.05
LOGIT_ATOL = 0.5


@pytest.fixture(scope="module")
def artifacts(model_location_generator):
    transformers = pytest.importorskip("transformers")
    model_id_or_path = str(model_location_generator(MODEL_ID, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
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
    classifier = XLMRobertaClassificationHead.from_state_dict(sd)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    hidden = encode_to_last_hidden(
        tt_model, input_ids, attention_mask, device=device, pad_token_id=tokenizer.pad_token_id
    )
    cls_hidden = hidden[:, 0, :][:BATCH_SIZE]
    tt_logit = classifier(cls_hidden).view(-1)

    torch.testing.assert_close(tt_logit, ref_logit, rtol=LOGIT_RTOL, atol=LOGIT_ATOL)
