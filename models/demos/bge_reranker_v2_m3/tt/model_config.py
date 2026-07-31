# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reranker-specific weight loading for BAAI/bge-reranker-v2-m3.

The XLM-RoBERTa encoder backbone is reused from
``models/demos/wormhole/bge_m3`` unchanged. That module's default loader,
however, resolves checkpoints as ``AutoModelForCausalLM`` and requires the
bge-m3 sparse/colbert head files, so it cannot load a
sequence-classification checkpoint directly.

To keep the shared bge-m3 backbone untouched, this module loads the reranker
checkpoint with ``AutoModelForSequenceClassification`` and returns a
state_dict that already contains both the encoder tensors (``roberta.*``,
consumed by the bge-m3 encoder) and the classification head
(``classifier.*``). The caller passes this state_dict into
``bge_m3.tt.common.create_tt_model``, which skips its own loader when a
state_dict is provided.
"""

from __future__ import annotations


def load_reranker_state_dict(model_name: str) -> dict:
    """Load the reranker checkpoint as a sequence-classification model.

    Returns a state_dict containing both the XLM-RoBERTa encoder weights and
    the classifier head. The presence of the classifier head is validated once,
    downstream, by ``XLMRobertaClassificationHead.from_state_dict`` (which raises
    if the tensors are missing); this loader intentionally does not repeat that
    check to avoid a redundant guard on the same condition.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype="auto")
    return model.state_dict()
