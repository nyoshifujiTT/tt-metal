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

import torch

from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import CLASSIFIER_KEYS


def load_reranker_state_dict(model_name: str) -> dict:
    """Load the reranker checkpoint as a sequence-classification model.

    Returns a state_dict containing both the XLM-RoBERTa encoder weights and
    the classifier head. Raises RuntimeError if the classifier head is absent
    (i.e. the checkpoint is not a sequence-classification model).
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype="auto")
    state_dict = model.state_dict()

    # Defensive check: AutoModelForSequenceClassification on the real
    # bge-reranker-v2-m3 always yields these tensors, so this does not trigger in
    # normal operation. It fails loudly if a wrong (non sequence-classification)
    # checkpoint is configured instead of silently producing bad scores.
    missing = [k for k in CLASSIFIER_KEYS if k not in state_dict]
    if missing:
        raise RuntimeError(
            f"{model_name} is missing sequence-classification head tensors: {missing}. "
            "bge-reranker-v2-m3 requires a classifier head."
        )
    return state_dict
