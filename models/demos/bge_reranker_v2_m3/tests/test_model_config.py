# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the reranker weight loader (host, no device, no download).

load_reranker_state_dict() must return a state_dict containing BOTH the
XLM-RoBERTa encoder tensors (roberta.*) and the sequence-classification head
(classifier.*). Validation that the classifier head is present lives in a single
place downstream (XLMRobertaClassificationHead.from_state_dict); this test checks
the loader passes the tensors through and that the head is the one guard that
rejects a classifier-less checkpoint. transformers is monkeypatched so the test
runs without downloading weights or touching a device.
"""

import sys
import types

import pytest
import torch

from models.demos.bge_reranker_v2_m3.tt import model_config
from models.demos.bge_reranker_v2_m3.tt.xlm_roberta_classification_head import (
    XLMRobertaClassificationHead,
)


class _FakeModel:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


def _install_fake_transformers(monkeypatch, state_dict):
    """Install a fake transformers module whose
    AutoModelForSequenceClassification.from_pretrained returns _FakeModel."""
    fake = types.ModuleType("transformers")

    class _AutoSeqCls:
        @staticmethod
        def from_pretrained(model_name, dtype="auto"):
            return _FakeModel(state_dict)

    fake.AutoModelForSequenceClassification = _AutoSeqCls
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _reranker_like_state_dict(with_classifier=True):
    sd = {
        "roberta.embeddings.word_embeddings.weight": torch.zeros(4, 8),
        "roberta.encoder.layer.0.attention.self.query.weight": torch.zeros(8, 8),
        "roberta.encoder.layer.0.output.dense.weight": torch.zeros(8, 8),
    }
    if with_classifier:
        sd.update(
            {
                "classifier.dense.weight": torch.zeros(8, 8),
                "classifier.dense.bias": torch.zeros(8),
                "classifier.out_proj.weight": torch.zeros(1, 8),
                "classifier.out_proj.bias": torch.zeros(1),
            }
        )
    return sd


def test_loader_returns_encoder_and_classifier(monkeypatch):
    sd = _reranker_like_state_dict(with_classifier=True)
    _install_fake_transformers(monkeypatch, sd)

    out = model_config.load_reranker_state_dict("BAAI/bge-reranker-v2-m3")

    # classifier head present
    for k in XLMRobertaClassificationHead.CLASSIFIER_KEYS:
        assert k in out
    # encoder backbone tensors present (consumed by the bge-m3 encoder)
    assert any(k.startswith("roberta.encoder.layer.") for k in out)
    assert any(k.startswith("roberta.embeddings.") for k in out)


def test_classifier_absence_is_rejected_once_by_the_head(monkeypatch):
    # The loader is a pass-through: it does not re-check the classifier head.
    sd = _reranker_like_state_dict(with_classifier=False)
    _install_fake_transformers(monkeypatch, sd)
    out = model_config.load_reranker_state_dict("BAAI/bge-reranker-v2-m3")
    assert not any(k.startswith("classifier.") for k in out)

    # The single validation point is the classification head loader.
    with pytest.raises(KeyError, match="classifier"):
        XLMRobertaClassificationHead.from_state_dict(out)
