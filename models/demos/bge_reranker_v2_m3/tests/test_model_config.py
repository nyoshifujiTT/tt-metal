# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for the reranker weight loader (host, no device, no download).

load_reranker_state_dict() must return a state_dict containing BOTH the
XLM-RoBERTa encoder tensors (roberta.*) and the sequence-classification head
(classifier.*), and must raise if the checkpoint has no classifier head.
transformers is monkeypatched so the test validates the extraction/validation
logic without downloading weights or touching a device.
"""

import sys
import types

import pytest
import torch

from models.demos.bge_reranker_v2_m3.tt import model_config
from models.demos.bge_reranker_v2_m3.tt.classifier_head import CLASSIFIER_KEYS


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
    for k in CLASSIFIER_KEYS:
        assert k in out
    # encoder backbone tensors present (consumed by the bge-m3 encoder)
    assert any(k.startswith("roberta.encoder.layer.") for k in out)
    assert any(k.startswith("roberta.embeddings.") for k in out)


def test_loader_raises_without_classifier(monkeypatch):
    sd = _reranker_like_state_dict(with_classifier=False)
    _install_fake_transformers(monkeypatch, sd)

    with pytest.raises(RuntimeError, match="classifier head"):
        model_config.load_reranker_state_dict("BAAI/bge-reranker-v2-m3")
