# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free construction tests for the bge-m3 embedding wrapper __init__.

``BgeM3ForEmbedding.__init__`` pops its embedding-specific options
(``sentence_pooling_method``/``return_*``) out of kwargs before calling
``super().__init__()`` (so they are not forwarded to the base initializer), then
assigns them to ``self`` only after the base initializer has run. These tests
pin that contract: the popped options land on the instance with the right
values/defaults, and they are not leaked into the base __init__ kwargs.

No device is opened; the base __init__ only needs a truthy ``device``.
"""

import unittest.mock as mock

import pytest

from models.demos.wormhole.bge_m3.demo import xlm_roberta_encoder as enc_mod
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding


@pytest.fixture(autouse=True)
def _stub_autoconfig(monkeypatch):
    """AutoConfig.from_pretrained would hit the network/HF hub; stub it out since
    these tests only exercise __init__ option handling, not the HF config."""
    import transformers

    class _Cfg:
        hidden_size = 1024

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: _Cfg())


def _make(**kwargs):
    # A non-None device is all the base __init__ requires (no device is opened).
    return BgeM3ForEmbedding(device=object(), **kwargs)


def test_default_embedding_options():
    model = _make()
    assert model.sentence_pooling_method == "mean"
    assert model.normalize_embeddings is False
    assert model.return_dense is True
    assert model.return_sparse is False
    assert model.return_colbert is False


def test_embedding_options_override():
    model = _make(
        sentence_pooling_method="cls",
        normalize_embeddings=True,
        return_dense=False,
        return_sparse=True,
        return_colbert=True,
    )
    assert model.sentence_pooling_method == "cls"
    assert model.normalize_embeddings is True
    assert model.return_dense is False
    assert model.return_sparse is True
    assert model.return_colbert is True


def test_base_init_runs_first_and_options_not_forwarded():
    """The base __init__ must run first (its attributes are present) and must not
    receive the popped embedding options in **kwargs."""
    seen_kwargs = {}
    real_init = enc_mod.XlmRobertaEncoder.__init__

    def spy_init(self, *args, **kwargs):
        seen_kwargs.update(kwargs)
        real_init(self, *args, **kwargs)

    with mock.patch.object(enc_mod.XlmRobertaEncoder, "__init__", spy_init):
        model = _make(sentence_pooling_method="cls", return_sparse=True)

    # Popped options were consumed by the subclass, never forwarded to super().
    for leaked in (
        "sentence_pooling_method",
        "normalize_embeddings",
        "return_dense",
        "return_sparse",
        "return_colbert",
    ):
        assert leaked not in seen_kwargs
    # Base initializer really ran (its attributes exist) and the subclass option
    # assignment happened after it.
    assert model.max_batch_size == 32
    assert model.sentence_pooling_method == "cls"
    assert model.return_sparse is True
