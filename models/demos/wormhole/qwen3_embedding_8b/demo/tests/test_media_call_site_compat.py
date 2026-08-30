# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The MEDIA call site must keep working after the hf_config change.

tt-inference-server's tt-media-server calls this wrapper directly
(tt_model_runners/embedding_runner.py::Qwen3Embedding8BRunner._load_model):

    Qwen3ForEmbedding(device=..., model_location_generator=...,
                      max_batch_size=..., max_seq_len=...,
                      act_dtype=..., weight_dtype=..., model_name=...)

It passes model_name and never passes hf_config, so making model_name default
to None must not break it: the config still has to be resolved from model_name.
Device-free -- only the config-resolution branch of __init__ runs.
"""

import types

import pytest
import transformers

from models.demos.wormhole.qwen3_embedding_8b.demo.model import Qwen3ForEmbedding


def _fake_device():
    return types.SimpleNamespace()


def test_media_call_site_resolves_config_from_model_name(monkeypatch):
    sentinel = object()
    seen = {}

    def _from_pretrained(name, *args, **kwargs):
        seen["name"] = name
        return sentinel

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _from_pretrained)

    # Exactly the kwargs tt-media-server passes.
    model = Qwen3ForEmbedding(
        device=_fake_device(),
        model_location_generator=lambda *a, **k: None,
        max_batch_size=32,
        max_seq_len=8192,
        act_dtype=None,
        weight_dtype=None,
        model_name="Qwen/Qwen3-Embedding-8B",
    )

    assert model.config is sentinel
    assert seen["name"] == "Qwen/Qwen3-Embedding-8B"


def test_neither_config_nor_name_is_an_error(monkeypatch):
    # The raise exists so a caller that supplies neither fails loudly instead of
    # silently loading the 8B config; it must not fire for the MEDIA call above.
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: pytest.fail("must not refetch when nothing was named"),
    )
    with pytest.raises(ValueError, match="hf_config"):
        Qwen3ForEmbedding(device=_fake_device())
