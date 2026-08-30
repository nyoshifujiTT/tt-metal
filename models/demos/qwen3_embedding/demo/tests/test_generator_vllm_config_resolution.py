# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free regression tests for Qwen3ForEmbedding config resolution.

The wrapper must use the ``hf_config`` handed in by the caller (vLLM /
tt-inference-server pass ``model_config.hf_config``) *as-is* and must never
re-fetch the config from a hardcoded repo id. Re-fetching was a real bug: it
redundantly downloaded a config the caller already had, ignored a pre-staged
local directory, and forced the 8B repo id regardless of the served checkpoint
(so it broke 0.6B and offline / pre-staged use).

These tests exercise only the config-resolution branch of ``__init__`` against a
tiny fake device, so no ttnn hardware is required. ``AutoConfig.from_pretrained``
is monkeypatched to detect any network/re-fetch attempt.
"""

import types

import pytest
import transformers

from models.demos.qwen3_embedding.tt.model import Qwen3ForEmbedding


def _fake_device():
    # __init__ only stores the device; nothing is called on it here.
    return types.SimpleNamespace()


def _forbid_from_pretrained(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            f"AutoConfig.from_pretrained must not be called when hf_config is "
            f"provided (args={args!r})"
        )

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _boom)


def test_hf_config_is_used_as_is(monkeypatch):
    # When the caller hands in a config, it must be used verbatim with no re-fetch.
    _forbid_from_pretrained(monkeypatch)
    sentinel = object()
    model = Qwen3ForEmbedding(device=_fake_device(), hf_config=sentinel)
    assert model.config is sentinel


def test_hf_config_takes_precedence_over_model_name(monkeypatch):
    # A stale/hardcoded model_name must never override the handed-in config.
    _forbid_from_pretrained(monkeypatch)
    sentinel = object()
    model = Qwen3ForEmbedding(
        device=_fake_device(),
        hf_config=sentinel,
        model_name="Qwen/Qwen3-Embedding-8B",
    )
    assert model.config is sentinel


def test_explicit_model_name_resolves_via_standard_hf(monkeypatch):
    # Without hf_config, the config is loaded from the *explicit* model_name via
    # standard HuggingFace resolution (local cache / pre-staged dir honoured).
    seen = {}
    resolved = object()

    def _fake(name, *args, **kwargs):
        seen["name"] = name
        return resolved

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _fake)
    model = Qwen3ForEmbedding(device=_fake_device(), model_name="/local/Qwen3-Embedding-0.6B")
    assert model.config is resolved
    assert seen["name"] == "/local/Qwen3-Embedding-0.6B"


def test_missing_config_and_name_fails_loud(monkeypatch):
    # No hardcoded 8B default: with neither hf_config nor model_name, fail loud.
    _forbid_from_pretrained(monkeypatch)
    with pytest.raises(ValueError):
        Qwen3ForEmbedding(device=_fake_device())
