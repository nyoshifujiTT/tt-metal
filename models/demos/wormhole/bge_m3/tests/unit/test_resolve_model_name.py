# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared checkpoint resolution helper (host only)."""

from models.demos.wormhole.bge_m3.tt.common import resolve_model_name

REPO_ID = "BAAI/bge-reranker-v2-m3"


def test_hf_model_env_wins_over_everything(monkeypatch):
    monkeypatch.setenv("HF_MODEL", "/local/snapshot")

    def _unexpected(*args, **kwargs):
        raise AssertionError("model_location_generator must not be consulted when HF_MODEL is set")

    assert resolve_model_name(REPO_ID, _unexpected) == "/local/snapshot"


def test_falls_back_to_repo_id_without_generator(monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)

    assert resolve_model_name(REPO_ID) == REPO_ID
    assert resolve_model_name(REPO_ID, None) == REPO_ID


def test_uses_generator_with_ci_v2_download(monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    seen = {}

    def _generator(model_version, **kwargs):
        seen["model_version"] = model_version
        seen.update(kwargs)
        return "/mnt/MLPerf/bge-reranker-v2-m3"

    resolved = resolve_model_name(REPO_ID, _generator)

    assert resolved == "/mnt/MLPerf/bge-reranker-v2-m3"
    assert seen["model_version"] == REPO_ID
    # CIv2 checkpoints must be downloaded on demand, with the long timeout the
    # large reranker/embedding snapshots need.
    assert seen["download_if_ci_v2"] is True
    assert seen["ci_v2_timeout_in_s"] == 1800


def test_returns_str_even_if_generator_returns_path(monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    from pathlib import Path

    resolved = resolve_model_name(REPO_ID, lambda *a, **k: Path("/mnt/MLPerf/x"))

    assert isinstance(resolved, str)
    assert resolved == "/mnt/MLPerf/x"
