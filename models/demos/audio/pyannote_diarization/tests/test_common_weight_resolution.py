# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Unit tests for the community-1 weight resolution helper.

Pure-python: no device, no torch, no network. Asserts that model identity comes
from ``HF_MODEL`` and that model location prefers ``model_location_generator``
over a Hugging Face download.
"""
import os

from models.demos.audio.pyannote_diarization import common


def test_repo_id_defaults_to_the_official_gated_repo(monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert common.repo_id() == "pyannote/speaker-diarization-community-1"


def test_repo_id_honours_hf_model(monkeypatch):
    monkeypatch.setenv("HF_MODEL", "pyannote-community/speaker-diarization-community-1")
    assert common.repo_id() == "pyannote-community/speaker-diarization-community-1"


def test_hf_token_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert common.hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert common.hf_token() == "hf_dummy"


def test_resolve_model_dir_prefers_the_location_generator(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    seen = {}

    def fake_generator(model_version, **kwargs):
        seen["model_version"] = model_version
        seen["kwargs"] = kwargs
        return tmp_path

    assert common.resolve_model_dir(fake_generator) == str(tmp_path)
    assert seen["model_version"] == "pyannote/speaker-diarization-community-1"
    # CIv2 large-file cache must be opted into explicitly.
    assert seen["kwargs"]["download_if_ci_v2"] is True


def test_resolve_model_dir_falls_back_when_generator_has_no_local_copy(monkeypatch):
    """The fixture echoes the model id back when nothing is cached locally."""
    calls = {}

    def fake_snapshot_download(repo_id, token=None):
        calls["repo_id"] = repo_id
        calls["token"] = token
        return "/snapshot/dir"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")

    got = common.resolve_model_dir(lambda model_version, **kwargs: model_version)

    assert got == "/snapshot/dir"
    assert calls["repo_id"] == "pyannote/speaker-diarization-community-1"
    assert calls["token"] == "hf_dummy"


def test_resolve_weights_joins_the_checkpoint_relpath(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    got = common.resolve_weights(common.EMBEDDING_RELPATH, lambda model_version, **kwargs: tmp_path)
    assert got == os.path.join(str(tmp_path), "embedding/pytorch_model.bin")
