# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Shared weight resolution for the pyannote speaker-diarization-community-1 demo.

Model identity and model location are kept separate, matching the tt-metal
convention: ``HF_MODEL`` (or the default repo id) says *which* model, while
``model_location_generator`` says *where* to read it from (the CIv2 large-file
cache, the CIv1 ``/mnt/MLPerf`` mirror, or the local Hugging Face cache).

``pyannote/speaker-diarization-community-1`` is gated, so resolving it from the
Hub needs ``HF_TOKEN``. Callers without a token can point ``HF_MODEL`` at the
ungated mirror ``pyannote-community/speaker-diarization-community-1``, which
carries the same checkpoint SHAs.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_REPO_ID = "pyannote/speaker-diarization-community-1"

# Sub-checkpoints of the pipeline repo, as laid out on the Hugging Face Hub.
EMBEDDING_RELPATH = "embedding/pytorch_model.bin"
SEGMENTATION_RELPATH = "segmentation/pytorch_model.bin"
PIPELINE_RELPATH = "config.yaml"


def repo_id() -> str:
    """Which model to use: ``HF_MODEL`` when set, else the official repo id."""
    return os.environ.get("HF_MODEL") or DEFAULT_REPO_ID


def hf_token() -> Optional[str]:
    """Token for the gated repo; ``None`` lets huggingface_hub use its cache."""
    return os.environ.get("HF_TOKEN") or None


def resolve_model_dir(model_location_generator=None) -> str:
    """Return a local directory holding the community-1 pipeline.

    Resolution order:

    1. ``model_location_generator`` (the tt-metal fixture) when supplied, which
       covers the CIv2 large-file cache and the CIv1 ``/mnt/MLPerf`` mirror.
    2. A Hugging Face snapshot download of :func:`repo_id`, honouring
       ``HF_HOME`` / ``HF_TOKEN``.
    """
    if model_location_generator is not None:
        located = str(model_location_generator(repo_id(), download_if_ci_v2=True))
        # The fixture echoes the model id back when it has no cached copy; in
        # that case fall through to a real Hugging Face resolve.
        if os.path.isdir(located):
            return located

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id(), token=hf_token())


def resolve_weights(relpath, model_location_generator=None) -> str:
    """Return a local path to one checkpoint inside the community-1 repo."""
    return os.path.join(resolve_model_dir(model_location_generator), relpath)
