# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The prompt-embedding gather must not do redundant host copies.

The gather now lives in _embed_prompt (device by default, host as fallback); the
audio splice that follows it is the only remaining host step, mirroring what the
CUDA reference does in embed_input_ids/_merge_multimodal_embeddings. The old
`self.text_embed[input_ids].clone()` copied the (S, 2048) prompt embedding twice,
costing ~37 ms of the ~83 ms host work per request, so no clone may come back --
wherever the gather lives.
"""

import inspect
import os


import torch

TT_DIR = os.path.join(os.path.dirname(__file__), "..", "tt")


def _source():
    with open(os.path.join(TT_DIR, "generator_vllm.py")) as fh:
        return fh.read()


def test_no_redundant_clone_of_the_prompt_embedding():
    src = _source()
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "self.text_embed[input_ids].clone()" in ln], (
        "advanced indexing already returns a fresh tensor; the extra .clone() "
        "doubles the host-side copy for every request"
    )
    assert "self.text_embed[input_ids]" in src, "the host fallback gather must remain"


def test_gather_result_is_writable_without_clone():
    """Guard the assumption the fix relies on: the gather owns its storage."""
    table = torch.randn(64, 8)
    ids = torch.tensor([1, 2, 3])
    out = table[ids]
    out[0] = 0.0
    assert not torch.equal(table[1], out[0]), "in-place write must not alias the table"
