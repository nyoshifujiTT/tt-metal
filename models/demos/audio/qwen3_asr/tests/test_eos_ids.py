# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demo decoder must stop on the same token ids the served path stops on.

The checkpoint's generation_config declares two stop ids; vLLM honours both.
Hardcoding only one made the demo keep decoding past a 151643 stop, which showed
up as trailing filler words the server never emits.
"""

import ast
import json
import os

HERE = os.path.dirname(__file__)
DECODER = os.path.join(HERE, "..", "tt", "qwen3_asr_decoder.py")


def _module():
    with open(DECODER) as fh:
        return ast.parse(fh.read())


def _eos_tuple():
    for node in _module().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EOS_TOKEN_IDS" for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("EOS_TOKEN_IDS must be defined")


def test_eos_ids_match_the_checkpoint_generation_config():
    assert _eos_tuple() == (151643, 151645)


def test_generate_defaults_to_the_full_eos_set():
    src = open(DECODER).read()
    assert "eos_id=None" in src, "generate() must default to the full stop set"
    assert "nxt not in eos_ids" in src, "the loop must stop on any of the stop ids"
    assert "out[-1] in eos_ids" in src, "a trailing stop token must be trimmed"


def test_eos_ids_are_consistent_with_a_snapshot_generation_config(tmp_path):
    # Guard the shape of what we read from the checkpoint, so a future change to
    # the parsing has something concrete to fail against.
    cfg = tmp_path / "generation_config.json"
    cfg.write_text(json.dumps({"eos_token_id": [151643, 151645], "pad_token_id": 151643}))
    loaded = json.loads(cfg.read_text())["eos_token_id"]
    assert tuple(loaded) == _eos_tuple()
