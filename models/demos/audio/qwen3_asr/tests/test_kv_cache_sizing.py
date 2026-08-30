# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The adapter must tell the plugin how much KV cache it actually needs.

vllm-tt-plugin sizes the paged KV cache from
``model_class.get_max_tokens_all_users``. When a model does not provide it the
plugin falls back to 131072 tokens, which for this model is 2052 blocks where
132 suffice (max_model_len 2048 x max_num_seqs 4 / block_size 64, plus the
planner's per-user padding). Every block is a zero tensor written to the device
during startup, so the 15.5x over-allocation was paid in wall-clock startup
time on every launch.

ASR capacity is knowable rather than heuristic: a request is capped at the
feature extractor's 30s window, so it can never exceed max_model_len.
"""

import ast
import os

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _fn():
    src = _read(ADAPTER)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_max_tokens_all_users":
            namespace = {}
            body = [n for n in node.body if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)]
            fn = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
                type_params=[],
            )
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            exec(compile(module, ADAPTER, "exec"), namespace)  # noqa: S102 - our own source
            raw = namespace["get_max_tokens_all_users"]
            # The source is a @classmethod; bind a dummy cls so the extracted
            # plain function can be called the way the plugin calls it.
            return lambda **kw: raw(None, **kw)
    raise AssertionError("get_max_tokens_all_users not found")


def test_the_adapter_declares_its_kv_capacity():
    src = _read(ADAPTER)
    assert "def get_max_tokens_all_users(" in src, "the plugin needs this to size the KV cache"


def test_capacity_is_max_model_len_times_max_num_seqs():
    fn = _fn()
    assert fn(max_model_len=2048, max_num_seqs=4) == 8192
    assert fn(max_model_len=2048, max_num_seqs=1) == 2048


def test_capacity_is_far_below_the_plugin_fallback():
    # The fallback is 131072 tokens; ours must be the real requirement.
    fn = _fn()
    assert fn(max_model_len=2048, max_num_seqs=4) < 131_072


def test_degenerate_inputs_do_not_produce_a_zero_cache():
    # A zero would make the planner allocate nothing and fail at runtime.
    fn = _fn()
    assert fn(max_model_len=0, max_num_seqs=0) >= 1
