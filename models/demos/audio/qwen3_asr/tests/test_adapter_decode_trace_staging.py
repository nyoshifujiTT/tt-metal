# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The adapter must not turn the eager decode warmup call into a capture.

tt-metal #55343 split decode trace setup in two. ``warmup_model_decode`` calls
``decode_forward`` once with ``enable_trace=False`` and ``prepare_trace=True``;
``Generator.decode_forward`` reaches ``_prepare_decode_trace_variant`` -- which
allocates the long-lived decode trace inputs -- only on that branch::

    if enable_trace:      ... _decode_forward_trace_text(...)
    elif prepare_trace:   ... _prepare_decode_trace_variant(...)
    else:                 ... _decode_forward_no_trace_text(...)

This adapter overrides ``decode_forward``. Forcing ``enable_trace`` ON in that
override skips the staging branch entirely, so the capture binds to buffers
allocated behind an already-live trace and every later request's prefill
rewrites them. Observed on the server: the first transcription after warmup is
correct and the rest are not, non-deterministically.

So the override may force the flag OFF (that decision is the adapter's:
QWEN3ASR_DECODE_TRACE=0) but must never force it ON.
"""

import ast
import os

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def _method(name):
    """The FunctionDef for ``name`` defined on the adapter class."""
    tree = ast.parse(_read(ADAPTER))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {ADAPTER}")


def _enable_trace_assignments(node):
    """Every value assigned to ``kwargs["enable_trace"]`` inside ``node``.

    Returned as (value_node, guarded) so a test can tell an unconditional
    assignment from one that only runs on a branch.
    """
    found = []
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "kwargs"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "enable_trace"
            ):
                guarded = any(
                    isinstance(parent, ast.If) and stmt in ast.walk(parent)
                    for parent in ast.walk(node)
                    if isinstance(parent, ast.If)
                )
                found.append((stmt.value, guarded))
    return found


def test_decode_forward_never_forces_the_trace_on():
    """``enable_trace=DECODE_TRACE`` is the exact shape that broke this.

    Asserting on the *value* rather than merely "is it guarded" is what
    distinguishes the fix from the bug: the old line assigned a flag that is
    True in the shipped configuration, so any check that only looked for an
    assignment stayed green through the outage.
    """
    for value, _guarded in _enable_trace_assignments(_method("decode_forward")):
        assert isinstance(value, ast.Constant) and value.value is False, (
            "decode_forward may only force enable_trace to the literal False; "
            "assigning DECODE_TRACE turns the eager staging call into a capture "
            "and _prepare_decode_trace_variant never runs (tt-metal #55343)"
        )


def test_decode_forward_only_disables_under_the_env_switch():
    """The one permitted override is the adapter's own opt-out."""
    assignments = _enable_trace_assignments(_method("decode_forward"))
    assert assignments, "decode_forward is expected to keep the DECODE_TRACE=0 opt-out"
    for _value, guarded in assignments:
        assert guarded, "the override must sit behind `if not DECODE_TRACE`, not run unconditionally"
    assert "if not DECODE_TRACE:" in ast.unparse(_method("decode_forward"))


def test_decode_forward_and_warmup_agree_on_the_policy():
    """Both overrides gate on the same condition.

    warmup_model_decode already had the correct shape. decode_forward being the
    only asymmetric one is what let the regression through, so pin that they
    match rather than pinning each in isolation.
    """
    warmup = ast.unparse(_method("warmup_model_decode"))
    decode = ast.unparse(_method("decode_forward"))
    for src in (warmup, decode):
        assert "if not DECODE_TRACE:" in src
        assert "kwargs['enable_trace'] = False" in src


def test_the_staging_branch_it_protects_still_exists_upstream():
    """If upstream drops prepare_trace, this rule is obsolete -- fail loudly.

    Without this the tests above would keep passing against a Generator that no
    longer has the branch, and the comment explaining *why* would quietly become
    false.
    """
    generator = os.path.join(
        HERE, "..", "..", "..", "..", "tt_transformers", "tt", "generator.py"
    )
    src = _read(generator)
    assert "elif prepare_trace:" in src, (
        "Generator.decode_forward no longer has the prepare_trace branch; "
        "re-derive the adapter's enable_trace policy against the new contract"
    )
    assert "_prepare_decode_trace_variant" in src


def test_warmup_utils_still_requests_staging_on_the_eager_call():
    """The other half of the contract: who sets prepare_trace, and when."""
    warmup_utils = os.path.join(
        HERE, "..", "..", "..", "..", "common", "warmup", "warmup_utils.py"
    )
    src = _read(warmup_utils)
    assert 'decode_kwargs["prepare_trace"] = True' in src
    assert 'if not enable_trace and hasattr(self, "_prepare_decode_trace_variant"):' in src
