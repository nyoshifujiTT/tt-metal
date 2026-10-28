# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The staging hook must be visible ON THE ADAPTER, not just on the Generator.

``warmup_utils.warmup_model_decode`` decides whether to stage the persistent
decode trace inputs with a capability probe::

    if not enable_trace and hasattr(self, "_prepare_decode_trace_variant"):
        decode_kwargs["prepare_trace"] = True

``self`` there is this adapter. The adapter *composes* a Generator in
``self._ttt_generator`` instead of inheriting from it, and defines no
``__getattr__``, so an attribute that exists on Generator is still absent here.
The probe then fails, prepare_trace is never set, and the eager pass runs a
plain untraced decode -- leaving the trace inputs to be allocated during the
capture instead of before it (tt-metal #55343).

This is invisible to any test that checks the Generator: the method is there.
It is only wrong at the seam, which is what these tests cover.
"""

import ast
import os

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")
ADAPTER_CLASS = "TTQwen3ASRForConditionalGeneration"
PROBED_ATTR = "_prepare_decode_trace_variant"


def _read(path):
    with open(path) as fh:
        return fh.read()


def _adapter_class():
    for node in ast.walk(ast.parse(_read(ADAPTER))):
        if isinstance(node, ast.ClassDef) and node.name == ADAPTER_CLASS:
            return node
    raise AssertionError(f"{ADAPTER_CLASS} not found")


def _methods(cls):
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_adapter_does_not_inherit_generator():
    """The premise. If this ever changes, the delegation below is redundant.

    Stated as a test rather than a comment so the reasoning cannot silently
    stop applying: were the adapter to start inheriting from Generator, the
    hook would be inherited and this whole file should be revisited.
    """
    bases = {ast.unparse(b) for b in _adapter_class().bases}
    assert "Generator" not in bases, (
        "adapter now inherits Generator; re-derive whether the explicit "
        "delegation of the staging hook is still needed"
    )
    assert "__getattr__" not in _methods(_adapter_class()), (
        "a __getattr__ would forward the probe implicitly; re-derive this rule"
    )


def test_the_staging_hook_is_reachable_on_the_adapter():
    assert PROBED_ATTR in _methods(_adapter_class()), (
        f"{ADAPTER_CLASS} must expose {PROBED_ATTR}: warmup_model_decode probes "
        "it with hasattr() on the adapter, and the adapter does not inherit "
        "from Generator, so a Generator-side definition is not enough"
    )


def test_the_hook_delegates_rather_than_reimplementing():
    """It must forward to the composed Generator, unchanged.

    A local reimplementation would drift from the Generator's own staging and
    defeat the point; asserting the call target keeps it a pure delegation.
    """
    body = ast.unparse(_methods(_adapter_class())[PROBED_ATTR])
    assert f"self._ttt_generator.{PROBED_ATTR}(" in body, (
        "the hook must delegate to the composed Generator"
    )


def test_it_forwards_arguments_verbatim():
    """The Generator owns this signature; pinning it here would drift.

    Forwarding *args/**kwargs means an upstream signature change cannot break
    the seam silently by dropping an argument the adapter did not know about.
    """
    node = _methods(_adapter_class())[PROBED_ATTR]
    assert node.args.vararg is not None, "must forward *args"
    assert node.args.kwarg is not None, "must forward **kwargs"
    body = ast.unparse(node)
    assert "*args" in body and "**kwargs" in body


def test_the_probe_that_requires_this_still_exists_upstream():
    """If upstream stops probing, this rule is obsolete -- fail loudly."""
    warmup_utils = os.path.join(
        HERE, "..", "..", "..", "..", "common", "warmup", "warmup_utils.py"
    )
    src = _read(warmup_utils)
    assert f'hasattr(self, "{PROBED_ATTR}")' in src, (
        "warmup_model_decode no longer probes for the staging hook; "
        "re-derive how the decode trace inputs get staged"
    )


def test_every_generator_entry_point_the_runner_uses_is_delegated():
    """The seam is the recurring hazard, so cover it as a class, not one case.

    This hook was missed precisely because it is reached by capability probe
    instead of by an explicit call, so listing the known entry points guards
    the next one added the same way.
    """
    methods = _methods(_adapter_class())
    for name in ("decode_forward", "read_decode_output", "process_decode_output_host", PROBED_ATTR):
        assert name in methods, f"{name} must be present on the adapter"
        assert "self._ttt_generator." in ast.unparse(methods[name]), (
            f"{name} must reach the composed Generator"
        )
