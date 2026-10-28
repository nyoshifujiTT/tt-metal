# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What this adapter declares must be what the composed Generator reads.

``Generator.decode_forward`` gates the async-ahead token keep on::

    supports_async_decode = self.model_capabilities.get("supports_async_decode", False)

``self`` is the Generator, and this adapter composes one instead of
subclassing it, so the Generator falls back to its own class default. That
default omits ``supports_async_decode``, and ModelCapabilitiesMixin is explicit
that an absent key means "not supported".

Meanwhile the vLLM plugin reads the capabilities off the *adapter* class, where
``supports_async_decode`` is True, and enables async scheduling. The host token
and position then lag device sampling by one step, while the Generator declines
the keep that exists to recover the authoritative device token -- so a reset
step conditions on the stale host token.

Neither side is inspectable from the other, which is why this is pinned here.
"""

import ast
import os

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "..", "tt", "generator_vllm.py")
GENERATOR = os.path.join(HERE, "..", "..", "..", "..", "tt_transformers", "tt", "generator.py")
ADAPTER_CLASS = "TTQwen3ASRForConditionalGeneration"


def _read(path):
    with open(path) as fh:
        return fh.read()


def _class(path, name):
    for node in ast.walk(ast.parse(_read(path))):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _capabilities(cls):
    """The class-level ``model_capabilities`` dict, as a plain dict."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "model_capabilities" for t in stmt.targets
        ):
            return ast.literal_eval(stmt.value)
    raise AssertionError(f"{cls.name} declares no model_capabilities")


def test_the_generator_default_really_lacks_the_key():
    """The premise: inheritance would not have supplied it either.

    Without this the fix below looks like belt-and-braces. It is not: the key
    is genuinely absent upstream, and absent means False.
    """
    assert "supports_async_decode" not in _capabilities(_class(GENERATOR, "Generator")), (
        "Generator now declares supports_async_decode; re-derive whether the "
        "adapter still has to push its capabilities down"
    )


def test_the_adapter_declares_async_decode():
    """If this ever goes False the mismatch disappears and so does the need."""
    assert _capabilities(_class(ADAPTER, ADAPTER_CLASS)).get("supports_async_decode") is True


def test_the_adapter_pushes_its_capabilities_into_the_generator():
    init = next(
        n
        for n in _class(ADAPTER, ADAPTER_CLASS).body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    body = ast.unparse(init)
    assert "self._ttt_generator.model_capabilities = type(self).model_capabilities" in body, (
        "the composed Generator must read the capabilities this class declares; "
        "Generator.decode_forward resolves them off its own self, and this "
        "adapter is not a Generator subclass"
    )


def test_it_assigns_the_class_attribute_not_a_literal():
    """A restated literal would drift from the declaration silently.

    Pinning the *expression* is the point: copying the dict, or writing the
    keys out a second time, reintroduces exactly the two-sources-of-truth
    problem this guards against.
    """
    init = next(
        n
        for n in _class(ADAPTER, ADAPTER_CLASS).body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    for stmt in ast.walk(init):
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Attribute)
            and stmt.targets[0].attr == "model_capabilities"
        ):
            assert isinstance(stmt.value, ast.Attribute), (
                "assign the class attribute itself, not a copy or a restated dict"
            )
            assert ast.unparse(stmt.value) == "type(self).model_capabilities"
            return
    raise AssertionError("no assignment to the Generator's model_capabilities")


def test_the_gate_that_requires_this_still_exists_upstream():
    src = _read(GENERATOR)
    assert 'self.model_capabilities.get("supports_async_decode", False)' in src, (
        "Generator no longer gates on supports_async_decode; re-derive whether "
        "pushing the adapter's capabilities down is still required"
    )
