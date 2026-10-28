# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The demo must apply the same repetition penalty the served path is asked for.

The served eval sends the deployment preset (repetition_penalty=1.1). Decoding
the demo with plain greedy compares two decoding rules rather than two
front-ends, which is what was left of the demo-vs-server gap after the metric and
the stop ids were aligned.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))


def _fn():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "tt", "qwen3_asr_decoder.py")
    src = open(path).read()
    namespace = {"torch": torch}
    start = src.index("def apply_repetition_penalty")
    end = src.index("class Qwen3ASRDecoder")
    exec(compile(src[start:end], path, "exec"), namespace)  # noqa: S102 - our own source
    return namespace["apply_repetition_penalty"]


def test_penalty_is_a_noop_without_history_or_at_one():
    fn = _fn()
    logits = torch.tensor([1.0, -2.0, 3.0])
    assert torch.equal(fn(logits.clone(), [], 1.1), logits)
    assert torch.equal(fn(logits.clone(), [0, 2], 1.0), logits)
    assert torch.equal(fn(logits.clone(), [0, 2], None), logits)


def test_positive_logits_are_divided_and_negative_multiplied():
    fn = _fn()
    logits = torch.tensor([2.0, -2.0, 5.0])
    out = fn(logits.clone(), [0, 1], 2.0)
    assert out[0].item() == 1.0  # positive -> divided
    assert out[1].item() == -4.0  # negative -> multiplied
    assert out[2].item() == 5.0  # untouched


def test_penalty_can_change_the_argmax():
    fn = _fn()
    logits = torch.tensor([10.0, 9.5])
    assert int(logits.argmax()) == 0
    assert int(fn(logits.clone(), [0], 1.5).argmax()) == 1


def test_repeated_ids_are_penalised_once():
    fn = _fn()
    logits = torch.tensor([4.0, 1.0])
    once = fn(logits.clone(), [0], 2.0)
    twice = fn(logits.clone(), [0, 0, 0], 2.0)
    assert torch.equal(once, twice)


def test_prompt_ids_are_penalised_too():
    # vLLM's apply_penalties masks prompt_mask | output_mask, so prompt tokens
    # count as "seen". Penalising only generated ids is HF behaviour and decodes
    # differently from the served path.
    fn = _fn()
    logits = torch.tensor([4.0, 1.0])
    out = fn(logits.clone(), [0], 2.0)
    assert out[0].item() == 2.0


def test_out_of_range_ids_are_ignored():
    # Prompt ids include specials above the model's logit width in some configs;
    # they must not index out of bounds.
    fn = _fn()
    logits = torch.tensor([4.0, 1.0])
    out = fn(logits.clone(), [0, 999999], 2.0)
    assert out[0].item() == 2.0
    assert out[1].item() == 1.0


def test_generate_threads_prompt_ids():
    path = os.path.join(os.path.dirname(__file__), "..", "tt", "qwen3_asr_decoder.py")
    src = open(path).read()
    assert "prompt_ids=None" in src, "generate() must accept the prompt ids"
    assert "seen + out" in src, "the penalty must see prompt ids plus generated ids"




def test_eval_defaults_to_the_deployment_preset():
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "corpus_eval.py")
    src = open(path).read()
    assert 'os.environ.get("QWEN3ASR_EVAL_REPETITION_PENALTY", "1.1")' in src
    assert "repetition_penalty=a.repetition_penalty" in src
    assert "prompt_ids=input_ids" in src, "the eval must pass the prompt ids"


def test_it_matches_vllms_own_implementation():
    """The docstring claims vLLM semantics; check against vLLM, not a restatement.

    Every other test here encodes the rule as I understand it -- divide positive
    logits, multiply negative, penalise prompt ids too. That verifies internal
    consistency, not agreement with the thing being copied. vLLM ships the
    reference form as apply_repetition_penalties_torch(logits, prompt_mask,
    output_mask, penalties), so compare against it directly.

    Measured equal for penalties 1.1 / 1.5 / 2.0 with prompt ids {3, 7, 11} and
    output ids {7, 20} (max abs difference 5.96e-08, i.e. float rounding).

    Skips when vLLM is absent: this file is otherwise importable without it, and
    the demo path does not depend on vLLM being installed.
    """
    pytest = __import__("pytest")
    try:
        from vllm._custom_ops import apply_repetition_penalties_torch
    except Exception:  # pragma: no cover - depends on the environment
        pytest.skip("vLLM not installed in this environment")

    ours = _fn()
    torch.manual_seed(0)
    vocab = 64
    prompt_ids = [3, 7, 11]
    output_ids = [7, 20]

    for penalty in (1.1, 1.5, 2.0):
        logits = torch.randn(vocab)

        mine = ours(logits.clone(), set(prompt_ids) | set(output_ids), penalty)

        reference = logits.clone().unsqueeze(0)
        prompt_mask = torch.zeros(1, vocab, dtype=torch.bool)
        prompt_mask[0, prompt_ids] = True
        output_mask = torch.zeros(1, vocab, dtype=torch.bool)
        output_mask[0, output_ids] = True
        apply_repetition_penalties_torch(
            reference, prompt_mask, output_mask, torch.tensor([penalty])
        )

        assert torch.allclose(mine, reference[0], atol=1e-6), (
            f"penalty={penalty}: diverges from vLLM by "
            f"{(mine - reference[0]).abs().max():.2e}"
        )


def test_the_docstring_points_at_the_reference_it_copies():
    """Name the function, so the next reader can re-run the comparison.

    "vLLM's apply_penalties" is a description; the checkable artifact is
    _custom_ops.apply_repetition_penalties_torch, which the test above uses.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "tt", "qwen3_asr_decoder.py")
    src = open(path).read()
    start = src.index("def apply_repetition_penalty")
    doc = src[start : src.index('"""', src.index('"""', start) + 3)]

    assert "apply_repetition_penalties_torch" in doc, (
        "name the reference implementation, not just 'vLLM'"
    )
    assert "prompt_mask | output_mask" in doc, (
        "state the union that makes prompt ids count, which is the divergence "
        "from HF transformers this note exists to flag"
    )
