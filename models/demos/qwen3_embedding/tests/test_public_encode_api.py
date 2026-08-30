# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The two public entry points of the Qwen3-Embedding model wrapper.

``Qwen3ForEmbedding`` exposes exactly two ways to produce a result, and they
differ by which stage of the official three-stage model (Transformer ->
Pooling(lasttoken) -> Normalize) they stop at:

  * :meth:`encode` runs all three and returns the finished, unit-norm
    ``[batch, hidden]`` embedding. This is what a caller without a pooling
    layer of its own -- the standalone demo, tt-media-server -- wants.
  * :meth:`encode_token_hidden_states` stops before pooling and returns the flat
    ``[total_tokens, hidden]`` per-token layout, which is the input contract of
    vLLM's pooling runner (the ``Pooler`` does the pooling and normalization).

Both are public: the vLLM adapter is a separate class and must not have to
reach into a private helper to satisfy its contract.

These checks are device-free -- ``forward`` is replaced by a recorder, so only
the routing from each public method into it is exercised.
"""

import torch

import pytest

from models.demos.wormhole.qwen3_embedding_8b.demo.model import Qwen3ForEmbedding


class _Recorder(Qwen3ForEmbedding):
    """A wrapper whose forward records its arguments instead of running."""

    def __init__(self):  # noqa: D107 - deliberately skips the real __init__
        self.calls = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        return torch.zeros(1, 4)


def test_encode_runs_the_full_model_including_pooling_and_normalize():
    model = _Recorder()
    ids = torch.zeros(1, 8, dtype=torch.long)

    model.encode(ids)

    (call,) = model.calls
    assert call["embed_single_trace"] is True, (
        "encode() must fold slice + final norm + L2 normalize into the trace so the "
        "caller receives the finished, normalized embedding"
    )
    assert not call.get("return_full_hidden_states", False), "encode() returns the pooled embedding, not token hidden states"


def test_encode_token_hidden_states_stops_before_pooling():
    model = _Recorder()
    ids = torch.zeros(1, 8, dtype=torch.long)

    model.encode_token_hidden_states(ids)

    (call,) = model.calls
    assert call["return_full_hidden_states"] is True, (
        "the pooling runner indexes the flat token axis; a pooled return would be misread"
    )
    assert not call.get("embed_single_trace", False), (
        "pooling and normalization belong to the caller's Pooler on this path"
    )


def test_both_public_methods_forward_the_attention_mask():
    model = _Recorder()
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8)

    model.encode(ids, mask)
    model.encode_token_hidden_states(ids, mask)

    assert all(call["attention_mask"] is mask for call in model.calls)


def test_the_public_api_is_public():
    # A private helper would force the vLLM adapter -- a different class -- to
    # reach into the base's internals to meet the pooling contract.
    for name in ("encode", "encode_token_hidden_states"):
        assert not name.startswith("_")
        assert callable(getattr(Qwen3ForEmbedding, name))


class _NoDevice(Qwen3ForEmbedding):
    """Enough of the wrapper to reach the stage check, without a device."""

    def __init__(self):  # noqa: D107
        self.max_batch_size = 4
        self.max_seq_len = 128


def test_forward_refuses_to_produce_an_unnormalized_pooled_vector():
    # Qwen3-Embedding's modules.json ends in a Normalize stage, so a pooled but
    # unnormalized vector is not one of the model's defined results. Callers
    # that reached it had to know to normalize afterwards, and one did not.
    model = _NoDevice()

    with pytest.raises(ValueError, match="encode"):
        model.forward(torch.zeros(1, 8, dtype=torch.long))
