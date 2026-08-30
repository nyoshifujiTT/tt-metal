# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The device Pooler's contract, checked without a device.

``Qwen3EmbeddingDevicePooler`` stands where vLLM's own Pooler would: the
pooling runner hands it the model's per-token hidden states and a
``PoolingMetadata``, and it returns one served vector per request. What it adds
is doing the selection and the normalize with ttnn ops, so only the finished
vector crosses to the host.

The ttnn path is exercised through a stand-in that records the ops it was asked
for; the torch path -- taken when the model handed back host tensors -- is
exercised for real, since the two must agree on which row they pick and on
whether they normalize.
"""

import types

import pytest
import torch

from models.demos.qwen3_embedding.tt.pooler import Qwen3EmbeddingDevicePooler


def _metadata(prompt_lens, last_token_indices=None):
    cursor = None
    if last_token_indices is not None:
        cursor = types.SimpleNamespace(last_token_indices=torch.tensor(last_token_indices))
    return types.SimpleNamespace(
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.int64),
        pooling_cursor=cursor,
    )


def _owner():
    """A wrapper stand-in, holding the model the way the real one does."""

    def l2_normalize_hidden(hidden):
        hidden.normalized = True
        return hidden

    model = types.SimpleNamespace(
        args=types.SimpleNamespace(dim=4),
        l2_normalize_hidden=l2_normalize_hidden,
    )
    # The wrapper keeps a list of per-device models.
    return types.SimpleNamespace(model=[model])


def test_reports_the_embed_task():
    # inspect_model_cls enumerates a model's pooling tasks through this.
    pooler = Qwen3EmbeddingDevicePooler(_owner())
    assert pooler.get_supported_tasks() == {"embed"}


def test_pooling_before_the_model_is_built_says_so():
    # The Pooler is built during model construction, before the wrapper builds
    # the transformer on its first forward.
    pooler = Qwen3EmbeddingDevicePooler(types.SimpleNamespace(model=None))

    with pytest.raises(RuntimeError, match="before the model was built"):
        pooler(_FakeTTNNHidden(rows=2, width=4), _metadata([2]))


def test_picks_each_requests_last_token_from_the_cursor():
    pooler = Qwen3EmbeddingDevicePooler(_owner(), pooler_config=types.SimpleNamespace(normalize=False))
    # Two requests of 3 and 2 tokens, concatenated on the token axis.
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    out = pooler(hidden, _metadata([3, 2], last_token_indices=[2, 4]))

    assert torch.equal(out[0], hidden[2])
    assert torch.equal(out[1], hidden[4])


def test_derives_the_last_token_from_prompt_lengths_without_a_cursor():
    pooler = Qwen3EmbeddingDevicePooler(_owner(), pooler_config=types.SimpleNamespace(normalize=False))
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    out = pooler(hidden, _metadata([3, 2]))

    # Running total of the prompt lengths ends one past each request's last token.
    assert torch.equal(out[0], hidden[2])
    assert torch.equal(out[1], hidden[4])


@pytest.mark.parametrize(
    ("pooler_config", "expected"),
    [
        (None, True),
        (types.SimpleNamespace(normalize=None), True),
        (types.SimpleNamespace(normalize=True), True),
        (types.SimpleNamespace(normalize=False), False),
    ],
)
def test_normalize_follows_the_resolved_pooler_config(pooler_config, expected):
    # vLLM documents normalize as defaulting to True and leaves it None when
    # nothing overrode it, so only an explicit False turns it off.
    pooler = Qwen3EmbeddingDevicePooler(_owner(), pooler_config=pooler_config)
    hidden = torch.ones(2, 4)

    (row,) = pooler(hidden, _metadata([2]))

    assert bool(abs(row.norm().item() - 1.0) < 1e-5) is expected


class _FakeTTNNHidden:
    """A hidden state that is not a torch tensor, like ttnn.Tensor."""

    def __init__(self, rows, width):
        self.shape = (1, 1, rows, width)
        self.sliced_at = None
        self.normalized = False

    def device(self):  # ttnn.Tensor.device is a method, not a torch attribute
        raise AssertionError("device() must not be read as a torch attribute")


def test_pools_a_device_tensor_with_ttnn_ops(monkeypatch):
    hidden = _FakeTTNNHidden(rows=5, width=4)

    def fake_slice(tensor, start, end):
        tensor.sliced_at = (start, end)
        return tensor

    # Substitute the three ops the pooler uses. ttnn's own are native bindings
    # and cannot be reassigned on the module, which is why the pooler looks them
    # up through one indirection.
    fake_ops = types.SimpleNamespace(
        slice=fake_slice,
        get_device_tensors=lambda tensor: [tensor],
        to_torch=lambda tensor: torch.ones(1, 1, 1, 4),
    )

    pooler = Qwen3EmbeddingDevicePooler(_owner())
    monkeypatch.setattr(pooler, "_device_ops", lambda: fake_ops)
    (row,) = pooler(hidden, _metadata([5], last_token_indices=[4]))

    # The row was selected on device, at the request's last token...
    assert hidden.sliced_at == ((0, 0, 4, 0), (1, 1, 5, 4))
    # ...normalized on device...
    assert hidden.normalized is True
    # ...and only the finished vector came back as torch.
    assert isinstance(row, torch.Tensor)
    assert row.shape == (4,)
