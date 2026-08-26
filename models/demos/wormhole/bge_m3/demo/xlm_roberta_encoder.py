# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM wrapper base for XLM-RoBERTa encoder models.

Both the bge-m3 embedding wrapper and the bge-reranker-v2-m3 cross-encoder run
the same XLM-RoBERTa encoder backbone (``create_tt_model``) and expose the same
vLLM plumbing: device/vllm_config resolution, ``initialize_vllm_model``, lazy
model construction, request validation and the ``get_max_*`` helpers.

That plumbing lives here so each model only implements what actually differs:
its ``forward`` (pooling vs classification), ``get_embedding_dim`` and any extra
per-model state. Subclasses customise weight loading and post-init via the
``_load_state_dict`` / ``_post_initialize`` hooks.

Placed in the bge-m3 package because bge-m3 and the reranker are the only two
users today (see the reranker classifier-head note); promote to a neutral
location if a third XLM-RoBERTa encoder model appears.
"""

from __future__ import annotations

import abc
from typing import Iterator, Optional

import torch

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length


class XlmRobertaEncoder(abc.ABC):
    """Common vLLM plumbing for TT XLM-RoBERTa encoder models.

    Execution is single-device only. ``mesh_device`` is accepted for API
    compatibility, but the encoder builds and runs exactly one model on one
    device. It implements neither data parallelism nor any other multi-device
    mode (tensor parallelism, etc.): the only parallelism-related argument it
    takes is ``tt_data_parallel``, and no tensor-parallel or sharding argument
    exists, so no other multi-device path can even be requested. ``__init__``
    therefore only needs to reject ``tt_data_parallel > 1`` (the one way a caller
    could ask for multiple devices) rather than silently running single-device.

    TODO(data-parallel): add a real multi-device execution path and lift the
    ``tt_data_parallel > 1`` guard in __init__ once it is implemented and tested.
    """

    # ---- pure-virtual model interface (subclasses must implement) ----
    @abc.abstractmethod
    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        """Run the model on a tokenized request and return its output.

        The two subclasses differ here: the bge-m3 wrapper pools into an
        embedding dict, the reranker cross-encoder returns relevance logits.
        """

    @abc.abstractmethod
    def get_embedding_dim(self) -> int:
        """Output width vLLM should expect from ``forward``."""

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "",
        vllm_config=None,
        prefix: str = "",
        tt_data_parallel: int = 1,
        **kwargs,
    ):
        del prefix, kwargs

        # Execution is single-device only (see class docstring). tt_data_parallel
        # is the only parallelism knob accepted (there is no tensor-parallel/
        # sharding argument), so rejecting tt_data_parallel != 1 is sufficient to
        # rule out every multi-device request rather than silently downgrading it.
        if tt_data_parallel != 1:
            raise NotImplementedError(
                f"{type(self).__name__} only supports single-device execution "
                f"(tt_data_parallel=1); got tt_data_parallel={tt_data_parallel}"
            )

        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device
        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # Accepted for API compatibility; execution stays single-device (values
        # other than 1 are rejected above).
        self.tt_data_parallel = tt_data_parallel
        self.dtype = dtype
        self.model_name = model_name

        # Always set the attribute, None when running outside vLLM (the demo /
        # test path). Setting it only when present made ``self.vllm_config``
        # raise AttributeError instead of reading as None, so callers had to
        # guess whether it existed.
        self.vllm_config = vllm_config

        self._is_initialized = False
        self.model_args = None
        self.model = None
        self.state_dict = None
        self.tokenizer = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = 8192,
        model_location_generator=None,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
        vllm_config=None,
        dtype=ttnn.bfloat16,
        **kwargs,
    ):
        if optimizations is not None:
            raise ValueError(f"Optimizations are not supported for {cls.__name__}")

        if vllm_config is not None:
            return cls(
                device=mesh_device,
                model_location_generator=model_location_generator,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
                tt_data_parallel=tt_data_parallel,
                dtype=dtype,
                **kwargs,
            )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            dtype=dtype,
            **kwargs,
        )

    # ---- hooks for subclasses ----
    def _load_state_dict(self):
        """Return a state_dict to hand to create_tt_model, or None to let the
        backbone load its own weights. Overridden by models that build the
        state_dict themselves (e.g. the reranker seq-classification loader)."""
        return None

    def _post_initialize(self) -> None:
        """Hook run after the encoder is built (e.g. to build a classifier head)."""
        # Optional hook: no-op by default (an overridable "do nothing", not an
        # abstract method -- bge-m3 deliberately does not override it).
        return None

    def _initialize_model(self) -> None:
        if self._is_initialized and self.model is not None:
            return

        if self.state_dict is None:
            self.state_dict = self._load_state_dict()

        self.model_args, self.model, self.state_dict = create_tt_model(
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            state_dict=self.state_dict,
            hf_model_name=self.model_name,
        )
        self.tokenizer = self.model_args.tokenizer
        self._post_initialize()
        self._is_initialized = True

    def _validate_request(self, batch_size: int, padded_seq_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if padded_seq_len > self.max_seq_len:
            raise ValueError(f"Padded sequence length {padded_seq_len} exceeds max_seq_len {self.max_seq_len}")

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def warmup_model_prefill(self, *args, **kwargs) -> None:
        """Compiles one full forward per long-sequence device width.

        Kernels are JIT-compiled per shape, and the compile lands on whichever
        request first uses an unseen shape: measured on Blackhole p150 at seq
        8192, a first-time 16-row forward took 62.6 s against an 8.0 s steady
        state, and a first-time 11-row forward 28.1 s against 5.4 s. Serving must
        not expose that to a user request, so the widths are compiled here, at
        the hook vLLM already reserves for warmup
        (``compile_or_warm_up_model`` -> ``warmup_model``), which is where the
        decoder models do their warmup too.

        Scope is the long-sequence widths (``BGE_M3_LONG_SEQ_WIDTHS``): that is
        the set this encoder can choose between, so it is where an unseen shape
        can appear at request time, and it is small and fixed by construction,
        which is what makes exhaustive warmup possible.

        TODO(short-seq-warmup): the shorter sequence buckets each execute one
        fixed width, so they cannot surprise a *later* request the way the long
        widths could, but the first request per bucket still pays its compile.
        Warming them too means running up to 32 rows at 4096/6144 tokens, which
        is more tokens per forward than the 8192x16 shape the upstream
        circular-buffer fix (#41397) had to cap, so it needs its own validation
        before being added here.
        """
        self._initialize_model()
        pad_token_id = self.tokenizer.pad_token_id
        for width in BGE_M3_LONG_SEQ_WIDTHS:
            # Run the model's own forward, not just the encoder: whatever the
            # subclass does after the encoder (the reranker's device CLS
            # extraction and classification head) compiles its own kernels too,
            # and warming only the encoder left that cost on the first real
            # request (measured 9.9 s against a 0.4 s steady state at width 1).
            self.forward(
                torch.full((width, BGE_M3_LONG_SEQ_LEN), pad_token_id, dtype=torch.long),
                attention_mask=torch.ones((width, BGE_M3_LONG_SEQ_LEN), dtype=torch.long),
            )

    # ---- shared encoder execution (uses self.model / self.device) ----
    def _run_encoder_chunk(self, padded_inputs: dict[str, Optional[torch.Tensor]]) -> ttnn.Tensor:
        """Runs the TT encoder on one already-padded chunk, returning the ttnn output.

        Uses ``self.model`` (a BgeM3Model) and ``self.device`` (the mesh device
        this wrapper created the model on; BgeM3Model does not store it). The
        output is forced to TILE_LAYOUT so downstream heads can consume it.
        """
        device = self.device
        token_type_ids = padded_inputs.get("token_type_ids")
        position_ids = padded_inputs.get("position_ids")
        output = self.model(
            input_ids=to_ttnn_ids(padded_inputs["input_ids"], device=device),
            attention_mask=to_ttnn_ids(padded_inputs["attention_mask"], device=device),
            token_type_ids=(to_ttnn_ids(token_type_ids, device=device) if token_type_ids is not None else None),
            position_ids=(to_ttnn_ids(position_ids, device=device) if position_ids is not None else None),
        )
        if output.layout != ttnn.TILE_LAYOUT:
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        return output

    def _encode_in_chunks(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Drives the device pad/chunk contract, delegating each chunk to
        ``self._forward_chunk``.

        Template method shared by both consumers of the XLM-RoBERTa encoder
        (bge-m3 embedding pooling and the bge-reranker cross-encoder). It slices
        the request into device-sized chunks, pads each to the fixed
        (padded_batch_size, padded_seq_len) device shape, and calls the
        overridable primitive ``self._forward_chunk(padded_inputs,
        chunk_batch_size)`` for every chunk. The per-chunk results are returned as
        a list in request order; the caller combines them (dict concat for
        bge-m3, tensor concat for the reranker).
        """
        pad_token_id = self.tokenizer.pad_token_id
        batch_size, seq_len = input_ids.shape
        padded_seq_len = get_padded_sequence_length(seq_len)

        chunk_outputs = []
        for start, end in iter_execution_ranges(batch_size, padded_seq_len):
            target_padded_batch_size = get_target_padded_batch_size(batch_size, padded_seq_len, end - start)
            padded_inputs = _pad_chunk_inputs(
                input_ids[start:end],
                _slice_optional_batch_tensor(attention_mask, start, end),
                _slice_optional_batch_tensor(token_type_ids, start, end),
                _slice_optional_batch_tensor(position_ids, start, end),
                padded_seq_len=padded_seq_len,
                padded_batch_size=target_padded_batch_size,
                pad_token_id=pad_token_id,
            )
            chunk_outputs.append(self._forward_chunk(padded_inputs, end - start))
        return chunk_outputs

    @abc.abstractmethod
    def _forward_chunk(self, padded_inputs: dict[str, Optional[torch.Tensor]], chunk_batch_size: int):
        """Per-chunk primitive of the ``_encode_in_chunks`` template method.

        Turns one already-padded device chunk (the ttnn output of
        ``_run_encoder_chunk``) into this model's per-chunk result. This is where
        the two models genuinely diverge, so there is no meaningful shared
        default: the bge-m3 embedding wrapper pools on device and returns a
        per-chunk embedding dict, while the bge-reranker cross-encoder transfers
        the raw last hidden state to host. Each subclass therefore implements it.
        """


########################################################
# ENCODER PAD / CHUNK / EXECUTION HELPERS
########################################################
# Shared device pad/chunk contract for the XLM-RoBERTa encoder, used by both the
# bge-m3 embedding wrapper and the bge-reranker-v2-m3 cross-encoder.

# Long-sequence path uses fixed 16-wide device execution regardless of max_batch_size.
BGE_M3_LONG_SEQ_LEN = 8192
BGE_M3_LONG_SEQ_CHUNK = 16
# Device widths the long-sequence path is allowed to execute at. Kernels are
# JIT-compiled per shape, and a compile costs tens of seconds on the request
# that triggers it, so the set of widths must stay small and fixed rather than
# following the request size exactly. Powers of two up to the 16-row chunk cap
# keep the wasted rows under 2x while bounding the number of shapes to five.
BGE_M3_LONG_SEQ_WIDTHS = (1, 2, 4, 8, 16)
# Short-sequence multi-request path pads to 32 rows for device execution.
BGE_M3_SHORT_SEQ_PADDED_BATCH = 32


def is_long_seq_8192(padded_seq_len: int) -> bool:
    return padded_seq_len == BGE_M3_LONG_SEQ_LEN


def get_target_padded_batch_size(original_batch_size: int, padded_seq_len: int, chunk_batch_size: int) -> int:
    """
    Device padding width for TT execution of one chunk.

    At the long sequence length the encoder must not exceed
    ``BGE_M3_LONG_SEQ_CHUNK`` rows per forward (the circular-buffer limit the
    upstream bge-m3 fix in #41397 was added for), but it does not have to reach
    it: device time is linear in the number of rows, so padding a narrower chunk
    up to 16 rows just pays for rows that are masked out anyway. Pad up to the
    smallest allowed width instead, which leaves the 16-row upper bound intact.

    The width is rounded up to one of ``BGE_M3_LONG_SEQ_WIDTHS`` rather than set
    to the exact row count, because each distinct width JIT-compiles its own
    kernels: an unseen width costs tens of seconds on the request that hits it
    (measured 62.6 s falling to 7.97 s over repeats at 16 rows, 28.1 s to 5.36 s
    at 11 rows). A fixed five-width set can be compiled once and never surprises
    a later request, while still wasting under 2x rows in the worst case.

    Measured on Blackhole p150 at seq 8192, one forward: 1 row 398 ms, 2 rows
    793 ms, 3 rows 1458 ms, 5 rows 2366 ms, 7 rows 3330 ms, 8 rows 3666 ms,
    16 rows 7295 ms - and the resulting logit is bit-identical at every width,
    because the padded rows carry attention_mask=0.
    """
    if is_long_seq_8192(padded_seq_len):
        capped = min(chunk_batch_size, BGE_M3_LONG_SEQ_CHUNK)
        return next(width for width in BGE_M3_LONG_SEQ_WIDTHS if width >= capped)
    if original_batch_size == 1:
        return 1
    return BGE_M3_SHORT_SEQ_PADDED_BATCH


def get_execution_chunk_size(original_batch_size: int, padded_seq_len: int) -> int:
    """
    Number of real batch rows per forward. For long sequences, capped at 16; a
    tail chunk holding fewer rows is padded only to its own row count (see
    get_target_padded_batch_size).
    """
    if is_long_seq_8192(padded_seq_len):
        return BGE_M3_LONG_SEQ_CHUNK
    if original_batch_size == 1:
        return 1
    return BGE_M3_SHORT_SEQ_PADDED_BATCH


def iter_execution_ranges(
    original_batch_size: int,
    padded_seq_len: int,
) -> Iterator[tuple[int, int]]:
    """Yields (start, end) batch slices for the original request."""
    chunk = get_execution_chunk_size(original_batch_size, padded_seq_len)
    for start in range(0, original_batch_size, chunk):
        yield (start, min(start + chunk, original_batch_size))


def _pad_tensor(tensor: torch.Tensor, padded_seq_len: int, pad_value: int = 0) -> torch.Tensor:
    batch_size, seq_len = tensor.shape
    if seq_len == padded_seq_len:
        return tensor

    padded = torch.full(
        (batch_size, padded_seq_len),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, :seq_len] = tensor
    return padded


def _pad_batch_tensor(tensor: torch.Tensor, padded_batch_size: int, pad_value: int = 0) -> torch.Tensor:
    batch_size = tensor.shape[0]
    if batch_size == padded_batch_size:
        return tensor

    padded = torch.full(
        (padded_batch_size, *tensor.shape[1:]),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:batch_size] = tensor
    return padded


def _slice_optional_batch_tensor(
    tensor: Optional[torch.Tensor],
    start: int,
    end: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[0] == 1:
        return tensor
    return tensor[start:end]


def to_ttnn_ids(ids: torch.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _pad_chunk_inputs(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    *,
    padded_seq_len: int,
    padded_batch_size: int,
    pad_token_id: int,
) -> dict[str, Optional[torch.Tensor]]:
    """Pads one batch slice to (padded_batch_size, padded_seq_len) for the device.

    Shared by every caller of the encoder: applies sequence-length padding
    (128/1024/2048/8192 alignment) and batch padding to the fixed device width.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": _pad_batch_tensor(
            _pad_tensor(input_ids, padded_seq_len, pad_value=pad_token_id),
            padded_batch_size,
            pad_value=pad_token_id,
        ),
        "attention_mask": _pad_batch_tensor(
            _pad_tensor(attention_mask, padded_seq_len, pad_value=0),
            padded_batch_size,
            pad_value=0,
        ),
        "token_type_ids": _pad_batch_tensor(
            _pad_tensor(token_type_ids, padded_seq_len, pad_value=0),
            padded_batch_size,
            pad_value=0,
        )
        if token_type_ids is not None
        else None,
        "position_ids": _pad_batch_tensor(
            _pad_tensor(position_ids, padded_seq_len, pad_value=pad_token_id),
            padded_batch_size,
            pad_value=pad_token_id,
        )
        if position_ids is not None
        else None,
    }
