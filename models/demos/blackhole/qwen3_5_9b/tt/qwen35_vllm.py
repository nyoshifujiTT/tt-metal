# tt/qwen35_vllm.py
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM wrapper for Qwen3.5-9B: a thin tt_transformers Generator subclass.

Qwen3.5-9B is a hybrid model: 8 full-attention layers (paged KV, stateless across prefill) plus
24 Gated DeltaNet (GDN) layers carrying a recurrent + conv state that accumulates across the whole
sequence. Standard tt_transformers models are stateless beyond paged KV, so the standard contract
assumes token-padding is numerically free and the decode trace is position-general. Neither holds
for GDN — which is the root of every place this model must diverge.

What conforms to the standard Generator (Llama/DeepSeek/Qwen-VL):
  - Decode, end to end. The model implements the standard decode contract (prepare_inputs_decode /
    ttnn_decode_forward / process_output_decode); current_pos and page_table are device input
    tensors the standard replay updates per step. The inherited WarmupForwardMixin captures the
    decode trace at position 0 during warmup; Generator.decode_forward replays it at serving.

What must diverge, and why:
  - Prefill bucketing. GDN forbids token-padding inside its recurrent scan, so prefill pads to a
    fixed bucket and passes an EXACT valid_len (the standard contract only plumbs get_last_token,
    floored to a 32-multiple — too lossy for the GDN mask). See model.prefill_masked_bucket.
  - Chunk-outer trace. At 128K a whole-length prefill trace is infeasible; we capture ONE
    2048-token chunk trace and replay it N times, carrying GDN/KV state in place. See
    model.prefill_traced_chunked / capture_prefill_trace_chunked.
  - State-reset guard. The stock trace capture runs the forward twice (compile + capture), which
    advances GDN state non-idempotently. Harmless only because every new sequence re-zeros the
    bound GDN buffers before consuming a token (model._reset_gdn_state_for_new_sequence).

Generator drives decode; all prefill is model-owned (prefill_masked_bucket / prefill_traced_chunked)
via prefill_dispatch. GDN recurrent state and the attention KV caches are model-bound, so the
kv_cache contract param is accepted but unused.
"""
import math
import os
from typing import Mapping, Optional

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.common import create_tt_model
from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prefill_dispatch
from models.tt_transformers.tt.generator import Generator

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

_PREFILL_WARMUP_CHUNK = 2048
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    # Enable native image input (video stays disabled). vLLM's Qwen3VL processor turns both
    # image_url and base64 data URIs into pixel_values + image_grid_thw, so URL/base64 parity
    # is handled by the standard processor (no custom handling here).
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1, "video": 0}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen35ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B/27B on Blackhole (native image via HF vision tower)."""

    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):
        # The OpenAI chat renderer resolves the TT-prefixed arch to THIS class and calls
        # get_placeholder_str to insert the image marker into the prompt. The base
        # SupportsMultiModal default returns None (drops the image), so the MM processor
        # would find no <|image_pad|> to expand. Mirror vLLM's native Qwen3VL markers so
        # rendering inserts <|vision_start|><|image_pad|><|vision_end|>, which the standard
        # Qwen3VL processor then expands to the correct number of image tokens.
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV-cache token capacity = served context length (B=1).

        Qwen3.5/3.6 serve one sequence at a time (max_concurrency=1), so the whole
        paged KV cache belongs to a single user and its capacity is exactly the
        served context length — max_model_len, i.e. the catalog's max_context.
        Deriving from max_model_len, instead of the inherited 131072 fallback, lets
        these models serve at the full requested ISL (e.g. 256K = 262144): the
        chunk-outer prefill and the full-KV page-table sizing in
        warmup_model_prefill already scale to whatever KV cache vLLM allocates.
        The * max_num_seqs keeps the all-user semantics correct if B ever grows.
        """
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Resolution order: MODEL_WEIGHTS_DIR (tt-inference-server Docker convention) →
        # HF_MODEL → vLLM's hf_config._name_or_path (the hub id). Resolve a hub id to a
        # local snapshot dir (AutoConfig on a bare hub id is unreliable in this transformers).
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(name_or_path)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        inst = cls([model], [args], mesh_device)
        # Disable the generator's SPLIT sampling-trace path. With it on (default), decode_forward
        # forces sampling_module.enable_internal_trace=True and _capture_decode_trace_text calls
        # sampling_module.capture_trace(tt_out_tok=device_inputs[0]) — this port's rank-2 decode
        # token tensor — violating the sampling op's rank-4 preallocated-output contract. With it
        # off, sampling runs INSIDE the model decode trace (ttnn_decode_forward → sample), which
        # is what this port implements.
        inst.enable_split_sampling = False
        # Qwen3-VL pattern: build the on-device (TT) vision encoder NOW, during model init,
        # i.e. BEFORE warmup captures the prefill/decode traces. If the vision encoder is built
        # lazily on the first request (after the decode trace is parked), its device-buffer
        # allocations collide with the parked trace and every warm request returns a frozen/
        # wrong vision embedding. Constructing it up front makes the trace capture happen against
        # the post-vision allocator state, so eager vision at request time no longer corrupts it.
        if os.getenv("TT_QWEN35_NATIVE_VISION") and os.environ.get("QWEN35_TT_VISION", "1") != "0":
            try:
                inst._ensure_tt_vision()
            except Exception as _e:  # pragma: no cover - fall back to lazy build if eager fails
                logger.warning(f"[QWEN35_VISION] eager vision build failed ({_e!r}); will build lazily")
        return inst

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs."""
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    # ── Native image (multimodal) support ─────────────────────────────
    def _ensure_hf_vision(self):
        """Lazily load the HF reference model (vision tower + text embed_tokens) once.

        The TT text decoder has no vision tower; we run the reference Qwen3.5 vision encoder
        (model.model.visual) and text embedding (model.model.language_model.embed_tokens) in
        torch on host to produce merged input embeddings, then feed them into the TT decoder.
        This is the reference multimodal computation (perf tradeoff of host vision), not a bypass.
        """
        if getattr(self, "_hf_ref", None) is not None:
            return
        import os as _os
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as _Ref
        mp = _os.environ.get("MODEL_WEIGHTS_DIR") or _os.environ.get("HF_MODEL") or self.model[0].args.CKPT_DIR
        cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
        ref = _Ref.from_pretrained(mp, config=cfg, torch_dtype=torch.float32, device_map="cpu").eval()
        self._hf_ref = ref
        self._hf_visual = ref.model.visual
        self._hf_embed_tokens = ref.model.language_model.embed_tokens
        self._image_token_id = int(getattr(ref.config, "image_token_id", 248056))

    def _ensure_tt_vision(self):
        """Lazily build the on-device (TT) vision encoder once.

        The heavy Qwen3.5 vision transformer blocks (depth=27 ViT) are executed on the p150x4
        mesh via the image-shipped device implementation
        (models/demos/qwen3_vl/tt/DropInVisionTransformer), which is a drop-in for the HF
        Qwen visual tower: forward(pixel_values, grid_thw). Patch-embed / pos-embed / rope
        preprocessing stay on host (tiny), matching the reference generator_vllm wiring. Only
        the transformer blocks + patch-merger — the >99% of vision FLOPs — run on TT.
        """
        if getattr(self, "_tt_visual", None) is not None:
            return
        # HF reference visual is required as the preprocessing/weight source for the TT wrapper.
        self._ensure_hf_vision()
        from models.demos.qwen3_vl.tt.model import DropInVisionTransformer
        from models.demos.qwen3_vl.tt.model_config import VisionModelArgs
        model = self.model[0]
        vc = self._hf_ref.config.vision_config
        try:
            from models.tt_transformers.tt.model_config import DecodersPrecision as _DP
            _vis_opt = _DP.performance(vc.depth, model.args.CKPT_DIR)
        except Exception:
            _vis_opt = None
        vision_model_args = VisionModelArgs(
            self.mesh_device,
            max_batch_size=getattr(model.args, "max_batch_size", 1),
            max_seq_len=getattr(model.args, "max_seq_len", 4096),
            optimizations=_vis_opt,
        ) if _vis_opt is not None else VisionModelArgs(
            self.mesh_device,
            max_batch_size=getattr(model.args, "max_batch_size", 1),
            max_seq_len=getattr(model.args, "max_seq_len", 4096),
        )
        # Match the served checkpoint's vision depth (Qwen3.5-27B ViT depth=27).
        vision_model_args.hf_config.vision_config.depth = vc.depth
        # DropInVisionTransformer consumes the HF visual state_dict (auto-remapped internally)
        # and runs the transformer blocks on the mesh device.
        self._tt_visual = DropInVisionTransformer(self._hf_visual, vision_model_args)
        self._tt_vision_out_hidden = int(vc.out_hidden_size)

    def _tt_vision_encode(self, pixel_values, image_grid_thw):
        """Run the TT device vision encoder and return image embeds as torch [num_img_tokens, H].

        Mirrors reference generator_vllm.forward_single_user usage: the wrapper returns a
        mesh-sharded (dim=1) ttnn tensor over out_hidden_size, which we gather back to host with
        ConcatMeshToTensor(dim=1) and slice to the true out_hidden_size.
        """
        self._ensure_tt_vision()
        # Ensure any parked/async decode-trace work on cq0 has fully completed before issuing
        # eager vision ops, so vision does not read/execute against in-flight trace state
        # (warm-request vision-output-freeze fix).
        ttnn.synchronize_device(self.mesh_device)
        grid = image_grid_thw
        if grid.dim() == 1:
            grid = grid.unsqueeze(0)
        image_embeds_tt, _deepstack = self._tt_visual.forward_single_user(
            pixel_values.to(torch.float32), grid_thw=grid.to(torch.int32)
        )
        out = ttnn.to_torch(
            image_embeds_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1),
        )
        ttnn.deallocate(image_embeds_tt)
        ttnn.synchronize_device(self.mesh_device)
        # out: [num_img_tokens, out_hidden_size * num_shards_along_dim1]; keep the real width.
        out = out[:, : self._tt_vision_out_hidden].to(torch.float32)
        return out

    def _build_mm_input_embeds(self, input_ids, pixel_values, image_grid_thw):
        """Return (input_embeds [1,T,hidden] torch, (cos,sin) mrope tuple) for an image prompt.

        input_ids: torch [1, T]. pixel_values/image_grid_thw: as produced by the vLLM Qwen3VL
        processor. Uses transformers' native inputs_embeds + masked_scatter merge and get_rope_index
        for mrope position ids.
        """
        self._ensure_hf_vision()
        with torch.no_grad():
            if os.environ.get("QWEN35_TT_VISION", "1") != "0":
                # On-device (TT) vision encoder — the heavy ViT runs on the p150x4 mesh.
                import time as _t
                _t0 = _t.time()
                image_embeds = self._tt_vision_encode(pixel_values, image_grid_thw)
                print(f"[QWEN35_VISION] backend=TT encode={_t.time()-_t0:.3f}s "
                      f"grid={image_grid_thw.tolist()} tokens={image_embeds.shape[0]}", flush=True)
            else:
                # Host (CPU torch) reference vision encoder — fallback / parity baseline.
                import time as _t
                _t0 = _t.time()
                ve = self._hf_visual(pixel_values.to(torch.float32), grid_thw=image_grid_thw)
                image_embeds = ve.pooler_output if hasattr(ve, "pooler_output") else (ve[0] if isinstance(ve, (tuple, list)) else ve)
                image_embeds = image_embeds.to(torch.float32)  # [num_img_tokens, hidden]
                print(f"[QWEN35_VISION] backend=CPU encode={_t.time()-_t0:.3f}s "
                      f"grid={image_grid_thw.tolist()} tokens={image_embeds.shape[0]}", flush=True)
            inputs_embeds = self._hf_embed_tokens(input_ids.to(torch.long))  # [1, T, hidden]
            mask = (input_ids == self._image_token_id)
            n_img = int(mask.sum().item())
            assert n_img == image_embeds.shape[0], (
                f"image token count {n_img} != vision embeds {image_embeds.shape[0]}"
            )
            merged = inputs_embeds.clone()
            merged[mask] = image_embeds.to(merged.dtype)
            # mrope position ids for the whole sequence, then cos/sin in the rope_tp layout.
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids[mask] = 1
            attention_mask = torch.ones_like(input_ids)
            position_ids, _ = self._hf_ref.model.get_rope_index(
                input_ids=input_ids.to(torch.long),
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            x = torch.empty((1,), dtype=torch.float32)
            cos, sin = self._hf_ref.model.language_model.rotary_emb(x, position_ids)
        return merged, (cos, sin)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        model = self.model[0]

        def _nonempty(v):
            # A genuine multimodal payload requires a non-empty tensor/list. vLLM may pass
            # pixel_values=None (or an empty list/0-elem tensor) for text-only requests and
            # during text warmup; treat those as text so we do not enter the image path.
            if v is None:
                return False
            if isinstance(v, (list, tuple)):
                return len(v) > 0 and any(_nonempty(x) for x in v)
            if hasattr(v, "numel"):
                return v.numel() > 0
            return True

        pv = kwargs.get("pixel_values")
        grid = kwargs.get("image_grid_thw")
        # Require BOTH pixel_values AND image_grid_thw to be genuinely present. The vision
        # merge/rope needs grid_thw; a pixel_values without grid (or vice versa) is not a
        # valid image request and must not enter the MM prefill path.
        has_mm = _nonempty(pv) and _nonempty(grid)
        if has_mm and model.num_devices > 1:
            return self._prefill_forward_tp_mm(model, tokens, page_table, prompt_lens, kwargs)
        if model.num_devices > 1:
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens)
        logits = prefill_dispatch(model, tokens, page_table, prompt_lens, use_trace=kwargs.get("enable_trace", False))
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section;
        # zero deltas are correct for our text-only port.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, rope_deltas

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens):
        """TP (B=1) paged prefill via the model-owned masked fixed-bucket path.

        prefill_traced_chunked rounds the prompt up to a fixed bucket and masks the GDN to the
        EXACT valid_len, so prefill runs one of a bounded, pre-warmed program set (the
        compile-clobbers-trace fix) — for <=2048 prompts it is entirely the masked bucket (no
        chunk trace needed). Longer prompts replay the chunk-outer trace (Milestone B). Returns
        host logits [1, 1, vocab] gathered to a single replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logits = model.prefill_traced_chunked(tokens, page_table, actual_len=T)  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_mm(self, model, tokens, page_table, prompt_lens, kwargs):
        """TP prefill for an IMAGE request: build merged text+vision input embeddings on host
        (HF reference vision + embed_tokens) and feed them into the TT decoder via the model's
        input_embeds path. Text decode is the SAME working reference path; only the input
        embeddings carry the image. Returns (logits [1,1,vocab], rope_deltas)."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        input_ids = tokens[:1, :T].to(torch.long)

        # Gather this request's pixel_values / grid (vLLM passes per-user lists).
        pv = kwargs["pixel_values"]
        grid = kwargs["image_grid_thw"]
        pv0 = pv[0] if isinstance(pv, (list, tuple)) else pv
        if isinstance(pv0, (list, tuple)):
            pv0 = torch.concat([p for p in pv0], dim=0)
        grid0 = grid[0] if isinstance(grid, (list, tuple)) else grid
        if isinstance(grid0, (list, tuple)):
            grid0 = torch.stack([g.to(dtype=torch.int32) for g in grid0], dim=0)
        grid0 = grid0.to(torch.int32)
        if grid0.dim() == 1:
            grid0 = grid0.unsqueeze(0)

        merged_embeds, _rope = self._build_mm_input_embeds(input_ids, pv0, grid0)
        # merged_embeds: [1, T, hidden] torch. Feed into the reference short-prompt masked path.
        # rope_cos_sin left None for now (linear positions); mrope override added if image
        # localization proves insufficient.
        logits = model.prefill_traced_chunked(
            input_ids, page_table, actual_len=T, input_embeds=merged_embeds, rope_cos_sin=None
        )
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        return logits, torch.zeros(1, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
        # Both single-device and TP serve TRACED decode. The decode trace is valid for TP
        # because GDN state lives in fixed in-place buffers (reset_state_inplace preserves
        # addresses), and it no longer collides with prefill: TP prefill runs the bounded,
        # pre-warmed masked-bucket program set (warmed before the trace parks), so a request
        # never compiles a new program that could clobber the parked decode trace.
        # Standard path (default): the decode trace is captured at position 0 during warmup by the
        # inherited WarmupForwardMixin, then replayed here by Generator.decode_forward — identical
        # to Llama/DeepSeek/Qwen-VL. current_pos and page_table are device input tensors the
        # standard replay updates per step, so a pos-0 capture is position-general.
        #
        # Dormant fallback (QWEN35_DECODE_PRIME=1, paired with the warmup_model_decode no-op): if
        # the pos-0 warmup capture ever proves insufficient for GDN, lazily capture the decode
        # trace on the FIRST decode, at the REAL post-prefill position and recurrent state, via
        # prime_decode_trace (GDN-state snapshot/restore around the stock two-pass capture). This
        # should not be needed — every new sequence re-zeros the GDN state at prefill
        # (model._reset_gdn_state_for_new_sequence), so the warmup capture's residue can't leak in.
        if os.environ.get("QWEN35_DECODE_PRIME") == "1" and not getattr(self, "_decode_trace_primed", False):
            from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prime_decode_trace

            # Set the guard BEFORE calling so the re-entrant decode_forward it triggers skips
            # priming and just performs the capture.
            self._decode_trace_primed = True
            tokens = args[0] if args else kwargs.get("tokens")
            start_pos = args[1] if len(args) > 1 else kwargs.get("start_pos")
            prime_decode_trace(self, self.model[0], tokens, start_pos, kwargs.get("page_table"))

        # On-device sampling rope-freeze guard: this port drives decode rope from HOST-computed
        # cos/sin passed in per step (prepare_decode_inputs_host), NOT a device rope table with
        # ttnn.plus_one. The stock Generator._decode_forward_trace_text only re-copies host
        # inputs when `reset_inputs` is set, and with sampling_on_device=True it computes
        # reset_inputs = (reset_batch or not sampling_on_device) = False — so cos/sin would
        # freeze at the captured position and every sampled token would use pos-0 rope. Force a
        # host-input refresh each sampling step by clearing prev_page_table (the Generator then
        # takes its `prev_page_table is None` branch and re-copies inputs). Cost is one host->
        # device copy per step — the same refresh the non-sampling path already does every step.
        if kwargs.get("sampling_params") is not None and getattr(self.model[0], "sampling", None) is not None:
            self.prev_page_table = None
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Single-device AND TP share this path: capture_prefill_trace_chunked dispatches to its
        # TP fork (_capture_prefill_trace_chunked_tp) when num_devices>1. The capture compiles the
        # per-chunk programs AND warms the bounded masked-bucket program set (short prompts + the
        # long-prompt tail) before the decode trace is parked, so a real request only ever replays
        # already-compiled programs — the compile-clobbers-trace fix — and long prompts replay the
        # chunk-outer trace (bounded host dispatch, the 128K path) instead of the eager fallback.
        #
        # The plugin's warmup_model() is two-phase: it calls this first with
        # enable_trace=False (compile), then resets ``already_warmed_up_prefill``
        # and calls again with enable_trace=True (capture). Only the traced phase
        # captures the chunk-prefill trace; capture_prefill_trace_chunked compiles
        # its own programs before capturing, so the non-traced phase is a no-op.
        # The guard attribute MUST be named ``already_warmed_up_prefill`` so the
        # plugin's between-phase reset (model_runner.warmup_model) clears it.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the captured chunk-trace page table to the FULL allocated KV cache
        # (max_model_len worth of blocks), so served ISL matches the tt-metal demo's
        # 128K — not a hardcoded 4096. The chunk-outer trace still captures only one
        # _PREFILL_WARMUP_CHUNK-token chunk, so this is just a larger page-table
        # tensor, not more compute/trace memory. kv_cache[0][0] is the first attention
        # layer's K cache, shape [max_num_blocks, n_kv_heads, block_size, head_dim].
        if kv_cache:
            # Round up to a multiple of 32: the paged/chunked SDPA requires the page-table
            # width (stick size) to be % 32 == 0 (the allocated block count carries a slack
            # block, e.g. 257, which is not 32-aligned). prefill_traced_chunked pads each
            # request's page table up to this width before replay.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, *args, **kwargs):
        # Standard path (default): defer to the inherited WarmupForwardMixin, which captures the
        # paged-SDPA + GDN decode trace at position 0 during warmup. Qwen sets
        # _supports_on_device_sampling=False, so the orchestrator passes can_sample_on_device=False
        # and exactly one greedy trace is captured; serving replays it with per-step input updates.
        #
        # Dormant fallback (QWEN35_DECODE_PRIME=1): skip the warmup capture entirely; decode_forward
        # primes the trace lazily at the real post-prefill position instead. See decode_forward.
        if os.environ.get("QWEN35_DECODE_PRIME") == "1":
            return
        return super().warmup_model_decode(*args, **kwargs)
