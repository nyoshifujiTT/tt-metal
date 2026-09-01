# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT (tt-metal) vLLM adapter for Qwen3-ASR-1.7B.

Bridges the standalone tt-metal Qwen3-ASR pipeline (audio encoder + Qwen3 text
decoder, see ../tt/audio_encoder.py and ../tt/qwen3_asr_decoder.py) onto the TT
vLLM backend so neosophie/Qwen3-ASR-1.7B-JA is served through the OpenAI
``/v1/audio/transcriptions`` endpoint.

Design (decoder-only + multimodal-embeds injection, the qwen3_vl pattern):
  * multimodal processing / prompt building is delegated to vLLM's own
    ``Qwen3ASRMultiModalProcessor`` (registered below) and the
    ``SupportsTranscription`` classmethods copied from vLLM's reference
    ``qwen3_asr.py`` — so ``/v1/audio/transcriptions`` builds exactly the same
    token stream (audio_pad x N spliced between <|audio_start|>/<|audio_end|>).
  * ``prefill_forward`` runs, per user: mel -> TT audio encoder -> audio embeds,
    splices them into the host text-embedding table at the audio-token
    positions, and drives a paged single-user text prefill on pre-merged
    embeddings (base decoder's ``prepare_inputs_prefill`` embed step replaced).
  * ``decode_forward`` / ``allocate_kv_cache`` reuse the standard tt_transformers
    paths. The text decoder is a plain Qwen3-1.7B (1D RoPE); the ASR config
    carries an ``mrope_section`` so vLLM marks it ``uses_mrope`` and the plugin
    runs the ``request_specific_rope`` contract — we satisfy it by returning a
    zero ``rope_deltas`` and ignoring ``rope_deltas_all_users`` in decode.
"""
import os

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import Generator as TTTGenerator
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache
from models.tt_transformers.tt.model_config import ModelArgs

from .qwen3_asr_decoder import DECODE_TRACE as _DECODER_DECODE_TRACE
from .qwen3_asr_decoder import PREFILL_PIN_LEN as _DECODER_PREFILL_PIN_LEN
from .qwen3_asr_decoder import Qwen3ASRDecoder

from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration as _RefQwen3ASR,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

AUDIO_TOKEN_ID = 151676

# Decode tracing is ON by default (fast-dispatch/replay decode). It can be
# disabled with ``QWEN3ASR_DECODE_TRACE=0`` for platforms/boards where the
# tt-metal decode ND-hang (tt-inference-server #3105, PR #44118 regression;
# untraced eager decode still hangs per #40592) is reproducible under a
# reused decode trace.
#
# History: this defaulted OFF to mirror the upstream standalone server on the
# original board (10.160.20.103), where a persistent decode trace wedged the
# long-lived service within ~9-26 requests. That instability turned out to be
# specific to that board: on the delivery machine (p150, 172.27.44.85) decode
# trace ON sustained conc=4 soaks of 300 and 600 requests (900 total) with
# zero wedges and health=200 throughout, so the hang no longer gates the
# default. The capture is taken UP FRONT at warmup (never lazily on the first
# request, which would hit tt-metal's "Allocating device buffers is unsafe due
# to the existence of an active trace" path).
DECODE_TRACE = _DECODER_DECODE_TRACE

# All prefills are padded to ONE fixed bucket length so the long-lived server
# only ever compiles/executes a single prefill program shape. tt-metal has a
# length-keyed program-cache collision (see qwen3_asr_decoder.py and
# tenstorrent/tt-metal#49451): mixing prefills that land in different padded
# buckets (e.g. 128 vs 1024 from get_padded_prefill_len) corrupts/hangs the
# decoder for later requests. A single 1024-token pin covers up to ~30s clips
# (30s -> ~390 audio tokens + prompt < 1024), the WhisperFeatureExtractor cap.
# Override with QWEN3ASR_PREFILL_PIN if a deployment needs a different bucket.
PREFILL_PIN_LEN = _DECODER_PREFILL_PIN_LEN

# The audio encoder runs conv2d(batch_size=n_chunks) + SDPA(seq=n_chunks*13)
# where n_chunks = ceil(mel_frames / 100), i.e. its program shape scales with
# the audio length. Feeding variable-length mel to a long-lived server makes
# the encoder compile/execute many shapes and corrupts the device over a
# variable-length workload (the standalone server avoids this by pinning every
# request to a fixed audio length). Pin the mel to a fixed number of frames
# (multiple of 100 = whole chunks) so the encoder is single-shape. 1500 frames
# = 15s covers typical ASR turns; longer clips fall back to their own shape.
# The encoder output is sliced to the real audio-token count during splice, so
# padding frames do not change the transcript. Override via QWEN3ASR_MEL_PIN.
MEL_CHUNK = 100  # mel frames per encoder chunk (N_WINDOW*2)
PIN_MEL_FRAMES = int(os.environ.get("QWEN3ASR_MEL_PIN", "3000"))

def _pin_mel(mel, target_frames):
    """Return ``mel`` (n_mels, T) at exactly ``target_frames`` frames.

    vLLM hands us the mel already sliced to the real audio, so unlike the
    standalone server (which zero-extends the WAVEFORM and lets the feature
    extractor emit the tail) we have to supply the padding ourselves. Measured
    against the unpadded CPU reference on TED clips, the options rank:
    extractor's own tail 0.0904 < zero padding 0.0971 < constant silence column
    0.0988. We cannot recover the extractor's tail from a pre-sliced mel, so
    replicate the last real frame: it is a real frame from this recording rather
    than a synthetic constant, and the padded positions are masked out of
    attention anyway (see encode_mel's valid_frames).
    """
    n_frames = mel.shape[1]
    if n_frames >= target_frames:
        return mel[:, :target_frames]
    tail = mel[:, -1:].expand(mel.shape[0], target_frames - n_frames)
    return torch.cat([mel, tail], dim=1)

# Optionally gather the prompt embeddings on device (ttnn.embedding via the
# decoder's own embedding table) instead of indexing a host fp32 copy.
#
# OFF by default, and that default is measured rather than assumed. The CUDA
# reference does embed on the accelerator, and in isolation the device gather is
# ~10x faster here (10.5 ms host vs 1.1 ms device for a 149-token prompt, output
# bit-identical). But that microbenchmark repeats a single prompt length, while
# real traffic varies it. Feeding the raw length compiles a program per length
# and churns the program cache (TED throughput 6.08 -> 3.48 audio-s/s), and
# pinning to the prefill bucket to avoid that makes the device gather ~1024 rows
# for a ~149-token prompt, which still lands below the host path (5.79 vs 6.08).
#
# So on this hardware the host gather is genuinely the faster choice end-to-end.
# Set QWEN3ASR_DEVICE_EMBED=1 to use the device path (e.g. once a length-agnostic
# embedding avoids both the cache churn and the padded work).
DEVICE_EMBED = os.environ.get("QWEN3ASR_DEVICE_EMBED", "0").strip().lower() in ("1", "true", "yes", "on")


def _is_full_asr_snapshot(path):
    """True if ``path`` holds a full Qwen3-ASR checkpoint (audio_tower.* +
    thinker.* weights), as opposed to an already-extracted plain Qwen3 decoder."""
    try:
        import json

        cfg = json.load(open(os.path.join(path, "config.json")))
    except Exception:
        return False
    return cfg.get("model_type") == "qwen3_asr"


def _resolve_audio_snapshot(hf_config):
    """Directory that holds the full Qwen3-ASR HF snapshot (audio_tower.* +
    processor). Prefer an explicit env override, else HF_MODEL (the standard
    served weights dir), else the vLLM model path."""
    p = os.environ.get("QWEN3ASR_AUDIO_SNAPSHOT")
    if p:
        return p
    hf_model = os.environ.get("HF_MODEL")
    if hf_model and _is_full_asr_snapshot(hf_model):
        return hf_model
    return getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", "")


def _ensure_text_decoder(snapshot_dir):
    """Return a directory holding the extracted plain-Qwen3 text decoder.

    If ``snapshot_dir`` is already an extracted decoder (model_type=qwen3), use it
    as-is. If it is a full Qwen3-ASR snapshot, extract the thinker text decoder
    into a cache dir (idempotent) so ModelArgs can load a plain Qwen3 checkpoint.
    Mirrors the standalone server's auto-extraction so the adapter works with the
    standard served weights, not just a pre-extracted checkpoint.
    """
    if not _is_full_asr_snapshot(snapshot_dir):
        # Already a plain decoder checkpoint.
        return snapshot_dir

    import glob
    import hashlib
    import json
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    cache_root = os.environ.get("TT_CACHE_PATH") or os.path.join(
        os.path.expanduser("~"), ".cache", "qwen3_asr"
    )
    tag = hashlib.md5(os.path.abspath(snapshot_dir).encode()).hexdigest()[:12]
    out_dir = os.path.join(cache_root, f"qwen3_asr_text_decoder_{tag}")
    done_marker = os.path.join(out_dir, "model.safetensors")
    if os.path.exists(done_marker) and os.path.exists(os.path.join(out_dir, "config.json")):
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for fpath in sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors"))):
        with safe_open(fpath, "pt") as h:
            for k in h.keys():
                if k.startswith("thinker.model."):
                    sd["model." + k[len("thinker.model.") :]] = h.get_tensor(k)
                elif k == "thinker.lm_head.weight":
                    sd["lm_head.weight"] = h.get_tensor(k)
    if not sd:
        raise RuntimeError(
            f"No thinker.* text-decoder weights found in {snapshot_dir}; "
            "cannot extract Qwen3-ASR text decoder."
        )
    save_file(sd, done_marker, metadata={"format": "pt"})
    # Plain Qwen3-1.7B text config (matches the standalone extractor). Read the
    # thinker text_config from the snapshot to stay faithful to the weights.
    src_cfg = json.load(open(os.path.join(snapshot_dir, "config.json")))
    tc = src_cfg.get("thinker_config", {}).get("text_config", {})
    text_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": tc.get("hidden_size", 2048),
        "intermediate_size": tc.get("intermediate_size", 6144),
        "num_hidden_layers": tc.get("num_hidden_layers", 28),
        "num_attention_heads": tc.get("num_attention_heads", 16),
        "num_key_value_heads": tc.get("num_key_value_heads", 8),
        "head_dim": tc.get("head_dim", 128),
        "vocab_size": tc.get("vocab_size", 151936),
        "rope_theta": tc.get("rope_theta", 1000000.0),
        "max_position_embeddings": tc.get("max_position_embeddings", 65536),
        "hidden_act": tc.get("hidden_act", "silu"),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-06),
        "attention_bias": tc.get("attention_bias", False),
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }
    json.dump(text_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        src = os.path.join(snapshot_dir, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))
    return out_dir


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class TTQwen3ASRForConditionalGeneration(WarmupForwardMixin, SupportsMultiModal, SupportsTranscription):
    # --- SupportsTranscription contract (delegated to vLLM reference impl) ---
    supported_languages = _RefQwen3ASR.supported_languages
    supports_transcription = True
    supports_transcription_only = False

    get_placeholder_str = _RefQwen3ASR.get_placeholder_str
    get_generation_prompt = _RefQwen3ASR.get_generation_prompt
    get_speech_to_text_config = _RefQwen3ASR.get_speech_to_text_config
    post_process_output = _RefQwen3ASR.post_process_output

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
        **kwargs,
    ) -> int:
        """All-user KV-cache token capacity.

        The plugin sizes the paged KV cache from this number. Without it the
        fallback in ``get_num_available_blocks_tt`` is 131072 tokens, which for
        this model means 2052 blocks where 132 suffice (max_model_len 2048 x
        max_num_seqs 4 / block 64, plus the planner's per-user padding) - a 15.5x
        over-allocation that costs real startup time, because every block is a
        zero tensor written to device up front.

        ASR requests are bounded: a single clip is capped at the feature
        extractor's 30s window, so no request can exceed ``max_model_len``. The
        exact capacity is therefore known rather than guessed.
        """
        return max(int(max_model_len), 1) * max(int(max_num_seqs), 1)

    def __init__(self, decoder, model_args, mesh_device, audio_params, text_embed, tokenizer=None):
        # composition over the tt_transformers Generator (as qwen3_vl does)
        self._ttt_generator = TTTGenerator([decoder], [model_args], mesh_device, tokenizer=tokenizer)
        self.audio_params = audio_params
        self.text_embed = text_embed  # host embed_tokens.weight (vocab, dim), float32

    # --- passthrough properties expected by the plugin/generator ---
    @property
    def model(self):
        return self._ttt_generator.model

    @property
    def model_args(self):
        return self._ttt_generator.model_args

    @property
    def mesh_device(self):
        return self._ttt_generator.mesh_device

    @property
    def cache_path(self):
        return self._ttt_generator.model_args[0].model_cache_path

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        assert tt_data_parallel == 1, "Qwen3-ASR TT adapter currently supports tt_data_parallel=1"

        # Resolve the full Qwen3-ASR snapshot (audio_tower.* + thinker.*). Under
        # the standard server path HF_MODEL points at the full served snapshot;
        # under the standalone/manual path it may already be an extracted decoder.
        snap = _resolve_audio_snapshot(hf_config)
        # Text decoder: a plain Qwen3-1.7B checkpoint. Auto-extract it from the
        # full snapshot when needed (idempotent, cached), mirroring the
        # standalone server, so ModelArgs loads a config transformers recognizes.
        decoder_dir = _ensure_text_decoder(
            os.environ.get("HF_MODEL", "") if _is_full_asr_snapshot(os.environ.get("HF_MODEL", "")) else snap
        )

        # ModelArgs reads HF_MODEL; point it at the extracted decoder for the
        # duration of construction, then restore so the audio-tower resolution
        # and any downstream code see the original value.
        prev_hf_model = os.environ.get("HF_MODEL")
        os.environ["HF_MODEL"] = decoder_dir
        try:
            model_args = ModelArgs(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
            state_dict = model_args.load_state_dict()
            decoder = Qwen3ASRDecoder(
                args=model_args,
                dtype=ttnn.bfloat8_b,
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
                use_paged_kv_cache=True,
            )
        finally:
            if prev_hf_model is None:
                os.environ.pop("HF_MODEL", None)
            else:
                os.environ["HF_MODEL"] = prev_hf_model

        # Audio encoder weights from the full Qwen3-ASR snapshot (audio_tower.*)
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        sys.path.insert(0, os.path.join(here, "..", "reference"))
        import audio_encoder as tt_enc  # noqa: E402
        import audio_encoder_ref as ref  # noqa: E402

        w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
        audio_params = tt_enc.preprocess_weights(w, mesh_device)
        cls._encode_mel = staticmethod(tt_enc.encode_mel)

        # Host text-embedding table (for splicing audio embeds). Load from the
        # extracted decoder checkpoint (tie_word_embeddings=true).
        from safetensors import safe_open

        with safe_open(os.path.join(decoder_dir, "model.safetensors"), "pt") as h:
            text_embed = h.get_tensor("model.embed_tokens.weight").float()

        return cls(decoder, model_args, mesh_device, audio_params, text_embed, tokenizer=model_args.tokenizer)

    # --- KV cache ---
    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)

    # --- prefill ---
    def _merge_embeds(self, input_ids, input_features, feature_len):
        """input_ids: (S,) long; input_features: (128, T) mel; feature_len: int mel frames.
        Returns merged host embeds (S, dim) float32."""
        mel = input_features
        if mel.dim() == 3:
            mel = mel[0]
        mel = mel[:, :feature_len].float()
        # Pin the encoder input to a fixed frame count (whole chunks) so the audio
        # encoder always runs one program shape. PIN_MEL_FRAMES defaults to 3000
        # (=30s, the WhisperFeatureExtractor chunk cap), so real ASR turns are
        # never actually cut; longer single-shot clips are the caller's
        # responsibility to chunk (vLLM max_audio_clip_s). The padded positions
        # are masked out of attention below, and the real audio-token count
        # (n_mask) governs how many encoder rows are spliced in.
        n_frames = mel.shape[1]
        mel = _pin_mel(mel, PIN_MEL_FRAMES)
        # Pass the REAL frame count so the encoder masks the pinned padding out of
        # attention. Without the mask the padded tail participates as ordinary
        # key/value positions and shifts the encoder output.
        audio_embeds = type(self)._encode_mel(
            mel, self.audio_params, self.mesh_device, valid_frames=n_frames
        ).float()  # (Senc, dim)
        inp = self._embed_prompt(input_ids)
        mask = input_ids == AUDIO_TOKEN_ID
        n_mask = int(mask.sum())
        if audio_embeds.shape[0] > n_mask:
            audio_embeds = audio_embeds[:n_mask]
        elif audio_embeds.shape[0] < n_mask:
            pad = torch.zeros(n_mask - audio_embeds.shape[0], audio_embeds.shape[1])
            audio_embeds = torch.cat([audio_embeds, pad], 0)
        inp[mask] = audio_embeds.to(inp.dtype)
        return inp

    def _embed_prompt(self, input_ids):
        """Prompt token ids -> embeddings (S, dim) as torch.

        Prefers the decoder's on-device embedding table, matching what the CUDA
        reference does in embed_input_ids. Falls back to the host table when
        QWEN3ASR_DEVICE_EMBED=0 or if the device path is unavailable for this
        model build.
        """
        if DEVICE_EMBED:
            embd = getattr(self.model[0], "embd", None)
            if embd is not None:
                S = input_ids.shape[0]
                # Pin the embedding to the prefill bucket, exactly like the mel
                # input (PIN_MEL_FRAMES) and the prefill itself (PREFILL_PIN_LEN).
                # Feeding the raw per-request length here compiles a new program
                # for every distinct prompt length and churns the program cache:
                # measured 6.08 -> 3.48 audio-s/s on TED before pinning, versus
                # 6.14 after. Only the first S rows are ever read by the caller.
                S_pin = max(get_padded_prefill_len(S), PREFILL_PIN_LEN) if S <= PREFILL_PIN_LEN else get_padded_prefill_len(S)
                padded_ids = torch.zeros(S_pin, dtype=torch.int32)
                padded_ids[:S] = input_ids.to(torch.int32)
                tt_ids = ttnn.from_torch(
                    padded_ids.reshape(1, 1, 1, S_pin),
                    device=self.mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                tt_emb = embd(tt_ids)
                out = ttnn.to_torch(tt_emb, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
                ttnn.deallocate(tt_emb)
                ttnn.deallocate(tt_ids)
                # (1,1,S,dim) -> (S,dim); the decoder consumes bf16, and the
                # splice below writes audio rows in the same dtype.
                return out.reshape(S_pin, -1)[:S, : self.text_embed.shape[1]].float()

        # Host fallback. Advanced indexing already returns a fresh tensor, so no
        # .clone() here: that used to copy the whole (S, 2048) prompt embedding a
        # second time (83.0 ms -> 45.6 ms per request when removed).
        return self.text_embed[input_ids]

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, enable_trace=False, **kwargs):
        gen = self._ttt_generator
        decoder = self.model[0]
        batch = tokens.shape[0]
        vocab = self.model_args[0].vocab_size
        output_logits = torch.zeros(batch, 1, vocab)

        # audio features are gathered by the plugin as per-user lists
        feats = kwargs.get("input_audio_features")
        feat_lens = kwargs.get("audio_feature_lengths")

        for user_id in range(batch):
            seq_len = int(prompt_lens[user_id])
            last_token_idx = seq_len - 1
            input_ids = tokens[user_id, :seq_len].to(torch.int64)

            uf = feats[user_id] if feats is not None else None
            if isinstance(uf, list):
                uf = uf[0] if uf else None
            ufl = feat_lens[user_id] if feat_lens is not None else None
            if isinstance(ufl, list):
                ufl = ufl[0] if ufl else None
            if ufl is not None and torch.is_tensor(ufl):
                ufl = int(ufl.reshape(-1)[0])
            elif ufl is not None:
                ufl = int(ufl)

            if uf is not None:
                merged = self._merge_embeds(input_ids, uf, ufl if ufl is not None else uf.shape[-1])
            else:
                # Text-only prompt: same embedding path, just no audio splice.
                merged = self._embed_prompt(input_ids)

            # Pin to the fixed bucket (round the natural padded length UP to the pin
            # so every request shares one prefill program shape). If a prompt is
            # larger than the pin (rare, >~30s single-shot), fall back to the
            # natural bucket for that request.
            natural = get_padded_prefill_len(seq_len)
            prefill_seq_len = max(natural, PREFILL_PIN_LEN) if natural <= PREFILL_PIN_LEN else natural
            if merged.shape[0] < prefill_seq_len:
                pad = torch.zeros(prefill_seq_len - merged.shape[0], merged.shape[1], dtype=merged.dtype)
                merged = torch.cat([merged, pad], 0)

            if page_table is not None:
                page_table_user = gen._get_prefill_user_page_table(
                    page_table[user_id : user_id + 1], kv_cache[0], seq_len
                )
            else:
                page_table_user = None

            # We drive one single-user prefill per request. page_table_user is
            # the 1-row slice for THIS user (row 0 holds this user's block IDs),
            # so the KV-cache fill must address row 0: paged_fill_cache treats
            # batch_idx as the page-table row index and this page table has
            # exactly one row. Passing the GLOBAL user_id made batch_idx=user_id
            # index past the 1-row page table (TT_FATAL "Batch idx must be within
            # the page_table batch size") for every request after slot 0, which
            # crashed EngineCore under max_num_seqs>1. Use user_id=0 for the
            # single-user path; the correct physical KV block is still selected
            # by the block IDs carried in page_table_user row 0.
            # Qwen3ASRDecoder.prepare_inputs_prefill follows the qwen3_vl
            # "tokens is actually embeddings" pattern, so hand it the merged
            # embeddings directly; page_table/batch_size/user_id pass straight
            # through to the base preparation.
            inputs = decoder.prepare_inputs_prefill(
                merged.unsqueeze(0),
                page_table=page_table_user,
                batch_size=1,
                user_id=0,
            )
            prefill_input, rot_g, rot_l, page_table_tt, *_ = inputs
            tt_logits = decoder.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_g,
                rot_mats_local=rot_l,
                user_id=0,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache[0],
                batch_size=1,
            )
            logits = decoder.process_output_prefill(
                ttnn.from_device(tt_logits), last_token_idx=(last_token_idx % 32)
            )
            output_logits[user_id] = logits
            ttnn.deallocate(tt_logits)
            # Free the per-request merged-embeds prefill input. It is a fresh
            # ~1024x2048 bf16 device tensor built every request and is consumed
            # entirely by the prefill forward (decode reads only the paged KV
            # cache, not this tensor), so leaving it allocated leaks DRAM across
            # requests until the device fragments/wedges. Safe because prefill is
            # never traced (trace_mode=none / prefill warmup is a no-op).
            try:
                ttnn.deallocate(prefill_input)
            except Exception:
                pass

        rope_deltas = torch.zeros(batch, dtype=torch.int64)
        return output_logits, rope_deltas

    # --- decode ---
    def decode_forward(self, *args, **kwargs):
        kwargs.pop("rope_deltas_all_users", None)  # 1D RoPE: no mrope deltas needed
        # Force the adapter's own trace policy regardless of the runner's
        # trace_mode: ON by default (fast-dispatch/replay decode), OFF only when
        # QWEN3ASR_DECODE_TRACE=0. Pinning it here (rather than following the
        # plugin's global trace_mode) keeps the decode-trace decision a single,
        # explicit adapter-level switch.
        kwargs["enable_trace"] = DECODE_TRACE
        return self._ttt_generator.decode_forward(*args, **kwargs)

    # --- warmup ---
    # Prefill is NOT traced (the qwen3_vl pattern): every request runs the audio
    # encoder, which allocates fresh device buffers (conv2d / from_torch / ...).
    # Capturing a prefill trace around those dynamic allocations is unsupported,
    # so prefill warmup is a no-op.
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False) -> None:
        logger.warning("Warmup model prefill is a no-op for Qwen3-ASR TT adapter (prefill is not traced)")

    # Decode warmup runs so every decode op is COMPILED into the program cache up
    # front (this alone keeps steady-state stable, per the worklog:
    # "trace_mode=none + warmup -> 50/50, 30/30 no wedge"), and with the default
    # DECODE_TRACE=1 it also CAPTURES the decode trace before any request, so the
    # capture never happens lazily on the first one -- that would hit tt-metal's
    # "Allocating device buffers is unsafe due to the existence of an active
    # trace" path.
    #
    # The runner calls this TWICE: once with enable_trace=False to compile, then
    # again with enable_trace=True to capture. Forcing the trace on in the first
    # call made both passes do the capture work, and the compile pass alone cost
    # 162 s of a 327 s startup. Honour the caller's flag and only override it to
    # disable tracing when DECODE_TRACE=0.
    def warmup_model_decode(self, *args, **kwargs) -> None:
        if not DECODE_TRACE:
            kwargs["enable_trace"] = False
        super().warmup_model_decode(*args, **kwargs)

    # --- device output helpers used by the plugin ---
    def read_decode_output(self, tt_out, async_read=False):
        return self._ttt_generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens=False):
        return self._ttt_generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def __del__(self):
        try:
            self._ttt_generator.__del__()
        except Exception:
            pass
