#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Corpus-level eval of the metal/demo path, matching asr_ja_eval.py's metric.

The two front-ends over the same weights are not bit-exact (tt-metal documents
that decode SDPA reduction counts change the result), so the honest parity claim
is "same corpus CER", not "same string on one clip". This runs the demo decoder
over the same manifest the served-path eval uses and reports the same numbers.
"""
import argparse, json, os, re, sys, time, unicodedata

import torch
import soundfile as sf
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.whisper import WhisperFeatureExtractor

MDIR = os.environ["TT_METAL_HOME"] + "/models/demos/audio/qwen3_asr"
sys.path.insert(0, MDIR + "/reference")
sys.path.insert(0, MDIR + "/tt")

import ttnn  # noqa: E402
from models.tt_transformers.tt.model_config import ModelArgs  # noqa: E402
from models.tt_transformers.tt.common import PagedAttentionConfig  # noqa: E402
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache  # noqa: E402
import audio_encoder as tt_enc  # noqa: E402
import audio_encoder_ref as ref  # noqa: E402
from qwen3_asr_decoder import Qwen3ASRDecoder, decoder_weight_dtype  # noqa: E402

ASR_TAG = "<asr_text>"
PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
MEL_PIN = int(os.environ.get("QWEN3ASR_MEL_PIN", "3000"))

# The served (vLLM) path always runs paged KV, so the demo front-end has to run it
# too or the two are not comparable: paged and non-paged decode dispatch to
# different SDPA kernels (paged_scaled_dot_product_attention_decode vs
# scaled_dot_product_attention_decode) and produce different transcripts for the
# same audio. Defaults mirror the server launch (--block_size 64).
PAGED_KV = os.environ.get("QWEN3ASR_EVAL_PAGED_KV", "1").strip().lower() in ("1", "true", "yes", "on")
PAGE_BLOCK_SIZE = int(os.environ.get("QWEN3ASR_EVAL_PAGE_BLOCK", "64"))
PAGE_MAX_BLOCKS = int(os.environ.get("QWEN3ASR_EVAL_PAGE_MAX_BLOCKS", "2048"))

# The served path builds ModelArgs with max_batch_size = --max_num_seqs (4 in the
# shipped tt-inference-server spec). max_batch_size selects the decode program
# shape, so a demo built at batch 1 runs a different decode kernel configuration
# than the server even for a single request.
EVAL_MAX_BATCH = int(os.environ.get("QWEN3ASR_EVAL_MAX_BATCH", "4"))

# Sampling parameters the served-path eval client sends (the deployment preset).
# Scoring the demo with plain greedy while the server runs with a repetition
# penalty compares two different decoding rules, not two front-ends.
REPETITION_PENALTY = float(os.environ.get("QWEN3ASR_EVAL_REPETITION_PENALTY", "1.1"))


def build_paged_kv(model, max_seq_len):
    """Return (page_table, kv_cache) for the demo's single user, or (None, None)."""
    if not PAGED_KV:
        return None, None
    # use_paged_kv_cache=True means the *caller* owns the cache (that is how vLLM
    # drives it), so allocate it exactly like the served path does rather than
    # reaching for attention.layer_past, which is only populated in the
    # self-allocating non-paged mode.
    args = model.args
    # Same per-buffer shape the plugin computes in _kv_cache_shape:
    # (num_blocks, num_kv_heads // min(num_devices, num_kv_heads), block_size, head_size).
    num_devices = max(int(getattr(args, "num_devices", 1) or 1), 1)
    shape = (
        PAGE_MAX_BLOCKS,
        args.n_kv_heads // min(num_devices, args.n_kv_heads),
        PAGE_BLOCK_SIZE,
        args.head_dim,
    )
    # allocate_vllm_kv_cache returns list[submesh][layer][k, v]. Generator indexes
    # it as kv_cache[model_id], so keep the submesh dimension; stripping it here
    # makes the ops read a layer's [k, v] pair as if it were the submesh list and
    # then treat n_kv_heads as the block count ("max_num_blocks=8").
    kv_cache = allocate_vllm_kv_cache(
        shape,
        dtype=torch.bfloat16,
        num_layers=args.n_layers,
        dp_model=[model],
        tt_cache_path=args.model_cache_path,
    )
    # One live user in slot 0, but the decode program is built for
    # args.max_batch_size, so the page table needs that many rows. Give every
    # slot its own disjoint block range so idle slots can never alias slot 0.
    blocks = -(-max_seq_len // PAGE_BLOCK_SIZE)
    batch = int(getattr(args, "max_batch_size", 1) or 1)
    page_table = torch.arange(batch * blocks, dtype=torch.int32).reshape(batch, blocks)
    return page_table, kv_cache


def feat_len(T):
    leave = T % 100
    f = (leave - 1) // 2 + 1
    return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (T // 100) * 13


def pin_mel(mel, target_frames):
    """Return ``mel`` at exactly ``target_frames`` frames.

    WhisperFeatureExtractor already emits a fixed 3000-frame mel: it zero-extends
    the WAVEFORM to its 30s window and runs the filterbank over the whole thing,
    so the frames past the real audio are genuine silence frames, not a constant.
    Re-deriving that tail ourselves is both unnecessary and worse -- measured
    against the unpadded CPU reference, the extractor's own tail lands at 0.0904
    max-abs while zero padding gives 0.0971 and a constant silence column 0.0988.
    So only slice/keep here; never synthesise the padding.
    """
    if mel.shape[1] >= target_frames:
        return mel[:, :target_frames]
    # Shorter than the pin only if the caller already truncated the extractor
    # output; fall back to edge replication rather than inventing zeros.
    tail = mel[:, -1:].expand(mel.shape[0], target_frames - mel.shape[1])
    return torch.cat([mel, tail], dim=1)


def parse_asr(t):
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", t, flags=re.DOTALL)
    return m.group(2).strip() if m else t.strip()


# Japanese CER normalisation. This MUST stay identical to the served-path eval
# client (tt-inference-server's asr_ja_eval.py), otherwise the two front-ends are
# scored with different metrics and the numbers cannot be compared: stripping
# only whitespace here while the served eval also strips punctuation made the
# demo look 8 CER points worse (0.184 vs 0.100) for output that is largely the
# same text with different punctuation.
_NORM_STRIP = re.compile(r"[\s、。,\.\?!？！「」『』（）\(\)\[\]【】…・:;：；\"'`~〜ー－\-]")


def norm_ja(s):
    """Same normalisation asr_ja_eval.py applies before CER (NFKC + punctuation strip)."""
    s = unicodedata.normalize("NFKC", s)
    return _NORM_STRIP.sub("", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--language", default="Japanese")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    items = []
    with open(a.manifest) as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if a.limit:
        items = items[: a.limit]
    print(f"loaded {len(items)} clips", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained(a.snapshot)
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    dec_tok = AutoTokenizer.from_pretrained(a.ckpt)
    with safe_open(os.path.join(a.ckpt, "model.safetensors"), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight").float()
    pad_id = tok.convert_tokens_to_ids(AUDIO_PAD)
    w = ref.load_audio_tower_weights(snap_dir=a.snapshot, dtype=torch.float32)

    import jiwer

    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=32768)
    recs, refs, hyps = [], [], []
    t_start = time.time()
    audio_total = 0.0
    try:
        enc_params = tt_enc.preprocess_weights(w, dev)
        args = ModelArgs(dev, max_batch_size=EVAL_MAX_BATCH, max_seq_len=2048)
        sd = args.load_state_dict()
        dtype = decoder_weight_dtype()
        paged_config = (
            PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=PAGE_MAX_BLOCKS) if PAGED_KV else None
        )
        model = Qwen3ASRDecoder(
            args,
            dtype,
            dev,
            sd,
            args.weight_cache_path(dtype),
            paged_attention_config=paged_config,
            use_paged_kv_cache=PAGED_KV,
        )
        page_table, kv_cache = build_paged_kv(model, args.max_seq_len)

        for i, it in enumerate(items):
            path = it.get("wav") or it.get("audio") or it.get("audio_filepath") or it["path"]
            ref_txt = it.get("ref") or it.get("text") or it.get("reference") or ""
            try:
                wav, sr = sf.read(path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                # Standard ASR speed metrics divide by the ORIGINAL waveform
                # duration, not by a padded/feature length: RTF = processing /
                # audio, RTFx = audio / processing (vLLM's own ASR benchmark and
                # the Open ASR Leaderboard both do this). Deriving it from the mel
                # frame count instead silently changes the denominator - measured
                # 1649.4 s of "audio" for clips that are really 1892.0 s, a 1.147x
                # error that made this eval incomparable with the served one.
                clip_seconds = len(wav) / float(sr)
                feats = fe([wav], sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                mel = feats["input_features"][0].float()
                nf = int(feats["attention_mask"][0].sum()) if "attention_mask" in feats else mel.shape[1]
                ntok = feat_len(nf)
                # Keep the extractor's own fixed-width mel. Slicing it to the real
                # frames and re-padding ourselves is strictly worse: the extractor
                # zero-extends the WAVEFORM and runs the filterbank over the whole
                # 30s window, so its tail is genuine silence, while any padding we
                # synthesise is not (measured vs the unpadded CPU reference:
                # extractor tail 0.0904, zero pad 0.0971, constant column 0.0988).
                mel = pin_mel(mel, MEL_PIN)
                prompt = (
                    f"<|im_start|>user\n{PLACEHOLDER}<|im_end|>\n"
                    f"<|im_start|>assistant\nlanguage {a.language}{ASR_TAG}"
                )
                ids = tok.encode(prompt)
                p = ids.index(pad_id)
                input_ids = torch.tensor(ids[:p] + [pad_id] * ntok + ids[p + 1 :], dtype=torch.long)

                t0 = time.time()
                t_enc0 = time.time()
                ae = tt_enc.encode_mel(mel, enc_params, dev, valid_frames=nf).float()
                t_encode = time.time() - t_enc0
                inp = embed[input_ids].clone()
                mask = input_ids == pad_id
                nm = int(mask.sum())
                ae = ae[:nm] if ae.shape[0] > nm else torch.cat([ae, torch.zeros(nm - ae.shape[0], ae.shape[1])], 0)
                inp[mask] = ae.to(inp.dtype)
                t_gen0 = time.time()
                out = model.generate(
                    inp.unsqueeze(0),
                    max_new_tokens=a.max_new_tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    repetition_penalty=a.repetition_penalty,
                    prompt_ids=input_ids,
                )
                t_gen = time.time() - t_gen0
                lat = time.time() - t0

                hyp = parse_asr(dec_tok.decode(out, skip_special_tokens=False))
                audio_total += clip_seconds
                refs.append(norm_ja(ref_txt))
                hyps.append(norm_ja(hyp))
                recs.append({"path": path, "ref": ref_txt, "hyp": hyp, "lat_s": round(lat, 3), "enc_s": round(t_encode, 3), "gen_s": round(t_gen, 3), "toks": len(out)})
            except Exception as exc:  # noqa: BLE001
                recs.append({"path": path, "error": repr(exc)})
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(items)}", flush=True)
    finally:
        ttnn.close_device(dev)

    wall = time.time() - t_start
    cer = float(jiwer.cer(refs, hyps)) if refs else None
    report = {
        "manifest": a.manifest,
        "clips": len(items),
        "ok": len(refs),
        "fail": len(items) - len(refs),
        "corpus_cer": round(cer, 4) if cer is not None else None,
        "wall_s": round(wall, 2),
        "audio_s": round(audio_total, 1),
        # Standard ASR speed metrics. rtfx = audio / processing (higher is
        # faster, >1 means faster than real time); rtf is its reciprocal. Both
        # use the ORIGINAL waveform duration, matching vLLM's ASR benchmark and
        # the Open ASR Leaderboard so the numbers can be compared with published
        # ones. NOTE: this eval drives the model directly and sequentially, so
        # its rtfx is a single-stream figure and is NOT interchangeable with a
        # served benchmark's, which includes HTTP and can run requests
        # concurrently.
        "rtfx": round(audio_total / wall, 3) if wall else None,
        "rtf": round(wall / audio_total, 4) if audio_total else None,
    }
    with open(a.output, "w") as fh:
        json.dump({"report": report, "samples": recs}, fh, ensure_ascii=False, indent=2)
    print("REPORT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
