# Qwen3-ASR-1.7B on Tenstorrent (Blackhole / P150a)

Port of `Qwen/Qwen3-ASR-1.7B` to ttnn. Target: a single Blackhole (P150a; on the
dev host = one chip of a P300 posing as a P150). No upstream tt-metal branch exists
for this model — fresh port. Closest structural reference is `models/demos/qwen3_vl`
(Qwen3 decoder + encoder tower + projector + multimodal token splice); the audio
front-end borrows from `models/demos/audio/whisper`.

## Architecture (verified against HF config.json + qwen_asr modeling)

Single `Qwen3ASRForConditionalGeneration` ("Thinker"), three parts:

1. **AuT audio encoder** (`thinker.audio_tower`, ~300M):
   WhisperFeatureExtractor mel (`num_mel_bins=128`) →
   `conv2d1/2/3` (3×3, stride 2, pad 1, `downsample_hidden_size=480`, GELU) = 8× downsample →
   `conv_out` Linear(480·16 → `d_model=1024`, no bias) →
   `+ SinusoidsPositionEmbedding` (`max_source_positions=1500`) →
   24 × `Qwen3ASRAudioEncoderLayer` (`d_model=1024`, `heads=16`, `ffn=4096`, GELU, qkv bias=True) →
   `ln_post` LayerNorm → `proj1` Linear(1024→1024) → GELU → `proj2` Linear(1024→`output_dim=2048`).
   **Windowed attention**: bidirectional within blocks defined by `cu_seqlens`;
   block size = `n_window_infer=800` mel frames (offline). `conv_chunksize=500`.
2. **Projector** = `proj1`/`proj2` (audio 1024 → LLM hidden 2048).
3. **Qwen3 text decoder** (`thinker.model`, = Qwen3-1.7B): `hidden=2048`, `28` layers,
   `16/8` heads (GQA), `head_dim=128`, `intermediate=6144`, SiLU, RMSNorm `eps=1e-6`,
   **qk-norm**, RoPE `theta=1e6`, `vocab=151936`, `max_pos=65536`.
4. **Multimodal glue**: processor = WhisperFeatureExtractor + Qwen2Tokenizer; audio
   embeddings replace placeholder `audio_token_id=151676` (`audio_start=151669`) in the
   input-embedding sequence, then standard causal prefill + greedy decode. 30 languages.

## Prefill seqlen rule
Prefill embeds are padded to a **multiple of 512** (`tt/qwen3_asr_decoder.py:prefill_logits`), min 512.
Trailing pad rows are causal-masked from the last real token, so padding does not change the last real
token's logits. Two reasons for 512 specifically:
- Attention shards seqlen across the core grid and each shard must be tile(32)-aligned; a 128-multiple
  can yield 48-row shards → TT_FATAL. 256 satisfies alignment on its own.
- **Different 512-buckets cannot be mixed in one long-lived process** — see *Known limitations* below.
  The decoder MLP reshapes prefill `x` to `[1, S_pad//512, 512, -1]` for `S_pad >= 512`, so different
  padded lengths differ only in the batch dim `-3`, which the prefill matmul program-cache hash does not
  distinguish (512→1024 TT_FATALs). Prompts run `13.0 * seconds + 13` tokens (measured), so forcing
  min-512 pins every request to the single `[1,1,512,d]` program shape **for clips up to ≈38 s** and
  sidesteps the collision there; see *Known limitations* for what happens past that length.

## Known limitations

**Length-keyed prefill corruption → fixed-length prefill workaround.**
Interleaving prefills whose padded lengths fall in *different* 512-buckets corrupts / crashes the decoder
in one long-lived process. **Root cause (confirmed on device, Blackhole P150, 2026-07-07):** a tt-metal
**program-cache collision across the MLP prefill reshape**, not a bug in this model's code.

`models/tt_transformers/tt/mlp.py` reshapes the prefill activation to `[1, S_pad//512, 512, -1]` when
`S_pad >= prefill_len_cutoff` (512 on Blackhole). So a 512-pad prefill is `[1, 1, 512, d]` and a 1024-pad
prefill is `[1, 2, 512, d]` — they differ **only in the batch dim `-3`**, which the downstream matmul's
(`ttnn.experimental.minimal_matmul` / the attention `wo` matmul) program-cache hash does not distinguish.
The program compiled for the first bucket is then wrongly reused for the second.

Reproduced (see the repro under the PR discussion):
- A **1024-token prefill in isolation runs fine** (verified directly, all drivers).
- A **512-token prefill followed by a 1024-token prefill `TT_FATAL`s** in the attention output matmul
  (`a_shape[-1] == b_shape[-2]`, "width=3072 height=2048") — the reused program has the wrong shape.
- On the current tree a 256-pad vs 512-pad mix no longer reproduces corruption (partially improved
  upstream), but the 512↔1024 collision above is deterministic.

Why the shipped model works despite this: for the clip lengths this model is served with, prompts stay
inside one bucket. Measured against the running vLLM server by reading
`vllm:request_prompt_tokens_sum` per request:

| clip | prompt tokens | bucket |
|---|---|---|
| 5 s | 78 | 512 |
| 11 s | 156 | 512 |
| 15 s | 208 | 512 |
| 30 s | 403 | 512 |
| 38 s | **520** | 1024 |
| 45 s | **611** | 1024 |

That is `13.0 * seconds + 13` tokens (the mel front-end yields 100 frames/s and the conv stack
downsamples 8×), so **512 is crossed at ≈38 s**, not never. Treating that bound as an invariant
was true of the 14 s-pinned standalone server, not of the vLLM path, which accepts whatever it is
given up to `max_model_len`.

Crossing it is nonetheless safe here, and for a reason worth stating: the encoder input is pinned
to `PIN_MEL_FRAMES` (3000 = 30 s) in `tt/generator_vllm.py`, so the *encoder* is single-shape
regardless, and the 38 s and 45 s requests above returned HTTP 200 with `/health` still 200
afterwards. What the 512-multiple rule buys is that each bucket is entered by padded length alone;
a process that only ever sees ≤38 s clips never leaves `[1,1,512,d]`.

The workaround is therefore "pin to one bucket for the lengths actually served", enforced at two
layers:
- **Op level** (`tt/qwen3_asr_decoder.py`): pad every prefill to a 512-multiple, min 512.
- **Server level** (`server/qwen3_asr_server.py`, `FIXED_INFER_SEC = 14.0`): pin every `_infer` to a
  fixed 14 s audio length (pad short clips with silence, silence-chunk long audio into ≤14 s windows), so
  every request stays in the 512 bucket. Cost: a small accuracy trade-off from more/shorter chunks
  (full-clip CER 0.045 → 0.065, accepted for stability) and wasted compute on padded silence for short
  clips. See `server/LONGFORM_DESIGN.md` for the tiered chunking design.

Removing the fixed-14 s pin (to allow long single-shot / variable-length prefill) requires the tt-metal
program-cache fix — the batch dim `-3` must be part of the prefill matmul program hash. This cannot be
fixed at the model layer (bucketing still collides). Tracking issue + repro:
`docs/prefill_program_cache_collision_issue.md` in this PR.

## Install

Two environments, deliberately separate:

```bash
# 1) device side (server, demos, tests) — on top of a built tt-metal
pip install -r models/demos/audio/qwen3_asr/requirements.txt
pip install --no-deps -r models/demos/audio/qwen3_asr/requirements-processor.txt

# 2) CPU reference / golden tooling — its own venv, NEVER the tt-metal env
python3 -m venv /tmp/qwen3-asr-ref
/tmp/qwen3-asr-ref/bin/pip install -r models/demos/audio/qwen3_asr/requirements-reference.txt
```

`qwen-asr` is pinned and installed with `--no-deps` on the device side because it declares
an older `transformers` than tt-metal pins. Only its processor (prompt template + log-mel)
is used there, and `reference/qwen_asr_processor.py` imports that module without executing
the package `__init__` chain that would pull in the CPU modeling stack. The reference
tooling does need the full package, hence its own venv.

**Encoder golden requires a chunk-aligned clip (no partial final chunk).**
The AuT front-end consumes mel in 1 s chunks (100 frames at the 10 ms hop) and emits 13 encoder
rows per chunk. The CPU reference masks the audio tower's output down to `feature_lens`, so on a
clip that does not fill whole chunks the reference emits fewer rows than the ttnn port's
chunk-aligned output (e.g. a 7.62 s clip: reference `7*13 + ceil(62/8) = 99` rows vs the port's
`8*13 = 104`), and because the windowed-attention blocks then differ the encoder PCC drops to
~0.96 across *all* rows — not just the trailing ones. The port does not implement the reference's
partial-final-chunk masking; the shipped pipeline sidesteps it instead by padding every request to
a fixed length and trimming the encoder output to the processor's audio-token count
(`server/_infer`, `demo/`, `tests/test_e2e.py`). Consequence for the PCC suite: generate goldens
from a **whole-second** clip (`reference/dump_reference.py` defaults to 7.0 s and warns otherwise;
`tests/test_audio_encoder.py` fails with this explanation if the golden is misaligned).
Implementing feature-lens masking in the ttnn encoder would remove the constraint.

## Reference golden

`reference/dump_reference.py` (run in the reference venv above) loads the CPU model, hooks
submodules, transcribes a short clip, and saves per-stage tensors + `manifest.json`. The
defaults need nothing outside a clean checkout (an in-repo 16 kHz wav; output to
`$QWEN3ASR_GOLDEN_DIR` or `/tmp/qwen3_asr_golden` — tensors are large, so they stay out of
the repo). Captured stages:
`conv2d1`, `conv_out`, `enc_layer0`, `ln_post`, `audio_tower`/`proj2` (= audio embeds),
`lm_head` (prefill + decode logits), plus end-to-end token text.

Verified shapes at the defaults, i.e. the 7.0 s slice `DEFAULT_DUR` takes from the in-repo
clip (read back off `$QWEN3ASR_GOLDEN_DIR`):

| tensor | shape |
|---|---|
| `conv2d1` | `(7, 480, 64, 50)` |
| `conv_out` | `(7, 13, 1024)` |
| `enc_layer0`, `ln_post` | `(91, 1024)` |
| `audio_tower` / `proj2` (audio embeds) | `(91, 2048)` |
| `inputs_embeds` | `(109, 2048)` |
| `lm_head` (prefill logits) | `(1, 109, 151936)` |

The leading dims track the clip: 7 whole 1 s chunks × 13 encoder rows = 91 audio rows, and 109
is those plus the 18-token prompt template. A different `--dur` moves every number here, so
quote the duration alongside any shape (an earlier revision of this section listed 12 s shapes
while `DEFAULT_DUR` was 7.0, which cannot be reproduced by running the documented command).

**Two generators, both required.** `dump_reference.py` covers the encoder stages and
`lm_head`; `reference/extract_text_decoder.py` produces the text-decoder checkpoint the
decoder tests load *and* the remaining goldens they read — `inputs_embeds.npy` (the audio
embeds already spliced into the prompt sequence, which is an input to the language model
and so cannot come from a forward hook on it) and `position_ids.npy`. Run both into the
same golden dir, or `test_decoder.py` fails with `golden tensor not found:
inputs_embeds.npy`:

```bash
export QWEN3ASR_SNAP_DIR=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots
export QWEN3ASR_GOLDEN_DIR=/tmp/qwen3_asr_golden
export QWEN3ASR_TEXT_DECODER=/tmp/qwen3_asr_text_decoder

/tmp/qwen3-asr-ref/bin/python models/demos/audio/qwen3_asr/reference/dump_reference.py
/tmp/qwen3-asr-ref/bin/python models/demos/audio/qwen3_asr/reference/extract_text_decoder.py
```

The second prints `[save] inputs_embeds (109, 2048)` and writes the checkpoint
(`config.json`, `model.safetensors`, tokenizer) that `QWEN3ASR_TEXT_DECODER` points at.

## Tests and CI

```bash
# staged artifacts (regenerate with the reference venv, see above)
export QWEN3ASR_SNAP=<hf-hub>/models--Qwen--Qwen3-ASR-1.7B/snapshots
export QWEN3ASR_GOLDEN_DIR=/tmp/qwen3_asr_golden
export QWEN3ASR_TEXT_DECODER=/tmp/qwen3_asr_text_decoder
pytest models/demos/audio/qwen3_asr/tests/test_audio_encoder.py
pytest models/demos/audio/qwen3_asr/tests/test_decoder.py
QWEN3ASR_E2E_WAV=<16k-mono.wav> QWEN3ASR_E2E_TEXT="<expected words>" \
  pytest models/demos/audio/qwen3_asr/tests/test_e2e.py -s
```

The e2e clip needs no external asset: `reference/dump_reference.py` already
defaults to an in-repo 16 kHz mono wav, and the goldens record what the CPU
model transcribes it as, so the same pair drives the e2e test.

```bash
QWEN3ASR_E2E_WAV=models/demos/audio/whisper/demo/dataset/conditional_generation/17646385371758249908.wav \
QWEN3ASR_E2E_TEXT="driver of the vehicle" \
  pytest models/demos/audio/qwen3_asr/tests/test_e2e.py -s
```

The golden tensors and the extracted text-decoder checkpoint are too large for the repo, so
the fixtures skip when they are absent — convenient on a dev box, but on a runner that is
supposed to have them staged a skip would hide exactly the breakage these tests exist to
catch. Set **`QWEN3ASR_REQUIRE_ARTIFACTS=1`** to turn every such skip into a hard failure.

Use that flag whenever this suite runs unattended (a pipeline leg, a nightly, a bring-up
script): point `QWEN3ASR_GOLDEN_DIR` / `QWEN3ASR_TEXT_DECODER` at the staged artifacts and set
the flag, and a missing artifact, a dependency break or a `tt_transformers` API change fails
loudly instead of reporting a green skip. `tests/test_e2e.py` needs no staged audio — it runs
on the in-repo clip above.

> Wiring this into the shared model pipelines (`tests/pipeline_reorg/`) is deliberately left
> out of this PR; see the follow-ups in the PR description.

## Corpus eval (front-end parity)

`eval/corpus_eval.py` transcribes a manifest with the ttnn front-end and reports
corpus CER, so a demo run can be compared against a served run on the same
clips and with the same metric.

`TT_METAL_HOME` must be exported: the script reads it to locate its own
`reference/` and `tt/` packages, and raises `KeyError` without it. The snapshot
is the full Qwen3-ASR checkout (audio tower plus processor config); the ckpt is
the text decoder extracted from it by `reference/extract_text_decoder.py`.

```bash
export TT_METAL_HOME=/path/to/tt-metal
QWEN3ASR_SNAP_DIR=$HOME/.cache/huggingface/hub/models--neosophie--Qwen3-ASR-1.7B-JA/snapshots/<rev>

python models/demos/audio/qwen3_asr/eval/corpus_eval.py \
  --manifest /path/to/manifest.jsonl \
  --snapshot $QWEN3ASR_SNAP_DIR \
  --ckpt /path/to/extracted_text_decoder \
  --output /tmp/report.json
```

`HF_MODEL` does not have to be set: the script points it at `--ckpt` while it
builds `ModelArgs` and restores it afterwards, the same handling
`tt/generator_vllm.py` uses on the served path.

Each manifest line is JSON. `{"wav": "...", "ref": "..."}` is the canonical
form, but the reader accepts the field names other corpora ship with, so an
existing manifest usually needs no rewriting:

| field | keys tried, in order |
|---|---|
| audio path | `wav`, `audio`, `audio_filepath`, then `path` (**required** -- a line with none of these raises `KeyError`) |
| reference text | `ref`, `text`, `reference`, else `""` (an empty reference still scores, and drags CER to 1.0 for that clip) |

Defaults are chosen to match a vLLM serving run rather than to look good in
isolation:

| knob | default | why |
|---|---|---|
| `QWEN3ASR_EVAL_PAGED_KV` | `1` | paged and non-paged decode use different SDPA kernels |
| `QWEN3ASR_EVAL_PAGE_BLOCK` | `64` | same `--block_size` a vLLM server is launched with |
| `QWEN3ASR_EVAL_PAGE_MAX_BLOCKS` | `2048` | size of the paged KV pool this eval allocates: `empty_kcache_paged_attention(2048, 8, 64, 128)`. A manifest whose longest clip needs more blocks than this fails at allocation, not at decode |
| `QWEN3ASR_EVAL_MAX_BATCH` | `4` | `max_batch_size` selects the decode program shape |
| `QWEN3ASR_EVAL_REPETITION_PENALTY` | `1.1` | the sampling rule the serving request asks for |
| `QWEN3ASR_MEL_PIN` | `3000` | mel frames every clip is padded to, so the encoder sees one shape and the prefill program cache is not churned. **Read by the served path too** (`tt/generator_vllm.py`, as `PIN_MEL_FRAMES`), so changing it here without changing it there stops the two front-ends being comparable |

CER is computed after NFKC normalisation and punctuation stripping. Scoring one
side with a stricter normalisation than the other is not a model difference: it
accounted for most of an apparent 8-point CER gap during bring-up.

Speed is reported as `rtfx` (audio seconds processed per wall-clock second) and
its reciprocal `rtf`, both divided by the ORIGINAL waveform duration. That is the
definition vLLM's own ASR benchmark and the Open ASR Leaderboard use, so the
numbers can be held against published ones. Deriving the duration from the mel
frame count instead inflates it by the extractor's 30s padding — measured 1892.0 s
of real audio reported as 1649.4 s, a 1.147x error.

`rtfx` here is a **single-stream** figure: this eval drives the model directly,
one clip at a time, and times only encode + decode. A served benchmark measures a
different thing (HTTP and multipart included, requests possibly concurrent), so
the two `rtfx` values describe different workloads and must not be compared as if
they were the same measurement.

## Served-path knobs (`tt/`)

The ttnn modules under `tt/` are what a vLLM server actually runs, and they
read five environment variables of their own. None of them has to be set --
the defaults are the served configuration -- but an operator who does set one
is changing what the deployment computes, so they are listed rather than left
to be discovered by reading the source.

| knob | default | why |
|---|---|---|
| `QWEN3ASR_PREFILL_PIN` | `512` | every prefill is padded to this one bucket. tt-metal's prefill matmul has a length-keyed program-cache collision (tenstorrent/tt-metal#49451): mixing a 512-padded and a 1024-padded prefill in one process `TT_FATAL`s, and where it survives the same audio transcribes differently depending on the bucket. 512 covers this model's prompts (~30 s clip -> ~390 audio tokens + prompt). Raise it for longer single-shot clips -- but raise it for **every** front-end, or they stop being comparable |
| `QWEN3ASR_DECODE_TRACE` | `1` | capture a decode trace. Untraced, every decode step pays full per-op host dispatch: 489 ms/token vs 113 ms traced on p150. Set `0` only where the long-run trace instability in "Known limitations" bites |
| `QWEN3ASR_DECODER_DTYPE` | `bfloat8_b` | ttnn dtype for the decoder weights (`bfloat16` is the only other accepted value; anything else raises). The demos and the served path take it from the same helper, so it moves both |
| `QWEN3ASR_DEVICE_EMBED` | `0` | keep the text-embedding gather on the host. The device gather is ~10x faster in isolation (1.1 ms vs 10.5 ms for a 149-token prompt) but real traffic varies the prompt length, so it compiles a program per length and churns the cache (TED 6.08 -> 3.48 audio-s/s); pinning it to the prefill bucket instead pads a ~149-token gather to ~1024 rows and still loses (5.79 vs 6.08). Set `1` once a length-agnostic embedding avoids both |
| `QWEN3ASR_AUDIO_SNAPSHOT` | *(unset)* | directory holding the full Qwen3-ASR snapshot (audio tower + processor config). Unset, the adapter uses `HF_MODEL` when that is a full snapshot, else the vLLM model path -- which is why a normal deployment never sets it |

`QWEN3ASR_MEL_PIN` (3000) is read here too; it is documented with the corpus
eval above because both front-ends have to agree on it.
