# Qwen3.8-Flash-Next on Blackhole

A TTNN port of Qwen3.8-Flash-Next (48 hybrid layers: gated delta-net attention, sparse attention with an indexer,
a 256-expert MoE routed from a BF4 corpus; one multi-token-prediction layer) served from one 1x4 mesh of Blackhole
chips: an OpenAI-compatible chat server with chunked prefill, contexts to 262,144 tokens, greedy or sampled decode,
thinking and tool calls in the Qwen chat template.

This repository is a tt-metal checkout that carries the runtime fixes the model needs (section 1).  Clone it, build it
the standard way, download the checkpoint and start the server; the first start converts the weights into its caches.
No prebuilt archive, no pinned binary, no host-specific configuration.

| hardware | profile | status |
|---|---|---|
| QuietBox, 4x p150c (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04 (startup acceptance 96/96 against the CPU, 19.6 tokens/s at 32k) and 2026-09-06 from a fresh clone; `docs/PROOFS.md` |
| 4x p150 in one host, ethernet line | `p150-line` | the numbers below were measured on it; no fresh-clone run is recorded (section 9, `docs/PROOFS.md`) |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | verified 2026-09-19 (acceptance identical to the 4x p150 boxes), 2026-09-23 (the compact expert layout) and 2026-09-29 (two DRAM readers per bank on its mixed-harvest dies: the 4x p150 rows, 68.7 tok/s on the 560-token chat prompt with `--mtp 4`); section 7, `docs/PROOFS.md` |

## Performance (4x p150)

| path | measured | notes |
|---|---|---|
| prompt prefill | 820-870 tok/s (128-row chunks on the three-ring expert stream, 2026-09-30, sampling on, a 32k context; 750-790 with the fused local combine before it; 410 in 32-row chunks alone); 3,003 tok/s with `--prefill-slab 2048`; with `--mtp 4` 2,645 tok/s (2026-09-27, tolerance-class; pinned `A3-slab2048-mtp4-32k`) | 2026-09-30 (the chunk's expert stream on three rings: 31,732 tokens in 36.57 s against 39.35, pins 12/12, `docs/PREFILL.md`); 2026-09-26 (block-shared attention + one-pass combine + three-ring expert stream + GDN row programs defaults: 31,716 tokens in 10.56 s); the slab is tolerance-class against the chunk bodies (`docs/PREFILL.md`: the ms per prompt token, the TTFTs, the one-call expert stream; on long chat prompts the two forms' greedy streams part at token 30 / 0 of a 2k / 20k document, `docs/NUMERICS.md`) |
| decode, one stream | 38.6 tok/s greedy, 36.5 tok/s sampled | 25.9 ms per token greedy (2026-09-26: the router tail's live-row exp and the MoE dense composite, bitwise: `docs/NUMERICS.md`), flat with depth; the sampled figure is the 2026-09-25 measurement; the defaults are listed below the table |
| decode, 4 / 8 streams | 36.1 / 30.6 tok/s per user (144 / 244 aggregate) | the batched-decode lane body measured directly, 27.74 / 32.72 ms per step (2026-10-01: the fused GDN step reads a head's shared tiles once per head and multicasts them, bitwise; 28.41 / 34.07 before); the chat server serves one stream (component class: `docs/NUMERICS.md`) |
| decode with MTP (`--mtp 4`) | greedy 68.06 tok/s on a 560-token chat prompt, 64.16 on a 177-token multi-turn chat, 104.32 code (256-token answers), 111.40 json and 50.06 prose over their whole answers (153 / 218 tokens, EOS honoured; 2.94 / 4.75 / 2.68 / 2.09 / 4.45 tokens per pass; the stream leaves the CPU reference where plain decode does, 12 of 12 records) against 38.6 plain greedy; `--mtp 3` (2026-09-25) 45.5 / 55.4 / 60.6 / 30.7; `--mtp 5` opt-in (`docs/NUMERICS.md`); sampled 17.5 / 16.2 ms per token (the card profiles, thinking / non-thinking) | answers measured client-side with EOS honoured on the 4-chip p150 line (2026-09-28, the fork runtime built at `5eac9c778edb`; the served default set: the GDN verify-rows fold `gdn_rows_scan` with its gate scalars in one SFPU pass, the MoE rows programs on two rings, the gated-residual read's re-associated norm `gr_recip_last`; the pass p50 41.9 ms pipelined on the 560-token chat prompt; a 256-token row whose answer runs past 256 tokens is measured to 256, a row over a shorter answer stops at its end); speculative drafting with exact acceptance for greedy requests and, by default on an `--mtp --sampling` server, sampled ones; the fold against the wrap over 43 EOS-honoured prompts +2.44 % +- 1.09 % tokens per second (paired mean +1.36 +- 0.61, 2026-09-26); the fold program's gate scalars in one SFPU pass (2026-09-27) 111.0 -> 85.8 us per GDN layer and -1.16 ms per verify pass in the census, pins identical; the wrap stays the opt-out (`QWEN38_FUSED_OFF=gdn_rows_scan`: with the rings and `gr_recip_last` the wrap read 64.35 / 56.77 / 102.2 / 103.89 / 50.18 at 2.98 / 2.56 / 4.65 / 4.78 / 2.24 tokens per pass, sampled 17.6 / 19.3 ms, on 2026-09-27 on the same line class; `docs/NUMERICS.md`) |
| decode, 4 concurrent requests with MTP (`--lanes 4 --mtp 4`, opt-in, greedy only) | 132.2 tok/s aggregate on 4 x the 560-token chat prompt at once (36.5 / 39.4 / 42.9 / 47.1 per request) against 57.6 for the same four served in turn; the mixed acceptance batch 37.7 / 46.6 / 74.1 / 28.1 per user | 2026-09-29, a 1x4 p150 line at 32k, greedy, EOS honoured, byte-identical to the single stream's (13/13 rows); TTFT 1.10 / 2.27 / 3.47 / 4.65 s in a four-request burst against 0.91; an admission runs in segments between the other lanes' passes (`--lanes-stall-budget` 0.5 s: their longest pause 0.62 s, where the 2026-09-28 form froze them 1.35 s per admission at 134.0 tok/s aggregate); host load 0.9-1.6; `docs/SERVER.md`, `docs/NUMERICS.md` |
| image prompt (512 x 512 PNG, 256 image tokens, 282 prompt tokens) | TTFT 0.59 s end to end, streamed (server-side 0.52 s: prefill 0.50 s; tower 0.048 s of which 0.030 s device) against 0.52 s for a 282-token text prompt on the same server | 2026-09-29, a 1x4 p150 line at 32k, greedy, thinking off, host load 2-5; the tower's eight row buckets add 42 s to READY with the host's kernel cache warm (both hosts 42-44 s, MEASURED); `docs/SERVER.md` |
| contexts | 32k, 64k, 128k, 256k | 256k is single-user; MTP fits at 32k, 64k and 128k |
| correctness | bitwise repeatable; 96/96 greedy token match against the CPU reference on the acceptance prompt | chunked prefill is tolerance-class against the CPU reference on all 48 layers (`docs/NUMERICS.md`) |

The decode defaults since 2026-09-25 (each one's measurement and switch: `docs/NUMERICS.md`): BF8 dense weights, the
fused GDN step, the compact expert layout, the MoE post program's one-read routing, the MoE dense composite (the router
top-k beside the shared expert's eltwise, down linear and dispatch untilize as one program); 9 decode chains as fused programs,
the gated-residual read as two programs with its gathers inside them, the router tail's top-k on one core per token
group, two cores reading each DRAM bank in the decode linears.  A sampled request draws on the device after the top-32
candidate row (27.4 ms per token on both model cards; the host sampler's law, gated in `docs/NUMERICS.md`) on one-stream
`--sampling` servers; an `--mtp` server samples on the host; `--host-sampler` restores host sampling (`docs/SERVER.md`).

## 1. What you need

- Four Blackhole chips in one host as one 1x4 mesh: a QuietBox (4x p150c in an ethernet ring), four p150 cards in one
  host as an ethernet line, or four chips of a larger host (`--devices`); tt-kmd and the firmware bundle the chips
  shipped with (19.4.1.0 or later).
- This repository, built (section 2): tt-metal main `28238f903b` (2026-09-06; the device token streams are bitwise
  those of the previous base `d04395ed86` of 2026-08-29) plus the runtime fixes the model needs, not on main, without an upstream
  equivalent:
  - moe_compute: `Fix empty-rank moe compute metadata ownership` (the tilize writer; routed experts at batch 1),
    `Fix fused MoE source buffer double counting`, the local output path's replica rings and packed routing lists
    (`docs/PREFILL.md`; the DRAM-bank-to-worker query; the DRAM-sharded matmul's second reader per bank placed on a
    1x4 mesh, 2026-09-16)
  - the all-gather and fabric guards: `Guard all-gather scatter state initialization`, `Preserve ring connections in
    all-gather endpoint guard`, `Fix all-gather endpoint no-target connection access`, `Clear fabric router packet
    tags on teardown`
  - the four `#23023` commits (`Bind BF4 cache loads to verified file descriptors`, `Reject lexical aliases for
    descriptor loads`, `Fail closed BF4 cache shape and cleanup`, `Bind BF4 tensorbin payload and exact types`):
    `ttnn.load_tensor` on `/proc/self/fd` paths, used by `ttnn/bf4.py`

  The server admits only a `ttnn` imported from this checkout's own build and records the checkout's commit, tree and
  extension digest with every run (`tools/runtime_admission.py`; `docs/SERVER.md`).
- The checkpoint: 360 GB (131 safetensors shards, the tokenizer, the chat template; section 3).
- Disk under `--cache-root`: 69 GB for the BF4 expert cache (built once, shared by every context, kept across runtime
  rebuilds), about 23 GB for the 32k context and 10 GB for each other allocated context (the converted non-expert
  weights and the model I/O cache), about 1.3 GB of JIT kernel cache (`docs/SERVER.md`, disk and memory).
- Host memory: 64 GB is comfortable (the first start converts one MoE layer at a time, about 10 GB of host tensors in
  flight).  The optional CPU reference (`tools/run_full_cpu_oracle.py`) needs 170-240 GB and is not a user step.
- Build tooling: what tt-metal's `install_dependencies.sh` installs (clang-20 or gcc-12, cmake, ninja, python 3.10, `uv`).

## 2. Build

The standard tt-metal flow from a fresh clone (about 10 minutes on 64 cores; on the QuietBox's 32 cores `build_metal.sh`
took 684 s and `create_venv.sh` 95 s, 2026-09-06):

    git clone https://github.com/sjettTT/tt-qwen-3.8-flash-next.git
    cd tt-qwen-3.8-flash-next
    git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors
    ./build_metal.sh
    ./create_venv.sh

`build_metal.sh` takes its defaults (Release, clang-20 with libstdc++; `--toolchain-path cmake/x86_64-linux-gcc-12-toolchain.cmake`
for gcc).  `create_venv.sh` makes `python_env/` with torch and installs `ttnn` from this tree in editable mode (on a
later update it asks before reusing an existing `python_env/`; answer `y`), so `python_env/bin/python -c 'import ttnn'`
resolves inside the checkout; the launcher uses that interpreter and sets `TT_METAL_HOME` itself.

## 3. The checkpoint

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out /data/Qwen3.8-Flash-Next

fetches `Qwen/Qwen3.8-Flash-Next` from ModelScope at the release commit `2741eec155d03a8ce151b993ccce1a7b1e398d6b`
(145 files, 360,023,351,829 bytes) as parallel 128 MiB range requests (`--threads`, default 32), verifies every file's
SHA-256 against the ModelScope listing (saved as `<out>/.download/files.json`) and resumes after an interruption.  The
port pins an earlier ModelScope revision label (`checkpoint.PINNED_CHECKPOINT_REVISION`) whose weights, config,
tokenizer and chat template are byte-identical to the release commit; the server checks the files by digest (every
shard's header, the index, the file and tensor manifests, `config.json`, the tokenizer, `chat_template.jinja`) and
refuses a checkpoint that differs, with the digests printed.

`--verify-only` checks a copy already on the host the same way (every file present hashed against the listing, a marker
per verified file, nothing fetched); a following run fetches only the files absent or different.  `tools/verify_checkpoint_files.py
--checkpoint DIR --modelscope-tree <out>/.download/files.json --output report.json` re-hashes a download later (standard
library only, any `python3`); `tools/checkpoint_budget.py` prints the per-device residency budget; `tools/safetensors_metadata.py`
lists tensors.

## 4. Start the server

Every decode token reads sixteen 320-byte rows of the PLE n-gram table, which stays in the checkpoint (104 GB in 33
shards of layer 1: 51.2B parameters, over 100 GB at BF16) and is read through the page cache; the port assumes a host
whose RAM holds it (performance with the table on disk is not measured).  Read it once before the start (13 s from a
warm cache on the QuietBox; `docs/SERVER.md`):

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint /data/Qwen3.8-Flash-Next

Then the server; `--profile` is `tt-quietbox`, `p150-line` or `qb2` (section 7), and `tools/` in this README is
`models/demos/blackhole/qwen38_flash_next/tools/`:

    models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile tt-quietbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

The launcher prints the checkout it runs from (commit, modified or not), the interpreter and the `ttnn` extension, the
profile, the device set, the context and the run directory (`<cache-root>/runs/<stamp>/`, which holds `READY`,
`phase-markers.jsonl`, `requests.jsonl`, `acceptance.json` and, at shutdown, `result.json` and `STOPPED`).

**The first start** converts the routed experts of all 49 MoE layers into the BF4 cache
(`<cache-root>/caches/bf4-experts/`, 69 GB in the compact expert layout: about 27 s per layer on a 4x p150 host, the
49 under half an hour; the QuietBox 2026-09-25: 1448 s), then builds the caches of the chosen context, compiles the
kernels and captures the traces.  `--prepare-only --bf4-stage-limit N` converts at most N missing layers and stops, for
a machine that bounds a job's wall time.  The cache is keyed by the checkpoint and the converter's sources, not by the
tt-metal revision: a rebuilt runtime keeps it, and every start checks one re-packed expert against it (`docs/SERVER.md`).

**Warm starts** reach `READY` in about five minutes.  `--acceptance` replays the twelve shipped CPU greedy records
(`tools/acceptance/greedy-prompts/`) against the CPU before the server listens; `--require-json-96` refuses to serve
unless the `json` record matches the CPU 96/96.  The other records leave the CPU stream after 2-75 tokens; a replay
that leaves earlier than the pinned table is a regression (`docs/NUMERICS.md`).

| launcher flag | meaning |
|---|---|
| `--profile tt-quietbox\|p150-line\|qb2` | the hardware profile: the mesh graph descriptor, the device set, the route |
| `--allocated-context 32768\|65536\|131072\|262144` | the resident build (KV caches, RoPE tables and the context limit; the limit is the context minus 64 for the consumed EOS step); default 32768 |
| `--acceptance`, `--require-json-96` | replay the twelve CPU greedy records at start; refuse to serve unless `json` matches 96/96 |
| `--prepare-only --bf4-stage-limit N` | convert at most N missing expert layers into the BF4 cache and stop |
| `--long-chunks` | accepted for compatibility: the 128-row prefill chunks where the prompt allows are the served default since 2026-09-29 (section 6, bitwise the 32-row chunks); with `--mtp` the MTP layer's rows run inside the 128-row chunks too |
| `--prefill-slab 2048` | prefill slabs of 2048 rows ahead of the 128-row chunks: one matmul per dense linear, the routed experts in one `moe_compute` call on three rings, tolerance-class against the chunk bodies (`docs/PREFILL.md`); off by default; combines with `--mtp 4` since 2026-09-27 (+99 ms per traced slab, 2,645 tok/s at 31.7k prompt tokens; pinned `A3-slab2048-mtp4-32k`; the TTFT table in `docs/PREFILL.md`) |
| `--mtp 3\|4\|5` | speculative drafting on greedy requests (section 6); off by default; `QWEN38_MTP_DRAFTS_PER_REQUEST=1` opens the k = 4 and k = 5 chains together and a request picks with `extra_body.mtp_drafts` (`docs/SERVER.md`) |
| `--port`, `--host` | the listening port; `--host` default `0.0.0.0`: the QuietBox and p150-line profiles serve the LAN |
| `--serve-seconds N` | stop after N seconds (a drain: the request in flight gets its reply) |
| `--sampling` / `--no-sampling` | the launcher passes `--sampling`: sampled requests are served, a request naming no sampling field is still the bitwise greedy stream; `--no-sampling` refuses sampling fields with HTTP 400 (+0.3 ms per token saved) |
| `--stall-seconds N` | the watchdog on zero device progress, 300 through the launcher; 0 disables it (`docs/SERVER.md`) |
| `--validate-only` | run the checks and the CPU preparation without opening the mesh (a profile whose route is derived at start, `p150-line`, still needs the chips present) |
| `--devices A,B,C,D` | run `p150-line` on four other KMD device nodes (four chips of a larger host) |
| `--python` | another interpreter of this checkout |

## 5. Talk to it

OpenAI-compatible HTTP on the port you chose:

    curl -s http://<host>:8000/health
    curl -s http://<host>:8000/v1/models
    curl -s http://<host>:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "Qwen/Qwen3.8-Flash-Next",
      "messages": [{"role": "user", "content": "Why does ice float?"}],
      "max_tokens": 256, "stream": true}'

The command-line client:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<host>:8000/v1 --thinking --tools

`POST /v1/chat/completions` (streaming or one document), `GET /v1/models` and `GET /v1/models/<id>`, `GET /health`.  The request rules in short
(`docs/SERVER.md` has them in full):

| request | rule |
|---|---|
| `messages` | the prompt, exactly: the server adds no system prompt, and the device prompt is the reference render (`tokenizer.apply_chat_template`), so `usage.prompt_tokens` is the count the client computes itself |
| `messages` content parts | `text` and `image_url` (`data:` URLs only; `detail` low / auto / high; 64 .. 16,384 image tokens per image, counted in `usage.prompt_tokens`; also under `--lanes`); video parts get HTTP 400; on a server whose tower is not resident, image parts get HTTP 400 `vision_unavailable` naming the reason (`docs/SERVER.md`) |
| `GET /v1/models/<id>` | the model object (`context_length`, `max_model_len`, the limits), the same document as the list's entry; another id is 404 `model_not_found` |
| `max_tokens` / `max_completion_tokens` | default and limit: the remaining context (the context limit less the prompt); a prompt over the limit gets HTTP 400 `context_length_exceeded` |
| `stream`, `stop`, `ignore_eos`, `seed`, `logprobs` | as in OpenAI; `logprobs` are relative to the read candidate row, not the vocabulary |
| `tools` / `tool_choice` | OpenAI shape, `tool_calls` finish reason; arguments are typed by the tool's parameter schema |
| `enable_thinking` (default true), `reasoning_effort`, `thinking_budget` | reasoning streams as `reasoning_content`; `chat_template_kwargs` means the same as the top-level fields |
| sampling | no sampling field: greedy, the argmax stream bitwise equal to the greedy loop the acceptance replay measures (`temperature 0` and `greedy: true` are the same path); `temperature > 0` samples with it (`top_p` 1.0, `top_k` 20, no penalties unless given); another field alone (`top_p`, `top_k`, `min_p`, a penalty, `seed`) samples with the model card's profile for the thinking mode (`/health.sampling_defaults`), so `seed` alone is a reproducible sampled stream |
| `response_format` other than `text`, `logit_bias`, `parallel_tool_calls: false` | refused with HTTP 400 rather than dropped; unknown fields are logged |
| follow-up turns | the device keeps its committed prefix and prefills only the new turn; with thinking on it restores the prompt-end snapshot instead (`docs/SERVER.md`) |
| concurrency | one request decodes at a time; up to four wait in the queue (`queue_wait_seconds` in `usage`), the fifth gets HTTP 503 |

A client that hangs up is noticed at the next device step; a streaming request gets an SSE keepalive every 30 s
through the queue wait and the prefill; the stop signal (`--serve-seconds`, SIGTERM) drains.  Hang-ups, stalled
readers, deadlines, the stall watchdog and the `/health` fields: `docs/SERVER.md`.

## 6. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Each context has its own
  component and model I/O caches under `--cache-root` (the BF4 expert cache is shared); 256k leaves about 750 MB per
  device free and is single-user (`docs/PREFILL.md` has the long prompts' times to the first token).
- The chunked prefill runs 128-row chunks where the prompt allows (the remainder in 32-row chunks) by default since 2026-09-29;
  `--long-chunks` changes nothing: 1.16 ms per prompt token through the server (a 6,874-token prompt in 8.0 s, 2026-09-30, three-ring
  expert form) against 3.3 with 32-row chunks alone, the same tokens bitwise; the MTP stream is unchanged (`docs/NUMERICS.md`).
- `--mtp 3|4|5` drafts K tokens per pass with exact acceptance; off by default.  It fits at 32k, 64k and 128k at any k,
  not at 256k (94 MB free per bank against the pair's 128 MiB contiguous).  The committed stream is not bitwise with plain decode
  on 3 of the 12 acceptance prompts: near-ties within one bf16 step, not a defect (`docs/NUMERICS.md` has the indices
  and the pinned table).  On an `--mtp --sampling` server sampled requests draft too, by exact speculative sampling
  (plain sampling's law kept; greedy requests run the fused verify, the pinned greedy stream, sampled ones the split form;
  `docs/SERVER.md` has the seed, fingerprint and `--mtp-gdn-anchor` rules); `QWEN38_MTP_SAMPLED=0` restores the 1-row sampled loop.

## 7. QuietBox 2

A p300 card is two Blackhole dies joined on the card; a QuietBox 2 (2x p300c) has four dies in one ring (the two
on-card links and the two Warp400 links), so it is one 1x4 instance, served with `--profile qb2` (section 4).
The profile exports `tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto` (a 1x4 LINE over three of the four ring
links, two channels per link as in tt-metal's `p300_x2` descriptor): tt-metal classifies a p300 cluster that is not
exactly two or four dies as CUSTOM and refuses to open without one.  The route is derived at start from the fabric's
chip order and recorded in `READY` beside the ring walk (our box: `(1, 0, 3, 2)` where the walk gives `(0, 1, 2, 3)`).
Its dies are harvested differently (one serves four of its banks from another worker column), so until 2026-09-29 the
server read each DRAM bank with one core there.  The two-reader form is now stock on such a mesh: the first die's
bank-to-worker placement is shared by every die (a reader addresses its bank by id, so the odd die reads those banks from
a worker one to three NoC hops further), recorded in `READY` (`dram_workers_per_bank`, `dram_workers_placement`, which
also names the second reader's placement rule, `row+x` since 2026-09-29: `docs/NUMERICS.md`) and warned
once by tt-metal; one reader remains only where the placement cannot be validated (a die with another worker or DRAM
grid, or a bank count other than eight), the reason in `READY` (`dram_workers_fallback`).  Measured on our box 2026-09-29
with the release container's arguments (`--mtp 4 --long-chunks --prefill-slab 2048`, host load 1-2.4): one reader 63.9
tok/s on the 560-token chat prompt, two readers 68.7 (the 4x p150 line's 68.3); json 105.2 -> 113.1, multi-turn 59.9 ->
64.8, prose 46.6 -> 50.3, code 97.6 -> 104.9; plain decode 38.5 -> 40.5; the 13-record startup replay identical
(`docs/NUMERICS.md`).
Verified on our box on 2026-09-19 (`READY` 194 s after launch, `json` 96/96, the divergence table identical to the 4x
p150 boxes) and 2026-09-23 (the compact expert layout: 9.17 GB more free per device), and on a contributor's box from
a fresh clone on 2026-09-07 (`docs/PROOFS.md`).

## 8. Layout

| where | what |
|---|---|
| `chat.py` `checkpoint.py` `config.py` `reference.py` | the checkpoint, the chat template, the torch reference model |
| `diagnostic_bf4.py` | the binder of a CPU-staged BF4 expert corpus |
| `tt/` | torch reference components (the CPU oracle every test compares against) |
| `ttnn/` | the device model: builder, layers, GDN, QSA, MoE/BF4, embedding, sampling, MTP |
| `tools/run_qwen38_chat_server.sh`, `qwen38_chat_cli.py`, `qb_mesh_smoke.py` | the launcher; the client; open the mesh and check the route without the model |
| `tools/qwen38_chat_server.py`, `qwen38_chat_session.py`, `qwen38_chat_protocol.py`, `qwen38_sampling_step.py` | the HTTP server; the traced decode chain and the mesh open; the request/reply protocol; the sampling step |
| `tools/qwen38_asgi.py` | the server behind uvicorn: the container form the tt-model package serves (`docs/SERVER.md`) |
| `tools/runtime_admission.py`, `live_decode_diagnostic.py` | the runtime identity (this checkout's build); the CPU preparation and the live construction (the BF4 conversion on the first start) |
| `tools/hardware_profiles.py`, `physical_route.py`, `resident_decode.py`, `evidence_records.py` | the profiles (tt-quietbox, p150-line, tt-quietbox-2); the route derivation; the chain's fixed points; the run records |
| `tools/download_checkpoint.py`, `prewarm_ple_table.py`, `verify_checkpoint_files.py`, `checkpoint_budget.py`, `safetensors_metadata.py` | the ModelScope download with SHA-256 verification; the n-gram pre-warm and the checkpoint tools of sections 3 and 4 |
| `tools/acceptance/` | the CPU greedy records the startup replay compares against, the two-step CPU oracle |
| `tools/qwen38_reference_corpus.py` | the teacher-forced reference corpus Q38-REF-v1 (`tools/reference/`) and its columns: HF on the CPU, the `tt/` oracle, a served chain's agreement records; `score` |
| `tools/stage_full_bf4_cpu.py`, `verify_full_bf4_cpu.py`, `bind_full_bf4_corpus_cpu.py`, `probe_full_bf4_binding_cpu.py` | a BF4 expert corpus staged on the CPU (produce, verify, bind, probe) |
| `tools/ci/q38_ci.py` | the regression harness: pins.json, baselines/, job runner, verdicts, seeding |
| `tests/` | the no-device tests (set `QWEN38_CHECKPOINT` for the checkpoint-reading ones) |
| `docs/` | PROOFS, NUMERICS, SERVER, TESTING, PREFILL (the "More" section below) |

Run the tests from the repository root (`docs/TESTING.md` has the regression harness and the reference corpus):

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

## 9. What is verified, and the known limits

- Verified runs: the hardware table above and `docs/PROOFS.md` (the QuietBox 2026-09-04 and 2026-09-06, the QuietBox 2
  2026-09-07, 2026-09-19 and 2026-09-23).  No fresh-clone run is recorded for a p150 line; this README's numbers were
  measured on 4x p150 hosts.
- Numerics (`docs/NUMERICS.md`, the 2026-09-25 pinned table): `json` 96/96 against the CPU, the other eleven records
  leave the CPU greedy stream between token 2 and 75; MTP is not bitwise with plain decode on 3 of 12 prompts (near-ties).
- 256k context is single-user; MTP is not available at 256k.
- One request decodes at a time by default (the traced chain is single-stream); the queue holds four more.  `--lanes B`
  (2..8, needs `--mtp`, greedy only) serves B at once through the MTP lane chain; an admission's prefill runs in segments
  between the other lanes' passes under `--lanes-stall-budget` (`docs/SERVER.md`).
- Images: `image_url` parts of at most 16,384 image tokens each (the stock 4096 x 4096 px maximum); served under `--lanes` too (2026-09-29, byte-identical to the single stream); every
  start runs the tower's eight row buckets once (about 44 s of READY, about 100 s more on a host whose kernel cache has none of the
  tower's programs); a DRAM shortfall starts the server text-only and says so in READY and `/health` (`docs/SERVER.md`).
- The first start is long (the 69 GB expert conversion); a host that kills long jobs needs `--prepare-only
  --bf4-stage-limit N` runs first.
- Python 3.10 (`create_venv.sh` default); Linux x86_64.

## 10. Speculative decoding on SPEED-Bench

NVIDIA's SPEED-Bench (arXiv 2604.09557, the successor to Spec-Bench), scored by its own framework over the served endpoint
(`--mtp 4`, greedy, thinking off, 4,096 tokens per turn, a 1x4 p150 line at 32k, 2026-09-30 at `f5dc3c0544`): mean acceptance
length 3.25 tokens per step over 672 of the 880 prompts (math, humanities and STEM partial, their source being gated); coding 4.20,
math 4.18, RAG 3.63, multilingual 3.49, reasoning 3.48, STEM 3.17, summarization 3.02, QA 2.97, writing 2.81, humanities 2.76,
roleplay 2.24.  The paper's rows, the throughput split and the plain-decode arm: `docs/NUMERICS.md`.

## 11. Decode limit on one Blackhole Galaxy: 5 軸（DP, PP, TP, EP, DCP）の限界性能と、そこから採る形

この節は、32 chip の BH Galaxy で本モデルの decode を回すときの**限界性能**（物理値と HF 定義だけから出る 1 token の時間と機械全体の tokens/s）と、その限界を出す**形**（5 軸の分け方、chip の中身、1 token の時間線）を、詳細設計と実装がそのまま使える粒度で書く。並び: §11.1 目標と指標 → §11.2 設計の入力（モデル・routing の実態・機械の物理値・系列長） → §11.3 確定原則（設計規則） → §11.4 解析モデル（時間の式） → §11.5 限界性能の答え（系列長別、ctx 192 の詳細） → §11.6 採る形の詳細設計（tsu 側 / agg 側） → §11.7 今の実装との差と近づける順番 → §11.8 未確定と決着方法 → §11.9 生成・検証・運用。本文の数値・順位は全て生成器 `render_free5d_v8.py` が `free5d_v8.py` の結果と census から決定的に生成する（手書きの数値は無い）。改訂 8 は単品と会合の時間を仕様の命令数 × cycle の段の積み上げ（`design/stages.py`、`design/comm_model.py`: 受け渡しの飛行・RISC の検知と発行・pack・unpack・SFPU、chip 内の NoC 区間、200G の hop 306 ns）に置き換えたもの（改訂 7 までは単品ごとの固定の段を 0 としていた誤り）。経緯・指摘・撤回は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §9.4〜§13（追記専用）、生成器と census は同じ共有ディレクトリ（`BH-GLX_Qwen3.8-Flash-Next_free5d_v8_20261027.py`、`..._render_free5d_v8_20261027.py`、`..._design_stages_20261027.py`、`..._design_comm_model_20261027.py`、`..._design_census_stage2_batches_20261026.py`、`..._routing_census_20261023.py`、`..._routing_stats_20261023.py`）。README と食い違ったら新しい方の作業記録が正しい。

### 11.1 目標と指標

- **tsu**（tokens/s/user）= R_eff / T_tok。T_tok は 1 user の 1 token が出る間隔（1 pass の時間）。**agg**（tokens/s、機械全体）= DP × b × R_eff / T_pass。R_eff = 1（k=0）/ 1.376（MTP k=1: verify 2 行/user で 1 pass に出る token の期待値。本件プロンプト集合 ISL 128 の採択率）。
- **限界** = 物理値と HF 定義だけから出る値（§11.4 の式）。**効率** = 実測 / 同じ形の限界。目標は限界の形へ実装を寄せ、効率を上げること。今の配備（DP8 × TP4、lanes 16、k=1、ISL 128 / OSL 128、128 同時）は実測 1,186.9 tok/s で、同じ形の限界 9,638 の **12.3%**（§11.7）。
- 条件は並列列挙する（代表仮定を置かない）: 系列長 ctx ∈ {1（KV 無視）, 128, 8,192, 32,768, 262,144（native `max_position_embeddings`）, 1,000,000（YaRN factor 4、モデルカードの最大長）} と 192（今の配備の計測条件 = ISL 128 + decode の途中、実測との比較用のみ）；routed expert の先読みの当たり方 4 ケース（§11.4.5）；MTP k ∈ {0, 1}；batch b は 1〜2,048（agg の漸近まで）。
- この節で決めること: 系列長ごとの tsu 最大・agg 最大とその形（§11.5）、採る形の詳細（§11.6）、今の実装から限界へ近づける順番（§11.7）。決めないこと: 精度（`docs/NUMERICS.md`）、prefill（`docs/PREFILL.md`）、server（`docs/SERVER.md`）。

### 11.2 設計の入力

#### 11.2.1 モデル（HF 定義、配備の数値形式: dense bf8 1,088 B/tile、hc mixer・router・norm bf16、routed expert BF4 576 B/tile）

H = 2560、48 層 = 36 GDN（gated delta net、value head 48 × 128、qk head 16 × 128、状態 fp32）+ 12 QSA（query 24 head × 256、kv 2 head、indexer 4 head × 128 で index block（圧縮 4）を採点し budget 2,048 行を選ぶ sparse attention）、全層 MoE（512 expert top-10 × 640 + shared 640）、hyper-connection 4 枝（残差 [rows, 4H]、低ランク 320）、PLE、vocab 248,320、MTP 1 層。

| 単品（HF 定義） | バイト [MiB] |
|---|---|
| GDN 層（36 層）: in_proj qkv [2·16·128+48·128, H] bf8 | 26.56 |
| GDN: in_proj z [48·128, H] bf8 | 15.94 |
| GDN: out_proj [H, 48·128] bf8 | 15.94 |
| GDN: a/b 射影 [48, H]×2 bf8 + conv/norm/A_log/dt_bias | 0.33 |
| QSA 層（12 層）: q+gate 射影 [2·24·256, H] bf8 | 31.88 |
| QSA: k/v 射影 [2·256, H]×2 bf8 | 2.66 |
| QSA: o_proj [H, 24·256] bf8 | 15.94 |
| QSA: indexer 射影 [4·128+128, H] bf8 + norm | 1.66 |
| MoE（全 48 層 + MTP）: shared expert up/gate/down [640, H]×3 bf8 + gate | 4.99 |
| MoE: router [512, H] bf16 | 2.50 |
| MoE: routed expert 1 個 up/gate [2·640, H] + down [H, 640] BF4（× 512 × 49 層） | 2.64 |
| hyper-connection（層ごとに 2 ブロック）: down [320, 4H] + up [4H, 320] bf16 + inject + norm | 12.60 |
| 層の残り: hc 2 ブロック + norm 2 本 | 25.21 |
| PLE（層 2 の後）: [4H, H] + [H, H] bf16 + 表 | 62.64 |
| final mixer [320, 4H]×2 bf8 + norm | 6.66 |
| lm_head [248320, H] bf8 | 644.14 |
| MTP 層（QSA + MoE + 射影 2 本 + mixer 2 本 + hc） | 104.73 |
| **dense 合計（GDN 36 × 91.5 + QSA 12 × 84.8 + PLE + final + lm_head + MTP）** | **5,128（5.378 GB）** |
| **routed expert 合計（512 × 49 層 × 2.64）** | **66,150（69.36 GB）** |
| ユーザ状態: GDN 状態 S [48,128,128] fp32 × 36 層（+ conv tap 3 行 bf16） | 110.1/user 常駐、1 pass 読み+書き 220.2 |
| ユーザ KV: QSA 1 層 1 token K/V 2 head × 256 × 2 × bf16 + index key 128/4 | 2.06 KiB/token/層 = ctx 192 で 4.6、262k で 6,336、1M で 24,170 MiB/user |

#### 11.2.2 routing の実態（census: 本件プロンプト集合 34 item × 256 decode 位置の層別 top-10、CPU oracle）

routed の分子は一様仮定ではなく census の実測: 同時 B 行の実 distinct expert 数、EP 群に丸ごと置いたときの最繁忙群、tile 回し置きのときの最繁忙 chip、2 段先読み（層 l の router を 2 ブロック早い / 1 ブロック早い残差で回す）の集合とヒット率（h2 = P2 のヒット率、h12 = P1 ∪ P2 のヒット率）。rows > 256 は外挿（distinct の減衰則と偏り則）。

| B（行） | 実 distinct expert/層（一様仮定） | 2 段先読みのヒット率 h2 / h12 | 最繁忙 group（EP2 / EP4 / EP8、expert 数） | 最繁忙 chip（tiles 形 TPx8 / TPx16 / TPx32、tile 数; 均等なら distinct×20/TPx） | tiles TPx8 の集合（P2 / P1−P2 / 外れ / 無駄読み、tile） |
|---|---|---|---|---|---|
| 1 | 10.0（10.0） | 0.64 / 0.81 | 6.2 / 4.2 / 3.0 | 27 / 15 / 9（25 / 12 / 6） | 26.0 / 8.1 / 5.4 / 12.1 |
| 1x2 | 16.5（19.8） | 0.66 / 0.83 | 9.8 / 6.2 / 4.2 | 44 / 24 / 13（41 / 21 / 10） | 42.8 / 12.7 / 8.3 / 19.5 |
| 16x2 | 183.3（239.6） | 0.73 / 0.87 | 95.9 / 51.3 / 28.3 | 466 / 237 / 124（458 / 229 / 115） | 435.6 / 98.8 / 64.0 / 130.7 |
| 32x2 | 270.2（367.1） | 0.77 / 0.89 | 139.4 / 73.3 / 39.3 | 684 / 346 / 179（676 / 338 / 169） | 634.3 / 119.7 / 76.5 / 144.6 |
| 128x2 | 419.0（508.7） | 0.86 / 0.94 | 212.7 / 109.1 / 56.4 | 1054 / 530 / 269（1048 / 524 / 262） | 978.5 / 108.4 / 68.2 / 99.2 |
| 256 | 426.2（508.7） | 0.86 / 0.94 | 216.1 / 110.7 / 57.2 | 1071 / 539 / 274（1065 / 533 / 266） | 994.0 / 106.3 / 67.2 / 93.7 |

#### 11.2.3 機械の物理値（設計の入力。出典は一次資料（tt-isa-documentation、EthernetSsDesignSpec、GDDR6 のデータシート）と 27B README §3 の最新版）

**全定数の監査表（計画 v2 D-1、規則 R14。生成器 `tools/design/audit.py` の行がそのまま本表）**。生成器（`tools/design`、`tools/limit`）のモジュール直下の全定数を 1 行ずつ、値・種類・出典・状態で示す。状態 spec = 一次資料に値がある、derived = spec から式で出す、hw = この実機から読んだ（`tools/design/hw/`）、recount = 本設計の命令数・hop 数で、§12 の本文（D-2〜D-4）から数え直す、choice = 設計・模型の選択（§11.3 / §12 で列挙または理由付け）、stat = 本件プロンプト集合の統計、build = ビルド物でしか分からない（カーネルの大きさ、R2 の例外、K-1 で合否）、load = 負荷時にfirmware が決める（AI クロック、R2 の例外、K-1 で合否）、nosource = 探索して資料に値が無い（作業記録 10-28 に探索範囲）ので設計がその値に依らないことを D-2 で示す、name = ラベル・列挙。**この表の値は 10-28 の監査後のもの。下の箇条書きと §11.4 以降・§12 の数値は改訂 8 の生成値（N_RD 16、P_PACK 16、softplus 101、recip 6/群、harvest 14 通りの最悪）のままで、D-7 の再生成まで更新しない（規則 R9）。**

| 定数 | 値 | 種類 | 出典 | 状態 |
|---|---|---|---|---|
| H | 2560 | model | HF config.json text_config hidden_size | spec |
| L, L_GDN, L_QSA, LAYER | 48, 36, 12 | model | HF config.json text_config num_hidden_layers, layer_types (full_attention_interval 4) | spec |
| GDN_QK, GDN_V, GDN_HD | 16, 48, 128 | model | HF config.json text_config linear_num_key_heads, linear_num_value_heads, linear_key/value_head_dim | spec |
| CONV_K | 4 | model | HF config.json text_config linear_conv_kernel_dim | spec |
| QSA_Q, QSA_KV, QSA_HD | 24, 2, 256 | model | HF config.json text_config num_attention_heads, num_key_value_heads, head_dim | spec |
| ROPE | 64 | model | HF config.json text_config head_dim 256 x partial_rotary_factor 0.25 | derived |
| IDX_Q, IDX_KV, IDX_HD, IDX_BUDGET, IDX_COMP | 4, 1, 128, 2048, 4 | model | HF config.json text_config indexer_n_heads, indexer_kv_heads, indexer_head_dim, indexer_budget, indexer_compress_ratio | spec |
| E, TOPK (K) | 512, 10 | model | HF config.json text_config num_experts, num_experts_per_tok | spec |
| E_INT, SH_INT | 640, 640 | model | HF config.json text_config moe_intermediate_size, shared_expert_intermediate_size | spec |
| V | 248320 | model | HF config.json text_config vocab_size | spec |
| HC, HC_LOW, HCD | 4, 320, 10240 | model | HF config.json text_config hc_count, hc_lowrank; HCD = HC x H | spec |
| PLE_LAYER, PLE_NGRAM_PARTS, PLE_DIM | 2, 128, 2560 | model | HF config.json text_config ple_layer_ids [2], split_ngram_parts, ple_embed_dim | spec |
| PLE_HEADS, PLE_HEAD_DIM | 16, 160 | model | HF config.json text_config (ngram_size 3 - 1) x heads_per_ngram 8 = 16; ple_embed_dim / 16 = 160 | derived |
| PLE_CONV_K, PLE_DIL | 4, 3 | model | HF config.json text_config ple_conv_kernel_size; dilation = ngram_size (tt/ple.py:163, the HF n-gram conv) | spec |
| MTP layers | 1 | model | HF config.json text_config mtp_num_hidden_layers | spec |
| BF8 (TB bf8) | 1088 B/tile | format | tt-metal blockfloat_common.cpp: 1,024 datum bytes + 64 shared-exponent bytes (16 values share one) | spec |
| BF4 (TB bf4) | 576 B/tile | format | same file: 512 datum bytes + 64 exponent bytes | spec |
| BF16, FP32 | 2, 4 B | format | IEEE / bfloat16 | spec |
| TILE | 32 | format | 32 x 32 tile of 4 16 x 16 faces (tt-isa-documentation be83310 BlackholeA0/TensixTile/TensixCoprocessor/MVMUL.md) | spec |
| N_TILES, EXPERT_TILE_COLS | 20 | format | E_INT 640 / 32 | derived |
| EXPERT_TILE, GU, DN | 138240 B, 2/3, 1/3 | format | expert (2 x 640 x 2560 + 2560 x 640) x 0.5625 B / 20 columns; gate/up 2 of 3 matrices | derived |
| CORES | 130 | resource | this machine: every chip 13 of 14 Tensix columns enabled (tools/design/hw/hw_facts_a03u07.json) | hw |
| XS, YS | x 1-7,10-16 minus the chip's harvested column; y 2-11 | resource | tt-isa-documentation be83310 BlackholeA0/NoC/Coordinates.md + hw_facts (harvested x per chip) | hw |
| L1 per core | 1536 KiB | resource | tt-isa-documentation be83310 BlackholeA0/TensixTile/L1.md; MEM_L1_SIZE | spec |
| firmware (in L1_FIXED) | 40 KiB | resource | MEM_MAP_END = 40,960 B (tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h, compiled /tmp/q38/mm.c) | spec |
| kernel (in L1_FIXED) | 100 KiB | resource | kernel binaries + config: a build product (rule R2 exception); default config 69 KiB (bh_hal_tensix.cpp) | build |
| DRAM_CHIP | 8 x 4,278,190,080 B x 0.90 | resource | blackhole_140_arch.yaml dram_bank_size x 8 banks (= 31.875 GiB; the code's 31.9 is rounded, correction 58); 0.90 = reserve for runtime / trace (choice) | spec+choice |
| BW_DRAM, BW (DRAM) | 512 GB/s | resource | 8 GDDR6 instances x 2 x16 channels x 16 Gbps (tt-smi ENABLED_GDDR 0xff, DDR_SPEED 16,000, hw_facts); read + write share it (JESD250) | derived |
| BANK | 47.4 B/cycle | resource | 512 GB/s / 8 banks / 1.35 GHz | derived |
| BANK_EPS | 3 DRAM tiles per bank | resource | blackhole_140_arch.yaml dram (x 0 and 9) | spec |
| G | 4096 B | choice | DRAM request granule >= 2 KiB contiguous (27B README section 3 GDDR rule (1): one ACT per 2 KiB, tRC/tFAW not binding; K4ZAF325BC Table 98) | choice |
| CLK | 1.35 GHz | clock | tt-smi AICLK fmax 1350 MHz (hw_facts); firmware throttles above TDP 130 W (seen 1306-1312 MHz under other users' load) | load |
| NIU_OUT, NIU_IN, HOP (NoC), FLIT, NOC, NOC_FLIT, LINK | 5, 5, 9 cycles, 64 B | noc | tt-isa-documentation be83310 BlackholeA0/NoC/README.md:62-66 (NIU->router ~5, router->router 9, router->NIU ~5, 64 B a cycle) | spec |
| NX, NY | 17, 12 | noc | blackhole_140_arch.yaml grid; tt-isa-documentation be83310 BlackholeA0/NoC/Coordinates.md | spec |
| N_RD | 8 cycles (code still 16) | noc | SharePoint SOC 'bh noc deepdive' slide 21: write requests fetch flits from L1, 8 cycles of round trip | spec |
| ETH (channel coordinates) | per chip | noc | hw_facts: logical ch0-7 -> NoC0 x [1,16,2,15,3,4,13,12] (x 14 and 5 harvested; chip 18: ch6 x 5). gen_core_map.py's list (x 14 as ch5) is wrong for this machine: correction 59 | hw |
| UP_OWN, UP_HOPS, MC_HOPS, DRAM_HOPS | 7/11/14, 14, 17, 14 | noc | worst over 14 hypothetical harvests (rule R13 violation); recount on this machine's placement at D-3 | recount |
| NOC_ETH_CORE, NOC_HUB_ETH, NOC_ETH_ALL, NOC_ETH_ETH, NEAR_ETH_MIN | 6, 2, 22, 3, 91 | noc | placement placeholders; recount on this machine's placement and Ethernet tiles at D-3/D-4 | recount |
| NOC_RELAY (limit/mm_cost) | 92 | noc | revision-5 relay value used by mm_cost.tree_level (step6 TREE); replace by stages.py's stage sums at D-2 | recount |
| MVMUL_LAT | 5 | tensix | tt-isa-documentation be83310 BlackholeA0/TensixTile/TensixCoprocessor/MatrixUnit.md:21-25; Dst.md:112 (5 cycles before a written 8x16 block is readable) | spec |
| PHASES | bf8 2, bf4 1, bf16 4 | tensix | fidelity phases of the deployed formats (HiFi2, LoFi, HiFi4: MVMUL.md, SrcASrcB.md; ttnn/decode_matmul.py dense plan) | spec+choice |
| WEIGHT_UNPACK, UNP_BYTES, UNPACK_BYTES, ELT | 8 / 8 / 16 cycles a tile, 128 B/cycle, 16 | tensix | tt-isa-documentation be83310 BlackholeA0/UNPACR_Regular.md (BH throttle: 8-bit x8 = 128 B, 16-bit x4 '2x' = 128 B, <= 8 rows a cycle) | spec |
| UNP_ADDR | 2 | tensix | WormholeB0 UNPACR_Regular.md:735 (2 cycles of address calculation; BH page refers to WH: 'similar, not identical') | spec |
| SFP_LAT | 2 | tensix | tt-isa-documentation be83310 BlackholeA0/TensixCoprocessor/VectorUnit.md table; SFPMAD.md:70 (auto stall 1 cycle) | spec |
| PACK_ROWS | 4 | tensix | tt-isa-documentation be83310 BlackholeA0/TensixCoprocessor/PACR.md (4 Dst rows of 16 datums a cycle) | spec |
| SEM, STALL_LAG | 1, 1 | tensix | tt-isa-documentation be83310 BlackholeA0/TensixCoprocessor/SyncUnit.md, STALLWAIT.md:27 | spec |
| RISC_L1_LOAD, RISC_NOC_REG, RISC_FENCE | 8, 7, 1 | tensix | tt-isa-documentation be83310 BlackholeA0/TensixTile/BabyRISCV/README.md load latency table (L1 miss >= 8, NoC regs >= 7); MemoryOrdering.md:57-59 | spec |
| ACT silu, sigmoid, exp, rsqrt | 54, 52, 40, 30 cycles a group | tensix | repo llk_sfpu compiled with sfpi 7.74.0 -O3 -mcpu=tt-bh-tensix, counted by tools/design/sfpu/sfpu_count.py (worklog 10-28) | derived |
| ACT softplus | 52 (code 101) | tensix | same count, repo's current softplus (correction: 101 was the 27B copy's older version) | derived |
| ACT recip | 9 + 5 n for n groups (code 6 a group) | tensix | SFPLOADMACRO pipeline of _calculate_reciprocal_fast_24b_5c_ (correction 57) | derived |
| ACT mul, add | 4, 4 | tensix | SFPLOAD + op + nop (latency 2) + SFPSTORE (VectorUnit.md) | derived |
| P_PACK | 17 bf16 / 18 fp32 (code 16) | tensix | tt-metal #59810 / #58704 (LLK team, BH pack paths read against the RTL): packer drain 16-17 bf16, 18 fp32 | spec |
| P_UNP | no BH value (code 17 = NEO) | tensix | searched: tt-isa, Glean (TA Unpackers Performance = NEO RTL, #59809, #58713, TEN-4865), Glean chat: no BH value. Design must not depend on it (D-2) | nosource |
| ETH_POLL, NOC_ISSUE, EXPERT_LOOKUP | 22, 10, 10 | risc | poll = 2 x (fence 1 + L1 load 8 + compare/branch 2); issue = 3 stores + cmd_buf-ready load 7; lookup = L1 load + add | derived |
| DRISC_PAGE | 110 | risc | 27B core_plan.h kDriscPageCycles (instruction count x ISA latencies); recount from our DRISC sequence at D-2 | recount |
| ETH_SEND | 31 | risc | 27B core_plan.h kEthDm0PacketCycles (4 TXQ registers + reads + cmd_ongoing, EthernetTxRx.md); recount from our ERISC sequence at D-4 | recount |
| GDDR_PIPE | 405 cycles (0.3 us) | dram | controller/PHY cycles not in Rambus GDDR6 Controller UG 1.03, BH GDDR Architecture, SOC MAS 4.6 (Glean chat 10-28); only O2O Tdram ~392 (empirical). Keep off the chain (D-2/D-3) | nosource |
| BW (link), BW_LINK | 25 GB/s | link | BH Galaxy runs 200G (Confluence syseng 'Overview of Blackhole ASIC -> Blackhole Galaxy'); one channel one way | spec |
| CH, LINKS | 2 | link | this machine's I-0 link map: 2 channels per neighbour per direction (tools/design/hw/probe_links_a03u07.txt) | hw |
| HOP (chip) | 306 ns + medium | link | EthernetSsDesignSpec 2.13 Table 16 280 ns (400G, L1->remote L1, excludes wire/backplane) + RS-FEC 25.6 ns at 200G (27B README 3); medium per link class at D-4 (AEC DSP latency unpublished) | spec+recount |
| SAF, SAF_RATE | 96 B/cycle | link | EthernetSsDesignSpec 2.8.4 (TX reads L1 by 6 x 128-bit ports); store-and-forward 2.8 | spec |
| HDR | 56 B | link | preamble/SFD 8 + Ethernet 14 + TT-link 18 (tt-isa-documentation be83310 BlackholeA0/EthernetTile/EthernetTxRx.md:21) + FCS 4 + IPG 12 | spec |
| PKT | 2048 B | choice | packet payload <= 4,096 B (EthernetSsDesignSpec 1.2.10); 2 KiB chosen | choice |
| TORUS_R, TORUS_C | 8, 4 | link | this machine's I-0 map: 8 rings of 4 + vertical rings of 8 (hw_facts torus groups) | hw |
| ROWS, ROWS_AR | 2 | choice | B = 1, MTP k = 1 (token + verify row) | choice |
| TP, TPX, PP, KVG, GDR_UNITS | 8, 8, 4, 4, 96 | choice | the chosen form DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]; KVG = TP8 / 2 kv heads; 6 value heads x 16 units of 8 columns | choice |
| WAVES | 10 | choice | partial sums streamed in 10 waves (section 11.3 rule 17) | choice |
| PIECE | 512 KiB | choice | DRAM read piece of step7 (consumption-order issue) | choice |
| ROUTED_LOOKAHEAD, ROUTED_RESERVE_LAYERS | 2 | choice | routed reserve of 2 layers in L1 (fastest of 1-5, section 12.6) | choice |
| L1_MM, L1_GDR, L1_KVC, STREAM_MULT | 80, 48, 64 KiB, x1 | choice | section 11 stream regions (only when the configuration needs them, correction 33) | choice |
| PKT_CAP, SUBMESH, CASES_K, FORMS, HC_FORM | - | choice | enumeration / numerical method of the generators | choice |
| T_BUSIEST, X_DISTINCT, h12, h2 | 43.9 tiles, 16.5, 0.83, 0.66 (rows 1x2) | stat | census of the prompt set (34 items x 256 positions, CPU oracle, routing/stats.json) | stat |
| R_EFF, TOKENS_PER_PASS | 1.376 | stat | MTP k=1 tokens per pass on the prompt set (ISL 128) | stat |
| GB, MiB, KiB, KIND, FMT, VALNAME, CHIPS, TILE_PAIRS, STAGES, FIRST, PIECES, FANIN, KMAX_POOL, FRAMES_SINK, _MESH, _MEET, _MMS | - | name | labels, caches and enumerations | name |

状態の内訳: build 1, choice 9, derived 11, hw 5, load 1, name 1, nosource 2, recount 5, spec 34, spec+choice 2, spec+recount 1, stat 2


- DRAM 512 GB/s/chip（読み書きの和）、容量 31.9 GiB の 90% = 30.8 GB を使う（runtime・trace・CB に 10%）。
- chip 間 link 25 GB/s × 2 本/方向 × 4 方向（8 × 4 torus、submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）、1 hop 306 ns（EthernetSsDesignSpec Table 16 の 400G の 280 ns + 200G の RS-FEC の codeword の組 25.6 ns）+ 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、1 packet 4,096 B + 56 B。chip 内の段（`comm_model.py`）: 中継 138 cycle + flit（受け側の検知 22 + 同じ側の対の Ethernet タイルへの NoC 書き 10 + N_RD + 3 hop + 送り側の検知 22 + 送り 31）; 会合の起点・終点は種類ごと: 足し込み系（AR、RS）の起点 = 部分和を作った owner（Ethernet タイルから 6 hop 以内の 91 コア以上の中に置く）の 2 本の書き + 6 hop + 検知 + 送り = 169 cycle + flit、足し込みの段 179 cycle; 集め（AG）の起点 = pool の hub（Ethernet タイルから 2 hop）→ Ethernet + 検知 + 送り = 107 cycle + flit、終点 = 受けた Ethernet タイルから 2 本の multicast で全 130 コアへ（最悪 22 hop）+ 検知 = 278 cycle + flit; 群間の受け渡し = 足し込み系の起点 + 受けた chip での配り。
- Tensix 130 core × 1.35 GHz: 行列積の 32×32 tile-MAC = max(MVMUL の命令数, 重み tile の SrcA への unpack)（MVMUL.md / MatrixUnit.md: MVMUL は SrcB 8 行 × SrcA 16×16 を 1 cycle に 1 本、fidelity の phase ごとに 1 本; UNPACR_Regular.md: SrcA へ 8 行/cycle（8・4 bit）、4 行/cycle（16 bit））、fidelity は配備の重みの形式どおり（bf8 HiFi2、BF4 LoFi、bf16 HiFi4）→ 行 ≤ 8 で bf8 8 / BF4 8 / bf16 16 cycle、32 行で 32 / 16 / 64 cycle（`mm_cost.py`。改訂 5 までは全形式一律 HiFi2 32 cycle で、b=1 の 2 行には過大だった誤り）。attention / indexer の行列積は bf16 の K/V を HiFi4 で 64 cycle/tile。単品の時間は段の積み上げ（`stages.py`、改訂 8）: 行列積の pool = 実際に作れるコアの割付（`pool_plan.py`: 列ごとのコア数 / 線形 Stream-K / 列丸ごとの最速、列ごとの木をコアの実際の終わりから追う）の MAC + 送り手と木と owner（SFPU の fp32 finalize、pack、部分和の Dst への unpack と SFPU の足し込み、NoC 書き、owner の後処理）+ 出力の受け渡し（owner → hub 7〜14 hop、hub の検知と発行、2 本の multicast 17 hop、受け手の検知: 13×10 の格子・14 通りの harvest の最悪）; 要素演算は owner の finalize に融合（SFPU の fp32 関数の命令数）; GDR は (value head, 8 列) の単位で S を Dst に置き SFPU で 2 行を直列に 1111 cycle; router の top-10 は 16 owner の並列 + 選択コアの併合 743.0 cycle。仕様は RISC の load 8 cycle（L1）/ 7（NoC レジスタ）、NoC の hop 5 + 9 + 5 cycle、MVMUL・SFPU の遅延（MatrixUnit.md、VectorUnit.md）、poll は fence + load の 2 周期。文書に段数の無い遅延は予算（この予算の場合には）: packer P_PACK 16、unpacker P_UNP 17、NIU の書きのペイロード読み N_RD 16 cycle、GDDR の読みの pipeline 0.3 µs。routed の発行の段（`stages.routed_issue`）: router の単品の終わり → 選択コアが併合の直後に id を 8 bank の DRISC へ 8 本の unicast（DRAM タイルへは multicast できない: NoC/README.md:32）→ DRISC の表引きと 1 頁 110 cycle（27B `core_plan.h`）→ GDDR の pipeline → 最初の頁（2 KiB、最悪 14 hop）がコアへ = 0.53 µs。
- L1 1,536 KiB/core − 固定専有 140 KiB/core（firmware 40 + kernel 100）= 177.2 MiB/chip。ストリーム領域は構成が要るものだけ（改訂 8 の訂正 33）: KV/index チャンク 2 面 64 KiB/core（ctx > 1）、GDR 状態単位 2 面 + 出力 48 KiB/core（状態単位が 1 周の 130 コアに入らない b のとき）; 行列積の重みは枠の列に常駐するので tile のバッファは持たない。受け場所（全コアの同じ番地: 旗、x の受け 2 × rows × 最大の pool の K × 2 B、Stream-K の部分和、会合の受け 2 面、受け渡しの受け; `stages.landing_core`）。

#### 11.2.4 系列長で変わるもの

QSA 1 行 1 層の KV 読み = 選択 K/V min(ctx, 2048) 行 × 2 head × 256 × 2 × bf16 + index block ctx/4 × 128 × bf16（indexer は全 block を採点する）: 1 user 1 pass で ctx 1: 0、128: 3.2 MB、8k: 57、32k: 75、262k: 252、1M: 818 MB。ctx > 2,048 の選択（HF 定義 `reference.py`: 完全な index block の pooled key を 4 head の relu(q·k) の和で採点し、上位 512 block = 2,048 行 + 不完全な末尾を選ぶ）の段（`stages.index_select`: 採点の行列積 + fp32 の score の閾値を基数探索、DCP では回ごとの数の all-reduce）、1 行 1 層 1 chip（DCP1）: 8k 7.4 µs、262k 11.8 µs、1M 24.5 µs。選んだ K/V は選択の後にしか読めない（gather の発行 773 cycle + 読み/BW）。常駐 KV: 1 user で 262k 6.6 GB、1M 25.3 GB（1 chip の DRAM の 82% → DCP が容量のためにも要る）。YaRN は RoPE の周波数を変えるだけで decode のバイト・演算を変えない。GDN 状態は系列長に依らない。

### 11.3 確定原則（本件の設計規則。詳細設計と実装はこれに従い、外れるときは本節を先に直す）

**A. 限界と分母**
1. 限界は物理値（§11.2.3）と HF 定義（§11.2.1）だけから立てる。実装の実測値（collective の時間、expert の実効帯域、fabric の α、packet 固定費、τ）を限界に入れない。実測は効率の分子にだけ使う。
2. 単品の時間は仕様の命令数 × cycle の段の積み上げ（`stages.py`: 受け渡しの飛行・検知・pack・unpack・SFPU を含む。改訂 7 までの「op 固定費は数えない」は、仕様上要る段を 0 にした誤り）。実測値は限界に入れない（実測は実装の評価にだけ使う）。文書に段数の無い遅延だけを予算として置き、「この予算の場合には」と断る。27B の値は、その主題の最後の版を確かめてから使う。
3. 設計原則（「DRAM を止めない」等）を条件として課さない。構成は token 時間と agg だけで比べ、寝る chip があっても速ければ採る。27B の設計選択（TP4 × PP8 × 7 周）を写さないが、その形（周、空いている群が次の枠を L1 に読む）は候補として列挙する。
4. 仮定は「この場合には」で断り、代表が置けない所は並列列挙する（routed 4 ケース、系列長 6 点、hop ×1/×2/×4、ストリーム領域 ×0/×1/×2）。
5. README の数値・順位は全て生成値。同じ入力から再生成して results・README が byte 一致し `git status` が空になるまで周回し、周回ごとに本文を通読する。

**B. 軸**（chip 数 = DP × PP × S。S の内側の TP_a × DP_a = EP × TP_x、DCP、周 R は chip を増やさない）
6. 5 軸は DP / PP / TP / EP / DCP。DCP は vLLM の decode context parallel（`--decode-context-parallel-size`: TP rank の中で同じ kv head を複製する rank の内側で KV を系列方向に分け、world size を増やさない。GQA では max DCP = TP / kv head 数）。chip を増やす context parallel（prefill CP、Ring Attention、Ulysses）は decode の本件では列挙しない。構成ラベルは `DP × PP(R) × S chip[TP_a | EP × TPx | DCP/kv-group] b=…` で chip 数を明示する。
7. TP は head / 列を分け、DCP は 1 つの kv head の系列を分ける（直交）。GDN の状態は系列長に依らないので DCP の対象外で、value head で分ける。
8. EP は expert を群に丸ごと置き、層の時間は census の最繁忙群で決め、他の群は寝る（時間で評価）。dispatch / combine の all-to-all は行を持たない chip に expert の分担があるとき（DP_a > 1）に課し、EP の値には依らない（packet 模型の全対の流れ、dispatch は expert の gate/up が波で追いかけ、combine は down の間に波で流す露出）。expert の演算は gate/up の対と K を連結した down の累算で数え、expert ごとに段を分けない（原則 19）。
9. expert の列分割は 2 形を評価して速い方を採る: slice（expert ごとに TPx 等分、tile padding あり、寝る chip 無し）と tiles（20 tile を群の chip に e mod TPx から回し置き、padding 無し、偏りは census の最繁忙 chip）。
10. PP は固定段（R = 1）と周（層+tail を PP × R 枠に切り、枠 f を群 f mod PP に置く）の両方を列挙する。段の chip は空いている間に自段の dense を L1 に先読みするので、1 token の DRAM 時間には全 chip の帯域が効く。micro-batch 数 n_mb は 1〜min(PP, b) の自由変数。段の境はバイト均等と鎖均等の速い方、受け渡しの hop 数は段の形（8 × 4 torus 上の配置）から。

**C. MoE**
11. routed expert は router の後にしか読めない（dense だけ先読み可）。先読みは **2 段**: 層 l の router を 2 ブロック早い残差に当てた集合 P2 を層 l−1 の間に、1 ブロック早い集合 P1 の差分を層 l の attention ブロックの間に読み、外れ A − P1 − P2 は router 後（発行レイテンシ 0.53 µs）に読む。層の時間 = 前段の鎖 + [発行] + max(expert 段の鎖, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。L00 の expert は確定（embedding 行）だが pass 開始後にしか発行できず、MTP 層の expert は本 pass の argmax 後にしか予測できない（構造的な露出）。
12. routed の分子は census（§11.2.2）。全 expert 読みは群の 512/EP expert を毎 step 読む形で偏り無し。
13. 露出した routed 読みを二重に数えない（占有 = max(演算+通信, 先読み分/BW) + 露出分/BW）。

**D. メモリ**
14. L1 は積み上げで勘定する: 固定専有 → 構成が要るストリーム領域（ctx > 1 の KV チャンク、1 周に入らない GDR 単位）→ 受け場所（全コアの同じ番地: 旗、x の受け 2 × rows × 最大の pool の K × 2 B（次の単品の入力全体、2 面）、Stream-K の部分和 = 構成の pool の最大 fan-in 個、会合の受け 2 面、受け渡しの受け; `stages.landing_core`）→ 残差（列分割なら [rows, 4H]/TP_a、複製なら全体）→ expert 片 2 枚 → 最終段の logits chunk → 残りを常駐 dense と routed 先読みバッファに分ける（分け方は 0 / 1 層分 / 2 層分 / 残り全部の 4 通りを列挙）。空き時間の先読みは常駐の残り全部から routed 2 層分を取り置いた分まで（枠の dense は鎖が消費した順に空く; §12 で取り置き 1〜5 層を比べた最速）。dense 5.38 GB の 32 分割 160 MiB は L1 に全部は置けない。
15. 候補を「物理的に劣るはず」という理由で列挙から外さない。選択は (時間, pass あたりの DRAM バイト) の辞書順だけで決める。
16. 残差は L1 に入らなければ DRAM に置き、層ごとの往復を DRAM 車線に加える。容量は expert 分 + 段の dense 分 + ユーザの状態/KV ≤ 30.8 GB/chip（PP > 1 では自段の層分だけ）。

**E. 通信と演算**
17. 通信（会合 = collective 1 回）は packet 模型で数える（`comm_model.py`: 隣の chip へ 1 方向 2 チャネル × 25 GB/s（チャネルごとに FIFO、packet は先に空くチャネルへ、chunk は最少 packet 数と 2 の倍数の等分の速い方）、hop 306 ns、送り手の蓄積転送 96 B/cycle、packet 2 KiB + 56 B、chip 内の段（会合の種類ごとの起点・終点: 足し込み系は owner の起点と足し込みの段、集めは hub の起点と全コアへの配り; 中継、§11.2.3）、経路は縦 → 環の短い向き、形は BFS 木の multicast / 直送 / 木の reduce の速いもの、all-reduce は AG+局所 と RS+AG の速い方）。鎖に乗る露出: 入力 x の all-gather は届いた K chunk から次の射影が追いかける（露出 = 最後の chunk + 残りの仕事 − 射影）、部分和の reduce-scatter / all-reduce は 10 波で行列積の間に流す（露出 = 行列積を超えた分）、hc 読みの部分和の all-reduce も down の間に波で流す。投機 router の会合は鎖の外。
18. 層の会合は hyper-connection の形から数え、形は軸として列挙する: **A**（配備 `read_tp4`: 残差を列で TP_a 分割、down K 分割 → 部分和 AR（波）、up N 分割 → x の AG、出力 RS（波））、**B**（残差を複製: 出力 AR（波）に attention 出力の折り畳み F_i h_i（[1280, 768] bf16/chip/ブロック）を相乗りさせ MoE 読みの low-rank を局所に）、**Bn**（B + attention 読みの D を N 分割 + low-rank の AG）、**B+D**（B + D を複製）。B 系は受け渡しで残差の AG が 1 本増える。router は複製（会合無し、dense +126 MB/chip）と列分割（候補の AG 1 本）の両形、DP_a > 1 なら a2a 2 本、DCP > 1 なら merge 1 本（QSA 層）。
19. b=1 の演算は行列積の鎖で決まる: 出力 tile が core 数より少なければ K を分けて部分和を木で owner へ集める。コアの割付は実際に作れるもの（K 単位は整数、列ごとに K が違う）の最速で、閉じた式（仕事 / 130 + 一様な木）では数えない。出力は owner → hub → 全 core の 2 本の multicast で次の単品へ配る（依存する単品の出力列はどれも入力の全 K に依存するので、次の単品は配りの最初の群を待つ）。同じ入力を読む行列積は並べて走る（pool）。MoE は gate/up を tile の対（owner で silu × up × score）、routed の down を K = expert tile + shared の連結の 1 つの累算（owner で routed + gate 付き shared）にする。rows > 32 では 32 行 tile ごとの expert の和集合で数える形と、expert ごとに行を集める形（scatter-add）の速い方。GDR の単位は core に 1 つずつ乗り、verify 行は再帰で直列。段の cycle は `stages.py`（改訂 8）。

**F. 条件と結論の出し方**
20. 系列長は今の配備の計測条件（192）を設計条件にせず、§11.1 の 6 点を並べる。
21. agg は b を漸近（ユーザ状態の往復 95,605 t/s）まで振り、容量で落ちる b を明記する。「EP=1・DCP=1 が最速」のような結論を比較せずに前提から出さない。
22. tail は全単品を数える: final mixer（+ 部分和 AR + x の AG）、lm_head、局所 argmax（7.5 tile/core + NoC 集約）+ 候補の AG、MTP の mixer・fc・MTP 層・draft lm_head + argmax。PLE は演算と統計の AR × 2。
23. 群間の受け渡しは枠の境で生きている行（§11 は層の境で切るので、残差 1/TP_a + 次の層の投機の id 2 組; pass の折り返しは token id だけ）を、8 × 4 torus 上の実際のブロック配置（蛇行順、chip の対応は鏡映 4 通りの最小）で群ごとに packet 模型で流す（起点は部分を作った owner、終点は受けた chip の全コアへの配り）。§12 の枠は単品の境で切るので、境ごとに生きている行を数える（§12.5）。
24. 限界モデルは層の中で DRAM 車線を平滑化する上限（max(鎖, routed/BW)）で、L1 の取り置きと読み待ちを時間軸で追わない。時間軸で車線を追う詳細設計（§12）は、routed の読みが層境を越えて積み上がる分、周数を増やした形の受け渡し、dense を複製した形の L1 の圧迫の分だけ長くなりうる。設計の到達値（効率 100% の分母）は §12、§11 は形の比較と上限。

### 11.4 解析モデル（時間の式。実装は `free5d_v8.py`、docstring に全式）

#### 11.4.1 軸と列挙

32 = DP × PP × S、S = TP_a × DP_a = EP × TP_x、DCP | max(1, TP_a/2)、周 R ∈ {1, 2, 4}（PP > 1）、n_mb ∈ 1〜min(PP, b)、b ∈ {1, 2, …, 2048}。各構成に対し router 2 形 × expert 分割 2 形 × L1 の分け方 4 通り × 段の切り方 2 通り × 段の形（PP2 の 16 chip は 4×4 と 8×2）を評価し、(T_tok, DRAM/pass) の辞書順で速いものを採る。全 6510 構成。

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） |
|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食い、1 chip の dense 分担が DP 倍で常駐分が減る） | 無し |
| PP(R) | 層（+ lm_head）を PP × R 枠に切り、枠 f を群 f mod PP に | 群は自枠の dense のうち L1 に入る分を常駐、空いている間に次の枠を先読み、残りと routed・状態・KV を窓で流す | 段間受け渡し（残差 [rows, 4H]/TP、段の並びで 1〜3 hop）× PP × R |
| TP_a | 行列を head / 列で TP_a 枚に（形 A は残差も列で分ける: `read_tp4` の形; B 系は残差を複製: 原則 18） | 1/TP_a | 形 A: hc 統計 AR × 2 + 入力 AG × 2 + 出力 RS × 2（B 系: 原則 18） |
| EP × TPx | expert を群に丸ごと、群内を slice か tiles で | expert は 1 回 | DP_a > 1 のとき dispatch + combine の a2a；最繁忙群が時間を決める |
| DCP | kv-head 複製群の中で系列を分ける（各 chip は群の全 q head を自分の欠片で解く: attention の演算は減らず、KV の読みと index の採点・選択が 1/DCP） | 無関係 | q の AG + 部分 (o, m, l) の送りと併合（QSA 層） |

#### 11.4.2 層の時間と段の時間

- 層 = 前段（hc 読み 2 本、GDN または QSA、shared + router の射影、top-10、それらの collective）+ [発行レイテンシ] + max(expert 段の鎖（expert の up/gate → down + shared down、RS）, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。会合の露出は原則 17（x の AG は K chunk で射影が追いかける、部分和の RS / AR は 10 波で行列積の間に流す）、a2a（dispatch: hub の起点 + 全コアへの配り、combine: owner の起点 + owner の段）と DCP の q の AG・部分 (o, m, l) の送り + 併合（SFPU）は packet 模型で鎖に全部載せる。L00 と MTP 層は routed の構造的露出（原則 11）を足す。
- 枠 f の占有 occ_f = max(Σ層 + 受け渡し, 窓 = (常駐しない dense − 空き時間の先読み + micro-batch の routed + 状態/KV + 残差の往復)/BW)。群 g の DRAM/pass = Σ(群の枠) [常駐しない dense × n_mb + 状態/KV + routed × n_mb + 残差往復 × n_mb]/BW。
- **T_pass = max(max_g 群の DRAM/pass, max_g Σ(群の枠) occ_f × n_mb, Σ_f occ_f)**。空き時間の先読みは「T_pass − 群の占有」を枠に配り、常駐の残り − routed 2 層分まで（不動点; §12 の取り置きと同じ）。T_tok = T_pass。tail = 原則 22、受け渡し = 原則 23。

#### 11.4.3 L1 と DRAM の勘定（1 chip）

L1 177.2 MiB − ストリーム領域（ctx > 1 の KV チャンク、1 周に入らない GDR 単位）− 受け場所 − 残差 − expert 片 2 枚 − [最終段: logits chunk 32 行 × V/TP_a × fp32] = 残り → 常駐 dense + routed 先読みバッファ。DRAM 容量: expert 69.4 GB/(PP × EP × TPx) + 段の dense + user × (状態 0.115 GB/PP + KV) ≤ 30.8 GB。

#### 11.4.4 通信と演算の単価

会合（packet 模型、b=1 の行、submesh 4 / 8 / 16 / 32 chip）: x の AG 1.21 / 1.58 / 2.38 / 2.76 µs、hc 部分和の AR（枝ごと [2, 4×325] fp32）1.44 / 2.16 / 3.19 / 4.54、出力部分和 [2, H] bf16 の RS 1.08 / 1.51 / 2.40 / 2.79（鎖に乗る露出はこれより短い: TP8 で AG_x 0.92、RS の波 1.41、AR の波 1.53 µs）。行列積の鎖（1 chip、b=1、K 分割の木込み）: GDN qkv/z/a/b pool TP8 0.73 µs、TP32 0.59 µs。GDR 1 単位（value head, 8 列）1,111 cycle（k=1: 2 行の直列）。tile padding: expert 640 列を TPx 分割 → TPx16 1.6 倍、TPx32 1.6 倍（slice 形のみ）。

#### 11.4.5 routed の 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に読む分 |
|---|---|---|---|
| hit | routed を router の前に読み終える（2 段先読みが全部当たる; L00 は pass 開始後、MTP 層は自分の attention 後） | distinct × 2.76 MB | 0 |
| census | 2 段先読みの実測集合（B=1×2 行: 実 43.9 tile/層のうち P2 42.8、P1−P2 12.7、外れ 8.3、無駄読み 19.5（tiles 形 TPx8 最繁忙 chip）; ヒット率 h12 B=1×2 0.83、B=256 0.94） | A ∪ P1 ∪ P2 × 2.76 MB/expert | 外れ分 + 発行レイテンシ |
| miss | router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 毎 step 群の 512/EP expert 全部を読む（router に依存しない、偏り無し） | 512/EP × 2.76 MB | 0 |

### 11.5 限界性能の答え

#### 11.5.1 系列長ごとの最大（k=1、全ヒット；census / 全ミスを併記）

| ctx | tsu 最大（全ヒット） | 構成 | T_tok [ms]; DRAM/pass / 演算 / 通信 | tsu 最大（census / 全ミス） | agg 最大（全ヒット） | 構成 | users | T_pass [ms] | agg 最大（census / 全ミス） | DP1 で容量に入る最大 b |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,909 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.721; 0.481 / 0.434 / 0.258 | 1,542 / 1,378 | 76,724 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 4096 | 73.5 | 75,633 / 76,724 | 2,048 |
| 128 | 1,812 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.759; 0.613 / 0.447 / 0.258 | 1,527 / 1,365 | 73,970 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 4096 | 76.2 | 73,931 / 73,945 | 2,048 |
| 8,192 | 1,487 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.925; 0.535 / 0.627 / 0.258 | 1,302 / 1,146 | 52,659 | `DP1 x PP1 x 32chip[TP2 x DPa16 | EP32 x TPx1] b=2048` | 2048 | 53.5 | 52,142 / 52,659 | 2,048 |
| 32,768 | 1,458 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.944; 0.669 / 0.630 / 0.258 | 1,269 / 1,142 | 36,905 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1 | DCP2/kv-group] b=512` | 512 | 19.1 | 35,908 / 36,905 | 512 |
| 262,144 | 1,330 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group] b=1` | 1.034; 0.693 / 0.694 / 0.285 | 1,198 / 1,082 | 16,387 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP4/kv-group] b=128` | 128 | 10.7 | 15,786 / 16,387 | 128 |
| 1,000,000 | 1,190 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | 1.156; 0.711 / 0.791 / 0.311 | 1,075 / 989 | 6,266 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=32` | 32 | 7.0 | 5,772 / 6,259 | 32 |

b ごとの tsu 最良（全ヒット、k=1）:

| ctx \ b | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| 1 | 1,909 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,378 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 540 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 256 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP2/kv-group]` | 141 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP16 x TPx2]` | 61 `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1]` |
| 128 | 1,812 `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8]` | 1,356 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 534 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group]` | 251 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32]` | 135 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP16 x TPx2]` | 56 `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1]` |
| 8,192 | 1,487 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,163 `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 456 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group]` | 206 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP2/kv-group]` | 112 `DP1 x PP1 x 32chip[TP4 x DPa8 | EP8 x TPx4 | DCP2/kv-group]` | 44 `DP1 x PP1 x 32chip[TP2 x DPa16 | EP16 x TPx2]` |
| 32,768 | 1,458 `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8]` | 1,154 `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 453 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP2 x TPx16 | DCP4/kv-group]` | 205 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP4/kv-group]` | 107 `DP1 x PP1 x 32chip[TP4 x DPa8 | EP8 x TPx4 | DCP2/kv-group]` | 容量超過 |
| 262,144 | 1,330 `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]` | 1,104 `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]` | 411 `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 187 `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP4/kv-group]` | 容量超過 | 容量超過 |
| 1,000,000 | 1,190 `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 982 `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 331 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 容量超過 | 容量超過 | 容量超過 |

読み方: tsu 最大の形の軸（DP1 × PP4 × 8chip[TP8 | EP1 × TPx8]）は全系列長で同じで、周と DCP だけが変わる: 1: PP4(R2)×TP8 1,909、128: PP4(R4)×TP8 1,812、8,192: PP4(R2)×TP8 1,487、32,768: PP4(R4)×TP8 1,458、262,144: PP4(R4)×TP8×DCP2 1,330、1,000,000: PP4(R4)×TP8×DCP4 1,190。DCP は index block の採点と選択、K/V の読みを分けるが、attention の演算は減らず q の AG と部分の併合が要るので、長い系列でだけ効く（262,144 で DCP2 1,330 vs DCP1 1,173、1,000,000 で DCP4 1,190 vs DCP1 602）。agg 最大は常駐 KV が b を縛るので系列長とともに落ちる（DP1 で入る最大 b: 2,048 / 2,048 / 2,048 / 512 / 128 / 32）。全ミス（今の runtime の形）の tsu: 1,378 / 1,365 / 1,146 / 1,142 / 1,082 / 989。

#### 11.5.2 ctx 192（今の配備の計測条件）の詳細

- **agg 最大（全ヒット、k=1）**: **73,231 t/s** = `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048`（4096 users、T_pass 77.0 ms = 群の DRAM/pass: dense 2762 MB + 状態/KV 30,809 MB + routed 4,335 MB per chip；演算 45.0 + 通信 32.1 ms は窓の中；L1 常駐 0 / 段の dense 2762 MB、活性 80 MiB）。漸近 95,605 の 77%。census 73,192、全ミス 73,206、全 expert 読み 73,231。k=0 57,833。
- **tsu 最大（全ヒット、k=1）**: **1,810 t/s/u** = `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1`（T_tok 0.760 ms = 演算 0.447 + 通信 0.258 + 受け渡し + 窓；群の DRAM/pass 0.613 ms；周 R1 1,462 / R2 1,744 / R4 1,810；DCP1 1,810 / DCP2 1,736 / DCP4 1,684）。census 1,526（`DP1 x PP2(R4) x 16chip[TP16 | EP1 x TPx16] b=1`）、全ミス 1,365（`DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1`）、全 expert 読み 306。k=0 1,563。
- PP × TP の梯子（b=1、EP/DCP/周は自由）と hop 設計値への感度:

| PP × TP_a（b=1 の最良、EP/DCP/周/hc 形 自由） | 構成 | router / expert / 段の切り方 / 段の形 / hc | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | L1: 常駐 dense / 先読みバッファ / 空き時間の先読み [MB]（段の dense） | hop ×1 / ×2 / ×4 |
|---|---|---|---|---|---|---|---|---|---|
| PP1 × TP32 | `DP1 x PP1 x 32chip[TP32 | EP2 x TPx16] b=1` | replicated / slice / bytes / 8x4 / Bn | 1270 | 1.083 | 0.726 | 0.392 | 0.692 | 161 / 0.0 / 0 (385) | 1270 / 1000 / 673 |
| PP2 × TP16 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles / bytes / 8x2 / B+D | 1583 | 0.869 | 0.726 | 0.407 | 0.442 | 158 / 3.3 / 151 (440) | 1583 / 1192 / 777 |
| PP4 × TP8 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles / bytes / 2x4 / B+D | 1810 | 0.760 | 0.613 | 0.447 | 0.258 | 155 / 6.1 / 227 (377) | 1810 / 1494 / 1048 |
| PP8 × TP4 | `DP1 x PP8(R4) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | replicated / slice / chain / 1x4 / A | 1009 | 1.364 | 0.604 | 0.563 | 0.302 | 134 / 11.4 / 144 (354) | 1009 / 1008 / 938 |
| PP16 × TP2 | `DP1 x PP16(R2) x 2chip[TP2 | EP1 x TPx2] b=1` | split / slice / chain / 2x1 / A | 446 | 3.086 | 0.889 | 0.739 | 0.186 | 128 / 22.8 / 73 (492) | 446 / 445 / 446 |
| PP32 × TP1 | `DP1 x PP32 x 1chip[TP1 | EP1 x TPx1] b=1` | split / slice / chain / 1x1 / A | 164 | 8.399 | 1.417 | 1.179 | 0.000 | 144 / 0.0 / 0 (792) | 164 / 164 / 164 |

hop 306 ns を 1 / 2 / 4 倍にしたときの最良は PP4×TP8 / PP4×TP8 / PP4×TP8。ストリーム領域を 0 / 2 倍にすると tsu 最大点は 1,873 / 1,810。

- 各 TP の tsu 最良（b=1）:

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 382 | 3,059 (8) | `DP1 x PP8(R4) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | replicated / slice | 1009 | 1,009 (1) | 1.364; 0.604 / 0.563 / 0.302 |
| 8 | 798 | 3,190 (4) | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles | 1810 | 1,810 (1) | 0.760; 0.613 / 0.447 / 0.258 |
| 16 | 1305 | 2,611 (2) | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles | 1583 | 1,583 (1) | 0.869; 0.726 / 0.407 / 0.442 |
| 32 | 1270 | 1,270 (1) | `DP1 x PP1 x 32chip[TP32 | EP2 x TPx16] b=1` | replicated / slice | 1270 | 1,270 (1) | 1.083; 0.726 / 0.392 / 0.692 |

- 各 TP の tsu 最良（b=16; DP が違うと users が違うので agg は同列に比べない）:

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 82 | 10,474 (128) | `DP1 x PP8 x 4chip[TP4 | EP1 x TPx4] b=16 mb=8` | split / slice | 420 | 6,720 (16) | 3.276; 3.175 / 0.704 / 0.350 |
| 8 | 159 | 10,202 (64) | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=16 mb=4` | split / tiles | 439 | 7,024 (16) | 3.134; 3.021 / 0.744 / 0.481 |
| 16 | 271 | 8,673 (32) | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=16 mb=2` | split / tiles | 490 | 7,838 (16) | 2.809; 2.809 / 1.049 / 1.377 |
| 32 | 411 | 6,583 (16) | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=16` | replicated / tiles | 411 | 6,583 (16) | 3.344; 3.086 / 1.312 / 1.033 |

- ケース × k の最大:

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 57,833 | 14.1 | 4096 | `DP2 x PP1 x 16chip[TP8 x DPa2 | EP8 x TPx2 | DCP2/kv-group] b=2048` | 1563 | 1,563 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| hit | 1 | 73,231 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 1810 | 1,810 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 0 | 56,998 | 13.9 | 4096 | `DP2 x PP1 x 16chip[TP8 x DPa2 | EP8 x TPx2 | DCP2/kv-group] b=2048` | 1495 | 1,495 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 1 | 73,192 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 1526 | 1,526 | `DP1 x PP2(R4) x 16chip[TP16 | EP1 x TPx16] b=1` |
| miss | 0 | 57,833 | 14.1 | 4096 | `DP2 x PP1 x 16chip[TP8 x DPa2 | EP8 x TPx2 | DCP2/kv-group] b=2048` | 1228 | 1,228 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1` |
| miss | 1 | 73,206 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 1365 | 1,365 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1` |
| all | 0 | 57,833 | 14.1 | 4096 | `DP2 x PP1 x 16chip[TP8 x DPa2 | EP8 x TPx2 | DCP2/kv-group] b=2048` | 228 | 228 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |
| all | 1 | 73,231 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | 306 | 306 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |

- Pareto 前線（全ヒット、k=1; PP>1 4 点、EP>1 5 点、DCP>1 3 点、全 13 点）:

| agg [t/s] | tsu | users | 構成 | router / expert / hc | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73,231 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | replicated / slice / A | 76.963 | 76.003 | 45.008 | 32.056 | 0.000 | 0 / 0.0 (2762) | 32.0 | fitted |
| 69,451 | 33.9 | 2048 | `DP1 x PP1 x 32chip[TP2 x DPa16 | EP32 x TPx1] b=2048` | split / slice / A | 40.576 | 40.576 | 22.656 | 17.659 | 0.000 | 0 / 0.0 (2699) | 16.0 | fitted |
| 56,579 | 55.3 | 1024 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1] b=1024` | split / slice / Bn | 24.904 | 24.904 | 12.312 | 12.387 | 0.000 | 0 / 0.0 (1561) | 16.0 | fitted |
| 47,031 | 91.9 | 512 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1] b=512` | split / slice / A | 14.980 | 14.980 | 6.611 | 6.216 | 0.000 | 0 / 0.0 (1368) | 16.0 | fitted |
| 34,318 | 134.1 | 256 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP16 x TPx2] b=256` | replicated / slice / A | 10.265 | 10.177 | 4.293 | 4.666 | 0.000 | 0 / 0.0 (820) | 32.0 | fitted |
| 24,738 | 193.3 | 128 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=128` | split / tiles / A | 7.120 | 7.120 | 2.525 | 3.916 | 0.000 | 0 / 0.0 (710) | 13.5 | census |
| 15,922 | 248.8 | 64 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=64` | split / tiles / A | 5.531 | 5.531 | 1.914 | 1.923 | 0.000 | 0 / 0.0 (710) | 11.6 | census |
| 10,627 | 332.1 | 32 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP2/kv-group] b=32` | split / tiles / A | 4.143 | 4.143 | 1.139 | 1.421 | 0.000 | 48 / 0.0 (710) | 8.9 | census |
| 8,525 | 532.8 | 16 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group] b=16` | split / tiles / A | 2.583 | 2.583 | 0.940 | 1.588 | 0.000 | 33 / 17.1 (392) | 6.2 | census |
| 6,794 | 849.3 | 8 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=8 mb=2` | split / tiles / A | 1.620 | 1.617 | 0.666 | 0.929 | 0.000 | 98 / 11.2 (213) | 4.0 | census |
| 5,416 | 1354.1 | 4 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=4 mb=4` | split / slice / A | 1.016 | 1.000 | 0.446 | 0.465 | 0.000 | 154 / 6.8 (186) | 16.5 | census |
| 3,488 | 1744.0 | 2 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=2 mb=2` | replicated / tiles / B | 0.789 | 0.688 | 0.429 | 0.333 | 0.000 | 155 / 6.1 (236) | 2.2 | census |
| 1,810 | 1810.2 | 1 | `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles / B+D | 0.760 | 0.613 | 0.447 | 0.258 | 0.000 | 155 / 6.1 (377) | 2.2 | census |

- Pareto 前線（census、k=1）:

| agg [t/s] | tsu | users | 構成 | router / expert / hc | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73,192 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | split / slice / A | 77.004 | 76.940 | 44.803 | 32.276 | 0.673 | 0 / 0.0 (2699) | 32.0 | fitted |
| 68,556 | 33.5 | 2048 | `DP1 x PP1 x 32chip[TP2 x DPa16 | EP32 x TPx1] b=2048` | split / slice / A | 41.106 | 41.106 | 22.656 | 17.659 | 0.350 | 0 / 0.0 (2699) | 16.0 | fitted |
| 56,059 | 54.7 | 1024 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1] b=1024` | replicated / slice / A | 25.134 | 23.767 | 12.320 | 12.815 | 0.350 | 0 / 0.0 (1463) | 16.0 | fitted |
| 45,424 | 88.7 | 512 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP16 x TPx2] b=512` | split / slice / A | 15.510 | 15.510 | 6.772 | 8.236 | 0.350 | 0 / 0.0 (1368) | 32.0 | fitted |
| 32,954 | 128.7 | 256 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP8 x TPx4] b=256` | replicated / slice / A | 10.689 | 10.574 | 4.498 | 5.986 | 0.341 | 0 / 0.0 (820) | 62.2 | fitted |
| 23,394 | 182.8 | 128 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=128` | split / tiles / A | 7.529 | 7.529 | 2.537 | 3.916 | 0.282 | 0 / 0.0 (710) | 13.5 | census |
| 14,549 | 227.3 | 64 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=64` | split / tiles / A | 6.053 | 6.053 | 1.930 | 1.923 | 0.319 | 0 / 0.0 (710) | 11.6 | census |
| 9,342 | 291.9 | 32 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP2/kv-group] b=32` | split / tiles / A | 4.713 | 4.713 | 1.157 | 1.421 | 0.317 | 48 / 0.0 (710) | 8.9 | census |
| 7,113 | 444.5 | 16 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=16` | split / tiles / A | 3.095 | 3.095 | 1.021 | 1.722 | 0.274 | 30 / 19.9 (392) | 6.2 | census |
| 5,450 | 681.2 | 8 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=8` | split / tiles / A | 2.020 | 2.020 | 0.716 | 1.236 | 0.203 | 39 / 12.7 (242) | 3.8 | census |
| 4,531 | 1132.7 | 4 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=4 mb=4` | split / tiles / A | 1.215 | 1.215 | 0.469 | 0.518 | 0.545 | 153 / 7.7 (199) | 2.2 | census |
| 2,928 | 1464.1 | 2 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group] b=2 mb=2` | replicated / tiles / B | 0.940 | 0.842 | 0.438 | 0.360 | 0.273 | 153 / 7.7 (236) | 2.2 | census |
| 1,526 | 1526.5 | 1 | `DP1 x PP2(R4) x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles / B+D | 0.901 | 0.808 | 0.410 | 0.442 | 0.089 | 157 / 4.2 (435) | 1.2 | census |

- Pareto 前線（全ミス = 今の runtime の形、k=1）:

| agg [t/s] | tsu | users | 構成 | router / expert / hc | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 73,206 | 17.9 | 4096 | `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` | replicated / slice / A | 76.989 | 76.003 | 45.009 | 32.056 | 8.493 | 0 / 0.0 (2762) | 32.0 | fitted |
| 69,451 | 33.9 | 2048 | `DP1 x PP1 x 32chip[TP2 x DPa16 | EP32 x TPx1] b=2048` | split / slice / A | 40.576 | 40.576 | 22.656 | 17.659 | 4.260 | 0 / 0.0 (2699) | 16.0 | fitted |
| 56,579 | 55.3 | 1024 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1] b=1024` | split / slice / Bn | 24.904 | 24.904 | 12.313 | 12.387 | 4.260 | 0 / 0.0 (1561) | 16.0 | fitted |
| 47,031 | 91.9 | 512 | `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1] b=512` | split / slice / A | 14.980 | 14.980 | 6.612 | 6.216 | 4.260 | 0 / 0.0 (1368) | 16.0 | fitted |
| 34,233 | 133.7 | 256 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP16 x TPx2] b=256` | replicated / slice / A | 10.290 | 10.177 | 4.293 | 4.666 | 4.260 | 0 / 0.0 (820) | 32.0 | fitted |
| 24,738 | 193.3 | 128 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=128` | split / tiles / A | 7.120 | 7.120 | 2.497 | 3.916 | 3.589 | 0 / 0.0 (710) | 13.5 | census |
| 15,922 | 248.8 | 64 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32] b=64` | split / tiles / A | 5.531 | 5.531 | 1.877 | 1.923 | 3.085 | 0 / 0.0 (710) | 11.6 | census |
| 10,627 | 332.1 | 32 | `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP2/kv-group] b=32` | split / tiles / A | 4.143 | 4.143 | 1.110 | 1.421 | 2.388 | 48 / 0.0 (710) | 8.9 | census |
| 7,511 | 469.4 | 16 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32] b=16` | replicated / tiles / Bn | 2.931 | 2.931 | 0.839 | 1.220 | 1.662 | 50 / 0.0 (559) | 6.2 | census |
| 5,628 | 703.6 | 8 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=8 mb=2` | replicated / tiles / B | 1.956 | 1.920 | 0.554 | 0.542 | 2.194 | 109 / 0.0 (293) | 4.0 | census |
| 4,101 | 1025.3 | 4 | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=4 mb=4` | replicated / tiles / A | 1.342 | 1.291 | 0.452 | 0.388 | 2.426 | 161 / 0.0 (248) | 2.2 | census |
| 2,487 | 1243.7 | 2 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=2 mb=2` | replicated / tiles / B | 1.106 | 0.863 | 0.389 | 0.560 | 0.675 | 161 / 0.0 (293) | 1.2 | census |
| 1,365 | 1364.7 | 1 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles / B+D | 1.008 | 0.720 | 0.408 | 0.442 | 0.338 | 161 / 0.0 (440) | 1.2 | census |

- 今日の実装拘束内（TP ∈ {4,8}、EP = TP、PP = DCP = 1、lanes × (k+1) ≤ 32）:

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,267 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 530 (2,121 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| hit | 1 | 9,638 (75 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 605 (2,420 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 0 | 1,906 (238 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 407 (1,629 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 1 | 7,668 (60 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 452 (1,808 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 0 | 2,267 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 530 (2,121 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 1 | 9,638 (75 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 605 (2,420 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 0 | 226 (57 tsu) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` | 57 (226 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 1 | 4,568 (36 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 75 (301 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |

### 11.6 採る形の詳細設計

#### 11.6.1 tsu 側: `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1`（ctx 192、k=1、全ヒット: 1,810 tsu）

- **chip の配置**: 32 chip を PP4 群 × 8 chip（8 × 4 torus 上の 2×4 ブロック、群の直径 3 hop、群間の受け渡し 2 hop）。周 R4: 層 + tail を 16 枠に切り（bytes 均等）、枠 f を群 f mod 4 に置く。1 token は 16 枠を順に通り、各群は 1 pass に 4 回番が回る。
- **群の中（TP8）**: GDN は value head 6 本/chip（状態 S もその chip の DRAM）、qk head 2 本/chip；QSA は query head 3 本/chip、kv head 2 本を 4 枚ずつが複製（DCP1: ctx 192 では分けない）；hc は形 B+D（残差を複製、attention 出力に折り畳みを相乗り）；shared expert は列で 8 分割；router は **複製**（collective 無し）；routed expert は EP1 × TPx8 の **tiles** 形（20 tile を e mod 8 から回し置き、padding 無し、最繁忙 chip は census で 2.19 expert 相当/層）。
- **枠**（dense は群の DRAM にあり、空いている間に次の枠を L1 のバッファへ先読み; routed・状態は窓で）:

| 枠 | chip 群 | 層 | dense [MB] | 状態+KV [MB] | routed [MB] | 鎖 [ms] | 占有 [ms] | 空き時間の先読み [MB] |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | L0〜L2 | 74.6 | 2.4 | 18.2 | 0.052 | 0.052 | 33.2 |
| 1 | 1 | L3〜L6 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 54.2 |
| 2 | 2 | L7〜L10 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 3 | 3 | L11〜L14 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 4 | 0 | L15〜L18 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.2 |
| 5 | 1 | L19〜L22 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 54.2 |
| 6 | 2 | L23〜L26 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 7 | 3 | L27〜L30 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 8 | 0 | L31〜L34 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.2 |
| 9 | 1 | L35〜L38 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 54.2 |
| 10 | 2 | L39〜L42 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 11 | 3 | L43〜L46 | 90.2 | 2.8 | 24.3 | 0.058 | 0.058 | 40.0 |
| 12 | 0 | L47〜L47 | 23.8 | 0.4 | 6.1 | 0.017 | 0.017 | 10.6 |
| 13 | 1 | tail | 106.7 | 0.0 | 6.1 | 0.047 | 0.047 | 64.1 |
| 14 | 2 | tail | 0.0 | 0.0 | 0.0 | 0.003 | 0.003 | 0.0 |
| 15 | 3 | tail | 0.0 | 0.0 | 0.0 | 0.001 | 0.001 | 0.0 |

- **1 chip の L1**（177.2 MiB）: 受け場所（旗・x の受け・部分和・会合と受け渡しの受け、`stages.landing_core`）+ 残差 + expert 片 2 枚 + 残り 160.6 MB = 常駐 dense 155 MB + routed 先読みバッファ 6.1 MB。空いている間の先読みは枠ごとに「残り − 常駐 − routed 2 層分」まで（枠の表の最右列、原則 14）。
- **1 chip の DRAM**（1 pass に読む量 922 MB = dense 591 + 状態/KV 34 + routed 297; 0.613 ms = 帯域の 81%）。常駐: expert 2.17 GB + 自群の dense + ユーザ状態。
- **1 層の通信**（TP8、hc 形 B+D、packet 模型の露出）: GDN 層 5.00 µs、QSA 層 5.00 µs、tail 15.4 µs → 1 pass 0.258 ms（原則 17/18）。群間受け渡し 最大 3.29 µs × 15 + pass の折り返し（生きている行: 層の境で残差 1/8 + 次の層の投機の id 2 組、折り返しは token id だけ; 原則 23）。
- **1 token の時間線**: 演算 0.447 ms（行列積の鎖が大半、GDR 0.040 ms、tail 込み）+ 通信 0.258 + 受け渡し ≤ 0.053 + 窓の超過 = 0.760 ms。routed（全ヒット）は 2 ブロック早い予測で前の層の裏に読まれ、露出は L00 と MTP 層の構造的な分だけ。詳細設計（§12.6、ctx 1、R2・採る hc の形）の値は 789.6 µs（同じ入力の時間軸模擬、§11 は層の中で DRAM を平滑化する上限なので短い: 原則 24）。
- **何が床か**: 通信 34%、演算 59%。hop が 2 倍なら PP4×TP8 に移る。TP32 × PP1 は通信 0.692 ms が床で 1,270。

#### 11.6.2 agg 側: `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048`（ctx 192、k=1、全ヒット: 73,231 t/s）

- **chip の配置**: DP2 複製 × 16 chip（4×4）、PP1、各複製が b=2048 users（rows 4096）。dense は TP2（複製 router）、routed は EP16 × TPx1（slice: TPx1 は 20 tile を割り切るので padding 無し；最繁忙群 32 expert/層 = 群の全部、全 expert を毎 step 読む形に近い）、DCP1（kv-head 複製群 1 枚で KV を分ける）。
- **1 chip の DRAM/pass** 77.0 ms = dense 2762 MB（常駐 0 MB の残り）+ 状態/KV 30,809 MB（2048 users × GDN 状態 231 MB 往復 /2 + KV）+ routed 4,335 MB。演算 45.0 ms・通信 32.1 ms は DRAM の窓に隠れる。
- **L1**: 活性 80 MiB（rows 4096 × 4 × H bf16）が常駐 dense を 0 MB に削る。残差 40 MiB は DRAM。
- **容量**: 22.5 GB/chip（expert 4.3 + dense + 2048 users の状態/KV）。b=2048 は ctx 192 まで、8k で DP1、32k で b=512、262k で 128、1M で 32 が容量の上限（§11.5.1）。
- **何が床か**: ユーザ状態の往復（漸近 95,605 t/s）。重みの読みはこの b でほぼ償却されている（77%）。

#### 11.6.3 系列長で形がどう変わるか

- b=1 の tsu 側: 1 → `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]`（1,909）、128 → `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8]`（1,812）、8,192 → `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]`（1,487）、32,768 → `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8]`（1,458）、262,144 → `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]`（1,330）、1,000,000 → `DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]`（1,190）。1M は常駐 KV 25 GB/user が DCP の枚数に分かれないと DRAM に入らない。
- agg 側は常駐 KV で b が縛られる: 1 → `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048`（76,724 t/s、DP1 で入る最大 b 2,048）、128 → `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048`（73,970 t/s、DP1 で入る最大 b 2,048）、8,192 → `DP1 x PP1 x 32chip[TP2 x DPa16 | EP32 x TPx1] b=2048`（52,659 t/s、DP1 で入る最大 b 2,048）、32,768 → `DP1 x PP1 x 32chip[TP4 x DPa8 | EP32 x TPx1 | DCP2/kv-group] b=512`（36,905 t/s、DP1 で入る最大 b 512）、262,144 → `DP1 x PP1 x 32chip[TP8 x DPa4 | EP1 x TPx32 | DCP4/kv-group] b=128`（16,387 t/s、DP1 で入る最大 b 128）、1,000,000 → `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=32`（6,266 t/s、DP1 で入る最大 b 32）。

### 11.7 今の実装との差と、限界へ近づける順番（ctx 192、k=1）

| 段階 | 値 |
|---|---|
| 今の配備の形（DP8 × TP4、lanes 16、k=1、router 後に expert を読む）の実測 | 1,186.9 tok/s（14.6 tok/s/user） |
| 同じ形の限界（全ミス）: T_pass 18.27 ms は段の DRAM/pass で、うち router 後の routed 読み 13.61 ms、1 chip の dense 1,368 MB のうち常駐 0 MB（残りを pass ごとに読む） | 9,638 tok/s（効率 12.3%） |
| 同じ形で routed を先読みしても（全ヒット） | 9,638（DRAM/pass 支配で不変） |
| 今日の実装拘束内（TP∈{4,8}, EP=TP, PP=DCP=1, lanes×2 ≤ 32）の最良: agg / tsu | 9,638（`DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16`）/ 605（`DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1`） |
| 拘束を外した b=16（同じ 16 users/instance）: TP16 × DPa2 + DCP4、expert tiles（EP1 × TPx32）、router 列分割、hc A | tsu 533、agg 8,525（`DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group] b=16`） |
| b=1 の tsu 最大: TP8、PP4 の 4 周、expert tiles（EP1 × TPx8）、router 複製、hc B+D、L1 に次の枠を先読み | 1,810（`DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1`） |
| agg 最大: DP2 × 16chip、TP2 × DPa8、expert slice（EP16 × TPx1）、router 複製、hc A、b=2048（4,096 users） | 73,231（漸近 95,605 の 77%） |

今の形の効率 12.3% の差（9,638 − 1,187）は限界モデルの外（実装の stall・admission・lane の同期・router 後の読みの直列化・dispatch）にあり、作業記録 §9.4 の分解で追う。限界側で形を変える順: (1) router 後の読みを 2 段先読みに（全ミス → census → 全ヒット。同じ形では DRAM/pass 支配で agg は変わらず、tsu 側で効く）、(2) b=1 の tsu 最大の形（`DP1 x PP4(R4) x 8chip[TP8 | EP1 x TPx8] b=1`）へ: expert tiles、router 複製、PP4 の 4 周で次の枠を L1 に先読み、(3) agg 側は `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` で b を漸近まで上げる。各段階の限界値は §11.5 の表の該当行。

### 11.8 未確定（決着方法）

- hop 306 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §11.5.2 の感度の範囲（×2 で PP4×TP8）。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（最大点 `DP2 x PP1 x 16chip[TP2 x DPa8 | EP16 x TPx1] b=2048` を含む）はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL 128）の採択率。投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖で、device の BF4 連鎖では未測。
- 段の値のうち文書に段数の無い遅延は予算（P_PACK 16、P_UNP 17、N_RD 16 cycle、GDDR の読みの pipeline 0.3 µs; この予算の場合には）。GDR 1 単位 1111 cycle（2 行）、行列積の割付・木・NoC の区間は仕様の命令数と hop から（`stages.py`、`pool_plan.py`）。予算はデバイスで上限を超えないことを合否で確かめる（K-1）。
- L1 の固定専有 140 KiB/core のうち kernel の 100 KiB は予算（ビルドの成果物、K-1 で入ることを確かめる）。ストリーム領域（ctx > 1 の KV チャンク、GDR 単位）を 0 / 2 倍にすると tsu 最大点 1,873 / 1,810。
- 枠の群への割付は f mod PP の固定 round-robin（群間の量の差が残る近似）。空いている段の chip が隣の段の routed を自分の DRAM から読んで ETH で渡す形（入口 8 link × 25 GB/s = 39% の DRAM 相当）は列挙していない。
- 系列長の掃引は k=1 の 4 ケースのみ。長文脈の prefill（chunked prefill、KV の書き込み）は decode の限界の外。1M は YaRN 前提で精度は限界の外。
- DCP の選択は全体の top-512 block（HF 定義どおり、閾値の基数探索を chip 間の all-reduce で揃える）で、選ばれた block は DCP の chip に round-robin で置いた系列の上でほぼ均等に分かれる、として各 chip の SDPA と K/V の読みを数える（この場合には）。偏ると最も多い chip が律速になる。

### 11.9 生成・検証・運用

- 生成: `routing_census.py`（CPU oracle、34 item）→ `routing_stats.py` → `routing/stats.json`、`design/census_stage2_batches.py` → `routing/stats_stage2.json`；`run_free5d_v8.sh`（`free5d_v8.py routing/stats.json <ctx> <out> <case:k>` を (ctx, ケース, k) ごとに 32 本並列: ctx 192 は 4 ケース × k 0/1、他の ctx は k=1；`merge_free5d_v8.py` が ctx ごとの json にまとめる）→ `render_free5d_v8.py` → 本節；`install_free5d_v8.sh` が README の §11 を差し替える。
- 検証: 同じ入力から最初から再生成し、stats・stats_stage2・results・design・README が byte 一致（`sha256sum -c routing/V8.sha256`、`scripts/cycle_v8.sh` が census → 列挙 → 設計 → 設置 → 検算を 1 周する）、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。訂正があれば最初から。
- 運用: 作業記録は追記専用・日本語・実時間 UTC、ユーザの発言と答え（誤答も）を 1 往復ごとに記録する。README は「現在の確定状態」だけを書き、誤りは上書きで直す。数値・順位は生成値だけ。限界に実測値を入れない。

## 12. 詳細設計（ctx 1、B=1、k=1）: `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1]`、hc 形 B の確定状態

この節は §11.6.1 の形を単品の段（コア・RISC・スレッドの役割、命令列、同期、NoC の経路と飛行）まで降ろして確定した設計（作業記録 §12 の手順 0〜7 と改訂 8 の段の設計 D-2〜D-7）で、実装がそのまま使う: 単品の割付（§12.1）、1 chip の L1 と DRAM の会計（§12.2）、枠（§12.3）、単品の時間 = 段の積み上げ（§12.4）、通信（§12.5）、鎖を追った定常状態 = 到達目標（§12.6）、実装への予算（§12.7）、限界モデル §11 との差（§12.8）、生成と検証（§12.9）。条件: 極短文（ctx 1: KV・index block を無視）、1 user、MTP k=1（verify 2 行）、数値形式は配備のもの。本文の数値は全て生成器 `scripts/design/*.py` の出力（`render_design.py` が本節を組む）。時間は全て仕様（tt-isa-documentation、EthernetSsDesignSpec ほか、§11.2.3）の命令数 × cycle から積み上げ、実測は入れない。文書に段数の無い 4 つの遅延は予算として置き、**この予算の場合には** の値: packer の段数 P_PACK 16 cycle、unpacker の段数 P_UNP 17 cycle、NIU の書き要求のペイロード読み N_RD 16 cycle、GDDR の読みの pipeline 0.3 µs（実装の段階でデバイスの合否確認で上限を超えないことを見る）。

**到達目標（この形の 100%）: 1 pass 789.6 µs = 1,743 tsu（全ヒット、hc 形 B、routed の取り置き 2 層）。** 内訳: 演算 421.2 + 会合の露出 328.6 + 読み待ち 10.9 + 受け渡し 26.0 + 折り返し 2.9 µs。DRAM 車線 91〜95%、L1 高水位 167〜168 MiB / 167.7。census（2 段先読みの census のヒット率）1003.9 µs = 1,371 tsu、全ミス（今の runtime の形）1322.0 µs = 1,041 tsu。

### 12.1 単品と割付（手順 0〜1）

1 pass の単品は 1358 個（HF forward の順: embed → L00〜L47（GDN 36 / QSA 12、各層 MoE、L02 の前に PLE）→ final mixer → lm_head → argmax → MTP 層（mixer、fc、QSA、MoE、lm_head））。全体 dense 5,128.0 MiB、routed expert 69.36 GB（49 層 × 512 × 2.64 MiB BF4）、GDN 状態 S 108.0 MiB/user fp32、conv tap 2,160 KiB/user。

| 単品の種類 | TP8 の分け方（1 chip の持ち分） | 会合 |
|---|---|---|
| hc 読み（gr.norm / down / up / gate / inject） | 残差 [2, 4H] を各 chip に複製（形 B）: attention 読みは down を K 分割（部分和 [2, 4×(320+4+1)] fp32、枝ごとなので scale が可換）、MoE 読みの low-rank は attention の出力の会合に相乗りした折り畳み F_i h_i で局所; up は N 分割、gate × normalized は局所、inject は down と pool | attention 読み: AR_down（波）→ AG_x（K chunk）; MoE 読み: 会合なし（AR_yf に相乗り） |
| GDN in_proj qkv / z / a / b | value head 6（key head 2: q,k 2×2×128 + v 6×128 = 1,280 列）、z 6 head、a/b 6 列（tile pad 32）: 同じ入力なので pool | 無し |
| conv1d / l2norm / GDR / norm (gated) | 自分の列・head（GDR は 6 head × 16 単位（8 列ずつ）= 96 単位を 96 core に、S は単位の Dst に fp32 で置き SFPU で更新）、S はその chip の DRAM（b=1: 往復 27.0 MiB/pass） | 無し |
| GDN out_proj / QSA o_proj | K 分割（6 value head / 3 q head の行）→ 部分和 [2, H] fp32 + 折り畳み F_i h_i [2, 1280]（形 B）→ bf16 で AR（波） | AR_yf（波） |
| QSA q+gate / k,v / indexer | q head 3（q+gate 1,536 列）、kv 1 head を群 4 枚で複製、indexer 複製（局所採点）: pool | 無し |
| MoE router / shared | router 複製（512 logit 局所、top-10 局所）、shared expert gate [1, H] も同じ pool、shared gate/up N 分割 1/8（80 列 = 3 tile、gate と up の同じ index の tile を対で 1 コア、owner で silu × up）、shared down K 分割（routed の down と K を連結した 1 つの累算、owner で gate 付きで足す） | 無し |
| routed expert gate_up / down | tiles 形: expert e の 20 列 tile を e mod 8 から回し置き（最繁忙 chip = census 1x2 行 43.9 tile/層）、gate/up は同じ index の tile の対（owner で silu × up × score）、down は持つ tile の K 行 32 本を expert をまたいで K に連結した 1 つの累算（出力 H 列） | AR_yf に相乗り（形 B: MoE ブロックの出力の会合） |
| PLE（L02 の前） | key_proj N 分割 1/8、value_proj N 分割 1/8、n-gram 行は各 chip が読む、norm の統計は AR | AR（統計） |
| lm_head / argmax | vocab 1/8（31,040 列）、局所 argmax（970 tile を 130 core）→ 候補 AG（8 × 2 行 × 8 B） | AG_argmax |
| 投機 router（層 l の router を 1 / 2 ブロック早い残差に） | 層 l の router を持つ群で、自前の hc 読み + 会合 + router（鎖の外）。L00 は確定（embedding 行から）、MTP 層は予測不可（本 pass の argmax に依存） | 形 B の attention 読みと同じ会合（鎖の外） |

数値形式（配備の `ttnn/decode_matmul.py` の dense plan と同じ）: dense（GDN・QSA・shared・lm_head・final / MTP の mixer（改訂 6 までは bf16 と数えていた誤り）・MTP の fc）bf8（1,088 B/tile）、各層の hc mixer・router・norm・PLE bf16、routed expert BF4（576 B/tile）、活性 bf16、S fp32、行列積の部分和 fp32 DEST、線上の部分和 bf16。行列積の fidelity は配備どおり重みの形式で決まる（bf8 HiFi2、BF4 LoFi、bf16 HiFi4）。合格線は `docs/NUMERICS.md` の acceptance（pinned 表、`json` 96/96）。K 分割の足す順・hc 統計の reduce の順・GDR の 4 単位分割は bf16 丸めの回数を変えるので、実装時に PCC と発散 index で判定する（机上で決めない）。

### 12.2 1 chip の L1 と DRAM の会計（手順 2）

1 chip の 1 pass（全層の和。1 chip はこのうち自群の枠の分）: dense 874.1 MiB（= 全体/8 + router 複製 + indexer 複製 + kv 射影の群内複製 + 形 B の折り畳み + 8 列の群と半 tile への詰めの padding: stored）、routed tile 296.4 MiB（最繁忙 chip、全ヒット）、S 13.5 MiB（往復 27.0）、行と作業域 26.3 MiB、L1 常駐状態（conv tap、PLE tap）292 KiB、DRAM だけの読み（embedding / n-gram 行）12.5 KiB。DRAM の動く量 1,197.5 MiB/pass/chip = 1 chip を直列に見て 2.453 ms。単品ごとの表は `routing/design_step2_account.md`。

### 12.3 枠（手順 3）

1 chip の L1 177.2 MiB（130 core × 1,536 KiB − 固定専有 140 KiB/core: firmware 40 + kernel 100 の予算）− 受け場所 9.24 MiB（全コアの同じ番地 72.8 KiB/core: 旗 64 語、x の受け 2 × rows × K 2,560 × bf16（multicast で受ける pool の最大 K の入力全体、2 面）、Stream-K の部分和 7 個（全 pool の木の最大 fan-in）、会合の受け 2 × 15,360 B、受け渡しの受け 20,512 B; `stages.landing_core`、番地は `dataflow/design/chip_l1.h`）− 常駐状態 0.29 = **枠の容量 167.7 MiB**（コアごとの枠の領域 1,323.2 KiB）。改訂 8 の訂正 33: §11 の限界模型の「ストリームバッファ 192 KiB/core」は §12 の段の設計に対応する buffer が無い（重みは枠の列に常駐、S は枠に置き単位の作業は Dst、ctx 1 は KV チャンク無し）ので持たない。枠の列 = 単品の重み（16 列の半 tile と 8 列の群に詰めた stored、形 B の折り畳み込み: 訂正 32・41）+ S + 枠の単品の作業域の最大 + 残差（単品の行は実行中だけ受け場所にある: 訂正 42）。1 pass の列（1 chip、全層の和）1,184.9 MiB → **8 枠 = 2 周**（全部入る最小の周: 27B の規則）。単品境でバイト均等に切る。枠に入るかは chip の和ではなくコアごとの最大で決まる（§12.4.1 の置き方: 14 通りの harvest の全部で、全枠のコアの最大 ≤ 枠の領域）:

| 枠 | 群 | 単品の範囲 | 列 [MiB] | 重み [MiB] | うち routed [MiB] | S [MiB] | DRAM/pass [MiB] | L1 の余り [MiB] |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | embed_tokens 〜 L06.gdn.norm | 148.4 | 146.0 | 36.3 | 2.2 | 150.6 | 19.3 |
| 1 | 1 | L06.gdn.out_proj 〜 L13.gdn.gr.norm | 148.7 | 147.1 | 42.3 | 1.5 | 150.1 | 19.0 |
| 2 | 2 | L13.gdn.gr.down 〜 L19.moe.gr.inject | 147.1 | 145.2 | 36.3 | 1.9 | 148.9 | 20.6 |
| 3 | 3 | L19.moe.router 〜 L26.gdn.in_proj_qkv | 149.2 | 147.2 | 42.3 | 1.9 | 151.0 | 18.5 |
| 4 | 0 | L26.gdn.conv1d 〜 L33.gdn.gr.norm | 149.1 | 147.1 | 42.3 | 1.9 | 150.9 | 18.6 |
| 5 | 1 | L33.gdn.gr.down 〜 L39.moe.gr.inject | 147.1 | 145.2 | 36.3 | 1.9 | 148.9 | 20.6 |
| 6 | 2 | L39.moe.router 〜 L46.gdn.in_proj_qkv | 149.2 | 147.2 | 42.3 | 1.9 | 151.0 | 18.5 |
| 7 | 3 | L46.gdn.conv1d 〜 mtp.lm_head | 146.0 | 145.4 | 18.1 | 0.4 | 146.1 | 21.7 |

群ごとの 1 chip の DRAM/pass 297.1〜301.4 MiB（偏り 1.5%）= 0.609〜0.617 ms。枠の列は dense が鎖に消費された順に空くので、群が空いている間（他 3 群の番）に次の枠の dense を**全量**先読みできる（先読みに使える L1 = 容量 − (列 − dense) = 123〜149 MiB/枠、手順 5）。routed tile は投機 router 後にしか読めないので窓の中（routed 2 層分 ≈ 11.6 MiB を L1 に取り置く: §12.6 で 1〜5 層を比べた最速）。

### 12.4 単品の時間 = 段の積み上げ（手順 4 / 改訂 8 D-2、1 chip、仕様のみ）

単品の時間は段の積み上げ（`stages.py`: 仕様の定数に出典を付け、単品の種類ごとに段の cycle を足す）。行列積の pool（同じ入力を読む行列積を並べる）の時間 = **実際に作れるコアの割付**の時間（`pool_plan.py`、改訂 8 の訂正 18: 閉じた式「仕事 / 130 + 一様な木」は K 単位が整数で列ごとに K が違うので作れない）。割付は 3 方式の最速: A（列ごとにコア数 q を貪欲に足し、各コアは K 単位の連続区間、部分和は heap の木で owner へ）、B（線形 Stream-K: (列, K 単位) の列を 130 の等 cycle 区間に単位の境で切る。複数のコアにまたがる列の owner は、その列が自分の区間の最後の列になるコア = またがる最初のコアで、区間が次の列へ続くコアは葉）、C（列を丸ごと、長い順に空いたコアへ）。列ごとの完了は、コアの実際の終わりから木を追って数える: MAC（tile-MAC = max(MVMUL の命令数, 重み tile の SrcA への unpack): MVMUL.md・MatrixUnit.md の 8 行 × 16 × 16 を 1 cycle、fidelity の phase ごとに 1 本、UNPACR_Regular.md の BH 128 B/cycle → 2 行で bf8 HiFi2 8、BF4 LoFi 8、bf16 HiFi4 16 cycle）+ 送り手（SFPU の fp32 finalize 45、pack 2 行 17、NoC 書き 10 + N_RD、2 hop の飛行）+ 木の段（poll 22、部分和の Dst への unpack、SFPU の足し込み）+ owner（finalize、pack、書き、後処理）。コアごとの割付は `dataflow/design/unit_assign.h`。MoE（訂正 19〜21）: gate/up は tile の対（同じ index の gate と up を同じコア）で owner が silu × up（× score）、routed の down は K = この chip の expert tile + shared の 80 列を連結した 1 つの累算で owner が routed と gate 付き shared を足す。+ 出力の受け渡し（owner → hub 7/11/14 hop（owner を hub の近くに置く。13×10 の格子・14 通りの harvest の最悪）、hub の検知と発行、NoC0 と NoC1 の 2 つの長方形への multicast 17 hop、受け手の検知）。飛行は 5 + 9 × hop + 5 + flit（NoC/README.md）。要素演算は owner の finalize に融合（SFPU の fp32 関数の命令数: silu 54、sigmoid 52、rsqrt 30 cycle/群）。GDR は (value head, 8 列) の単位 = S [128, 8] fp32 を Dst に置き 1 行 2 パスを SFPU で（列の和は SFPTRANSP で行レーンに複製）、2 行の直列で 1111 cycle。router の top-10 は 16 個の router owner がそれぞれ top-10 を取り選択コアで併合 743 cycle。SDPA（ctx 1）は kv head のコアで 3 q head × 2 行を解き o_proj のコアへ直接 multicast。argmax は各コアの約 8 logit 列 tile を SFPSWAP で併合して 1 hop。1 chip の鎖（step4 の単品の和、hc 形 A の単品で数える; 採る形の鎖は §12.6）: **層 L00（GDN + MoE）8.7 µs**（L00.gdn.gr.down 0.76、L00.gdn.gr.up 0.58、L00.gdn.in_proj_qkv 0.73、L00.gdn.conv1d 0.18、L00.gdn.gated 1.11、L00.gdn.norm 0.34、L00.gdn.out_proj 0.48、L00.moe.gr.down 0.76、L00.moe.gr.up 0.58、L00.moe.router 1.39、L00.moe.routed 0.68）、**層 L03（QSA + MoE）7.9 µs**、PLE 1.2、tail（mixer + lm_head × 2 + argmax + MTP の fc）19.0 → **1 pass の鎖 429.4 µs**。投機 router 1.22 µs/本（鎖の外）。単品ごとの表は `routing/design_step4_item_time.md`。

#### 12.4.1 置き方（コアの物理配置）と、置いた時間（改訂 8 の訂正 30〜44）

単品の行列積の pool の論理コア k を、harvest の場合ごとのラスター順の道 P（x の昇順に列、各列は y 2 → 11）の P[(k + o) mod 130] に置く（`raster.py`、`placement.py`）。後の論理コアは前の論理コアから右・下にあるので、子 → 親の部分和は NoC1（左・上）で、木の辺の hop は論理の距離と列の境で決まる。pool ごとの回し o は、枠の全 pool の置いた時間の和を最小にし、コアごとの L1（置いた重み + 枠の S・状態・残差・作業域の最大の分担）≤ 枠の領域 1,323.2 KiB を制約にする（局所探索 → はみ出す枠は min-max の均し → 入る範囲で時間を戻す）。pool の時間は、置いた後の実際の hop（木の各辺、owner → hub）で数え直し、方式（列ごと / 線形 Stream-K / 列丸ごと）も置き方の上で選び直す。設計値は 14 通りの harvest の最悪。採る形 B の 8 枠の全部が、14 通りの harvest のすべてで入る: コアの最大 1,320.4〜1,323.0 KiB。行列積の時間（1 pass、1 chip の単品の和）: 置いた値 287.9 µs、段の模型（hop の楽観の定数）296.1 µs。置き方は `dataflow/design/unit_assign.h` に出す（pool ごとのコアの割付と、harvest ごとの回し）。

### 12.5 通信（手順 6）

通信の時間は packet 模型（`comm_model.py`: 隣の chip へ 1 方向 2 チャネル × 25 GB/s（チャネルごとに FIFO、packet は先に空くチャネルへ、chunk は最少 packet 数と 2 の倍数の等分の速い方）、hop 306 ns（EthernetSsDesignSpec Table 16 の 400G の 280 ns + 200G の RS-FEC の codeword の組 25.6 ns）、送り手の蓄積転送 96 B/cycle、packet 2 KiB + 56 B、経路は縦 → 環の短い向き、collective は BFS 木の multicast / 直送 / 木の reduce の速い形）。chip 内の段（改訂 8 D-4、会合の種類ごと）: 足し込み系（AR、RS の部分和）の起点 = 部分和を作った owner（Ethernet タイルから 6 hop 以内にある 91 コア以上の中に置く: core_map の 14 通りの harvest で確認）が hub と Ethernet タイルへ 2 本の書き（2 × (発行 + N_RD)）+ 6 hop + 送り側の検知と送り 31 cycle = 169 cycle + flit、足し込みの段 = 受けた Ethernet → owner 6 hop + 検知 + Dst への unpack + SFPU の足し込み = 179 cycle; 集め（AG）の起点 = 作った pool の hub（Ethernet タイルから 2 hop の位置に選ぶ）→ Ethernet 2 hop + 検知と送り = 107 cycle + flit、終点 = 受けた Ethernet タイルが 2 本の multicast で全 130 コアへ（最悪 22 hop）+ 受け手の検知 = 278 cycle + flit; 中継 = 受け側の検知 + 同じ側の対の Ethernet タイルへ 3 hop の NoC 書き + 送り側の検知 + 送り = 138 cycle + flit; 群間の受け渡し = 足し込み系の起点 + 受けた chip で全コアへ配る段（集めの終点と同じ）。群 = 2×4 ブロック、群間は 2 hop（列の 2 chip が群間 link を共有）。

| 会合 | 中身 | payload/chip [B] | 速い形 | 時間 [µs] |
|---|---|---|---|---|
| AR_down | hc 読みの部分和 + 統計（枝ごと [2, 4×325] fp32） | 10,400 | AG_tree+local | 2.16 |
| AG_low | 低ランク行 40 本（形 Bn） | 320 | AG_tree | 1.46 |
| AG_x | ブロック入力 x [2, H/8] bf16 | 1,280 | AG_tree | 1.58 |
| RS_y | ブロック出力の部分和 [2, H] bf16 | 10,240 | RS_tree | 1.51 |
| AR_y | 同上の all-reduce（複製残差） | 10,240 | AG_tree+local | 2.14 |
| AR_yf | y + 折り畳み f [2, H+1280] bf16 | 15,360 | AG_tree+local | 2.44 |
| AG_ids | 予測 expert id 40 B | 40 | AG_tree | 1.42 |
| AG_argmax | argmax 候補 8 × 2 行 × 8 B | 128 | AG_tree | 1.43 |

鎖に乗る露出（会合の全体ではなく、演算と重ねて残る分）: AG_x は届いた K chunk から射影を進めるので露出 GDN 0.91 / QSA 0.85 / MoE 0.78 µs；RS_y / AR_down は出力を 10 波で出し波ごとに流すので露出 RS_y 1.41（attention）/ 1.41（MoE）、AR_down 1.52 µs。投機 router の会合（AR_down 2.16 + AG_x 1.58）は鎖の外。

hc の形（残差の置き方と down/up の分け方）を列挙して手順 7 で時間比較した:

| 形 | 1 層の hc 区間（出力行列積 + 会合 + down + 会合 + up + AG_x 露出、GDN / QSA）[µs] | pass [µs] | DRAM 増 [MiB/chip/pass] | 中身 |
|---|---|---|---|---|
| A | 10.93 / 10.87 | 535 | +0.0 | column-split residual: D K-split (AR_down), U N-split (AG_x), output RS_y (waves) |
| B | 9.35 / 9.29 | 457 | +13.4 | replicated residual: outputs AR_y (waves); the attention output carries the fold F_i h_i, so the MoE read's low-rank is local (its R_start term off the chain); the attention read's D K-split (AR_down waves), U N-split (AG_x) |
| Bn | 9.56 / 9.50 | 468 | +13.4 | B with the attention read's D N-split (40 rows from the whole row, AG_low) instead of K-split |
| B0 | 11.11 / 11.06 | 544 | +0.0 | B without the fold (both reads D K-split with AR_down waves) |
| B+D | 8.13 / 8.08 | 398 | +80.4 | B with the attention read's D replicated (no meeting; down on the whole row; +6.25 MiB per layer per chip) |
| B+D+U | 7.08 / 7.08 | 347 | +214.4 | B+D with both reads' U replicated (no AG_x; +12.5 MiB per layer per chip) |

| 群間の受け渡し（枠の境） | 生きている行 | bytes/chip | [µs] |
|---|---|---|---|
| (pass wrap) \| embed_tokens | token id 8 | 8 | 1.05 |
| L06.gdn.norm \| L06.gdn.out_proj | 残差 1/8 5,120 + inject 32 + gated norm の行 3,072 | 8,224 | 1.51 |
| L13.gdn.gr.norm \| L13.gdn.gr.down | 残差 1/8 5,120 + 正規化残差 5,120 | 10,240 | 1.62 |
| L19.moe.gr.inject \| L19.moe.router | 残差 1/8 5,120 + x [2, H] 10,240 + inject 32 | 15,392 | 1.87 |
| L26.gdn.in_proj_qkv \| L26.gdn.conv1d | 残差 1/8 5,120 + x [2, H] 10,240 + inject 32 + in_proj_qkv 5,120 | 20,512 | 2.07 |
| L33.gdn.gr.norm \| L33.gdn.gr.down | 残差 1/8 5,120 + 正規化残差 5,120 | 10,240 | 1.62 |
| L39.moe.gr.inject \| L39.moe.router | 残差 1/8 5,120 + x [2, H] 10,240 + inject 32 | 15,392 | 1.87 |
| L46.gdn.in_proj_qkv \| L46.gdn.conv1d | 残差 1/8 5,120 + x [2, H] 10,240 + inject 32 + in_proj_qkv 5,120 | 20,512 | 2.07 |

受け渡し合計 13.7 µs/pass（step 6b: 枠の切れ目ごとに、切れ目の前で作られ後で使われる値を HF のデータの流れから数える; 最大 20,512 B は枠の容量が受け場所に仮置きした 20,512 B 以下）（形 B 系は受け手で残差の AG +1.91 µs/境）。投機 router が群をまたぐ先頭層: 前の群が残差 1/8（5 KiB）を 2 hop 1.37 µs で送り、受け手（空いている群）が自前の重みで読み + router を回す。link の共有: 群内の会合（BFS 木）は環の E/W とブロック内の縦 link だけを使い、群間の縦 link（受け渡し、残差 hop、router と expert が別枠に落ちる切れ目 3・6 の id hop）とは物理的に別で、群間 link に乗る 3 本は時刻が重ならない（受け渡しは枠の最後、残差 hop と id hop は枠の途中）ので予定表は要らない。

### 12.6 鎖を追った定常状態 = 到達目標（手順 7）

`step7_chain.py`（離散事象、12 pass 流して後半 6 pass の平均）: 単品を HF 順に、前の単品の終わり + 会合の露出（+ 枠の境の受け渡し）から始め、読みが届くまで待つ（届いた分から消費、最後の片の後 t/8）。群ごとの DRAM 車線（512 GB/s）は 512 KiB の片を消費順に発行し、routed は投機 router のトリガ後、L1 は積み上げ（先の routed 2 層分の余地を取り置く = 27B 大原則 1 の本件版。取り置き 1〜5 層を比べた最速）。routed の発行時刻: P2 = 層 l−1 の attention 開始 + 自前の hc 読み 1.34 + 会合 + router 1.22 + 発行 0.53（router の結果 → 8 bank の DRISC へ 8 本の unicast（DRAM タイルへは multicast できない）→ 表引き + DRISC の 1 頁 → GDDR の pipeline 0.3 µs（予算）→ 最初の頁がコアへ; §12.7; 群をまたげば + 残差 hop）、P1 = 同じく層 l、外れ = router 後 + 0.53。L00 は pass 開始 + 3.5 µs（確定: 全行の hc 読み + router + 発行）、MTP 層は自分の attention 開始（予測不可）。

| ケース / hc 形 | T_pass [µs] | tsu | 演算 | 会合の露出 | 読み待ち（dense / routed） | 受け渡し | 車線 | L1 高水位 [MiB] |
|---|---|---|---|---|---|---|---|---|
| 全ヒット / A | **856.7** | **1,606** | 445.2 | 382.6 | 11.5（0.0 / 11.5） | 12.3 | 78〜81% | 167〜167 |
| 全ヒット / B | **789.6** | **1,743** | 421.2 | 328.6 | 10.9（0.0 / 10.9） | 26.0 | 91〜95% | 167〜168 |
| 全ヒット / Bn | **801.5** | **1,717** | 436.9 | 325.6 | 9.9（0.0 / 9.9） | 26.0 | 90〜94% | 167〜168 |
| 全ヒット / B+D | **937.3** | **1,468** | 441.3 | 254.5 | 196.4（97.0 / 99.5） | 41.0 | 89〜99% | 148〜168 |
| census（2 段先読み）/ A | **1017.9** | **1,352** | 445.2 | 382.6 | 160.1（0.0 / 160.1） | 12.3 | 72〜75% | 158〜168 |
| census（2 段先読み）/ B | **1003.9** | **1,371** | 421.2 | 328.6 | 212.7（0.0 / 212.7） | 26.0 | 77〜81% | 163〜168 |
| census（2 段先読み）/ Bn | **1012.6** | **1,359** | 436.9 | 325.6 | 208.3（0.0 / 208.3） | 26.0 | 77〜80% | 163〜168 |
| census（2 段先読み）/ B+D | **1007.1** | **1,366** | 441.3 | 254.5 | 262.9（0.1 / 262.8） | 41.0 | 89〜99% | 161〜168 |
| 全ミス（router 後）/ A | **1390.7** | **989** | 445.2 | 382.6 | 537.6（0.0 / 537.6） | 12.3 | 48〜49% | 160〜167 |
| 全ミス（router 後）/ B | **1322.0** | **1,041** | 421.2 | 328.6 | 536.5（0.0 / 536.5） | 26.0 | 54〜55% | 165〜168 |
| 全ミス（router 後）/ Bn | **1335.3** | **1,030** | 436.9 | 325.6 | 537.0（0.0 / 537.0） | 26.0 | 54〜55% | 165〜168 |
| 全ミス（router 後）/ B+D | **1276.8** | **1,078** | 441.3 | 254.5 | 532.4（0.0 / 532.4） | 41.0 | 66〜71% | 167〜168 |

| 枠 | 群 | 演算 [µs] | 始まり → 終わり [µs]（幅） | 会合の露出 | 読み待ち | 受け渡し |
|---|---|---|---|---|---|---|
| 0 | 0 | 55.7 | 0.0 → 106.5（106.5） | 45.4 | 5.4 | 0.00 |
| 1 | 1 | 54.4 | 109.9 → 207.0（97.1） | 42.6 | 0.0 | 3.42 |
| 2 | 2 | 55.0 | 210.5 → 309.1（98.6） | 43.5 | 0.0 | 3.53 |
| 3 | 3 | 56.1 | 312.8 → 411.5（98.6） | 42.6 | 0.0 | 3.77 |
| 4 | 0 | 56.1 | 415.5 → 514.2（98.7） | 42.6 | 0.0 | 3.98 |
| 5 | 1 | 55.0 | 517.7 → 616.3（98.6） | 43.5 | 0.0 | 3.53 |
| 6 | 2 | 56.1 | 620.0 → 718.7（98.6） | 42.6 | 0.0 | 3.77 |
| 7 | 3 | 32.9 | 722.7 → 786.7（64.0） | 25.7 | 5.4 | 3.98 |

読み: 採る形は全ヒット（効率 100% の分母のケース）の (T_pass, DRAM 増) の辞書順の最小 = **形 B**（§11.3 原則 15; A 856.7 µs / 重み 1,078.6 MiB/pass（全層の和、1 chip はその 1/4）・2 周、B 789.6 µs / 重み 1,170.5 MiB/pass（全層の和、1 chip はその 1/4）・2 周、Bn 801.5 µs / 重み 1,176.3 MiB/pass（全層の和、1 chip はその 1/4）・2 周、B+D 937.3 µs / 重み 1,451.9 MiB/pass（全層の和、1 chip はその 1/4）・3 周）。census と全ミスの最速はそれぞれ B（1003.9 µs）と B+D（1276.8 µs）。改訂 8 で会合 1 本の段（chip 内の検知・発行・NoC 区間、200G の FEC）が重くなり、残差を複製して会合を減らす B 系の利点が、折り畳み重みと受け手の残差 AG の分を上回った（改訂 7 では A が最良だった）。B+D は D の複製で列が増え、周も増える。本件 b=1 は「DRAM 車線 91〜95% + 鎖」の釣り合いの点にある。全ヒットでも残る routed の露出 10.9 µs は L00（pass 開始後にしか発行できない）と MTP 層（argmax 後にしか予測できない）の分で構造的。投機 router の演算 196 µs/pass は鎖の待ち 339 µs に入る。

### 12.7 実装への予算（効率 100% = §12.6 の値を出す条件）

- 単品の演算: §12.4 の段の積み上げ（`stages.py`）を §12.4.1 の置き方の上で数えた時間（14 通りの harvest の最悪）。1 pass 421.2 µs。各段の cycle がそのまま実装の予算で、コンパイル時定数のヘッダに固定する（設計の正はヘッダ: §12.9）。
- 会合: §12.5 の露出（chip 内の段を含む; AG_x は K chunk で射影が追いかける、RS/AR は 10 波で流す、投機の会合は鎖の外）。1 pass 328.6 µs。
- 受け渡し: 枠の境で生きている行だけを 2 hop（形 B は受け手で残差の AG; 26.0 µs/pass）。
- DRAM: 各群 297〜301 MiB/pass を 512 GB/s の 91〜95% で（`dram_plan.h`: 各コアは置いた重みを ≤ 2 本の流れ（bank、端点タイル、NOC1）で読み、bank は枠の 1/8 ずつ、link の使用率 ≤ 0.5 の整数計画; 片 512 KiB、消費順の発行、全コア共通の周期（I-1b' の 99.35% の形）、routed は 2 段先読みと外れのトリガで）。読み待ち 10.9 µs 以内。
- L1: 枠の容量 167.7 MiB、routed の取り置き 2 層分、高水位 168 MiB。
- 投機 router: 層 l の router を持つ群で、attention ブロック開始の残差から 2 本（l、l+1）を鎖の待ち時間に回す。ヒット率は census（h12 0.849、h2 0.698）で、外れ分は router 後に読む（census ケース 1003.9 µs がその場合の到達値）。
- 文書に段数の無い遅延の予算（この予算の場合には）: P_PACK 16、P_UNP 17、N_RD 16 cycle、GDDR の読みの pipeline 0.3 µs。実装の最初に、デバイスでこれらの上限を超えないことを合否で確かめる（測って値を決めるのではない）。
- routed の発行（`stages.routed_issue`）: router の単品の終わり → 選択コアが併合の直後に id を 8 bank の DRISC へ 8 本の unicast（DRAM タイルへは multicast できない: NoC/README.md:32）→ DRISC の表引きと 1 頁 110 cycle → GDDR の pipeline → 最初の頁（2 KiB、最悪 14 hop）がコアへ = 718 cycle（softmax と multicast の後の段と並行）。

### 12.8 限界モデル §11 との整合

§11（改訂 8: `free5d_v8.py`、本節と同じ段の模型 `stages.py` / `comm_model.py` を使う）の同じ軸の形（`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`、ctx 1、全ヒット）は 720.8 µs（1,909 tsu）= 演算 434.5 + 会合 258.3 + 受け渡し・窓 28.1、本節 789.6 µs（1,743 tsu）: 差 +68.7 µs。
§11 は層の中で DRAM 車線を平滑化する上限（層 = max(鎖, routed/BW)）で、L1 の取り置きと読み待ちを時間軸で追わない（§11.3 原則 24）。そのため §11 は周数を増やす形・dense を複製する形（hc B+D、周 R4）を速く見積もるが、時間軸で追う本節では、枠の境の受け渡しが倍になり（R4）、複製した dense が L1 の取り置きを圧迫して読み待ちが出る（B+D）ので、どちらも採る形（B・R2）より遅い（§12.6 の表、R4 は作業記録）。設計の到達値（効率 100% の分母）は本節の値、§11 は形の比較と上限。census と全ミスでも同様（§11 census 932.1 / 本節 1003.9 µs、全ミス 1156.1 / 1322.0 µs）。

### 12.9 生成と検証

- 生成: `scripts/design/run_design.sh`（step0 → step2 → step4 → step6 → step3 → step5 → census_stage2 → step7 の各ケース、入力は HF 定義・`routing/stats.json`・`routing/*.npz`・仕様の値だけ; 単品と会合の段は `stages.py`・`pool_plan.py`・`comm_model.py`; step3 は step6 の会合と受け渡しの最大と step4 の pool の fan-in から受け場所を引く）→ `render_design.py` → 本節；`install_design.sh` が README の §12 を差し替える。
- **設計の正はヘッダ**（`dataflow/design/`、`tools/design/gen_headers.sh` が生成; `dram_plan.h` は harvest・枠・コアごとの DRAM の読みの流れ（bank、端点タイル、NoC、バイト ≤ 2 本: bank 均等、link の使用率 ≤ 0.5 の整数計画）、`unit_assign.h` は (pool, harvest, 回し) ごとの方式・予算・仕事と親）: `core_map.h`（harvest 14 通りの hub・multicast の長方形・owner の順・Ethernet 近傍・中継の対）、`stage_costs.h`（仕様の定数・予算・段の cycle 予算）、`design_plan.h`（1 pass の単品と枠）、`chip_l1.h`（1 コアの L1 の番地）、`coll_routes.h`（2×4 群の BFS 木と群間の受け渡しの経路）、`unit_assign.h`（pool ごとのコアの割付と cycle 予算）。各ヘッダの static_assert と host 検査 `dataflow/design_tests/test_qwen38_design_host.cpp`（デバイス不要）がヘッダの継ぎ目を検査する。カーネルは設計値をこれらのヘッダからコンパイル時に引き、ランタイム引数は実行時の HW の事実（DRAM のバッファの base、harvest の場合番号と自コアの index、Ethernet のチャネル）だけにする。カーネルの値とヘッダの対応（H-3）: 役割・K の範囲・列・部分和の親・fan-in → `unit_assign.h`（harvest と自コアのラスター順の index から引く）; hub と multicast の長方形・owner の順 → `core_map.h`; 単品の形・枠の範囲 → `design_plan.h`; L1 の番地（x の受け・部分和・会合と受け渡しの受け・旗・枠の領域）→ `chip_l1.h`; 会合の木・方向・packet 数と大きさ・群間の受け渡しの経路 → `coll_routes.h`; DRAM の読みの bank・端点・NoC・バイト → `dram_plan.h`; 段の cycle 予算（デバイスでの合否の線）→ `stage_costs.h` と `unit_assign.h` の pool の予算。
- 検証: 同じ入力から再生成して `routing/design_*.json`・README・ヘッダが byte 一致、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。
- 作業記録: `BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §12〜（手順ごとの結果と見つけて直した誤り）。

## More

- `docs/PROOFS.md`: the fresh-clone release proof of 2026-09-06 on the QuietBox (build, checkpoint, first start,
  acceptance, `--mtp 4`, `--long-chunks`, 64k) with its run logs; the QuietBox 2 runs.
- `docs/NUMERICS.md`: the acceptance mechanism, the pinned divergence tables (plain, `--long-chunks`, MTP, teacher-forced),
  the fused decode kernels, their switches and throughput, the device sampler's law gate, what "bitwise" means here.
- `docs/SERVER.md`: the request rules in full, sampling, thinking, the prompt-end snapshot, the stall watchdog,
  `/health`, runtime admission, the BF4 cache identity, the n-gram table pre-warm, disk and memory.
- `docs/TESTING.md`: the no-device tests, the reference corpus Q38-REF-v1 and its scorer, the regression harness
  (`tools/ci/q38_ci.py`), how to run the acceptance gate.
- `docs/PREFILL.md`: the prefill chunk bodies and the 2048-row slab (`--prefill-slab`): what runs as one matmul, its numerics class, the expert forms, the served rates.
