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

この節は、32 chip の BH Galaxy で本モデルの decode を回すときの**限界性能**（物理値と HF 定義だけから出る 1 token の時間と機械全体の tokens/s）と、その限界を出す**形**（5 軸の分け方、chip の中身、1 token の時間線）を、詳細設計と実装がそのまま使える粒度で書く。並び: §11.1 目標と指標 → §11.2 設計の入力（モデル・routing の実態・機械の物理値・系列長） → §11.3 確定原則（設計規則） → §11.4 解析モデル（時間の式） → §11.5 限界性能の答え（系列長別、ctx 192 の詳細） → §11.6 採る形の詳細設計（tsu 側 / agg 側） → §11.7 今の実装との差と近づける順番 → §11.8 未確定と決着方法 → §11.9 生成・検証・運用。本文の数値・順位は全て生成器 `render_free5d_v3.py` が `free5d_v3.py` の結果と census から決定的に生成する（手書きの数値は無い）。経緯・指摘・撤回は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §9.4〜§10（追記専用）、生成器と census は同じ共有ディレクトリ（`BH-GLX_Qwen3.8-Flash-Next_free5d_v3_20261024.py`、`..._render_free5d_v3_20261024.py`、`..._routing_census_20261023.py`、`..._routing_stats_20261023.py`、`..._routing_stats_full_20261023.json`）。README と食い違ったら新しい方の作業記録が正しい。

### 11.1 目標と指標

- **tsu**（tokens/s/user）= R_eff / T_tok。T_tok は 1 user の 1 token が出る間隔（1 pass の時間）。**agg**（tokens/s、機械全体）= DP × b × R_eff / T_pass。R_eff = 1（k=0）/ 1.376（MTP k=1: verify 2 行/user で 1 pass に出る token の期待値。本件プロンプト集合 ISL 128 の採択率）。
- **限界** = 物理値と HF 定義だけから出る値（§11.4 の式）。**効率** = 実測 / 同じ形の限界。目標は限界の形へ実装を寄せ、効率を上げること。今の配備（DP8 × TP4、lanes 16、k=1、ISL 128 / OSL 128、128 同時）は実測 1,186.9 tok/s で、同じ形の限界 9,824 の **12.1%**（§11.7）。
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

routed の分子は一様仮定ではなく census の実測: 同時 B 行の実 distinct expert 数、EP 群に丸ごと置いたときの最繁忙群、tile 回し置きのときの最繁忙 chip、投機 routing（層 l の router を 1 ブロック早い hidden で回す）のヒット率。rows > 256 は外挿（distinct の減衰則と偏り則）。

| B（行） | 実 distinct expert/層（一様仮定） | 投機 routing（1 ブロック早い router）のヒット率 h | 最繁忙 group（EP2 / EP4 / EP8、expert 数） | 最繁忙 chip（tiles 形 TPx8 / TPx16 / TPx32、tile 数; 均等なら distinct×20/TPx） |
|---|---|---|---|---|
| 1 | 10.0（10.0） | 0.78 | 6.2 / 4.2 / 3.0 | 27 / 15 / 9（25 / 12 / 6） |
| 1x2 | 16.5（19.8） | 0.79 | 9.8 / 6.2 / 4.2 | 44 / 24 / 13（41 / 21 / 10） |
| 16x2 | 183.3（239.6） | 0.85 | 95.9 / 51.3 / 28.3 | 466 / 237 / 124（458 / 229 / 115） |
| 32x2 | 270.2（367.1） | 0.88 | 139.4 / 73.3 / 39.3 | 684 / 346 / 179（676 / 338 / 169） |
| 128x2 | 419.0（508.7） | 0.94 | 212.7 / 109.1 / 56.4 | 1054 / 530 / 269（1048 / 524 / 262） |
| 256 | 426.2（508.7） | 0.94 | 216.1 / 110.7 / 57.2 | 1071 / 539 / 274（1065 / 533 / 266） |

#### 11.2.3 機械の物理値（設計の入力、出典は 27B README §3 と同じ）

- DRAM 512 GB/s/chip（読み書きの和）、容量 31.9 GiB の 90% = 30.8 GB を使う（runtime・trace・CB に 10%）。
- chip 間 link 25 GB/s × 2 本/方向 × 4 方向（8 × 4 torus、submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）、1 hop 280 ns（EthernetSsDesignSpec Table 16）+ 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、1 packet 4,096 B + 56 B。
- Tensix 130 core × 1.35 GHz: 行列積 HiFi2 32 cycle/32×32 tile-MAC、要素演算 16 cycle/bf16 tile・32 cycle/fp32 tile、gated delta rule 1 組 (value head, user) 4,224 cycle（27B §9）→ 32 列 4 単位に分け 1 単位 256 + 800 × 行 cycle、router top-10 は logit tile ごと 10 pass × 32 cycle → NoC 集約 92 cycle → 10 pass。op 固定費は数えない（27B §9 と同じ）。
- L1 1,536 KiB/core − 固定専有 140 KiB/core（firmware 40 + kernel 100）= 177.2 MiB/chip。ストリームバッファ 1 core 192 KiB（行列積 tile 2 面 + 部分和 80、GDR 状態単位 2 面 + 出力 48、KV/index チャンク 2 面 64）= 24.4 MiB/chip。

#### 11.2.4 系列長で変わるもの

QSA 1 行 1 層の KV 読み = 選択 K/V min(ctx, 2048) 行 × 2 head × 256 × 2 × bf16 + index block ctx/4 × 128 × bf16（indexer は全 block を採点する）: 1 user 1 pass で ctx 1: 0、128: 3.2 MB、8k: 57、32k: 75、262k: 252、1M: 818 MB。index 走査の演算（1 行 1 層、1 chip）: 262k 24 µs、1M 93 µs。常駐 KV: 1 user で 262k 6.6 GB、1M 25.3 GB（1 chip の DRAM の 82% → DCP が容量のためにも要る）。YaRN は RoPE の周波数を変えるだけで decode のバイト・演算を変えない。GDN 状態は系列長に依らない。

### 11.3 確定原則（本件の設計規則。詳細設計と実装はこれに従い、外れるときは本節を先に直す）

**A. 限界と分母**
1. 限界は物理値（§11.2.3）と HF 定義（§11.2.1）だけから立てる。実装の実測値（collective の時間、expert の実効帯域、fabric の α、packet 固定費、τ）を限界に入れない。実測は効率の分子にだけ使う。
2. 単価は出典のあるものだけを使う（行列積 HiFi2 32 cycle、GDR 4,224 cycle/組、tile 読み 32 cycle、op 固定費なし）。出典の無い単価は置かない。
3. 設計原則（「DRAM を止めない」等）を条件として課さない。構成は token 時間と agg だけで比べ、寝る chip があっても速ければ採る。27B の設計選択（TP4 × PP8 × 7 周）を写さないが、その形（周、空いている群が次の枠を L1 に読む）は候補として列挙する。
4. 仮定は「この場合には」で断り、代表が置けない所は並列列挙する（routed 4 ケース、系列長 6 点、hop ×1/×2/×4、ストリームバッファ ×0/×1/×2）。
5. README の数値・順位は全て生成値。同じ入力から再生成して results・README が byte 一致し `git status` が空になるまで周回し、周回ごとに本文を通読する。

**B. 軸**（chip 数 = DP × PP × S。S の内側の TP_a × DP_a = EP × TP_x、DCP、周 R は chip を増やさない）
6. 5 軸は DP / PP / TP / EP / DCP。DCP は vLLM の decode context parallel（`--decode-context-parallel-size`: TP rank の中で同じ kv head を複製する rank の内側で KV を系列方向に分け、world size を増やさない。GQA では max DCP = TP / kv head 数）。chip を増やす context parallel（prefill CP、Ring Attention、Ulysses）は decode の本件では列挙しない。構成ラベルは `DP × PP(R) × S chip[TP_a | EP × TPx | DCP/kv-group] b=…` で chip 数を明示する。
7. TP は head / 列を分け、DCP は 1 つの kv head の系列を分ける（直交）。GDN の状態は系列長に依らないので DCP の対象外で、value head で分ける。
8. EP は expert を群に丸ごと置き、層の時間は census の最繁忙群で決め、他の群は寝る（時間で評価）。dispatch / combine の all-to-all は行を持たない chip に expert の分担があるとき（DP_a > 1）に課し、EP の値には依らない。expert ごとの op 固定費は課さない。
9. expert の列分割は 2 形を評価して速い方を採る: slice（expert ごとに TPx 等分、tile padding あり、寝る chip 無し）と tiles（20 tile を群の chip に e mod TPx から回し置き、padding 無し、偏りは census の最繁忙 chip）。
10. PP は固定段（R = 1）と周（層+tail を PP × R 枠に切り、枠 f を群 f mod PP に置く）の両方を列挙する。段の chip は空いている間に自段の dense を L1 に先読みするので、1 token の DRAM 時間には全 chip の帯域が効く。micro-batch 数 n_mb は 1〜min(PP, b) の自由変数。段の境はバイト均等と鎖均等の速い方、受け渡しの hop 数は段の形（8 × 4 torus 上の配置）から。

**C. MoE**
11. routed expert は router の後にしか読めない（dense だけ先読み可）。層の時間 = 前段の鎖 + [router 後に読む分があれば発行レイテンシ 0.16 µs] + max(expert 段の鎖, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。先読みは L1 の routed バッファの分まで。全ヒットでも routed は micro-batch の hidden に依存するので段の空き時間には読めず、最良でも 1 層前の裏。
12. routed の分子は census（§11.2.2）。全 expert 読みは群の 512/EP expert を毎 step 読む形で偏り無し。
13. 露出した routed 読みを二重に数えない（占有 = max(演算+通信, 先読み分/BW) + 露出分/BW）。

**D. メモリ**
14. L1 は積み上げで勘定する: 固定専有 → ストリームバッファ → 活性 4 × [rows, H] bf16 + 残差 [rows, 4H]/TP_a → expert 片 2 枚 → 最終段の logits chunk → 残りを常駐 dense と routed 先読みバッファに分ける（分け方は 0 / 1 層分 / 2 層分 / 残り全部の 4 通りを列挙）。dense 5.38 GB の 32 分割 160 MiB は L1 に全部は置けない。
15. 候補を「物理的に劣るはず」という理由で列挙から外さない。選択は (時間, pass あたりの DRAM バイト) の辞書順だけで決める。
16. 残差は L1 に入らなければ DRAM に置き、層ごとの往復を DRAM 車線に加える。容量は expert 分 + 段の dense 分 + ユーザの状態/KV ≤ 30.8 GB/chip（PP > 1 では自段の層分だけ）。

**E. 通信と演算**
17. 通信は物理式: ring の link 時間 (n−1)/n × T / (paths × 25 GB/s)、最後の tile のレイテンシ = hop × (280 ns + 蓄積転送) + 中継の NoC、tile ごとに流して層の演算の半分までに隠す。all-reduce は RS+AG（レイテンシ 2 本）と「部分和の all-gather + 局所 reduce」（1 本）の速い方。
18. 層の collective は配備 `read_tp4` の形から数える: hc 統計の all-reduce × 2、ブロック入力の all-gather × 2、ブロック出力の reduce-scatter × 2 = 6 本/層 + router 列分割なら候補の all-gather 1 本 + DP_a > 1 なら a2a 2 本 + DCP > 1 なら merge 1 本（QSA 層）。重みの分割と通信の本数は同じ形から出す（形を混ぜない）。router は複製（collective 無し、dense +126 MB/chip）と列分割の両形を評価。
19. b=1 の演算は行列積の鎖で決まる: 出力 tile が core 数より少なければ K を分けて部分和を NoC の木で足し（1 段 92 cycle + 64 flit + 32 cycle）、同じ入力を読む行列積は並べて走る（27B §9「和でなく max」）。GDR の単位は core に 1 つずつ乗り、verify 行は再帰で直列。

**F. 条件と結論の出し方**
20. 系列長は今の配備の計測条件（192）を設計条件にせず、§11.1 の 6 点を並べる。
21. agg は b を漸近（ユーザ状態の往復 95,605 t/s）まで振り、容量で落ちる b を明記する。「EP=1・DCP=1 が最速」のような結論を比較せずに前提から出さない。

### 11.4 解析モデル（時間の式。実装は `free5d_v3.py`、docstring に全式）

#### 11.4.1 軸と列挙

32 = DP × PP × S、S = TP_a × DP_a = EP × TP_x、DCP | max(1, TP_a/2)、周 R ∈ {1, 2, 4}（PP > 1）、n_mb ∈ 1〜min(PP, b)、b ∈ {1, 2, …, 2048}。各構成に対し router 2 形 × expert 分割 2 形 × L1 の分け方 4 通り × 段の切り方 2 通り × 段の形（PP2 の 16 chip は 4×4 と 8×2）を評価し、(T_tok, DRAM/pass) の辞書順で速いものを採る。全 6510 構成。

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） |
|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食い、1 chip の dense 分担が DP 倍で常駐分が減る） | 無し |
| PP(R) | 層（+ lm_head）を PP × R 枠に切り、枠 f を群 f mod PP に | 群は自枠の dense のうち L1 に入る分を常駐、空いている間に次の枠を先読み、残りと routed・状態・KV を窓で流す | 段間受け渡し（残差 [rows, 4H]/TP、段の並びで 1〜3 hop）× PP × R |
| TP_a | 行列を head / 列で TP_a 枚に（残差も列で分ける: `read_tp4` の形） | 1/TP_a | hc 統計 AR × 2 + 入力 AG × 2 + 出力 RS × 2 |
| EP × TPx | expert を群に丸ごと、群内を slice か tiles で | expert は 1 回 | DP_a > 1 のとき dispatch + combine の a2a；最繁忙群が時間を決める |
| DCP | kv-head 複製群の中で系列を分ける | 無関係 | merge 1 本（QSA 層） |

#### 11.4.2 層の時間と段の時間

- 層 = 前段（hc 読み 2 本、GDN または QSA、shared + router の射影、top-10、それらの collective）+ [発行レイテンシ] + max(expert 段の鎖（expert の up/gate → down + shared down、RS）, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。通信は link の合計が層の演算の半分までに隠れ、隠れない分 + 最後の tile のレイテンシが露出。
- 枠 f の占有 occ_f = max(Σ層 + 受け渡し, 窓 = (常駐しない dense − 空き時間の先読み + micro-batch の routed + 状態/KV + 残差の往復)/BW)。群 g の DRAM/pass = Σ(群の枠) [常駐しない dense × n_mb + 状態/KV + routed × n_mb + 残差往復 × n_mb]/BW。
- **T_pass = max(max_g 群の DRAM/pass, max_g Σ(群の枠) occ_f × n_mb, Σ_f occ_f)**。空き時間の先読みは「T_pass − 群の占有」を枠に配り、バッファ上限まで（不動点）。T_tok = T_pass。

#### 11.4.3 L1 と DRAM の勘定（1 chip）

L1 177.2 MiB − ストリームバッファ 24.4 MiB − 活性（b=1 で 40 KiB + 残差）− expert 片 2 枚 − [最終段: logits chunk 32 行 × V/TP_a × fp32] = 残り → 常駐 dense + routed 先読みバッファ。DRAM 容量: expert 69.4 GB/(PP × EP × TPx) + 段の dense + user × (状態 0.115 GB/PP + KV) ≤ 30.8 GB。

#### 11.4.4 通信と演算の単価

最後の tile のレイテンシ: 4 chip 0.68 µs、8 chip 1.07、16 chip 1.85、32 chip 2.23。all-reduce の 2 形: hc 統計の行（b=1 で 2.5 KiB）は AG+局所で 1 本、残差 [rows, H] は RS+AG。行列積の鎖（1 chip、b=1、K 分割の木込み）: GDN qkv/z/a/b pool TP8 1.10 µs、TP32 0.68 µs。GDR 1 単位 1,856 cycle（k=1）。tile padding: expert 640 列を TPx 分割 → TPx16 1.6 倍、TPx32 1.6 倍（slice 形のみ）。

#### 11.4.5 routed の 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に読む分 |
|---|---|---|---|
| hit | routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | 1 ブロック早い投機 routing の実測ヒット率（h: B=1 0.78、B=16×2 0.85、B=256 0.94）、外れ分は router 後に読み直す | distinct × 2.76 MB × (2−h) | (1−h) 分 + 発行レイテンシ |
| miss | router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 毎 step 群の 512/EP expert 全部を読む（router に依存しない、偏り無し） | 512/EP × 2.76 MB | 0 |

### 11.5 限界性能の答え

#### 11.5.1 系列長ごとの最大（k=1、全ヒット；census / 全ミスを併記）

| ctx | tsu 最大（全ヒット） | 構成 | T_tok [ms]; DRAM/pass / 演算 / 通信 | tsu 最大（census / 全ミス） | agg 最大（全ヒット） | 構成 | users | T_pass [ms] | agg 最大（census / 全ミス） | DP1 で容量に入る最大 b |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.719; 0.589 / 0.376 / 0.308 | 1,783 / 1,470 | 84,191 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 4096 | 66.9 | 83,576 / 84,191 | 2,048 |
| 128 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.719; 0.591 / 0.376 / 0.308 | 1,779 / 1,470 | 82,215 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 4096 | 68.6 | 81,628 / 82,215 | 2,048 |
| 8,192 | 1,902 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group] b=1` | 0.723; 0.607 / 0.376 / 0.312 | 1,726 / 1,469 | 58,892 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | 2048 | 47.9 | 58,590 / 58,892 | 2,048 |
| 32,768 | 1,889 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | 0.728; 0.601 / 0.376 / 0.317 | 1,708 / 1,468 | 42,281 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | 512 | 16.7 | 41,665 / 42,281 | 512 |
| 262,144 | 1,784 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | 0.771; 0.630 / 0.377 / 0.317 | 1,582 / 1,460 | 18,037 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | 128 | 9.8 | 17,647 / 18,037 | 128 |
| 1,000,000 | 1,579 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=1` | 0.872; 0.769 / 0.299 / 0.545 | 1,565 / 1,444 | 6,960 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | 32 | 6.3 | 6,657 / 6,960 | 32 |

b ごとの tsu 最良（全ヒット、k=1）:

| ctx \ b | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| 1 | 1,913 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,518 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 645 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 307 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 168 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 72 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 128 | 1,913 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,514 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 643 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 305 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 166 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 70 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 8,192 | 1,902 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]` | 1,458 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 613 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 283 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 138 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 52 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 32,768 | 1,889 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 1,440 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 603 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 274 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 130 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 容量超過 |
| 262,144 | 1,784 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 1,336 `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 524 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 215 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 容量超過 | 容量超過 |
| 1,000,000 | 1,579 `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 1,140 `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 372 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 容量超過 | 容量超過 | 容量超過 |

読み方: tsu 最大は全系列長で PP4 の 2 周 × TP8（1M だけ PP2 の 2 周 × TP16 + DCP8: 常駐 KV 25 GB/user を 8 枚に分けないと容量に入らない）で、DCP は ctx ≥ 8k で b=1 から効く（index block の走査が分かれる: 262k で DCP4 1,784 vs DCP1 1,502）。agg 最大は常駐 KV が b を縛るので系列長とともに落ちる（DP1 で入る最大 b: 2,048 / 2,048 / 2,048 / 512 / 128 / 32）。全ミス（今の runtime の形）の tsu は系列長に鈍感（1,470 / 1,470 / 1,469 / 1,468 / 1,460 / 1,444）で、TP32 × PP1 の通信床で決まる。

#### 11.5.2 ctx 192（今の配備の計測条件）の詳細

- **agg 最大（全ヒット、k=1）**: **81,253 t/s** = `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048`（4096 users、T_pass 69.4 ms = 群の DRAM/pass: dense 324 MB + 状態/KV 30,855 MB + routed 4,335 MB per chip；演算 19.5 + 通信 16.8 ms は窓の中；L1 常駐 68 / 段の dense 392 MB、活性 80 MiB）。漸近 95,605 の 85%。census 80,680、全ミス 81,253、全 expert 読み 81,255。k=0 60,392。
- **tsu 最大（全ヒット、k=1）**: **1,913 t/s/u** = `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`（T_tok 0.719 ms = 演算 0.376 + 通信 0.308 + 受け渡し + 窓；群の DRAM/pass 0.592 ms；周 R1 1,615 / R2 1,913 / R4 1,786；DCP1 1,913 / DCP2 1,903 / DCP4 1,890）。census 1,778（`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1`）、全ミス 1,470（`DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1`）、全 expert 読み 306。k=0 1,528。
- PP × TP の梯子（b=1、EP/DCP/周は自由）と hop 設計値への感度:

| PP × TP_a（b=1 の最良、EP/DCP/周 自由） | 構成 | router / expert / 段の切り方 / 段の形 | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | L1: 常駐 dense / 先読みバッファ / 空き時間の先読み [MB]（段の dense） | hop ×1 / ×2 / ×4 |
|---|---|---|---|---|---|---|---|---|---|
| PP1 × TP32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / slice / bytes / 8x4 | 1514 | 0.909 | 0.643 | 0.242 | 0.643 | 157 / 2.3 / 0 (360) | 1514 / 980 / 574 |
| PP2 × TP16 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles / bytes / 8x2 | 1610 | 0.855 | 0.371 | 0.297 | 0.532 | 156 / 3.3 / 0 (261) | 1610 / 1085 / 657 |
| PP4 × TP8 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles / bytes / 2x4 | 1913 | 0.719 | 0.592 | 0.376 | 0.308 | 0 / 159.4 / 209 (209) | 1913 / 1416 / 938 |
| PP8 × TP4 | `DP1 x PP8(R2) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | split / slice / chain / 1x4 | 1040 | 1.323 | 0.696 | 0.528 | 0.233 | 0 / 158.8 / 236 (294) | 1040 / 1039 / 1005 |
| PP16 × TP2 | `DP1 x PP16(R2) x 2chip[TP2 | EP1 x TPx2] b=1` | split / slice / chain / 2x1 | 464 | 2.963 | 1.107 | 0.853 | 0.099 | 0 / 157.5 / 219 (492) | 464 / 464 / 462 |
| PP32 × TP1 | `DP1 x PP32 x 1chip[TP1 | EP1 x TPx1] b=1` | split / slice / chain / 1x1 | 175 | 7.849 | 1.396 | 1.571 | 0.000 | 155 / 0.0 / 0 (792) | 175 / 175 / 175 |

hop 280 ns を 1 / 2 / 4 倍にしたときの最良は PP4×TP8 / PP4×TP8 / PP8×TP4。ストリームバッファを 0 / 2 倍にすると tsu 最大点は 1,913 / 1,913。

- 各 TP の tsu 最良（b=1）:

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 383 | 3,066 (8) | `DP1 x PP8(R2) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | split / slice | 1040 | 1,040 (1) | 1.323; 0.696 / 0.528 / 0.233 |
| 8 | 796 | 3,185 (4) | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles | 1913 | 1,913 (1) | 0.719; 0.592 / 0.376 / 0.308 |
| 16 | 1425 | 2,849 (2) | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles | 1610 | 1,610 (1) | 0.855; 0.371 / 0.297 / 0.532 |
| 32 | 1514 | 1,514 (1) | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / slice | 1514 | 1,514 (1) | 0.909; 0.643 / 0.242 / 0.643 |

- 各 TP の tsu 最良（b=16; DP が違うと users が違うので agg は同列に比べない）:

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 84 | 10,700 (128) | `DP1 x PP8 x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=16 mb=8` | replicated / slice | 454 | 7,265 (16) | 3.030; 3.030 / 0.679 / 0.200 |
| 8 | 166 | 10,612 (64) | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=16 mb=4` | split / tiles | 508 | 8,121 (16) | 2.711; 2.711 / 0.620 / 0.368 |
| 16 | 328 | 10,485 (32) | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=16 mb=2` | split / tiles | 579 | 9,259 (16) | 2.378; 2.378 / 0.557 / 0.722 |
| 32 | 602 | 9,630 (16) | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 642 | 10,270 (16) | 2.144; 2.144 / 0.483 / 1.053 |

- ケース × k の最大:

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 60,392 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1528 | 1,528 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| hit | 1 | 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1913 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 0 | 59,965 | 14.6 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1510 | 1,510 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 1 | 80,680 | 19.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1778 | 1,778 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` |
| miss | 0 | 60,392 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1159 | 1,159 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` |
| miss | 1 | 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1470 | 1,470 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` |
| all | 0 | 60,393 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=2048` | 227 | 227 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |
| all | 1 | 81,255 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=2048` | 306 | 306 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |

- Pareto 前線（全ヒット、k=1; PP>1 3 点、EP>1 5 点、DCP>1 11 点、全 13 点）:

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.364 | 69.364 | 19.537 | 16.826 | 0.000 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 34.912 | 34.912 | 9.115 | 20.639 | 0.000 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 19.803 | 19.803 | 4.602 | 10.809 | 0.000 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.248 | 12.248 | 2.669 | 5.733 | 0.000 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.352 | 8.352 | 1.698 | 3.198 | 0.000 | 151 / 0.0 (392) | 62.2 | fitted |
| 29,808 | 232.9 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 5.909 | 5.909 | 1.152 | 2.211 | 0.000 | 155 / 0.0 (392) | 13.5 | census |
| 19,468 | 304.2 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | split / tiles | 4.524 | 4.524 | 0.861 | 1.476 | 0.000 | 125 / 32.0 (392) | 11.6 | census |
| 13,772 | 430.4 | 32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | split / tiles | 3.197 | 3.197 | 0.654 | 1.201 | 0.000 | 133 / 24.7 (242) | 8.9 | census |
| 10,270 | 641.9 | 16 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 2.144 | 2.144 | 0.483 | 1.053 | 0.000 | 141 / 17.1 (242) | 6.2 | census |
| 7,936 | 992.0 | 8 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP2/kv-group] b=8` | split / tiles | 1.387 | 1.378 | 0.383 | 0.970 | 0.000 | 148 / 10.5 (242) | 3.8 | census |
| 6,050 | 1512.4 | 4 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=4 mb=4` | split / tiles | 0.910 | 0.910 | 0.391 | 0.368 | 0.000 | 153 / 6.1 (199) | 2.2 | census |
| 3,403 | 1701.5 | 2 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=2 mb=2` | split / tiles | 0.809 | 0.483 | 0.391 | 0.360 | 0.000 | 147 / 12.1 (186) | 2.2 | census |
| 1,913 | 1913.2 | 1 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles | 0.719 | 0.592 | 0.376 | 0.308 | 0.000 | 0 / 159.4 (209) | 2.2 | census |

- Pareto 前線（census、k=1）:

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80,680 | 19.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.857 | 69.857 | 19.537 | 16.826 | 0.501 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,153 | 39.1 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 35.159 | 35.159 | 9.115 | 20.639 | 0.254 | 112 / 0.0 (392) | 64.0 | fitted |
| 70,278 | 68.6 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 20.049 | 20.049 | 4.602 | 10.809 | 0.254 | 135 / 0.0 (392) | 64.0 | fitted |
| 56,386 | 110.1 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.494 | 12.494 | 2.669 | 5.733 | 0.254 | 146 / 0.0 (392) | 64.0 | fitted |
| 40,998 | 160.1 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.592 | 8.592 | 1.698 | 3.198 | 0.247 | 151 / 0.0 (392) | 62.2 | fitted |
| 28,757 | 224.7 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 6.125 | 6.125 | 1.152 | 2.211 | 0.224 | 155 / 0.0 (392) | 13.5 | census |
| 18,357 | 286.8 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | split / tiles | 4.797 | 4.797 | 0.861 | 1.476 | 0.282 | 125 / 32.0 (392) | 11.6 | census |
| 12,634 | 394.8 | 32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | split / tiles | 3.485 | 3.485 | 0.654 | 1.201 | 0.296 | 133 / 24.7 (242) | 8.9 | census |
| 9,186 | 574.2 | 16 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 2.397 | 2.397 | 0.483 | 1.053 | 0.261 | 141 / 17.1 (242) | 6.2 | census |
| 7,217 | 902.1 | 8 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=8` | split / tiles | 1.525 | 1.525 | 0.383 | 0.988 | 0.188 | 148 / 10.5 (242) | 3.8 | census |
| 5,377 | 1344.2 | 4 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=4 mb=2` | split / tiles | 1.024 | 0.988 | 0.353 | 0.620 | 0.240 | 154 / 6.0 (213) | 2.2 | census |
| 3,154 | 1577.2 | 2 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=2 mb=2` | replicated / tiles | 0.872 | 0.850 | 0.297 | 0.532 | 0.140 | 156 / 3.3 (269) | 1.2 | census |
| 1,778 | 1777.6 | 1 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | replicated / tiles | 0.774 | 0.624 | 0.376 | 0.317 | 0.128 | 0 / 159.4 (218) | 2.2 | census |

- Pareto 前線（全ミス = 今の runtime の形、k=1）:

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.364 | 69.364 | 19.537 | 16.826 | 8.475 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 34.912 | 34.912 | 9.115 | 20.639 | 4.241 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 19.803 | 19.803 | 4.602 | 10.809 | 4.241 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.248 | 12.248 | 2.669 | 5.733 | 4.241 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.352 | 8.352 | 1.698 | 3.198 | 4.123 | 151 / 0.0 (392) | 62.2 | fitted |
| 29,808 | 232.9 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 5.909 | 5.909 | 1.152 | 2.211 | 3.570 | 155 / 0.0 (392) | 13.5 | census |
| 18,771 | 293.3 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | replicated / tiles | 4.692 | 4.692 | 0.851 | 1.374 | 3.067 | 157 / 0.0 (510) | 11.6 | census |
| 12,385 | 387.0 | 32 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group] b=32` | replicated / tiles | 3.555 | 3.541 | 0.659 | 1.067 | 2.370 | 157 / 0.0 (510) | 8.9 | census |
| 8,422 | 526.4 | 16 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP2/kv-group] b=16` | replicated / tiles | 2.614 | 2.589 | 0.494 | 0.906 | 1.644 | 158 / 0.0 (510) | 6.2 | census |
| 6,269 | 783.6 | 8 | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=8 mb=4` | replicated / tiles | 1.756 | 1.634 | 0.452 | 0.308 | 4.431 | 159 / 0.0 (234) | 4.2 | census |
| 4,571 | 1142.7 | 4 | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=4 mb=4` | replicated / tiles | 1.204 | 1.133 | 0.376 | 0.308 | 2.353 | 159 / 0.0 (234) | 2.2 | census |
| 2,672 | 1335.9 | 2 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=2 mb=2` | replicated / tiles | 1.030 | 0.777 | 0.297 | 0.532 | 0.639 | 160 / 0.0 (269) | 1.2 | census |
| 1,470 | 1469.6 | 1 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / tiles | 0.936 | 0.598 | 0.244 | 0.643 | 0.185 | 159 / 0.0 (360) | 0.7 | census |

- 今日の実装拘束内（TP ∈ {4,8}、EP = TP、PP = DCP = 1、lanes × (k+1) ≤ 32）:

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,263 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 525 (2,101 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| hit | 1 | 9,824 (77 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 604 (2,417 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 0 | 2,117 (265 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 481 (1,925 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 1 | 8,795 (69 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 548 (2,194 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 0 | 2,263 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 525 (2,101 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 1 | 9,824 (77 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 604 (2,417 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 0 | 226 (56 tsu) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` | 56 (226 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 1 | 4,615 (72 tsu) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=16` | 76 (304 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |

### 11.6 採る形の詳細設計

#### 11.6.1 tsu 側: `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`（ctx 192、k=1、全ヒット: 1,913 tsu）

- **chip の配置**: 32 chip を PP4 群 × 8 chip（8 × 4 torus 上の 2×4 ブロック、群の直径 3 hop、群間の受け渡し 2 hop）。周 R2: 層 + tail を 8 枠に切り（bytes 均等）、枠 f を群 f mod 4 に置く。1 token は 8 枠を順に通り、各群は 1 pass に 2 回番が回る。
- **群の中（TP8）**: GDN は value head 6 本/chip（状態 S もその chip の DRAM）、qk head 2 本/chip；QSA は query head 3 本/chip、kv head 2 本を 4 枚ずつが複製（DCP1: ctx 192 では分けない）；hc の down は K 分割・up は N 分割（残差 [rows, 4H] は列で 8 分割）；shared expert は列で 8 分割；router は **複製**（collective 無し）；routed expert は EP1 × TPx8 の **tiles** 形（20 tile を e mod 8 から回し置き、padding 無し、最繁忙 chip は census で 2.19 expert 相当/層）。
- **枠**（dense は群の DRAM にあり、空いている間に次の枠を L1 のバッファへ先読み; routed・状態は窓で）:

| 枠 | chip 群 | 層 | dense [MB] | 状態+KV [MB] | routed [MB] | 鎖 [ms] | 占有 [ms] | 空き時間の先読み [MB] |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | L0〜L5 | 96.4 | 4.4 | 36.4 | 0.087 | 0.087 | 96.4 |
| 1 | 1 | L6〜L12 | 104.3 | 4.8 | 42.5 | 0.100 | 0.100 | 104.3 |
| 2 | 2 | L13〜L19 | 104.3 | 4.8 | 42.5 | 0.100 | 0.100 | 104.3 |
| 3 | 3 | L20〜L26 | 102.6 | 5.2 | 42.5 | 0.101 | 0.101 | 102.6 |
| 4 | 0 | L27〜L33 | 104.3 | 4.8 | 42.5 | 0.100 | 0.100 | 104.3 |
| 5 | 1 | L34〜L40 | 104.3 | 4.8 | 42.5 | 0.100 | 0.100 | 104.3 |
| 6 | 2 | L41〜L47 | 104.3 | 4.8 | 42.5 | 0.100 | 0.100 | 104.3 |
| 7 | 3 | tail | 99.0 | 0.0 | 6.1 | 0.030 | 0.030 | 99.0 |

- **1 chip の L1**（177.2 MiB）: ストリームバッファ 24.4 MiB + 活性 45 KiB + expert 片 2 枚 + 残り 159.4 MB = 常駐 dense 0 MB + routed 先読みバッファ 159.4 MB。次の枠の dense 最大 104.3 MB + 1 層の routed 6.1 MB がこのバッファに入る（余り 49 MB）。
- **1 chip の DRAM**（1 pass に読む量 1150 MB = dense 819 + 状態/KV 34 + routed 297; 0.592 ms = 帯域の 82%）。常駐: expert 2.17 GB + 自群の dense + ユーザ状態。
- **1 層の通信**（TP8、最後の tile のレイテンシ 1.07 µs）: hc 統計 AR × 2（AG + 局所、1 本ずつ）、ブロック入力 AG × 2、ブロック出力 RS × 2 = 6 本 = 6.4 µs/層 → 48 層 0.308 ms。群間受け渡し 0.79 µs × 8。
- **1 token の時間線**: 演算 0.376 ms（行列積の鎖が大半、GDR 0.049 ms）+ 通信 0.308 + 受け渡し 0.006 + 窓の超過 = 0.719 ms。routed（全ヒット）は 1 層前の裏でバッファに読まれ、露出 0。
- **何が床か**: 通信 43%、演算 52%。hop が 2 倍なら PP4×TP8 に移る。TP32 × PP1 は通信 0.643 ms が床で 1,514。

#### 11.6.2 agg 側: `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048`（ctx 192、k=1、全ヒット: 81,253 t/s）

- **chip の配置**: DP2 複製 × 16 chip（4×4）、PP1、各複製が b=2048 users（rows 4096）。dense は TP16（列分割 router）、routed は EP4 × TPx4（slice: TPx4 は 20 tile を割り切るので padding 無し；最繁忙群 128 expert/層 = 群の全部、全 expert を毎 step 読む形に近い）、DCP8（kv-head 複製群 8 枚で KV を分ける）。
- **1 chip の DRAM/pass** 69.4 ms = dense 324 MB（常駐 68 MB の残り）+ 状態/KV 30,855 MB（2048 users × GDN 状態 231 MB 往復 /16 + KV）+ routed 4,335 MB。演算 19.5 ms・通信 16.8 ms は DRAM の窓に隠れる。
- **L1**: 活性 80 MiB（rows 4096 × 4 × H bf16）が常駐 dense を 68 MB に削る。残差 5 MiB は L1。
- **容量**: 20.1 GB/chip（expert 4.3 + dense + 2048 users の状態/KV）。b=2048 は ctx 192 まで、8k で DP1、32k で b=512、262k で 128、1M で 32 が容量の上限（§11.5.1）。
- **何が床か**: ユーザ状態の往復（漸近 95,605 t/s）。重みの読みはこの b でほぼ償却されている（85%）。

#### 11.6.3 系列長で形がどう変わるか

- b=1 の tsu 側は 262k まで PP4(R2) × TP8 のまま DCP を 1 → 2 → 4 と増やす（kv-head 複製群 4 枚の内側）。1M では常駐 KV 25 GB/user が 8 枚に分かれないと DRAM に入らず、PP2(R2) × TP16 + DCP8 に移る（1,579 tsu）。
- agg 側は常駐 KV で b が縛られ、8k から DP1、32k で TP16 × DPa2 + EP8 + DCP8 の b=512、262k で b=128、1M では TP32 + DCP16 の b=32（6,960 t/s）。

### 11.7 今の実装との差と、限界へ近づける順番（ctx 192、k=1）

| 段階 | 値 |
|---|---|
| 今の配備の形（DP8 × TP4、lanes 16、k=1、router 後に expert を読む）の実測 | 1,186.9 tok/s（14.6 tok/s/user） |
| 同じ形の限界（全ミス）: T_pass 17.93 ms は段の DRAM/pass で、うち router 後の routed 読み 13.59 ms、1 chip の dense 1,368 MB のうち常駐 146 MB（残りを pass ごとに読む） | 9,824 tok/s（効率 12.1%） |
| 同じ形で routed を先読みしても（全ヒット） | 9,824（DRAM/pass 支配で不変） |
| 今日の実装拘束内（TP∈{4,8}, EP=TP, PP=DCP=1, lanes×2 ≤ 32）の最良: agg / tsu | 9,824（`DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16`）/ 604（`DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1`） |
| 拘束を外した b=16（同じ 16 users/instance）: TP32 + DCP16、expert tiles、router 列分割 | tsu 642、agg 10,270（`DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16`） |
| b=1 の tsu 最大: PP4 の 2 周 × TP8、expert tiles、router 複製、L1 に次の枠を先読み | 1,913（`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`） |
| agg 最大: DP2 × 16chip[TP16 | EP4 × TPx4 | DCP8]、b=2048（4,096 users） | 81,253（漸近 95,605 の 85%） |

今の形の効率 12.1% の差（9,824 − 1,187）は限界モデルの外（実装の stall・admission・lane の同期・router 後の読みの直列化・dispatch）にあり、作業記録 §9.4 の分解で追う。限界側で形を変える順: (1) router 後の読みを投機 routing で先読み（全ミス → census → 全ヒット。同じ形では DRAM/pass 支配で agg は変わらず、tsu 側で効く）、(2) expert を tiles 形で列分割し router を複製（TP8 以上）、(3) TP8 の段を 4 群 × 2 周に組み次の枠を L1 に先読み（b=1〜4）、(4) agg 側は DP2 × 16 chip の EP4 × DCP8 で b を漸近まで上げる。各段階の限界値は §11.5 の表の該当行。

### 11.8 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §11.5.2 の感度の範囲（×2 で PP4×TP8）。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（最大点 `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` を含む）はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL 128）の採択率。投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖で、device の BF4 連鎖では未測。
- GDR の 1 単位 256 + 800 × 行 cycle は 27B §9 の 4,224 cycle/(head, user) を 4 単位に分けた机上値（単位間の k'S の交換は数えていない）。行列積の K 分割の木の段数・NoC の値は設計値。
- L1 のストリームバッファ 1 core 192 KiB と固定専有 140 KiB/core、活性 4 × [rows, H] は設計値（感度: バッファ ×0 / ×2 で tsu 最大点 1,913 / 1,913）。
- 枠の群への割付は f mod PP の固定 round-robin（群間の量の差が残る近似）。空いている段の chip が隣の段の routed を自分の DRAM から読んで ETH で渡す形（入口 8 link × 25 GB/s = 39% の DRAM 相当）は列挙していない。
- 系列長の掃引は k=1 の 4 ケースのみ。長文脈の prefill（chunked prefill、KV の書き込み）は decode の限界の外。1M は YaRN 前提で精度は限界の外。
- DCP の局所 top-k（各 chip が budget/DCP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。

### 11.9 生成・検証・運用

- 生成: `routing_census.py`（CPU oracle、34 item）→ `routing_stats.py` → `routing/stats.json`；`free5d_v3.py routing/stats.json <ctx> <out> [k1]`（ctx 192 は 4 ケース × k 0/1、他の ctx は k=1）→ `render_free5d_v3.py` → 本節；`install_free5d_v3.sh` が README の §11 を差し替える。
- 検証: 同じ入力から最初から再生成し、stats・results・README が byte 一致（`sha256sum -c routing/H*.sha256`）、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。訂正があれば最初から。
- 運用: 作業記録は追記専用・日本語・実時間 UTC、ユーザの発言と答え（誤答も）を 1 往復ごとに記録する。README は「現在の確定状態」だけを書き、誤りは上書きで直す。数値・順位は生成値だけ。限界に実測値を入れない。

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
