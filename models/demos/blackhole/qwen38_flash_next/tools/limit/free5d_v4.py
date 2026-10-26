#!/usr/bin/env python
r"""Unconstrained five-axis (DP, PP, TP, EP, DCP) decode time model for Qwen3.8-Flash-Next on one BH Galaxy, revision 4.
Revision 4 = revision 3 + the eight corrections of the detailed design (README 12.8, worklog 12.5-12.9, 13):
  V4-1 meetings (collectives) are timed with the packet model comm_model.py (link 25 GB/s, hop 280 ns, store-and-forward
       16 ns per 2 KiB packet, relay 92 cycles + 32 flits, 56 B per packet, FIFO links, BFS-tree multicast / direct /
       tree reduce, the fastest form per payload), evaluated once per (submesh, payload) and tabulated.  The exposure
       on the chain: an all-gather of x is consumed chunk by chunk by the next projection (consume_overlap minus the
       projection's own time); a reduce-scatter / all-reduce of partial sums is sent in 10 waves during its matmul (the
       remainder beyond the matmul); the hc read's AR_down likewise in waves during the down matmul; the speculative
       routers' meetings are off the chain.  (Revision 3: last-tile latency per meeting, link time hidden to half.)
  V4-2 the routed read is a two-stage speculative prefetch: P2 (router l on the residual two blocks early, read during
       layer l-1), P1 \ P2 (one block early, during layer l's attention block), the misses after the router (+ issue
       latency); the wasted reads (P1 u P2) \ A are DRAM bytes.  The census case uses stats_stage2.json (the busiest
       chip/group of each set per batch and placement); hit = everything at P2 (L00 exact at the pass start, the MTP
       layer at its own attention block: both structurally exposed), miss = everything after the router.
  V4-3 the tail counts the MTP draft lm_head, the two mixers, the MTP fc and the PLE with their meetings (PLE stats AR
       x2, mixer AR + AG x2, argmax candidates AG); argmax is local (7.5 tiles per core + a NoC gather).
  V4-4 a hand-off carries the rows alive at the cut (the residual 1/TP_a + the block input x [rows, H] + inject) on the
       inter-group links shared by the column's chips (packet model); the pass wrap carries the token ids.
  V4-5 the hyper-connection form is an axis: A (column-split residual: AR_down waves + AG_x + RS_y waves), B
       (replicated residual: AR_y waves with the attention output's fold F_i h_i so the MoE read's low-rank is local;
       +[1280,768] bf16 per chip per attention block; the hand-off adds an AG of the residual), Bn (B with the
       attention read's D N-split + AG_low), B+D (B with D replicated: +[320,4H] bf16 per layer).  The fastest is kept.
  V4-6 the idle-time dense prefetch of a frame can take the whole frame (the L1 left after the frame's rows/work and
       a routed reserve of 3 layers), not the routed buffer minus one layer (H-22 replaced).
Everything else (axes, routing numerator, L1 accounting, the rounds, the matmul chain, GDR, top-10) is revision 3.

The fifth axis is vLLM's DECODE context parallelism (--decode-context-parallel-size): the KV cache of one sequence is
sharded inside the TP ranks that replicate the same kv head (vllm/config/parallel.py: "DCP does not expand the process
world size. Without PCP, DCP reuses TP ranks"; vllm/config/model.py: for GQA max DCP = TP / total kv heads).  It adds no
chips.  The chip-adding context parallelism of prefill/training (Ring Attention, Ulysses, Megatron CP) is not enumerated.
Physical values and the HF definition only.  No design principle is imposed: a configuration is judged by its token
time and its aggregate rate alone, idle chips allowed.  Changes from revision 2 (worklog section 10.H7/H8):
  H-5  gated delta rule: 27B README section 9 = 4,224 cycles per (value head, user) for a 16-tile fp32 state and one
       row: S into DEST 16x32 + S to SrcA twice 16x32x2 + g.S on the SFPU 16x1024/32 + 48 tile-MAC x 32 + S back to L1
       16x32 + the k/q/delta/o rows 8x16.  The R verify rows of one user are a recurrence (serial): the state moves once,
       the per-row part repeats: 1,024 + 3,200 R cycles.  The 4 independent 32-column units of a head (k'S, the delta,
       the outer-product update and q'S are column-local given the k, q, v, beta, g rows) take a quarter each:
       256 + 800 R cycles per unit; units are independent across cores: ceil(units / 130) rounds on the chain.
  H-6' hyper-connection (gated residual): the residual [rows, 4H] is split by hidden columns over the TP_a chips, `down`
       K-split, `up` N-split, `inject` K-split (the served read_tp4 form, no tile padding).  One all-reduce of
       [rows, 320 + 4 + 1] fp32 per read (the rms sum rides with the partial sums: the per-row scale commutes with the
       linear), then an all-gather of the block input [rows, H] bf16 (the projections need the whole row), and at the end
       of the block a reduce-scatter of the partial sums [rows, H] back to the column split so the write is local.  Per
       layer: 2 x (AR + AG + RS) = 8 last-tile latencies.  (The replicated-residual form has the same count with tile
       padding on `down` and 4x the residual in L1: never faster, not enumerated.)
  H-7  L1 residency: the stage's dense weights that fit the chip's L1 share for them (H-20) stay resident across passes
       and are never re-read; only the rest streams inside the micro-batch window (and is re-read per micro-batch).
  H-8  router top-10 over 512 logits: 10 selection passes of 32 cycles per fp32 tile (27B section 9: argmax = one tile
       read) on each logit tile in parallel, a NoC gather of the candidates (92 cycles), 10 passes on the merged tile;
       row-tiles are independent: max(one row-tile's chain, work / 130).  Two router forms are evaluated and the faster
       kept: replicated (2.6 MB/layer on every chip, no collective) or N-split (512/TP_a columns, tile padded, plus an
       all-gather of the candidates: one more latency per layer).
  H-9  matmul 32 cycles per 32x32 tile-MAC (HiFi2, 27B section 3); no per-op fixed cost (27B section 9 counts none).
  H-10 the residual lives in L1 when it fits, else in DRAM (read + write per block on the DRAM lane); the other
       activations of the micro-batch (block input, normalized rows, block output, partial sums: 4 x [rows, H] bf16)
       are counted against the L1 before the resident weights (H-20).
  H-13 an all-reduce is the faster of RS+AG (2 latencies) and the all-gather of the partial sums + local reduction
       (1 latency, (n-1) x the row on the links): for the hyper-connection statistics the latter wins at small rows.
  H-14 a pipeline stage (PP > 1) is idle while the micro-batch is elsewhere: in that time it prefetches the head of its
       non-resident dense part into the free L1 (H-22), so only the rest streams inside its window.
  H-15 the stage hand-off hop count follows the stage tiling of the 8x4 torus (the minimax matching of a block's chips
       to the next block's): PP 2 = 3 hops with 4x4 blocks or 1 hop with 8x2 blocks (both layouts evaluated), PP 4 = 2
       (2x4 blocks), PP 8 = 1 (rows), PP 16 = 2 (2x1 blocks), PP 32 = 1 (snake).  The pass wrap (token ids from the last
       stage to the first) is one more hand-off.
  H-16 the indexer projection (4 heads x 128 + 128 key) is replicated on the chips of a kv-head group: each chip scores
       its own 1/CP of the index blocks for all 4 heads without a collective (revision 2 split it by columns, which would
       need a reduction before the scores).
  H-17 the stage boundaries are chosen by the faster of two balancings: equal DRAM bytes per pass (throughput) or equal
       chain time (latency); both are evaluated.
  H-18 a matmul's chain on one chip: its output tiles are handed to the 130 cores; when there are fewer output tiles
       than cores (b=1 with TP 8-32) the K dimension is split so that every core carries work/130 tile-MACs, and the
       up to ceil(130/out) fp32 partial tiles of an output tile are tree-reduced over the NoC (per level: 92 cycles +
       64 flits + one 32-cycle DEST accumulate).  chain = min(no split: K tiles x 32, split: work/130 x 32 + levels x
       188).  The 27B README section 9 left this out ("under 1 us per matmul", negligible against 32 users' work); at
       b=1 it is a visible share of the chain here.
  H-20 L1 accounting (worklog 10.H20): per core 1,536 KiB = fixed 140 KiB (firmware 40 + kernel binary/config 100,
       27B section 3) + stream buffers (design values from the 27B section 9 unit descriptions: matmul activation tiles
       double-buffered 2 x 16 tiles x 2 KiB + output/partial tiles = 80 KiB; gated delta rule state unit 128 x 32 fp32
       = 16 KiB, in double-buffered + out = 48 KiB; KV / index-block chunks double-buffered 2 x 32 KiB = 64 KiB) + the
       chip's activations (4 x [rows, H] bf16 + the residual [rows, 4H]/TP_a) + the routed experts' stream buffer (two
       expert slices) + what is left, which is split between RESIDENT DENSE WEIGHTS and the ROUTED PREFETCH BUFFER.
       Four splits are evaluated and the fastest kept (ties broken by fewer DRAM bytes per pass): no buffer (all to
       dense), one layer's prefetchable routed bytes to the buffer, two layers', all of it a buffer (H-23).  (Revision 3 before H-20 reserved a single 16 MiB working area and
       declared the 32-way dense split L1-resident; it is not: 160 MiB against 153 MiB after the stream buffers.)
  H-21 the routed read of a layer: the prefetchable part (hit: all; census: the predicted set; all: all; miss: none)
       is read ahead up to the prefetch buffer; the rest streams inside the layer's expert phase, overlapped with the
       expert matmuls (the FPU consumes slices as they arrive), after the issue latency when it depends on the
       router.  layer = pre-expert chain (hyper-connection reads, GDN/QSA block, shared/router, top-10, their
       collectives) + [issue latency] + max(expert-phase chain, in-window routed bytes / BW), and never shorter than
       the layer's whole routed bytes / BW (the buffer holds at most one layer ahead).
  H-22 the idle-time dense prefetch of a pipeline stage (H-14) uses the routed prefetch buffer (empty while the stage is
       idle) minus one layer's routed need (the first layer's routed prefetch and the prefetched dense overlap).
  H-23 (worklog 10.H25/H28) the L1 left after the buffers is split into RESIDENT dense and the routed prefetch buffer;
       all four splits stay enumerated (a wide buffer can win where the routed bytes are large), and among forms with
       the same token time the one with fewer DRAM bytes per pass is kept (so "the dense re-read every pass" is chosen
       only when it is strictly faster).  The last stage's logits chunk (32 rows x V/TP_a fp32) comes out of its L1
       before both the resident share and the idle prefetch (the asymmetry that had picked the re-read form).
  H-24 (worklog 10.H29) the routed experts' column split has two forms, both evaluated: 'slice' (expert e's 640
       columns cut into TP_x equal, tile-padded slices: every chip of the group reads 1/TP_x of every expert of the
       group) and 'tiles' (expert e's 20 column tiles placed round-robin on the group's TP_x chips from chip e mod
       TP_x: no padding, the per-chip load is the census' busiest chip in tiles, routing_stats.py busiest_tiles).  In
       the tiles form a chip holds whole tiles of the up/gate (N) and the matching K rows of the down projection; the
       partial sums of the down projection ride the block's reduce-scatter as in the slice form.
  H-25 (worklog 10.H38) ROUNDS: the 27B README form (section 8-12) is a pipeline whose chip groups are visited R times
       per token: the layers (+tail) are cut into PP x R frames, frame f on chip group f mod PP, so one token crosses
       PP x R stages and every chip holds R frames.  Per chip the bytes are the same as R = 1; what changes: R x the
       hand-offs, and the L1 streaming buffer is refilled R times per pass while the group is idle (the idle-time
       prefetch cap is per frame).  R in {1, 2, 4} is enumerated for PP > 1 (PP = 1 has no idle time).  With R x PP = 32
       chips x rounds and TP_a = 1 this is the 27B pure-PP form; with TP_a = 4 the 27B TP4 x PP56 form.
  H-26 (worklog 10.H44) the context length is an enumerated axis, not a fixed 192: ctx in {1 (KV ignored), 8192, 32768,
       262144 (max_position_embeddings)}; 192 is kept only as the served measurement's condition.  Per row and QSA layer
       the KV read is min(ctx, 2048) selected K/V rows + ctx/4 index-key blocks (the indexer scans every block), so it
       saturates in the selection at 8k and grows with the index scan beyond; the resident KV per user (ctx x 2 kv
       heads x 256 x 2 x bf16 + index) enters the DRAM capacity check and caps b at long contexts.  The GDN state is
       context-free.
  H-19 matmuls that read the same input run side by side (27B section 9: "the max, not the sum"): their output tiles
       are pooled before the cores are handed out (H-18 on the pooled shape).  Pools per layer: the GDN in-projections
       (qkv, z, a, b); the QSA projections (q/gate, kv, indexer); the shared up/gate with the router; the routed experts'
       up/gate across the experts; the routed experts' down with the shared down.  The chain of a layer is the sum of
       its pools in dependency order.
  H-11 see H-8 (router forms).
  H-12 the dispatch/combine all-to-all is charged whenever DP_a > 1 (the rows of one TP group need experts that live on
       chips of another group, whatever EP is; revision 2 charged it only when EP > 1).  Bytes: every row goes to the
       TP_x column-slice holders of each of its 10 experts; the slices on the row's own TP group are local, so the
       per-chip egress per direction = rows_stage x 10 x TP_x x remote x H x 2 B / S with remote = 1 - 1/DP_a when the
       EP groups nest inside the TP groups (TP_x <= TP_a) and 1 - TP_a/TP_x otherwise; the combine returns the same
       bytes.  Destinations of a row = the chips holding a slice of any of its 10 experts = (expected distinct EP groups
       among 10 uniform draws) x TP_x, capped at S; the TP_a chips of the row's group share the sending.  The link time
       of an all-to-all counts the mean hop count of the submesh (a flow loads one link per hop).  With DP_a = 1 the rows
       are all-gathered inside the group and the combine is the block's reduce-scatter.  The CP merge is an all-gather
       of the partial (o, lse) (one latency).
Axes (32 = DP x PP x S, S = chips per pipeline stage):
  DP   replicas: independent user sets, no traffic between replicas, the weights stored DP times.
  PP   stages: the layers (+ lm_head/tail) are cut into PP x R frames (R = rounds, H-25); chip group g holds the frames
       f with f mod PP = g, so one token crosses PP x R stages and every chip group is visited R times per pass (R = 1:
       the plain pipeline; R > 1: the 27B README form, where a group streams its next frame into L1 while idle).  The b
       users are split into n_mb micro-batches (1..min(PP, b), a free choice) that occupy the stages in turn.  A group's
       resident dense weights (H-7) stay in L1; its non-resident dense part, the routed experts and the state/KV of the
       micro-batch stream inside its windows.
  TP   dense tensor parallel inside the stage: every dense matrix split over TP_a chips (K or N).  The GDN state follows
       the value heads (48 / TP_a per chip, 1 head per chip at 48, padded beyond).  The QSA query heads (24) and the 2 kv
       heads follow TP_a; chips sharing a kv head replicate its KV cache unless CP splits it.
  EP   expert parallel: the 512 routed experts placed whole on EP groups (512/EP per group), each group's experts split
       over the TP_x = S/EP chips of the group by columns.  A layer's time is set by the busiest group (census), the
       other groups idle for the difference.
  DCP  decode context parallel on the QSA layers (the code keeps the field name CP): inside each kv-head replication group (TP_a / 2 chips, or 1 when TP_a <= 2)
       the sequence (the compressed index-key blocks and the K/V cache) is split over CP chips; each chip scans its
       1/CP of the index blocks, selects its local top-k (budget/CP), attends over its local selection and the partial
       (o, logsumexp) are all-gathered and merged.  The GDN state is O(1) in the sequence: CP does not touch it.
Per chip, per layer, the three lanes:
  DRAM lane   bytes the chip reads+writes at 512 GB/s: non-resident dense, state x users_chip x heads/48 (read+write),
              KV x rows/CP, the busiest group's experts x pad / TP_x, the residual if it does not fit L1.
  compute     unit costs on the chip's share (130 cores): 32 cycles per tile-MAC, 16 per elementwise bf16 tile, 32 per
              fp32 tile, the gated delta rule units (H-5), the top-10 (H-8); no fixed cost.
  comm        physical: link 25 GB/s, 2 links per direction, 4 directions; one hop 280 ns + store-and-forward 2 KiB /
              (96 B/cycle); relay through a chip: NoC 92 cycles + 32 flits.  Ring all-gather / reduce-scatter over n
              chips: link (n-1)/n x T / (paths x 25 GB/s) with paths = the links usable by the group (2 per ring
              direction pair present, up to 8 for a 2-D block), last-tile latency hops x (280 ns + SAF) + relays.
              all-reduce = RS + AG as one stream (2 latencies).  All-to-all over S chips: per-chip egress / paths +
              diameter hops.  Tile-streamed: the link time hides under half the compute it feeds; the latency and the
              rest are exposed.
  routed      four cases -- 'miss' (known after the router: issue latency, then streamed with the expert matmuls),
              'census' (the one-block-early speculative router hits h: the predicted set is prefetchable, the (1-h) misses
              are read after the router), 'hit' (all known ahead: prefetchable), 'all' (every expert of the group every
              step: 512/EP experts, no skew, prefetchable); prefetch bounded by the L1 buffer (H-20, H-21).
Per frame f (chip group g = f mod PP):
  chain_f   = sum over its layers of the H-21 layer time + hand-off
  window_f  = (non-resident dense - idle-time prefetch + this micro-batch's routed + state/KV + residual traffic) / BW
  occ_f     = max(chain_f, window_f)
  dram_g    = sum over the group's frames of (non-resident dense x n_mb + state/KV of all users + routed x n_mb +
              residual traffic x n_mb) / BW  per pass
  T_pass    = max( max_g dram_g, max_g (sum of the group's occ_f) x n_mb, sum_f occ_f ) ;  T_tok = T_pass ;
  agg = DP x b x R_eff / T_pass ;  tsu = R_eff / T_tok.
Capacity: experts / (PP x EP x TP_x) + the chip's dense share + users' state and KV <= 0.9 x 31.9 GiB per chip.
Routing numerator: the census (32 decode items x 256 positions) for rows <= 256; beyond, labelled "fitted".
"""
import json
import math
import sys

GB = 1e9
MiB = 2**20
H = 2560
L = 48
L_GDN = 36
L_QSA = 12
V = 248320
E = 512
TOPK = 10
E_INTER = 640
SHARED_INTER = 640
GDN_QK_HEADS = 16
GDN_V_HEADS = 48
GDN_HD = 128
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HD = 256
IDX_HEADS = 4
IDX_KV = 1
IDX_HD = 128
IDX_BUDGET = 2048
IDX_COMPRESS = 4
HC = 4
HC_LOW = 320
HC_DIM = HC * H
BF4 = 576 / 1024
BF8 = 1088 / 1024
BF16 = 2.0
FP32 = 4.0
TILE = 32
LAYER_TYPES = ["gdn", "gdn", "gdn", "qsa"] * 12


def M(a, b, dt):
    return a * b * dt


BW_DRAM = 512e9
BW_LINK = 25e9
LINKS = 2
HOP = 280e-9
CLK = 1.35e9
SAF = 96
NOC = 64
NOC_RELAY = 92
PKT = 2048
CORES = 130
MM = 32
ELT = 16
FP32_TILE = 32


def gdr_unit(R):
    return 256 + 800 * R  # H-5: cycles per (value head, user, 32 columns of S) for R serial rows


TOPK_PASS = TOPK * FP32_TILE  # H-8: 10 selection passes over one fp32 tile
ISSUE_LAT = (
    2 * NOC_RELAY + PKT / NOC
) / CLK  # H-1: router result -> this chip's DRAM read issued: a NoC round trip + one packet
SPEC_L00 = (
    1.2e-6 + 1.33e-6 + ISSUE_LAT
)  # V4-2: L00's exact router at the pass start: a whole-row hc read (every chip has the embedding row) + the router + issue
L1_FIXED = 140 * 1024
L1_MM = 80 * 1024
L1_GDR = 48 * 1024
L1_KVC = 64 * 1024  # H-20, per core
L1_CHIP = CORES * (1536 * 1024 - L1_FIXED)
L1_STREAM = CORES * (L1_MM + L1_GDR + L1_KVC)
DRAM_CHIP = 31.9 * 1024**3 * 0.90
FRAMES_SINK = None  # when a dict, evaluate_form writes its per-frame numbers into it (renderer)
HC_FORM = ["A"]
PIECES = {}
FIRST = [0.0]  # V4-5: the hyper-connection form under evaluation; the matmul pieces of the last layer_compute
TOKENS_PER_PASS = {0: 1.0, 1: 1.376}
# weights
hc_block = 2 * M(HC_LOW, HC_DIM, BF16) + M(4, HC_DIM, BF16) + HC_DIM * BF16
W_GDN = dict(
    qkv=M(2 * GDN_QK_HEADS * GDN_HD + GDN_V_HEADS * GDN_HD, H, BF8),
    z=M(GDN_V_HEADS * GDN_HD, H, BF8),
    out=M(H, GDN_V_HEADS * GDN_HD, BF8),
    ab=2 * M(GDN_V_HEADS, H, BF8),
    small=(10240 * 4 + 48 * 2 + 128) * BF16,
)
W_QSA = dict(
    qg=M(2 * QSA_HEADS * QSA_HD, H, BF8),
    kv=2 * M(QSA_KV_HEADS * QSA_HD, H, BF8),
    o=M(H, QSA_HEADS * QSA_HD, BF8),
    idx=M(IDX_HEADS * IDX_HD + IDX_KV * IDX_HD, H, BF8),
    small=(256 * 2 + 128 * 2) * BF16,
)
W_MOE = dict(shared=3 * M(SHARED_INTER, H, BF8) + M(1, H, BF16), router=M(E, H, BF16))
EXPERT = M(E_INTER * 2, H, BF4) + M(H, E_INTER, BF4)
W_OTHER = 2 * hc_block + 2 * H * BF16
PLE = M(HC_DIM, H, BF16) + M(H, H, BF16) + 10240 * 4 * BF16 + 3 * 10240 * BF16
FINAL = 2 * M(HC_LOW, HC_DIM, BF8) + HC_DIM * BF16
LM_HEAD = M(V, H, BF8)
MTP = sum(W_QSA.values()) + sum(W_MOE.values()) + 2 * M(H, H, BF8) + 2 * M(HC_LOW, HC_DIM, BF8) + 2 * hc_block
S_USER_LAYER = (
    GDN_V_HEADS * GDN_HD * GDN_HD * FP32 + (2 * GDN_QK_HEADS * GDN_HD + GDN_V_HEADS * GDN_HD) * 3 * BF16
)  # per GDN layer per user (read once, written once)


def kv_read_row(ctx, cp):
    sel = min(ctx, IDX_BUDGET)
    return (sel * QSA_KV_HEADS * QSA_HD * 2 * BF16 + (ctx / IDX_COMPRESS) * IDX_KV * IDX_HD * BF16) / cp


def kv_write_row():
    return QSA_KV_HEADS * QSA_HD * 2 * BF16 + IDX_KV * IDX_HD * BF16 / IDX_COMPRESS


def kv_resident(ctx):
    return ctx * QSA_KV_HEADS * QSA_HD * 2 * BF16 + ctx / IDX_COMPRESS * IDX_KV * IDX_HD * BF16


DENSE_TOTAL = (
    L_GDN * (sum(W_GDN.values()) + sum(W_MOE.values()) + W_OTHER)
    + L_QSA * (sum(W_QSA.values()) + sum(W_MOE.values()) + W_OTHER)
    + PLE
    + FINAL
    + LM_HEAD
    + MTP
)
EXPERTS_TOTAL = 49 * E * EXPERT

SUBMESH = {1: (1, 1), 2: (2, 1), 4: (1, 4), 8: (2, 4), 16: (4, 4), 32: (8, 4)}


def diameter(n):
    r, c = SUBMESH[n]
    d = lambda m, ring: (m // 2 if m == ring else m - 1)
    return d(r, 8) + d(c, 4)


def paths(n):
    r, c = SUBMESH[n]
    return (2 if r >= 3 else (1 if r == 2 else 0)) * 2 + (2 if c >= 3 else (1 if c == 2 else 0)) * 2 or 2


def saf(b):
    return b / SAF / CLK


def last_tile(hops):
    return hops * (HOP + saf(PKT)) + max(0, hops - 1) * (NOC_RELAY + PKT / NOC) / CLK


import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/design")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comm_model as cm

_MESH = {}
_MEET = {}


def mesh_of(n):
    r, c = SUBMESH[n]
    if (r, c) not in _MESH:
        _MESH[(r, c)] = cm.Mesh(r, c)
    return _MESH[(r, c)]


PKT_CAP = (
    64 * 1024
)  # payloads above this are extrapolated linearly from two simulated points (the packet pipeline is linear in bytes once full)


def _meet_sim(kind, n, T):
    m = mesh_of(n)
    f = {"AG": cm.all_gather, "RS": cm.reduce_scatter, "AR": cm.all_reduce}[kind]
    return f(m, T)[0]


def meet(kind, n, T):
    """V4-1: seconds of one meeting of T bytes per chip over the n chips of the submesh (packet model, fastest form);
    kind AG (all-gather of T), RS (reduce-scatter of T partials), AR (all-reduce of T).  Beyond PKT_CAP the time is
    extrapolated linearly between the simulated points PKT_CAP/2 and PKT_CAP (the per-byte slope of the full pipeline).
    """
    if n <= 1 or T <= 0:
        return 0.0
    T = math.ceil(T / 256) * 256
    key = (kind, SUBMESH[n], T)
    if key not in _MEET:
        if T <= PKT_CAP:
            _MEET[key] = _meet_sim(kind, n, T)
        else:
            a = meet(kind, n, PKT_CAP / 2)
            b = meet(kind, n, PKT_CAP)
            _MEET[key] = b + (b - a) * (T - PKT_CAP) / (PKT_CAP / 2)
    return _MEET[key]


def ag_exposure(n, T, t_mm):
    """V4-1: an all-gather of T per chip consumed chunk by chunk by a projection of t_mm seconds: the exposure beyond t_mm.
    Above PKT_CAP the arrivals scale with the meeting time (the chunk arrival pattern is the same, stretched)."""
    if n <= 1 or T <= 0:
        return 0.0
    T = math.ceil(T / 256) * 256
    t_mm = round(t_mm * 2e7) / 2e7
    key = ("AGX", SUBMESH[n], T, t_mm)
    if key not in _MEET:
        m = mesh_of(n)
        tree = 2 * (NOC_RELAY + 64 + FP32_TILE) / CLK
        if T <= PKT_CAP:
            arr = cm.ag_arrivals(m, T, m.chips[0])
        else:
            base = cm.ag_arrivals(m, PKT_CAP, m.chips[0])
            sc = meet("AG", n, T) / max(meet("AG", n, PKT_CAP), 1e-12)
            arr = [a * sc for a in base]
        _MEET[key] = max(0.0, cm.consume_overlap(arr, t_mm, tree) - (t_mm + tree))
    return _MEET[key]


WAVES = 10


def wave_exposure(kind, n, T, t_mm):
    """V4-1: a meeting of T partials produced in WAVES waves during a matmul of t_mm seconds: the exposure beyond t_mm.
    Small payloads: every wave simulated with its ready time (the fastest form's wave pipeline).  Above PKT_CAP: the
    link pipeline is bandwidth-bound, exposure = max(last wave's meeting, whole meeting - t_mm)."""
    if n <= 1 or T <= 0:
        return 0.0
    T = math.ceil(T / 256) * 256
    t_mm = round(t_mm * 2e7) / 2e7
    key = ("W" + kind, SUBMESH[n], T, t_mm)
    if key not in _MEET:
        if T > PKT_CAP:
            _MEET[key] = max(meet(kind, n, T / WAVES), meet(kind, n, T) - t_mm)
        else:
            m = mesh_of(n)
            f = {"RS": cm.reduce_scatter, "AR": cm.all_reduce}[kind]
            _, form, _ = f(m, T)
            fl = []
            for w in range(1, WAVES + 1):
                t0 = t_mm * w / WAVES
                Pw = T / WAVES
                if form.startswith("AG_tree"):
                    fl += [(t0, root, m.bfs_tree(root), Pw, (root, w)) for root in m.chips]
                else:
                    Pd = Pw if form.startswith("AG_direct") else Pw / m.n
                    k = 0
                    for s in m.chips:
                        for d in m.chips:
                            if s != d:
                                fl.append((t0, s, cm.path_tree(m, s, d, k), Pd, (s, d, w)))
                                k += 1
            done = cm.simulate_mc(fl)[0] + cm.ADD_LEVEL
            _MEET[key] = max(0.0, done - t_mm)
    return _MEET[key]


def hand_off(n_cols_shared, T):
    """V4-4: T bytes from every chip of a block to its counterpart in the next block, 2 hops, the column's 2 chips
    sharing the inter-group link (packet model on a 4x4 stack); for 1-row blocks 1 hop.  Linear beyond PKT_CAP."""
    T = math.ceil(T / 256) * 256
    key = ("H", n_cols_shared, T)
    if key not in _MEET:
        if T > PKT_CAP:
            a = hand_off(n_cols_shared, PKT_CAP / 2)
            b = hand_off(n_cols_shared, PKT_CAP)
            _MEET[key] = b + (b - a) * (T - PKT_CAP) / (PKT_CAP / 2)
        else:
            big = cm.Mesh(4, 4) if n_cols_shared == 2 else cm.Mesh(2, 4)
            if n_cols_shared == 2:
                fl = [(0.0, (0, c), cm.path_tree(big, (0, c), (2, c)), T, (0, c)) for c in range(4)] + [
                    (0.0, (1, c), cm.path_tree(big, (1, c), (3, c)), T, (1, c)) for c in range(4)
                ]
            else:
                fl = [(0.0, (0, c), cm.path_tree(big, (0, c), (1, c)), T, (0, c)) for c in range(4)]
            _MEET[key] = cm.simulate_mc(fl)[0]
    return _MEET[key]


def coll(n, T):
    """revision 3 form kept for the a2a / CP merge (link time, last-tile latency)"""
    if n <= 1:
        return 0.0, 0.0
    return (n - 1) / n * T * (1 + 56 / 4096) / (paths(n) * BW_LINK), last_tile(math.ceil(diameter(n)))


def mean_hops(n):
    r, c = SUBMESH[n]

    def m(k, ring):
        return (sum(min(i, k - i) for i in range(k)) / k) if k == ring else (sum(i for i in range(k)) / k)

    return m(r, 8) + m(c, 4)


def all_to_all(S, per_chip):
    if S <= 1:
        return 0.0, 0.0
    return per_chip * (1 + 56 / 4096) * mean_hops(S) / (paths(S) * BW_LINK), last_tile(diameter(S))


def tile_pad(width, parts):
    return math.ceil(width / parts / TILE) * TILE * parts / width


class Routing:
    def __init__(self, path, path2=None):
        self.S = json.load(open(path))
        self.S2 = json.load(open(path2)) if path2 else None
        d128, d256 = self.S["batches"]["128"]["distinct_mean"], self.S["batches"]["256"]["distinct_mean"]
        self.decay = (E - d256) / (E - d128)
        self.d256 = d256
        self.miss256 = 1 - self.S["batches"]["256"]["hit_pred1"]

    def key(self, b, R):
        for k in ((f"{b}x{R}",) if R > 1 else ()) + (str(b * R),):
            if k in self.S["batches"]:
                return k
        return None

    def distinct(self, b, R):
        k = self.key(b, R)
        if k:
            return self.S["batches"][k]["distinct_mean"]
        rows = b * R
        return (
            E - (E - self.d256) * self.decay ** math.log2(rows / 256)
            if rows > 256
            else E * (1 - (1 - TOPK / E) ** rows) * 0.85
        )

    def hit(self, b, R):
        k = self.key(b, R)
        return self.S["batches"][k]["hit_pred1"] if k else 1 - self.miss256

    def busiest(self, b, R, groups):
        """experts of the busiest of `groups` equal groups (census), else fitted from the mean with the census skew law"""
        k = self.key(b, R)
        if k and str(groups) in self.S["batches"][k]["busiest"]:
            return self.S["batches"][k]["busiest"][str(groups)]["actual"], "census"
        mean = self.distinct(b, R) / groups
        return min(mean * (1 + 0.6 / math.sqrt(max(mean, 1))), E / groups), "fitted"

    def stage2(self, b, R, ep, tpx, xform):
        """V4-2: the busiest chip's (tiles form) or group's (slice form, in experts) sets per layer: A, P2, P1mP2, miss,
        wasted (census stats_stage2.json), else fitted from A with the B=256 ratios"""
        k = self.key(b, R)
        ref = self.S2["256"] if self.S2 else None
        if self.S2 and k and k in self.S2:
            d = self.S2[k]["tiles"][f"{ep}x{tpx}"] if xform == "tiles" else self.S2[k]["groups"][str(ep)]
            return dict(d), "census"
        A = self.busiest_tiles(b, R, ep, tpx)[0] if xform == "tiles" else self.busiest(b, R, ep)[0]
        r = (ref["tiles"]["1x8"] if xform == "tiles" else ref["groups"]["4"]) if ref else None
        if r is None:
            return dict(A=A, P2=0.0, P1mP2=0.0, miss=A, wasted=0.0), "fitted"
        s = A / max(r["A"], 1e-9)
        return {kk: r[kk] * s for kk in ("A", "P2", "P1mP2", "miss", "wasted")}, "fitted"

    def busiest_tiles(self, b, R, ep, tpx):
        """H-24: tiles (of 20 per expert) on the busiest chip of the tile placement over ep groups x tpx chips (census), else
        fitted: the mean tiles per chip x the census skew of the slice form at the same group count, capped at the group's whole
        """
        k = self.key(b, R)
        p = f"{ep}x{tpx}"
        if k and p in self.S["batches"][k].get("busiest_tiles", {}):
            return self.S["batches"][k]["busiest_tiles"][p]["actual"], "census"
        mean = self.distinct(b, R) * 20 / (ep * tpx)
        return min(mean * (1 + 0.6 / math.sqrt(max(mean / 20, 1))), E / ep * 20), "fitted"


def mm_t(rows, n, k):
    return math.ceil(rows / TILE) * math.ceil(n / TILE) * math.ceil(k / TILE)


def mm_chain(out, kt, work):
    """H-18: chain cycles on one chip for `work` tile-MACs over `out` output tiles with at most `kt` K tiles each"""
    if out >= CORES:
        return work / CORES * MM
    cpo = min(math.ceil(CORES / max(out, 1)), kt)
    levels = math.ceil(math.log2(cpo)) if cpo > 1 else 0
    return min(kt * MM, work / CORES * MM + levels * (NOC_RELAY + 4096 / NOC + FP32_TILE))


def mm_c(rows, n, k):
    """chain cycles for one [rows, k] x [k, n] matmul"""
    out = math.ceil(rows / TILE) * math.ceil(n / TILE)
    kt = math.ceil(k / TILE)
    return mm_chain(out, kt, out * kt)


def mm_pool(*mms):
    """H-19: matmuls run side by side (same input): their output tiles pooled, the K depth the largest"""
    out = sum(math.ceil(r / TILE) * math.ceil(n / TILE) for r, n, k in mms)
    kt = max(math.ceil(k / TILE) for r, n, k in mms)
    work = sum(math.ceil(r / TILE) * math.ceil(n / TILE) * math.ceil(k / TILE) for r, n, k in mms)
    return mm_chain(out, kt, work)


def topk_serial(rows, ntile):
    """H-8: cycles on the chain for the top-10 of `ntile` logit tiles per row-tile (local passes in parallel, NoC gather, merge)"""
    rt = math.ceil(rows / TILE)
    one = TOPK_PASS + NOC_RELAY + TOPK_PASS
    return max(one, rt * (ntile * TOPK_PASS + NOC_RELAY + TOPK_PASS) / CORES)


def layer_compute(l, rows, users, R, TP_a, TP_x, x_experts, ctx, cp, router_rep, x_tpx=None):
    """x_experts: expert-equivalents of routed columns on this chip (slice form: the busiest group's experts, each 1/TP_x
    wide; tiles form: busiest chip's tiles / 20 with x_tpx = 1)"""
    if x_tpx is None:
        x_tpx = TP_x
    """seconds on one chip of the stage, split into the pre-expert chain and the expert phase (H-21): the chain of the
    matmuls (H-18), elementwise work / 130 cores, the serial units (H-5, H-8)"""
    rt = math.ceil(rows / TILE)
    wloc = HC_DIM / TP_a
    c = 0.0
    s = 0.0
    # hyper-connection read x2 (H-6'): rms sum + scale on the local columns, down K-split, up N-split, gate x local, mean
    hc_form = HC_FORM[0]
    if hc_form == "A":
        down = mm_pool((rows, HC_LOW, wloc), (rows, HC, wloc))
        up = mm_c(rows, wloc, HC_LOW)
    elif hc_form == "Bn":
        down = mm_pool((rows, HC_LOW / TP_a, HC_DIM), (rows, HC, HC_DIM))
        up = mm_c(rows, wloc, HC_LOW)
    elif hc_form == "B+D":
        down = mm_pool((rows, HC_LOW, HC_DIM), (rows, HC, HC_DIM))
        up = mm_c(rows, wloc, HC_LOW)
    else:
        down = mm_pool((rows, HC_LOW, wloc), (rows, HC, wloc))
        up = mm_c(
            rows, wloc, HC_LOW
        )  # B: the attention read as A, the MoE read's low-rank local (the fold rides the output)
    s += 2 * (down + up)
    PIECES.clear()
    PIECES.update(down=down / CLK, up=up / CLK)
    c += 2 * (rt * (wloc / TILE) * 3 * ELT + rt * (wloc / TILE) * 2 * ELT)
    # hyper-connection write x2: residual += injection x block output (local columns)
    c += 2 * rt * (wloc / TILE) * 2 * ELT
    if LAYER_TYPES[l] == "gdn":
        proj = mm_c(rows, (2 * GDN_QK_HEADS * GDN_HD + 2 * GDN_V_HEADS * GDN_HD) / TP_a + 2 * TILE, H)
        s += proj  # H-19 pool: qkv, z, a, b
        units = math.ceil(GDN_V_HEADS / TP_a) * users * (GDN_HD / TILE)
        s += math.ceil(units / CORES) * gdr_unit(R)
        out = mm_c(rows, H, GDN_V_HEADS * GDN_HD / TP_a)
    else:
        sel = min(ctx, IDX_BUDGET) / cp
        proj = mm_c(
            rows,
            2 * QSA_HEADS * QSA_HD / TP_a
            + 2 * QSA_KV_HEADS * QSA_HD / max(1, TP_a // 2)
            + IDX_HEADS * IDX_HD
            + IDX_KV * IDX_HD,
            H,
        )
        s += proj  # H-19 pool: q/gate, kv, indexer
        c += (
            2 * rows * math.ceil(QSA_HEADS / TP_a) * math.ceil(sel / TILE) * (QSA_HD / TILE) * MM / TILE
            + rows * IDX_HEADS * math.ceil(ctx / IDX_COMPRESS / cp / TILE) * (IDX_HD / TILE) * MM / TILE
        )
        out = mm_c(rows, H, QSA_HEADS * QSA_HD / TP_a)
    if hc_form in ("B", "Bn", "B+D"):
        out = mm_pool(
            (rows, H, (GDN_V_HEADS * GDN_HD if LAYER_TYPES[l] == "gdn" else QSA_HEADS * QSA_HD) / TP_a),
            (rows, HC * HC_LOW, (GDN_V_HEADS * GDN_HD if LAYER_TYPES[l] == "gdn" else QSA_HEADS * QSA_HD) / TP_a),
        )  # V4-5: the fold F_i h_i pooled with the output projection
    s += out
    router = mm_c(rows, 2 * SHARED_INTER / TP_a + (E if router_rep else E / TP_a), H)
    s += router  # H-19 pool: shared up/gate, router
    PIECES.update(proj=proj / CLK, out=out / CLK, router=router / CLK)
    s += topk_serial(rows, E / TILE if router_rep else math.ceil(E / TP_a / TILE)) + (0 if router_rep else TOPK_PASS)
    pre = c / CORES / CLK + s / CLK
    c = 0.0
    s = 0.0
    rows_per_expert = max(1.0, rows * TOPK / E)
    s += mm_c(rows_per_expert, x_experts * 2 * E_INTER / x_tpx, H)  # H-19 pool: the experts' up/gate
    xdown = mm_pool((rows_per_expert, x_experts * H, E_INTER / x_tpx), (rows, H, SHARED_INTER / TP_a))
    s += xdown  # H-19 pool: the experts' down with the shared down
    PIECES.update(xdown=xdown / CLK)
    c += 2 * rt * (H / TILE) / TP_a * ELT * 4
    return pre, c / CORES / CLK + s / CLK


def PLE_COMPUTE(rows, TP_a):
    """V4-3: the PLE's projections on one chip (key [4H,2560] N-split, value [H,2560] N-split, pooled)"""
    return mm_pool((rows, HC_DIM / TP_a, 2560), (rows, H / TP_a, 2560)) / CLK


def balanced_cuts(costs, parts):
    n = len(costs)
    if parts == 1:
        return [(0, n)]
    if parts >= n:
        return [(i, i + 1) for i in range(n)] + [(n, n)] * (parts - n)
    lo, hi = max(costs), sum(costs)

    def ok(limit):
        cnt, acc = 1, 0.0
        for c in costs:
            if acc + c > limit:
                cnt += 1
                acc = c
            else:
                acc += c
        return cnt <= parts

    for _ in range(60):
        mid = (lo + hi) / 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    lim = hi * (1 + 1e-9)
    cuts, start, acc = [], 0, 0.0
    for i, c in enumerate(costs):
        if acc + c > lim and i > start:
            cuts.append((start, i))
            start, acc = i, c
        else:
            acc += c
    cuts.append((start, n))
    while len(cuts) > parts:
        j = min(range(len(cuts) - 1), key=lambda j: sum(costs[cuts[j][0] : cuts[j + 1][1]]))
        cuts[j : j + 2] = [(cuts[j][0], cuts[j + 1][1])]
    while len(cuts) < parts:
        cuts.append((n, n))
    return cuts


HAND_HOPS = {1: 0, 2: 3, 4: 2, 8: 1, 16: 2, 32: 1}


def evaluate(cfg, routing, ctx=192, k=1, case="miss"):
    """both router forms (H-8/H-11), both stage balancings (H-17) and, for PP 2 x 16 chips, both block layouts (H-15):
    the fastest is returned"""
    best = None
    layouts = (
        [((4, 4), 3), ((8, 2), 1)]
        if (cfg["PP"] == 2 and 32 // (cfg["DP"] * cfg["PP"]) == 16)
        else [(SUBMESH[32 // (cfg["DP"] * cfg["PP"])], HAND_HOPS[cfg["PP"]])]
    )
    for sub, hh in layouts:
        S = 32 // (cfg["DP"] * cfg["PP"])
        keep = SUBMESH[S]
        SUBMESH[S] = sub
        for rep in (0, 1):
            for bal in ("bytes", "chain") if cfg["PP"] > 1 else ("bytes",):
                for buf in (0,) if case == "miss" else (0, 1, 2, 3):
                    for xf in (
                        ("slice", "tiles") if cfg["TP_x"] >= 8 else ("slice",)
                    ):  # H-24 (TP_x 1, 2, 4 divide 20: the forms coincide)
                        for R in (1, 2, 4) if cfg["PP"] > 1 else (1,):  # H-25 rounds
                            if cfg["PP"] * R > L + 1:
                                continue
                            for hf in (
                                ("A", "B", "Bn", "B+D") if cfg["TP_a"] > 1 else ("A",)
                            ):  # V4-5 hyper-connection forms
                                HC_FORM[0] = hf
                                r = evaluate_form(cfg, routing, ctx, k, case, rep, bal, hh, buf, xf, R)
                                if best is None or (r["T_tok_ms"], r["dram_ms"]) < (best["T_tok_ms"], best["dram_ms"]):
                                    best = r
                                if hf == "A" and r["dram_ms"] >= r["T_tok_ms"] * 0.999:
                                    break  # DRAM-bound already: the B forms only add DRAM bytes
        SUBMESH[S] = keep
    return best


def evaluate_form(
    cfg, routing, ctx, k, case, router_rep, balance="bytes", hand_hops=1, buf_mode=0, xform="slice", rounds=1
):
    DP, PP, TP_a, DP_a, EP, TP_x, CP, b = (cfg[x] for x in ("DP", "PP", "TP_a", "DP_a", "EP", "TP_x", "CP", "b"))
    S = TP_a * DP_a
    assert S == EP * TP_x and DP * PP * S == 32
    R = k + 1
    rows = b * R
    rows_chip = rows / DP_a
    users_chip = b / DP_a
    R_eff = TOKENS_PER_PASS[k]
    heads_v = math.ceil(GDN_V_HEADS / TP_a)
    heads_q = math.ceil(QSA_HEADS / TP_a)
    h_chip = 2 if TP_a == 1 else 1
    hit = {"miss": 0.0, "census": routing.hit(b, R), "hit": 1.0, "all": 1.0}[case]
    pad_x = tile_pad(E_INTER, TP_x)
    pad_s = tile_pad(SHARED_INTER, TP_a)
    pad_r = tile_pad(E, TP_a)
    n_mb = cfg.get("n_mb", 1)
    rows_mb = rows_chip / n_mb
    users_mb = users_chip / n_mb
    dist_mb = routing.distinct(max(1, b // n_mb), R)
    if xform == "tiles":
        t_mb, src = routing.busiest_tiles(max(1, b // n_mb), R, EP, TP_x)
        x_mb = t_mb / 20.0
        chip_x = t_mb / 20.0 * EXPERT  # the busiest chip's bytes = its tiles x (expert bytes / 20), no padding
        x_all = (E / EP) * 20 / TP_x / 20.0 * EXPERT
    else:
        x_mb, src = routing.busiest(max(1, b // n_mb), R, EP)
        chip_x = x_mb * EXPERT * pad_x / TP_x
        x_all = (E / EP) * EXPERT * pad_x / TP_x
    # V4-2: the routed sets of a layer on the busiest chip (tiles) or group (slice): A, P2, P1mP2, miss, wasted
    st2, src2 = routing.stage2(max(1, b // n_mb), R, EP, TP_x, xform)
    unit = EXPERT / 20.0 if xform == "tiles" else EXPERT * pad_x / TP_x
    if case == "all":
        x_bytes_mb = x_all
        P2b = x_all
        P1b = 0.0
        Db = 0.0
    elif case == "hit":
        x_bytes_mb = chip_x
        P2b = chip_x
        P1b = 0.0
        Db = 0.0
    elif case == "miss":
        x_bytes_mb = chip_x
        P2b = 0.0
        P1b = 0.0
        Db = chip_x
    else:
        sc = chip_x / max(st2["A"] * unit, 1e-9)  # the census sets scaled to this placement's actual
        P2b = st2["P2"] * unit * sc
        P1b = st2["P1mP2"] * unit * sc
        Db = st2["miss"] * unit * sc
        wasted = st2["wasted"] * unit * sc
        x_bytes_mb = P2b + P1b + Db  # = A u P1 u P2 (the wasted reads are inside P2 and P1 \ P2)
    x_exposed_mb = Db
    # V4-1 meetings on the TP_a submesh (packet model); the a2a / CP merge keep the revision-3 ring form
    T_x = rows_mb * H * BF16 / TP_a
    T_y = rows_mb * H * BF16
    T_down = rows_mb * HC * (HC_LOW + HC + 1) * FP32
    T_low = rows_mb * HC_LOW / TP_a * FP32
    T_f = rows_mb * HC * HC_LOW * BF16
    rt_l, rt_t = coll(TP_a, rows_mb * TP_a * TOPK * 2 * FP32) if (not router_rep and TP_a > 1) else (0.0, 0.0)
    remote = (1 - 1 / DP_a) if TP_x <= TP_a else (1 - TP_a / TP_x)
    dest = min(S, EP * (1 - (1 - 1 / EP) ** TOPK) * TP_x)
    a2a_l, a2a_t = all_to_all(S, rows_mb * dest * remote * H * BF16 / TP_a) if DP_a > 1 else (0.0, 0.0)
    cp_l, cp_t = coll(CP, rows_mb * heads_q * (QSA_HD + 1) * FP32) if CP > 1 else (0.0, 0.0)
    hc_form = HC_FORM[0]
    # L1 accounting (H-20): fixed + stream buffers + activations + residual + two expert slices + logits chunk (tail)
    residual = rows_mb * HC_DIM * BF16 * (1.0 if hc_form != "A" else 1 / TP_a)
    act = 4 * rows_mb * H * BF16
    slice_ = EXPERT * pad_x / TP_x if xform == "slice" else EXPERT / 20 * math.ceil(20 / TP_x)
    free_ = L1_CHIP - L1_STREAM - act - 2 * slice_
    res_dram = residual > free_
    if not res_dram:
        free_ -= residual
    free_ = max(0.0, free_)
    logits_chunk = TILE * math.ceil(V / TP_a / TILE) * TILE * FP32
    res_traffic = 4 * residual if res_dram else 0.0  # per layer per micro-batch: 2 blocks x (read + write)
    extra_form = {
        "A": 0.0,
        "B": HC * HC_LOW * 768 * BF16 - HC_LOW * HC_DIM * BF16 / TP_a,
        "Bn": HC * HC_LOW * 768 * BF16 - HC_LOW * HC_DIM * BF16 / TP_a,
        "B+D": HC * HC_LOW * 768 * BF16
        - HC_LOW * HC_DIM * BF16 / TP_a
        + (HC_LOW * HC_DIM * BF16 - HC_LOW * HC_DIM * BF16 / TP_a),
    }[
        hc_form
    ]  # V4-5 per layer per chip
    buf = [0.0, P2b + P1b, 2 * (P2b + P1b), free_][buf_mode]
    buf = min(buf, free_)  # routed prefetch buffer (holds the stage-2/1 reads)
    P = P2b + P1b
    D = Db
    dense_l = []
    state_l = []
    routed_l = []
    chain_l = []
    comp_l = []
    comm_l = []
    first_l = []

    def block_meetings(l, pieces, gdnl):
        """V4-1/V4-5: the exposure on the chain of one layer's meetings (attention block + MoE block), seconds"""
        n = TP_a
        if n <= 1:
            return 0.0, 0.0
        agx_attn = ag_exposure(n, T_x, pieces["proj"])
        agx_moe = ag_exposure(n, T_x, pieces["router"])
        if hc_form == "A":
            ar = wave_exposure("AR", n, T_down, pieces["down"])
            rs_a = wave_exposure("RS", n, T_y, pieces["out"])
            rs_m = wave_exposure("RS", n, T_y, pieces["xdown"])
            pre = ar + agx_attn + rs_a + ar + agx_moe
            post = rs_m
        elif hc_form == "B":
            ar = wave_exposure("AR", n, T_down, pieces["down"])
            ary_a = wave_exposure("AR", n, T_y + T_f, pieces["out"])
            ary_m = wave_exposure("AR", n, T_y, pieces["xdown"])
            pre = ar + agx_attn + ary_a + agx_moe
            post = ary_m
        elif hc_form == "Bn":
            agl = meet("AG", n, T_low)
            ary_a = wave_exposure("AR", n, T_y + T_f, pieces["out"])
            ary_m = wave_exposure("AR", n, T_y, pieces["xdown"])
            pre = agl + agx_attn + ary_a + agx_moe
            post = ary_m
        else:
            ary_a = wave_exposure("AR", n, T_y + T_f, pieces["out"])
            ary_m = wave_exposure("AR", n, T_y, pieces["xdown"])
            pre = agx_attn + ary_a + agx_moe
            post = ary_m
        return pre, post

    def layer_time(pre, post, comm_p, comm_q, link_p, lat_p, link_q, lat_q, l_exposed=0.0):
        """pre/post: compute; comm_p/comm_q: the meetings' exposure (V4-1); link/lat: the a2a / router / CP meetings of
        revision 3 (hidden to half of the compute); l_exposed: the structural routed exposure (V4-2: L00 / MTP)"""
        hid_p = min(link_p, pre / 2)
        hid_q = min(link_q, post / 2)
        cp_ = comm_p + lat_p + (link_p - hid_p)
        cq_ = comm_q + lat_q + (link_q - hid_q)
        ahead = min(P, buf)
        t = pre + cp_ + (ISSUE_LAT if D > 0 else 0.0) + max(post + cq_, (P - ahead + D) / BW_DRAM) + l_exposed
        FIRST[0] = max(
            t, (P1b + D) / BW_DRAM
        )  # V4-2: as the first layer of a frame its P2 read overlaps the previous frame
        return max(t, (P + D) / BW_DRAM), cp_ + cq_

    _cache = {}
    for l in range(L):
        gdnl = LAYER_TYPES[l] == "gdn"
        if gdnl:
            dense = sum(W_GDN.values()) * heads_v / GDN_V_HEADS
        else:
            dense = (
                (W_QSA["qg"] + W_QSA["o"]) * heads_q / QSA_HEADS
                + W_QSA["kv"] * h_chip / 2
                + W_QSA["idx"]
                + W_QSA["small"]
            )  # H-16: indexer replicated
        dense += (
            W_MOE["shared"] * pad_s / TP_a
            + (W_MOE["router"] if router_rep else W_MOE["router"] * pad_r / TP_a)
            + W_OTHER / TP_a
            + (PLE / TP_a if l == 2 else 0)
            + extra_form
        )
        rowb = (
            users_chip * S_USER_LAYER * heads_v / GDN_V_HEADS * 2
            if gdnl
            else rows_chip * (kv_read_row(ctx, CP) + kv_write_row()) * h_chip / 2
        )  # all users, per pass
        if gdnl not in _cache:
            pre, post = layer_compute(
                l, rows_mb, users_mb, R, TP_a, TP_x, x_mb, ctx, CP, router_rep, TP_x if xform == "slice" else 1
            )  # tiles: x_mb expert-equivalents of whole columns
            _cache[gdnl] = (pre, post, block_meetings(l, dict(PIECES), gdnl))
        pre, post, (cm_p, cm_q) = _cache[gdnl]
        if l == 2:
            cm_p += 2 * meet("AR", TP_a, rows_mb * HC * FP32)  # V4-3: the PLE's two statistics all-reduces
        l_exp = (
            max(0.0, (P2b + P1b) / BW_DRAM - (pre + cm_p - SPEC_L00)) if (l == 0 and case != "miss") else 0.0
        )  # V4-2: L00's experts are exact but known only at the pass start (+ a whole-row hc read + router)
        t, comm = layer_time(
            pre,
            post,
            cm_p,
            cm_q,
            rt_l + a2a_l + (cp_l if not gdnl else 0.0),
            rt_t + a2a_t + (cp_t if not gdnl else 0.0),
            a2a_l,
            a2a_t,
            l_exp,
        )
        chain_l.append(t)
        first_l.append(FIRST[0])
        dense_l.append(dense)
        state_l.append(rowb)
        routed_l.append(x_bytes_mb * n_mb)
        comp_l.append(pre + post + (PLE_COMPUTE(rows_mb, TP_a) if l == 2 else 0.0))
        comm_l.append(comm)
    # V4-3: the tail = final mixer (+ AR_down + AG_x), lm_head, local argmax + candidates AG, [MTP: mixer, fc, the MTP layer, draft lm_head + local argmax + candidates AG]
    tail_dense = (FINAL + LM_HEAD + (MTP if k > 0 else 0)) / TP_a + (extra_form if k > 0 else 0.0)
    mixer = (
        (mm_pool((rows_mb, HC_LOW, HC_DIM / TP_a), (rows_mb, HC, HC_DIM / TP_a)) + mm_c(rows_mb, HC_DIM / TP_a, HC_LOW))
        / CLK
        + meet("AR", TP_a, T_down)
        + meet("AG", TP_a, T_x)
    )
    argmax = (
        math.ceil(math.ceil(V / TP_a / TILE) / CORES) * FP32_TILE + NOC_RELAY + math.ceil(CORES / TILE) * FP32_TILE
    ) / CLK + meet("AG", TP_a, rows_mb * TP_a * 2 * FP32)
    lm = mm_c(rows_mb, V / TP_a, H) / CLK + argmax
    tail_comm = meet("AR", TP_a, T_down) + meet("AG", TP_a, T_x) + meet("AG", TP_a, rows_mb * TP_a * 2 * FP32)
    if k > 0:
        pre, post = layer_compute(
            3, rows_mb, users_mb, R, TP_a, TP_x, x_mb, ctx, CP, router_rep, TP_x if xform == "slice" else 1
        )
        cm_p, cm_q = block_meetings(3, dict(PIECES), False)
        spec_lat = (
            PIECES["down"]
            + PIECES["up"]
            + (meet("AR", TP_a, T_down) + meet("AG", TP_a, T_x) if TP_a > 1 else 0.0)
            + PIECES["router"]
            + topk_serial(rows_mb, E / TILE) / CLK
            + ISSUE_LAT
        )
        l_exp = (
            max(0.0, (P2b + P1b) / BW_DRAM - (pre + cm_p - spec_lat)) if case != "miss" else 0.0
        )  # V4-2: the MTP layer's experts are predictable only from its own attention block (after the speculative read's own latency)
        t, comm = layer_time(pre, post, cm_p, cm_q, rt_l + a2a_l + cp_l, rt_t + a2a_t + cp_t, a2a_l, a2a_t, l_exp)
        fc = mm_c(rows_mb, 2 * H / TP_a, H) / CLK
        tail_chain = mixer + lm + mixer + fc + t + lm
        tail_comm += (
            meet("AR", TP_a, T_down) + meet("AG", TP_a, T_x) + comm + meet("AG", TP_a, rows_mb * TP_a * 2 * FP32)
        )
    else:
        tail_chain = mixer + lm
    comp_l.append(tail_chain - tail_comm)
    comm_l.append(tail_comm)
    costs = chain_l + [tail_chain]
    dbytes = dense_l + [tail_dense]
    sbytes = state_l + [0.0]
    rbytes = routed_l + [(x_bytes_mb * n_mb if k > 0 else 0.0)]
    NF = PP * rounds  # H-25: frames; frame f on chip group f % PP
    cuts = balanced_cuts(
        [d + s + r for d, s, r in zip(dbytes, sbytes, rbytes)] if balance == "bytes" else costs, NF
    )  # H-17
    hand_bytes = (
        rows_mb * HC_DIM / TP_a * BF16 + rows_mb * H * BF16 + rows_mb * HC * FP32
    )  # V4-4: the residual 1/TP_a + the block input x + inject
    hand = (
        (
            hand_off(2 if SUBMESH[S][0] >= 2 else 1, hand_bytes) * max(1, hand_hops / 2)
            + (meet("AG", TP_a, rows_mb * HC_DIM / TP_a * BF16) if hc_form != "A" else 0.0)
        )
        if NF > 1
        else 0.0
    )
    # the chip group's resident share is decided on its whole set of frames (same bytes as one round)
    grp = [[f for f in range(NF) if f % PP == g] for g in range(PP)]
    gw = [sum(sum(dbytes[cuts[f][0] : cuts[f][1]]) for f in fs) for fs in grp]
    glog = [any(cuts[f][1] > L for f in fs) for fs in grp]
    st_l = []
    resident = []
    caps = []
    wmax = max(gw)
    for f, (a, z) in enumerate(cuts):
        g = f % PP
        free_s = max(0.0, free_ - (logits_chunk if glog[g] else 0.0))
        buf_s = min(buf, free_s)  # H-23: the last stage's logits chunk first
        w = sum(dbytes[a:z])
        res = min(w, max(0.0, (free_s - buf_s)) * w / max(gw[g], 1.0))
        nonres = w - res  # the group's resident share, spread over its frames
        st = sum(sbytes[a:z])
        rt = sum(rbytes[a:z])
        nl = sum(1 for i in range(a, min(z, L)))
        fchain = (
            sum(costs[a:z]) + hand - ((costs[a] - first_l[a]) if a < L else 0.0)
        )  # V4-2: the frame's first layer's floor without its P2 read
        st_l.append((nonres, st, rt, nl * res_traffic, fchain))
        resident.append(res)
        caps.append(
            max(0.0, free_s - res - 3 * (P + D) * n_mb)
        )  # V4-6: the whole L1 left after the resident share and a 3-layer routed reserve
    dram_f = [(nonres * n_mb + st + rt + resd * n_mb) / BW_DRAM for nonres, st, rt, resd, chain in st_l]
    dram = [sum(dram_f[f] for f in fs) for fs in grp]  # per chip group per pass
    resident = [sum(resident[f] for f in fs) for fs in grp]

    def occ_of(pre):
        return [
            max(chain, (max(0.0, nonres - p) + rt / n_mb + st / n_mb + resd) / BW_DRAM)
            for (nonres, st, rt, resd, chain), p in zip(st_l, pre)
        ]

    def pass_of(occ):
        return max(max(dram), max(sum(occ[f] for f in fs) * n_mb for fs in grp), sum(occ))

    # H-14: while a chip group is idle (T_pass - its occupancy x n_mb, spread over its R x n_mb gaps) it prefetches up to
    # the buffer cap of each frame's non-resident part; the prefetch shortens the window, which shortens the idle time:
    # iterate downward from the cap to the fixed point (monotone, so it converges)
    pre = [
        min(cap, nonres) if PP > 1 else 0.0 for (nonres, st, rt, resd, chain), cap in zip(st_l, caps)
    ]  # H-22: the buffer minus one layer's routed
    for _ in range(16):
        occ = occ_of(pre)
        T_pass = pass_of(occ)
        idle = [max(0.0, (T_pass - sum(occ[f] for f in fs) * n_mb) / (n_mb * len(fs))) for fs in grp]
        new = [min(p, idle[f % PP] * BW_DRAM) for f, p in enumerate(pre)]
        if all(abs(a - b) < 1.0 for a, b in zip(new, pre)):
            break
        pre = new
    occ = occ_of(pre)
    T_pass = pass_of(occ)
    if FRAMES_SINK is not None:
        FRAMES_SINK.clear()
        FRAMES_SINK.update(
            dict(
                cuts=cuts,
                dense=[sum(dbytes[a:z]) for a, z in cuts],
                state=[sum(sbytes[a:z]) for a, z in cuts],
                routed=[sum(rbytes[a:z]) for a, z in cuts],
                nonres=[x[0] for x in st_l],
                chain=[x[4] for x in st_l],
                occ=list(occ),
                pre=list(pre),
                dram=list(dram),
                resident=list(resident),
                free=free_,
                buf=buf,
                grp=grp,
                hand=hand,
                comp=list(comp_l),
                comm=list(comm_l),
                chain_l=list(chain_l),
                T=T_pass,
            )
        )
    # capacity per chip: its share of the experts and its stage's dense, plus the resident state/KV of its users (B6)
    state_chip_user = L_GDN * S_USER_LAYER * heads_v / GDN_V_HEADS / PP
    kv_chip_user = L_QSA * kv_resident(ctx) * h_chip / 2 / CP / PP
    cap = (
        (EXPERTS_TOTAL if k > 0 else EXPERTS_TOTAL * 48 / 49) / (PP * EP * TP_x)
        + wmax
        + users_chip * state_chip_user
        + users_chip * kv_chip_user
    )
    feasible = cap <= DRAM_CHIP
    exp_one = (x_exposed_mb / BW_DRAM + ISSUE_LAT) if (case in ("miss", "census") and x_exposed_mb > 0) else 0.0
    src = src2 if case == "census" else src
    x_exposed_tot = (
        (L + (1 if k > 0 else 0)) * exp_one * n_mb
    )  # reported: every layer incl. the MTP layer, with the issue latency
    return dict(
        cfg=cfg,
        feasible=feasible,
        cap_GB=cap / GB,
        users=DP * b,
        rows=rows,
        agg=DP * b * R_eff / T_pass,
        tsu=R_eff / T_pass,
        slot_ms=T_pass * 1e3,
        T_tok_ms=T_pass * 1e3,
        dram_ms=max(dram) * 1e3,
        chain_ms=sum(occ) * 1e3,
        occ_max_ms=max(sum(occ[f] for f in fs) for fs in grp) * n_mb * 1e3,
        comp_ms=sum(comp_l) * 1e3,
        comm_ms=sum(comm_l) * 1e3,
        routed_exposed_ms=x_exposed_tot * 1e3,
        busiest=x_mb,
        dist=dist_mb,
        src=src,
        bytes_chip_GB=max(
            sum(
                sum(dbytes[cuts[f][0] : cuts[f][1]])
                + sum(sbytes[cuts[f][0] : cuts[f][1]])
                + sum(rbytes[cuts[f][0] : cuts[f][1]])
                for f in fs
            )
            for fs in grp
        )
        / GB,
        n_mb=n_mb,
        router="replicated" if router_rep else "split",
        resident_MB=max(resident) / 1e6,
        dense_chip_MB=wmax / 1e6,
        residual_in="DRAM" if res_dram else "L1",
        balance=balance,
        hc_form=hc_form,
        layout=f"{SUBMESH[S][0]}x{SUBMESH[S][1]}",
        prefetch_MB=max(sum(pre[f] for f in fs) for fs in grp) / 1e6,
        buf_MB=buf / 1e6,
        free_MB=free_ / 1e6,
        xform=xform,
        rounds=rounds,
    )


def enumerate_configs(bs=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)):
    out = []
    for DP in (1, 2, 4, 8, 16, 32):
        for PP in (1, 2, 4, 8, 16, 32):
            if 32 % (DP * PP):
                continue
            S = 32 // (DP * PP)
            for TP_a in (1, 2, 4, 8, 16, 32):
                if S % TP_a:
                    continue
                DP_a = S // TP_a
                for EP in (1, 2, 4, 8, 16, 32):
                    if S % EP:
                        continue
                    TP_x = S // EP
                    rep = max(1, TP_a // QSA_KV_HEADS)
                    for CP in (1, 2, 4, 8, 16):
                        if rep % CP:
                            continue
                        for b in bs:
                            if b % DP_a:
                                continue
                            for n_mb in (1, 2, 4, 8, 16, 32):
                                if n_mb > PP or n_mb > b or b % n_mb or (b // n_mb) % DP_a:
                                    continue
                                out.append(
                                    dict(DP=DP, PP=PP, TP_a=TP_a, DP_a=DP_a, EP=EP, TP_x=TP_x, CP=CP, b=b, n_mb=n_mb)
                                )
    return out


def label(c):
    """chips = DP x PP x S; inside the S chips of a stage: dense TP_a x DP_a, experts EP groups x TP_x, and CP splits the
    sequence inside each kv-head group (the TP_a/2 chips that replicate one of the 2 kv heads) -- EP, TP_x and CP add no chips
    """
    S = c["TP_a"] * c["DP_a"]
    return (
        f"DP{c['DP']} x PP{c['PP']}"
        + (f"(R{c['rounds']})" if c.get("rounds", 1) > 1 else "")
        + f" x {S}chip[TP{c['TP_a']}"
        + (f" x DPa{c['DP_a']}" if c["DP_a"] > 1 else "")
        + f" | EP{c['EP']} x TPx{c['TP_x']}"
        + (f" | DCP{c['CP']}/kv-group" if c["CP"] > 1 else "")
        + f"] b={c['b']}"
        + (f" mb={c['n_mb']}" if c.get("n_mb", 1) > 1 else "")
    )


def pareto(rs):
    pts = sorted([r for r in rs if r["feasible"]], key=lambda r: (-r["agg"], -r["tsu"]))
    out = []
    best = -1
    for r in pts:
        if r["tsu"] > best:
            out.append(r)
            best = r["tsu"]
    return out


CASES_K = [("miss", 0), ("miss", 1), ("census", 0), ("census", 1), ("hit", 0), ("hit", 1), ("all", 0), ("all", 1)]


def main(
    stats="/data/yoshifuji/qwen38/routing/stats.json",
    ctx=192,
    out="/data/yoshifuji/qwen38/routing/free5d_v4_results.json",
    cases="all",
    stats2="/data/yoshifuji/qwen38/routing/stats_stage2.json",
):
    """ctx: one context length; cases: 'all' (4 cases x k 0/1) or 'k1' (the four cases at k=1 only: the long-context sweep)"""
    ctx = int(ctx)
    routing = Routing(stats, stats2)
    cfgs = enumerate_configs()
    res = {}
    sel = CASES_K if cases == "all" else [(c, k) for c, k in CASES_K if k == 1]
    print(f"ctx={ctx}; configs {len(cfgs)}")
    for case, k in sel:
        if True:
            rs = [evaluate(c, routing, ctx, k, case) for c in cfgs]
            res[(case, k)] = rs
            f = [r for r in rs if r["feasible"]]
            ba = max(f, key=lambda r: r["agg"])
            bt = max(f, key=lambda r: r["tsu"])
            print(f"\n=== case {case} k={k} ===")
            for tag, r in (("agg max", ba), ("tsu max", bt)):
                print(
                    f"  {tag} agg {r['agg']:8.0f} tsu {r['tsu']:6.1f} users {r['users']:5d} {label(dict(r['cfg'],rounds=r['rounds']))} router {r['router']} hc {r['hc_form']} T {r['slot_ms']:.3f} ms (dram/pass {r['dram_ms']:.3f} / chain {r['chain_ms']:.3f} / stage x n_mb {r['occ_max_ms']:.3f}; comp {r['comp_ms']:.3f} comm {r['comm_ms']:.3f} routed_exp {r['routed_exposed_ms']:.3f}; dense/chip {r['dense_chip_MB']:.0f} MB resident {r['resident_MB']:.0f} MB residual {r['residual_in']}) busiest {r['busiest']:.1f} [{r['src']}]"
                )
            for r in pareto(rs):
                print(
                    f"    {r['agg']:8.0f} | {r['tsu']:6.1f} | {r['users']:5d} | {label(dict(r['cfg'],rounds=r['rounds'])):60s} | {r['router'][:3]} | T {r['slot_ms']:7.3f} dram {r['dram_ms']:7.3f} chain {r['chain_ms']:7.3f} occ {r['occ_max_ms']:7.3f} comp {r['comp_ms']:6.3f} comm {r['comm_ms']:6.3f} rx {r['routed_exposed_ms']:6.3f} res {r['resident_MB']:4.0f}/{r['dense_chip_MB']:4.0f} | {r['src']}"
                )
    json.dump(
        {
            f"{c}|k={k}": [dict(r, cfg=label(dict(r["cfg"], rounds=r["rounds"])), axes=r["cfg"]) for r in v]
            for (c, k), v in res.items()
        },
        open(out, "w"),
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
