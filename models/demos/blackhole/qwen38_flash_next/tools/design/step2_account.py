#!/usr/bin/env python
r"""Step 2 of the detailed design (worklog 12.2): the L1 use and the DRAM traffic of every unit ON ONE CHIP of the chosen
form `DP1 x PP4(R) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1] b=1 k=1` (rows 2), ctx 1.
Accounting rules (the 27B README section 7 rules applied to this model):
  - a unit's core needs its input row and its output row at once (both counted); fp32 partial sums are "work";
  - inputs / outputs are sums (a row read by two units is counted twice: the consumer owns a copy);
  - the TP8 split: GDN value heads 6 per chip (qk heads 2 per chip; the chip holds the q,k of the key heads its value heads
    use: value head v uses key head v//3, so 6 value heads = 2 key heads), QSA q heads 3 per chip and the kv heads
    replicated on the 4 chips of a kv group (DCP1), hc down K-split / up N-split / inject K-split on the chip's 1/8 of
    the residual columns, shared expert columns 1/8, router replicated or N-split (both), routed experts: the chip's tiles
    of each of the 10 experts (tiles form: busiest chip = census), lm_head vocab 1/8, PLE key N-split / value N-split;
  - rows the chip holds: the residual [2, 4H]/8 (its columns), block input [2, H] whole (after the all-gather), the head
    rows of its heads, partial sums [2, H] fp32 of its K-split projections, logits [2, V/8];
  - per-user state on this chip: GDN S of its 6 value heads x 36 layers (fp32 [128,128] = 64 KiB per head per layer),
    conv taps of its qkv columns, PLE taps of its residual columns; KV of its kv group (ctx 1: 2 rows);
  - DRAM traffic per pass on this chip: dense weights of its frames (if not resident), S read + S write of its heads,
    routed expert tiles (busiest chip), the n-gram rows (PLE) and the embedding row, the lm_head share.
Inputs: step0_units.json (the HF units), routing/stats.json (the census busiest chip in tiles), physical values.
Outputs: markdown (stdout) and step2_account.json (per unit per chip bytes for steps 3-7)."""
import json
import math
import sys

U0 = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step0_units.json"))
STATS = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/stats_H17.json"))
ROWS = U0["rows"]
TP = 8
TPX = 8
KVG = 4
H = 2560
HC = 4
HC_LOW = 320
HCD = HC * H
GDN_QK = 16
GDN_V = 48
GDN_HD = 128
QSA_Q = 24
QSA_KV = 2
QSA_HD = 256
E = 512
E_INT = 640
SH_INT = 640
V = 248320
IDX_Q = 4
IDX_KV = 1
IDX_HD = 128
PLE_DIM = 2560
BF8 = 1088 / 1024
BF4 = 576 / 1024
BF16 = 2.0
FP32 = 4.0
KiB = 1024
MiB = 1024**2
TILE = 32


def row(w, dt=BF16):
    return ROWS * w * dt


def tiles_n(n):
    return math.ceil(n / TILE)


# census: busiest chip in tiles for EP1 x TPx8 at rows 2 (key "1x2")
T_BUSIEST = STATS["batches"]["1x2"]["busiest_tiles"]["1x8"][
    "actual"
]  # tiles of 20 per expert on the busiest chip per layer
X_DISTINCT = STATS["batches"]["1x2"]["distinct_mean"]
EXPERT_TILE_COLS = 20


def share(u):
    """the chip's share of a unit: (weight bytes on this chip, input bytes, output bytes, work bytes, S bytes, L1-resident state bytes, dram-only reads) and the split name"""
    n = u["unit"]
    w = u["w"]
    fmt = u["w_fmt"]
    r = dict(
        unit=n,
        split="",
        w=0.0,
        in_=0.0,
        out=0.0,
        work=0.0,
        S=0.0,
        state_l1=0.0,
        dram_read=0.0,
        kind=u["kind"],
        layer=n.split(".")[0],
    )
    if ".gr.norm" in n:
        r.update(
            split="columns 1/8 of 4H (stats all-reduce)",
            w=w / TP,
            in_=row(HCD) / TP,
            out=row(HCD) / TP,
            work=row(HC, FP32),
        )
    elif ".gr.down" in n:
        r.update(
            split="K-split 1/8 (partial [2,320] fp32, all-reduce)",
            w=w / TP,
            in_=row(HCD) / TP,
            out=row(HC_LOW, FP32),
            work=row(HC_LOW, FP32),
        )
    elif ".gr.up" in n:
        r.update(
            split="N-split 1/8 of 4H (local)",
            w=w / TP,
            in_=row(HC_LOW, FP32),
            out=row(HCD) / TP,
            work=row(HCD, FP32) / TP,
        )
    elif ".gr.gate x normalized" in n:
        r.update(
            split="local columns 1/8 -> partial block input [2,H]/8 cols; all-gather -> [2,H] whole",
            w=0,
            in_=2 * row(HCD) / TP,
            out=row(H),
            work=0,
        )
    elif ".gr.inject" in n:
        r.update(
            split="K-split 1/8 (partial [2,4] fp32, rides the stats all-reduce)",
            w=w / TP,
            in_=row(HCD) / TP,
            out=row(HC, FP32),
            work=0,
        )
    elif ".gr.write" in n:
        r.update(
            split="local columns 1/8", w=0, in_=row(HCD) / TP + row(H) / TP + row(HC, FP32), out=row(HCD) / TP, work=0
        )
    elif ".in_proj_qkv" in n:
        r.update(
            split="N-split by head: 2 key heads (q,k) + 6 value heads = 2*2*128+6*128 = 1280 cols",
            w=w * (2 * 2 * GDN_HD + 6 * GDN_HD) / (2 * GDN_QK * GDN_HD + GDN_V * GDN_HD),
            in_=row(H),
            out=row(2 * 2 * GDN_HD + 6 * GDN_HD),
            work=row(2 * 2 * GDN_HD + 6 * GDN_HD, FP32),
        )
    elif ".conv1d" in n:
        cols = 2 * 2 * GDN_HD + 6 * GDN_HD
        r.update(
            split="its 1280 columns",
            w=w * cols / (2 * GDN_QK * GDN_HD + GDN_V * GDN_HD),
            in_=row(cols),
            out=row(cols),
            work=0,
            state_l1=cols * 3 * BF16,
        )
    elif ".in_proj_z" in n:
        r.update(
            split="N-split 6 value heads = 768 cols",
            w=w / TP,
            in_=row(H),
            out=row(6 * GDN_HD),
            work=row(6 * GDN_HD, FP32),
        )
    elif ".in_proj_a" in n or ".in_proj_b" in n:
        r.update(split="N-split 6 heads (6 cols, tile-padded to 32)", w=w / TP, in_=row(H), out=row(6, FP32), work=0)
    elif ".l2norm" in n:
        r.update(
            split="its 2 key heads -> 6 value heads", w=0, in_=row(2 * 2 * GDN_HD), out=row(2 * 6 * GDN_HD), work=0
        )
    elif ".gated delta rule" in n:
        r.update(
            split="6 value heads, S fp32 [128,128] each (DRAM round trip or resident)",
            w=0,
            in_=row(2 * 6 * GDN_HD) + row(6 * GDN_HD) + 2 * row(6, FP32),
            out=row(6 * GDN_HD, FP32),
            work=6 * 4 * GDN_HD * FP32 * 4,
            S=6 * GDN_HD * GDN_HD * FP32,
        )
    elif ".norm (gated)" in n:
        r.update(split="6 value heads", w=w, in_=row(6 * GDN_HD, FP32) + row(6 * GDN_HD), out=row(6 * GDN_HD), work=0)
    elif ".out_proj" in n:
        r.update(
            split="K-split 6 value heads (768 rows of K) -> partial [2,H] fp32, reduce-scatter",
            w=w / TP,
            in_=row(6 * GDN_HD),
            out=row(H, FP32),
            work=row(H, FP32),
        )
    elif ".indexer qk" in n:
        r.update(
            split="replicated (4 index heads scored locally, no collective)",
            w=w,
            in_=row(H),
            out=row(IDX_Q * IDX_HD + IDX_KV * IDX_HD),
            work=row(IDX_Q * IDX_HD + IDX_KV * IDX_HD, FP32),
        )
    elif ".index scoring" in n:
        r.update(split="local (ctx 1)", w=0, in_=row(IDX_Q * IDX_HD), out=0, work=0)
    elif ".q+gate proj" in n:
        r.update(
            split="N-split 3 q heads x (q 256 + gate 256) = 1536 cols",
            w=w * 3 / QSA_Q,
            in_=row(H),
            out=row(3 * 2 * QSA_HD),
            work=row(3 * 2 * QSA_HD, FP32),
        )
    elif ".k proj" in n:
        r.update(
            split="replicated on the kv group (1 kv head per group of 4; this chip holds its group's 1 head: 2 x 256 cols)",
            w=w / QSA_KV,
            in_=row(H),
            out=2 * row(QSA_HD),
            work=row(2 * QSA_HD, FP32),
        )
    elif ".partial RoPE" in n:
        r.update(
            split="3 q heads + 1 kv head",
            w=0,
            in_=row(3 * QSA_HD) + row(QSA_HD),
            out=row(3 * QSA_HD) + row(QSA_HD),
            work=0,
            state_l1=0,
        )
    elif ".SDPA" in n:
        r.update(
            split="3 q heads over the group's kv head (ctx 1: 2 rows)",
            w=0,
            in_=row(3 * QSA_HD) + 2 * row(QSA_HD) + row(3 * QSA_HD),
            out=row(3 * QSA_HD),
            work=row(3, FP32) * ROWS,
        )
    elif ".o_proj" in n:
        r.update(
            split="K-split 3 q heads (768 rows of K) -> partial [2,H] fp32, reduce-scatter",
            w=w / TP,
            in_=row(3 * QSA_HD),
            out=row(H, FP32),
            work=row(H, FP32),
        )
    elif ".router" in n and "speculative" not in n:
        r.update(
            split="replicated (512 logits local, top-10 local) [N-split form: 64 cols + candidates all-gather]",
            w=w,
            in_=row(H),
            out=row(E, FP32),
            work=row(E, FP32) + ROWS * 2 * 10 * FP32 * TP,
        )
    elif "speculative router" in n:
        r.update(split="same as the router (weight shared)", w=0, in_=row(H), out=row(E, FP32), work=row(E, FP32))
    elif ".shared gate_proj" in n:
        r.update(
            split="N-split 1/8 (80+80 cols, tile-padded to 96+96)",
            w=w / TP,
            in_=row(H),
            out=row(SH_INT / TP),
            work=row(2 * SH_INT / TP, FP32),
        )
    elif ".shared down_proj" in n:
        r.update(
            split="K-split 1/8 -> partial [2,H] fp32 (pooled with the experts' down, reduce-scatter)",
            w=w / TP,
            in_=row(SH_INT / TP),
            out=row(H, FP32),
            work=0,
        )
    elif "routed expert e: gate_up" in n:
        r.update(
            split=f"tiles form: this chip's tiles of each selected expert (busiest chip {T_BUSIEST:.1f} of {X_DISTINCT*EXPERT_TILE_COLS:.0f} tiles/layer); per tile: gate+up 2 x 32 cols",
            w=w * T_BUSIEST / EXPERT_TILE_COLS,
            in_=row(H),
            out=T_BUSIEST * TILE * ROWS * BF16,
            work=T_BUSIEST * 2 * TILE * ROWS * FP32,
        )
    elif "routed expert e: down" in n:
        r.update(
            split="tiles form: the matching 32 K-rows of down per tile -> partial [2,H] fp32 per expert, summed x score",
            w=w * T_BUSIEST / EXPERT_TILE_COLS,
            in_=T_BUSIEST * TILE * ROWS * BF16,
            out=row(H, FP32),
            work=row(H, FP32),
        )
    elif "routed + shared" in n:
        r.update(
            split="partial [2,H] fp32 (this chip) -> reduce-scatter to columns",
            w=0,
            in_=2 * row(H, FP32),
            out=row(H) / TP,
            work=0,
        )
    elif n.startswith("embed_tokens"):
        r.update(
            split="row lookup by the stage-0 chips (each its columns)",
            w=0,
            in_=0,
            out=row(HCD) / TP,
            work=0,
            dram_read=u["dram_row"] * ROWS / TP,
        )
    elif ".ple.n-gram" in n:
        r.update(
            split="16 head rows x 160 looked up (replicated: each chip reads the rows)",
            w=0,
            in_=0,
            out=row(PLE_DIM),
            work=0,
            dram_read=u["dram_row"] * ROWS,
        )
    elif ".ple.key_proj" in n:
        r.update(split="N-split 1/8 of 4H", w=w / TP, in_=row(PLE_DIM), out=row(HCD) / TP, work=row(HCD, FP32) / TP)
    elif ".ple.norm_query" in n:
        r.update(
            split="local columns 1/8 (gate partial -> stats all-reduce)",
            w=w / TP,
            in_=2 * row(HCD) / TP,
            out=row(HC, FP32),
            work=row(HC, FP32),
        )
    elif ".ple.value_proj" in n:
        r.update(
            split="N-split 1/8 of H -> its residual columns (4 branches x H/8)",
            w=w / TP,
            in_=row(PLE_DIM) + row(HC, FP32),
            out=row(HCD) / TP,
            work=row(H, FP32) / TP,
        )
    elif ".ple.norm_conv" in n:
        r.update(
            split="local columns 1/8 (stats all-reduce)",
            w=w / TP,
            in_=row(HCD) / TP,
            out=row(HCD) / TP,
            work=0,
            state_l1=u["state_l1"] / TP,
        )
    elif n.startswith("final hyper_connection_mixer"):
        r.update(split="as a hc read (K/N split)", w=w / TP, in_=row(HCD) / TP, out=row(H), work=row(HC_LOW, FP32))
    elif n.startswith("lm_head"):
        r.update(
            split="vocab N-split 1/8 (31,040 cols)", w=w / TP, in_=row(H), out=row(V / TP, FP32), work=row(V / TP, FP32)
        )
    elif n.startswith("argmax"):
        r.update(
            split="local argmax over its 31,040 + candidates all-gather (8 x 2 rows x 8 B)",
            w=0,
            in_=row(V / TP, FP32),
            out=ROWS * 4,
            work=ROWS * 8 * TP * 2,
        )
    elif n.startswith("mtp.hyper_connection_mixer"):
        r.update(split="as a hc read", w=w / TP, in_=row(HCD) / TP, out=row(H), work=row(HC_LOW, FP32))
    elif n.startswith("mtp.embed_tokens"):
        r.update(
            split="fc_embedding / fc_hidden N-split 1/8 -> hc init columns",
            w=w / TP,
            in_=row(H) + row(H),
            out=row(HCD) / TP,
            work=row(2 * H, FP32) / TP,
            dram_read=u["dram_row"] * ROWS / TP,
        )
    elif n.startswith("mtp.lm_head"):
        r.update(
            split="vocab N-split 1/8 (shared weight)", w=0, in_=row(H), out=row(V / TP, FP32), work=row(V / TP, FP32)
        )
    else:
        raise SystemExit("unmatched unit: " + n)
    return r


A = [share(u) for u in U0["units"]]


def layer_of(x):
    return x["layer"]


layers = []
for x in A:
    if not layers or layers[-1] != x["layer"]:
        layers.append(x["layer"])


def agg(pred):
    s = dict(w=0, in_=0, out=0, work=0, S=0, state_l1=0, dram_read=0)
    for x in A:
        if pred(x):
            for k in s:
                s[k] += x[k]
    return s


def fmt(s):
    return f"{s['w']/MiB:8.2f} | {s['in_']/KiB:7.1f} | {s['out']/KiB:7.1f} | {s['work']/KiB:7.1f} | {s['S']/KiB:7.1f} | {s['state_l1']/KiB:6.1f} | {s['dram_read']/KiB:6.1f}"


print(
    f"rows {ROWS}, chip = 1 of 8 in a TP8 group; routed tiles on the busiest chip per layer {T_BUSIEST:.2f} (census 1x2, EP1 x TPx8)"
)
print(
    "\n| unit | split on this chip | weight [MiB] | in [KiB] | out [KiB] | work [KiB] | S [KiB] | L1 state [KiB] | DRAM-only read [KiB] |"
)
print("|---|---|---|---|---|---|---|---|---|")
for x in A:
    if (
        x["layer"] in ("L00", "L03", "L02")
        or x["layer"] in ("final", "lm_head", "argmax")
        or x["layer"].startswith("mtp")
        and x["layer"] == "mtp"
        or x["unit"].startswith(("final", "lm_head", "argmax", "mtp.hyper", "mtp.embed", "mtp.lm"))
    ):
        print(
            f"| `{x['unit']}` | {x['split']} | {x['w']/MiB:.3f} | {x['in_']/KiB:.1f} | {x['out']/KiB:.1f} | {x['work']/KiB:.1f} | {x['S']/KiB:.1f} | {x['state_l1']/KiB:.1f} | {x['dram_read']/KiB:.1f} |"
        )
print(
    "\n| group | weight [MiB] | in [KiB] | out [KiB] | work [KiB] | S [KiB] | L1 state [KiB] | DRAM-only read [KiB] |"
)
print("|---|---|---|---|---|---|---|---|")
print(f"| GDN layer L00 (1 of 36) | {fmt(agg(lambda x: x['layer']=='L00' and '.gdn.' in x['unit']))} |")
print(f"| QSA layer L03 (1 of 12 +1 MTP) | {fmt(agg(lambda x: x['layer']=='L03' and '.qsa.' in x['unit']))} |")
print(f"| MoE block L00 (1 of 49) | {fmt(agg(lambda x: x['layer']=='L00' and '.moe.' in x['unit']))} |")
print(f"| PLE | {fmt(agg(lambda x: '.ple.' in x['unit']))} |")
print(f"| tail + MTP | {fmt(agg(lambda x: x['unit'].startswith(('final','lm_head','argmax','mtp.'))))} |")
tot = agg(lambda x: True)
print(f"| **one pass, this chip** | **{fmt(tot)}** |")
routed = sum(x["w"] for x in A if "routed expert" in x["unit"])
dense = tot["w"] - routed
rep_router = sum(x["w"] for x in A if ".router" in x["unit"] and "speculative" not in x["unit"])
rep_idx = sum(x["w"] for x in A if ".indexer qk" in x["unit"])
rep_kv = sum(x["w"] for x in A if ".k proj" in x["unit"])
print(
    f"\nroot per chip per pass: dense weights {dense/MiB:,.1f} MiB = the model's {U0['dense_bytes']/MiB:,.1f}/8 = {U0['dense_bytes']/8/MiB:,.1f} + replicated router {rep_router*7/8/MiB:.1f} + replicated indexer {rep_idx*7/8/MiB:.1f} + kv proj replicated on the 4-chip kv group {rep_kv*3/4/MiB:.1f} (check: {U0['dense_bytes']/8/MiB+rep_router*7/8/MiB+rep_idx*7/8/MiB+rep_kv*3/4/MiB:,.1f}); routed expert tiles {routed/MiB:.1f} MiB ({T_BUSIEST:.2f} tiles x 49 layers x {(2*E_INT*H*BF4+H*E_INT*BF4)/EXPERT_TILE_COLS/KiB:.1f} KiB/tile); S {tot['S']/MiB:.1f} MiB (read + write = {2*tot['S']/MiB:.1f}); L1-resident state {tot['state_l1']/KiB:.0f} KiB; DRAM-only reads {tot['dram_read']/KiB:.1f} KiB; rows in flight max {max(x['in_']+x['out']+x['work'] for x in A)/KiB:.0f} KiB (unit `{max(A,key=lambda x: x['in_']+x['out']+x['work'])['unit']}`)"
)
print(
    f"one pass, this chip, DRAM traffic if nothing is resident: {(dense+routed+2*tot['S']+tot['dram_read'])/MiB:,.1f} MiB = {(dense+routed+2*tot['S']+tot['dram_read'])/512e9*1e3:.3f} ms at 512 GB/s; the 8 chips of a group together {(dense+routed+2*tot['S']+tot['dram_read'])*8/MiB:,.1f} MiB; all 32 chips (4 groups) {(dense+routed+2*tot['S']+tot['dram_read'])*32/MiB:,.1f} MiB"
)
json.dump(
    dict(rows=ROWS, tp=TP, busiest_tiles=T_BUSIEST, units=A, layers=layers),
    open(sys.argv[3] if len(sys.argv) > 3 else "/tmp/q38/design/step2_account.json", "w"),
)
