#!/usr/bin/env python
r"""Step 4 of the detailed design (worklog 12.4): the time of every unit on one chip of the chosen form (rows 2, TP8,
tiles-form experts), from the physical values (README 11.2.3) only.
  matmul [M, K] x [K, N] on the chip's share: the per-format, per-row-count tile-MAC of mm_cost.py (revision 6: bf8
    HiFi2 / BF4 LoFi 8 cycles and bf16 HiFi4 16 cycles at rows <= 8; revision 5 used HiFi2 32 for every format).  Output tiles go to the 130 cores;
    when there are fewer output tiles than cores (every projection at rows 2 -> 1 row-tile) the K dimension is split over
    cores_per_tile = ceil(130 / out) cores so every core carries work/130 tile-MACs, and the fp32 partial tiles are
    tree-reduced over the NoC (per level: 92 cycles + 64 flits of a 4 KiB tile + a 32-cycle DEST accumulate).
    chain = min(no split: K-tiles x 32, split: work/130 x 32 + levels x 188 cycles).  Units that read the same input run
    side by side (their output tiles pooled): the GDN in-projections (qkv, z, a, b), the QSA projections (q+gate, k, v,
    indexer), the shared up/gate with the router, the experts' gate_up across experts, the experts' down with the shared
    down.  (README 11.3 E19, H-18/H-19.)
  gated delta rule: one (value head, user) is 4,224 cycles for one row (27B section 9), split into 4 independent 32-column
    units of 256 + 800 x rows cycles (the 2 rows are a recurrence); units on separate cores: 6 heads x 4 = 24 units on 24
    cores, one round.
  router top-10: per row-tile, 16 logit tiles -> 10 passes x 32 cycles each in parallel, NoC gather 92 cycles, 10 passes
    on the merged tile.
  argmax (greedy): the chip's 970 logit tiles (fp32, V/8 columns) spread over the 130 cores, 32 cycles per tile scanned,
    a NoC gather of the 130 (value, index) candidates (92 cycles), a final pass over the 5 candidate tiles; the 8 chips'
    candidates are all-gathered (step 6) and every chip picks the same token.
  elementwise: 16 cycles per bf16 tile moved through the FPU, 32 per fp32 tile, spread over the cores.
  DRAM: the unit's weight + S + reads at 512 GB/s (the chip's lane, for reference; the pass-level lane is step 5).
Outputs: the per-unit table (markdown) and step4_item_time.json."""
import json
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import mm_cost

A = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step2_account.json"))
U = A["units"]
ROWS = A["rows"]
S0 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step0_units.json"))["units"]
CLK = 1.35e9
CORES = 130
MM = 32
ELT = 16
FP32T = 32
NOC_RELAY = 92
NOC_FLIT = 64
TILE = 32
BW = 512e9
KiB = 1024
MiB = 1024**2
TP = 8
H = 2560
HC = 4
HC_LOW = 320
HCD = HC * H
GDN_HD = 128
QSA_HD = 256
E = 512
E_INT = 640
SH_INT = 640
V = 248320
IDX_Q = 4
IDX_HD = 128


def mm_c(M, N, K, fmt):
    return mm_cost.one(M, N, K, fmt)


def mm_pool(*mms):
    return mm_cost.pool(*mms)


def gdr_unit(R):
    return 256 + 800 * R


def topk(rows, ntile):
    return max(
        10 * FP32T + NOC_RELAY + 10 * FP32T,
        math.ceil(rows / TILE) * (ntile * 10 * FP32T + NOC_RELAY + 10 * FP32T) / CORES,
    )


def elt(bytes_, dt=2.0):
    return bytes_ / (TILE * TILE * dt) * (ELT if dt == 2.0 else FP32T) / CORES


T_BUSIEST = A["busiest_tiles"]


# per-unit cycles; pools are attached to the first unit of the pool and the others get 0 ("side by side")
def cycles(x):
    n = x["unit"]
    r = dict(mm=0.0, serial=0.0, elt_=0.0, pool="")
    if ".gr.norm" in n:
        r["elt_"] = elt(x["in_"]) * 3
    elif ".gr.down" in n:
        r["mm"] = mm_pool((ROWS, HC_LOW, HCD / TP, "bf16"), (ROWS, HC, HCD / TP, "bf16"))
        r["pool"] = "down + inject (same input)"
    elif ".gr.up" in n:
        r["mm"] = mm_c(ROWS, HCD / TP, HC_LOW, "bf16")
    elif ".gr.gate x normalized" in n:
        r["elt_"] = elt(x["in_"]) * 2
    elif ".gr.inject" in n:
        r["pool"] = "(in the gr.down pool)"
    elif ".gr.write" in n:
        r["elt_"] = elt(x["in_"]) * 2
    elif ".in_proj_qkv" in n:
        r["mm"] = mm_pool((ROWS, 1280, H, "bf8"), (ROWS, 6 * GDN_HD, H, "bf8"), (ROWS, TILE, H, "bf8"), (ROWS, TILE, H, "bf8"))
        r["pool"] = "qkv + z + a + b"
    elif ".in_proj_z" in n or ".in_proj_a" in n or ".in_proj_b" in n:
        r["pool"] = "(in the qkv pool)"
    elif ".conv1d" in n:
        r["elt_"] = elt(x["in_"]) * 4
    elif ".l2norm" in n:
        r["elt_"] = elt(x["in_"]) * 3
    elif ".gated delta rule" in n:
        r["serial"] = math.ceil(6 * 4 / CORES) * gdr_unit(ROWS)
    elif ".norm (gated)" in n:
        r["elt_"] = elt(x["in_"], 4.0) * 3
    elif ".out_proj" in n:
        r["mm"] = mm_c(ROWS, H, 6 * GDN_HD, "bf8")
    elif ".indexer qk" in n:
        r["mm"] = mm_pool((ROWS, 3 * 2 * QSA_HD, H, "bf8"), (ROWS, 2 * QSA_HD, H, "bf8"), (ROWS, IDX_Q * IDX_HD + IDX_HD, H, "bf8"))
        r["pool"] = "q+gate + k,v + indexer"
    elif ".index scoring" in n:
        r["elt_"] = elt(x["in_"])
    elif ".q+gate proj" in n or ".k proj" in n:
        r["pool"] = "(in the indexer pool)"
    elif ".partial RoPE" in n:
        r["elt_"] = elt(x["in_"]) * 2
    elif ".SDPA" in n:
        r["elt_"] = elt(x["in_"]) * 3
    elif ".o_proj" in n:
        r["mm"] = mm_c(ROWS, H, 3 * QSA_HD, "bf8")
    elif ".router" in n and "speculative" not in n:
        r["mm"] = mm_pool((ROWS, 2 * SH_INT / TP, H, "bf8"), (ROWS, E, H, "bf16"))
        r["serial"] = topk(ROWS, E / TILE)
        r["pool"] = "router + shared gate/up"
    elif "speculative router" in n:
        r["mm"] = mm_c(ROWS, E, H, "bf16")
        r["serial"] = topk(ROWS, E / TILE)
        r["pool"] = "(one block earlier, off the chain)"
    elif ".shared gate_proj" in n:
        r["pool"] = "(in the router pool)"
    elif ".shared down_proj" in n:
        r["pool"] = "(in the experts' down pool)"
    elif "routed expert e: gate_up" in n:
        r["mm"] = mm_c(1, T_BUSIEST * 2 * TILE, H, "bf4")
        r["pool"] = f"{T_BUSIEST:.1f} tiles x 2 cols pooled"
    elif "routed expert e: down" in n:
        r["mm"] = mm_pool((1, H * math.ceil(T_BUSIEST), TILE, "bf4"), (ROWS, H, SH_INT / TP, "bf8"))
        r["pool"] = f"{T_BUSIEST:.1f} down slices + shared down"
    elif "routed + shared" in n:
        r["elt_"] = elt(x["in_"], 4.0)
    elif n.startswith("embed_tokens"):
        r["elt_"] = 0
    elif ".ple.n-gram" in n:
        r["elt_"] = 0
    elif ".ple.key_proj" in n:
        r["mm"] = mm_pool((ROWS, HCD / TP, 2560, "bf16"), (ROWS, H / TP, 2560, "bf16"))
        r["pool"] = "key + value"
    elif ".ple.norm_query" in n:
        r["elt_"] = elt(x["in_"]) * 3
    elif ".ple.value_proj" in n:
        r["pool"] = "(in the key pool)"
    elif ".ple.norm_conv" in n:
        r["elt_"] = elt(x["in_"]) * 5
    elif n.startswith("final hyper") or n.startswith("mtp.hyper"):
        r["mm"] = mm_c(ROWS, HC_LOW, HCD / TP, "bf8") + mm_c(ROWS, H, HC_LOW, "bf8")
    elif n.startswith("lm_head") or n.startswith("mtp.lm_head"):
        r["mm"] = mm_c(ROWS, V / TP, H, "bf8")
    elif n.startswith("argmax"):
        r["serial"] = (
            math.ceil(math.ceil(V / TP / TILE) / CORES) * FP32T + NOC_RELAY + math.ceil(CORES / TILE) * FP32T
        )  # local argmax: each core scans its ~7.5 logit tiles (32 cycles per fp32 tile), NoC gather of the 130 candidates, final pass over 5 tiles
    elif n.startswith("mtp.embed_tokens"):
        r["mm"] = mm_c(ROWS, 2 * H / TP, H, "bf8")
    else:
        raise SystemExit("unmatched " + n)
    return r


rows = []
for x in U:
    c = cycles(x)
    t = (c["mm"] + c["serial"] + c["elt_"]) / CLK
    d = (x["w"] + 2 * x["S"] + x["dram_read"]) / BW
    rows.append(
        dict(
            unit=x["unit"],
            layer=x["layer"],
            t_us=t * 1e6,
            mm_us=c["mm"] / CLK * 1e6,
            serial_us=c["serial"] / CLK * 1e6,
            elt_us=c["elt_"] / CLK * 1e6,
            dram_us=d * 1e6,
            pool=c["pool"],
            spec=("speculative" in x["unit"]),
        )
    )


def show(pred, title):
    print(f"\n### {title}\n")
    print("| unit | compute [us] (matmul chain / serial / elementwise) | pool | DRAM on this chip [us] |")
    print("|---|---|---|---|")
    tt = 0
    dd = 0
    for r in rows:
        if pred(r):
            print(
                f"| `{r['unit'][:70]}` | {r['t_us']:.2f} ({r['mm_us']:.2f} / {r['serial_us']:.2f} / {r['elt_us']:.2f}) | {r['pool']} | {r['dram_us']:.2f} |"
            )
            if not r["spec"]:
                tt += r["t_us"]
            dd += r["dram_us"]
    print(f"| **sum (chain, speculative router excluded)** | **{tt:.2f}** | | **{dd:.2f}** |")
    return tt, dd


g = show(lambda r: r["layer"] == "L00" and ".gdn." in r["unit"], "GDN layer L00 (1 of 36), one chip")
q = show(lambda r: r["layer"] == "L03" and ".qsa." in r["unit"], "QSA layer L03 (1 of 12 + MTP), one chip")
m = show(lambda r: r["layer"] == "L00" and ".moe." in r["unit"], "MoE block L00 (1 of 49), one chip")
p = show(lambda r: ".ple." in r["unit"], "PLE, one chip")
t = show(
    lambda r: r["unit"].startswith(("final", "lm_head", "argmax", "mtp.hyper", "mtp.embed", "mtp.lm")),
    "tail (without the MTP layer's QSA+MoE), one chip",
)
tot = sum(r["t_us"] for r in rows if not r["spec"])
dtot = sum(r["dram_us"] for r in rows)
print(
    f"\none pass, one chip: compute chain {tot:.1f} us = {tot/1e3:.3f} ms (GDN 36 x {g[0]:.1f} + QSA 13 x {q[0]:.1f} + MoE 49 x {m[0]:.1f} + PLE {p[0]:.1f} + tail {t[0]:.1f}); DRAM lane if serial {dtot/1e3:.3f} ms; the group's 4 frames x 2 rounds share the pass: this chip is on the chain for {tot/4/1e3:.3f} ms of it"
)
print(
    f"speculative router per layer {next(r['t_us'] for r in rows if r['spec']):.2f} us (off the chain: runs one block earlier on the same chip)"
)
json.dump(rows, open(sys.argv[3] if len(sys.argv) > 3 else "/tmp/q38/design/step4_item_time.json", "w"))
