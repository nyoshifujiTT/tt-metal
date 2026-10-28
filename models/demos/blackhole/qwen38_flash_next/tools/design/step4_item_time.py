#!/usr/bin/env python
r"""Step 4 of the detailed design (worklog 12.4; revision 8 = the stage-level design D-2, worklog 10-27): the time of
every unit on one chip of the chosen form (rows 2, TP8, tiles-form experts), from the specifications only (stages.py:
instruction counts x specified cycles, plus the three budgets P_PACK / P_UNP / N_RD for pipeline depths no document gives).
  matmul pools (units reading the same input run side by side, their output tiles pooled): stages.pool_stage = the MACs
    (work / 130 at the per-format, per-row-count tile-MAC), the Stream-K senders (finalize, pack, NoC write) and the
    column owners (poll, unpack of the partials to Dst, SFPU add, finalize, pack, NoC write), then the hand-over of the
    finished columns: up to the hub (owners near the hub: 7 / 11 / 14 hops), the hub's poll and issue, the multicast to
    all 130 cores as two rectangles on NoC0 / NoC1 (17 hops, worst over the 14 harvest cases), the receivers' poll.
  elementwise work is fused into the owners' finalize (SFPU instruction counts of the fp32 functions, stages.ACT).
  gated delta rule: stages.gdr_unit = (value head, 8 columns of S fp32 in Dst), 2 passes a row on the SFPU, 2 rows in
    series; 96 units on 96 cores; + the hand-over to out_proj; the gated norm's per-head scale gathered over 7 hops.
  router top-10: stages.topk10 = each of the 16 router owners' top-10, merged on the selector, softmax of the 10.
  SDPA (ctx 1): on the kv-head cores, 3 q heads x 2 rows, the output multicast straight to the o_proj cores.
  argmax (greedy): each core's ~8 logit column tiles (SFPSWAP max + index), one hop of the candidates to the selector;
    the 8 chips' candidates are all-gathered (step 6).
  DRAM: the unit's weight + S + reads at 512 GB/s (the chip's lane, for reference; the pass-level lane is step 5).
Outputs: the per-unit table (markdown) and step4_item_time.json."""
import json
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import mm_cost
import stages as SG

A = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step2_account.json"))
U = A["units"]
ROWS = A["rows"]
S0 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step0_units.json"))["units"]
CLK = 1.35e9
CORES = 130
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


_MMS = []  # the matmul shapes of the unit being costed (written to the json for the H-1 L1 map)


def mm_c(M, N, K, fmt):
    _MMS.append([[M, N, K, fmt]])
    return mm_cost.one(M, N, K, fmt)


def mm_pool(*mms):
    _MMS.append([[list(x) if isinstance(x, tuple) else x for x in m] for m in mms])
    return mm_cost.pool(*mms)


T_BUSIEST = A["busiest_tiles"]


# per-unit cycles; pools are attached to the first unit of the pool and the others get 0 ("side by side")
def cycles(x):
    n = x["unit"]
    r = dict(mm=0.0, serial=0.0, elt_=0.0, pool="")
    if ".gr.norm" in n:
        r["elt_"] = SG.ACT["rsqrt"] + 2 * SG.ACT["mul"]  # D-2: 1/rms_g (part of SG.hc_read_elt)
    elif ".gr.down" in n:
        r["mm"] = mm_pool((ROWS, HC_LOW, HCD / TP, "bf16"), (ROWS, HC, HCD / TP, "bf16"))
        r["pool"] = "down + inject (same input)"
        r["elt_"] = SG.owner_act("silu") + SG.ACT["mul"]  # D-2: silu/4 in the owners' finalize
    elif ".gr.up" in n:
        r["mm"] = mm_c(ROWS, HCD / TP, HC_LOW, "bf16")
        r["elt_"] = SG.owner_act("sigmoid")  # D-2: sigmoid in the owners' finalize
    elif ".gr.gate x normalized" in n:
        r["elt_"] = 4 * 2 * SG.ACT["mul"]  # D-2: the 4 branches of a column on one owner: local
    elif ".gr.inject" in n:
        r["pool"] = "(in the gr.down pool)"
    elif ".gr.write" in n:
        r["elt_"] = 4 * 2 * SG.ACT["mul"]  # D-2: residual columns on their owner
    elif ".in_proj_qkv" in n:
        r["mm"] = mm_pool((ROWS, 1280, H, "bf8"), (ROWS, 6 * GDN_HD, H, "bf8"), (ROWS, TILE, H, "bf8"), (ROWS, TILE, H, "bf8"))
        r["pool"] = "qkv + z + a + b"
    elif ".in_proj_z" in n or ".in_proj_a" in n or ".in_proj_b" in n:
        r["pool"] = "(in the qkv pool)"
    elif ".conv1d" in n:
        r["elt_"] = SG.conv1d_elt()
    elif ".l2norm" in n:
        r["elt_"] = 0  # D-2: folded into the GDR unit (k.k, q.q with the passes)
    elif ".gated delta rule" in n:
        r["serial"] = SG.gdr_layer(ROWS, 6 * 16)  # D-2: 96 units of 8 columns on 96 cores + the hand-over
    elif ".norm (gated)" in n:
        r["elt_"] = SG.norm_gated()
    elif ".out_proj" in n:
        r["mm"] = mm_c(ROWS, H, 6 * GDN_HD, "bf8")
    elif ".indexer qk" in n:
        r["mm"] = mm_pool((ROWS, 3 * 2 * QSA_HD, H, "bf8"), (ROWS, 2 * QSA_HD, H, "bf8"), (ROWS, IDX_Q * IDX_HD + IDX_HD, H, "bf8"))
        r["pool"] = "q+gate + k,v + indexer"
    elif ".index scoring" in n:
        r["elt_"] = SG.index_ctx1_elt()
    elif ".q+gate proj" in n or ".k proj" in n:
        r["pool"] = "(in the indexer pool)"
    elif ".partial RoPE" in n:
        r["elt_"] = SG.rope_elt()
    elif ".SDPA" in n:
        r["elt_"] = SG.sdpa_ctx1(ROWS)
    elif ".o_proj" in n:
        r["mm"] = mm_c(ROWS, H, 3 * QSA_HD, "bf8")
    elif ".router" in n and "speculative" not in n:
        r["mm"] = mm_pool((ROWS, SH_INT / TP, H, "bf8", 2, SG.moe_act_elt(ROWS, False)), (ROWS, E, H, "bf16"), (ROWS, 1, H, "bf16"))
        r["serial"] = SG.topk10(ROWS, E // TILE)
        r["pool"] = "router + shared gate/up pairs (silu x up in their owners) + shared expert gate"
    elif "speculative router" in n:
        r["mm"] = mm_c(ROWS, E, H, "bf16")
        r["serial"] = SG.topk10(ROWS, E // TILE)  # rev8 read-through: the same stage-level top-10 as the router
        r["pool"] = "(one block earlier, off the chain)"
    elif ".shared gate_proj" in n:
        r["pool"] = "(in the router pool)"
    elif ".shared down_proj" in n:
        r["pool"] = "(in the experts' down pool)"
    elif "routed expert e: gate_up" in n:
        r["mm"] = mm_pool((1, T_BUSIEST * TILE, H, "bf4", 2, SG.moe_act_elt(ROWS, True)))
        r["pool"] = f"{T_BUSIEST:.1f} gate/up tile pairs (silu x up x score in their owners)"
    elif "routed expert e: down" in n:
        r["mm"] = mm_pool((ROWS, H, ((T_BUSIEST * TILE, "bf4"), (SH_INT / TP, "bf8")), "cat", 1, SG.moe_combine_elt(ROWS)))
        r["pool"] = f"down: K = {T_BUSIEST:.1f} expert tiles + shared 80 columns concatenated, routed + gated shared in the owners"
    elif "routed + shared" in n:
        r["elt_"] = 0  # correction 19: in the down owners' epilogue (SG.moe_combine_elt)
    elif n.startswith("embed_tokens"):
        r["elt_"] = 0
    elif ".ple.n-gram" in n:
        r["elt_"] = 0
    elif ".ple.key_proj" in n:
        r["mm"] = mm_pool((ROWS, HCD / TP, 2560, "bf16"), (ROWS, H / TP, 2560, "bf16"))
        r["pool"] = "key + value"
    elif ".ple.norm_query" in n:
        r["elt_"] = SG.ACT["rsqrt"] + 4 * SG.ACT["mul"] + SG.ACT["sigmoid"] + 20  # D-2: gate from the meeting's dot products
    elif ".ple.value_proj" in n:
        r["pool"] = "(in the key pool)"
    elif ".ple.norm_conv" in n:
        r["elt_"] = SG.ACT["rsqrt"] + 2 * 4 * 3 + SG.owner_act("silu") + 2 * SG.ACT["mul"]  # D-2: in the value owners' finalize
    elif n.startswith("final hyper") or n.startswith("mtp.hyper"):
        r["mm"] = mm_c(ROWS, HC_LOW, HCD / TP, "bf8") + mm_c(ROWS, H, HC_LOW, "bf8")
    elif n.startswith("lm_head") or n.startswith("mtp.lm_head"):
        r["mm"] = mm_c(ROWS, V / TP, H, "bf8")
    elif n.startswith("argmax"):
        r["serial"] = (
            math.ceil(math.ceil(V / TP / TILE) / CORES) * 2 * 4 + 9 + SG.flight(SG.UP_HOPS, 16) + SG.poll() + 9 * 4 + 9
        )  # D-2: each core's ~8 logit column tiles (2 rows = 2 groups each, SFPSWAP max + index), one hop of the 130 candidates to the selector, its 9 groups
    elif n.startswith("mtp.embed_tokens"):
        r["mm"] = mm_c(ROWS, 2 * H / TP, H, "bf8")
    else:
        raise SystemExit("unmatched " + n)
    return r


def form_pools(n, pools_a):
    """correction 39 (10-27 19:5x): the unit's pools per hyper-connection form (step 6's forms: A, B, Bn, B+D).  Only the
    hc read's down and the attention output differ:
      attention read's down (the hc read before a GDN / QSA block): A, B K-split [320+4, 4H/8]; Bn N-split [40 rows of
        D + inject, 4H] (AG_low); B+D replicated [320+4, 4H]
      MoE read's down (before a MoE block): K-split in every form; in B, Bn, B+D it runs at the attention block's start
        off the chain (g_b rides the attention output's all-reduce with the fold F_i h_i): same pool, same L1
      attention output (out_proj / o_proj): A plain; B, Bn, B+D pooled with the fold [4 x 320, heads x 128 (GDN) / 256 (QSA)] bf16"""
    out = {"A": pools_a}
    attn_read = (".gdn.gr.down" in n or ".qsa.gr.down" in n) and not n.startswith("mtp.hyper") and not n.startswith("final")
    moe_read = ".moe.gr.down" in n
    if attn_read:
        out["B"] = pools_a
        out["Bn"] = [[[ROWS, HC_LOW / TP, HCD, "bf16"], [ROWS, HC, HCD, "bf16"]]]
        out["B+D"] = [[[ROWS, HC_LOW, HCD, "bf16"], [ROWS, HC, HCD, "bf16"]]]
    elif moe_read:
        # correction 40: the MoE read's D K-split still runs (g_b = D_b[:, C_i] ((1+w) * r_b)[C_i] at the attention block's
        # start, off the chain, riding the attention output's all-reduce): its weights are in L1 and placed; its time is
        # off the chain (step 6's forms: moe block down 0)
        for f in ("B", "Bn", "B+D"):
            out[f] = pools_a
    elif ".out_proj" in n or ".o_proj" in n:
        k = 6 * GDN_HD if ".out_proj" in n else 3 * QSA_HD
        fold = [[ROWS, H, k, "bf8"], [ROWS, HC * HC_LOW, k, "bf16"]]
        for f in ("B", "Bn", "B+D"):
            out[f] = [fold]
    else:
        for f in ("B", "Bn", "B+D"):
            out[f] = pools_a
    return out


rows = []
for x in U:
    _MMS.clear()
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
            pools=[list(p) for p in _MMS],  # the unit's matmul pools (each a list of [M, N, K, fmt]); step 3 sizes the partial landing from them
            pools_by_form=form_pools(x["unit"], [list(p) for p in _MMS]),
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
