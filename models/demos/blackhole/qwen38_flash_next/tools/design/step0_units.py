#!/usr/bin/env python
r"""Step 0 of the detailed design (worklog 10.H61): the units (one HF operation each) of one decode pass of
Qwen3.8-Flash-Next, in forward order, with their weight bytes in the served number formats, the rows they consume and
produce (the edges), and the per-user state they touch.  Inputs: the HF definition as read from the CPU oracle
(tt/gdn.py, tt/qsa.py, tt/moe.py, tt/gr.py, tt/ple.py, tt/model.py, reference.py) and the served formats
(dense bf8 1,088 B/tile, hyper-connection mixers / router / norms bf16, routed experts BF4 576 B/tile, GDN state fp32).
Condition: ctx 1 (no KV), B=1, MTP k=1 -> rows = 2 (the user's token and its verify row).  No timing here.
Output: markdown tables (stdout) and a JSON with every unit (for the later steps)."""
import json
import sys

H = 2560
HC = 4
HC_LOW = 320
HCD = HC * H
L = 48
LAYER = ["gdn", "gdn", "gdn", "qsa"] * 12
GDN_QK = 16
GDN_V = 48
GDN_HD = 128
CONV_K = 4
QSA_Q = 24
QSA_KV = 2
QSA_HD = 256
ROPE = 64
IDX_Q = 4
IDX_KV = 1
IDX_HD = 128
IDX_BUDGET = 2048
IDX_COMP = 4
E = 512
TOPK = 10
E_INT = 640
SH_INT = 640
V = 248320
PLE_LAYER = 2
PLE_NGRAM_PARTS = 128
PLE_DIM = 2560
PLE_HEADS = 16
PLE_HEAD_DIM = 160
PLE_CONV_K = 4
PLE_DIL = 3
ROWS = 2  # B=1, k=1: the token row and the verify row
BF8 = 1088 / 1024
BF4 = 576 / 1024
BF16 = 2.0
FP32 = 4.0
KiB = 1024
MiB = 1024**2


def mm(n, k, dt):
    return n * k * dt  # weight bytes of a [n, k] linear in format dt (per element)


def row(w, dt=BF16):
    return ROWS * w * dt  # bytes of ROWS rows of width w


def hc_read(tag):
    """hyper-connection read: zero-centered rms norm over each of the 4 branches (group_size H), down [320, 4H] bf16,
    silu(/4), up [4H, 320] bf16, sigmoid gate x normalized, mean over the 4 branches -> block input [rows, H]"""
    return [
        dict(
            unit=f"{tag}.gr.norm (zero-centered RMS, 4 groups of H)",
            w=HCD * BF16,
            w_fmt="bf16",
            in_=row(HCD),
            out=row(HCD),
            work=0,
            kind="elt",
        ),
        dict(
            unit=f"{tag}.gr.down [320, 4H] (+ silu /4)",
            w=mm(HC_LOW, HCD, BF16),
            w_fmt="bf16",
            in_=row(HCD),
            out=row(HC_LOW, FP32),
            work=row(HC_LOW, FP32),
            kind="mm",
            M=ROWS,
            N=HC_LOW,
            K=HCD,
        ),
        dict(
            unit=f"{tag}.gr.up [4H, 320] (sigmoid)",
            w=mm(HCD, HC_LOW, BF16),
            w_fmt="bf16",
            in_=row(HC_LOW, FP32),
            out=row(HCD),
            work=row(HCD, FP32),
            kind="mm",
            M=ROWS,
            N=HCD,
            K=HC_LOW,
        ),
        dict(
            unit=f"{tag}.gr.gate x normalized, mean of 4 -> block input [rows, H]",
            w=0,
            w_fmt="-",
            in_=2 * row(HCD),
            out=row(H),
            work=0,
            kind="elt",
        ),
        dict(
            unit=f"{tag}.gr.inject [4, 4H] (sigmoid, x2)",
            w=mm(HC, HCD, BF16),
            w_fmt="bf16",
            in_=row(HCD),
            out=row(HC, FP32),
            work=0,
            kind="mm",
            M=ROWS,
            N=HC,
            K=HCD,
        ),
    ]


def hc_write(tag):
    return [
        dict(
            unit=f"{tag}.gr.write: residual += inject[b] x block output (4 branches)",
            w=0,
            w_fmt="-",
            in_=row(HCD) + row(H) + row(HC, FP32),
            out=row(HCD),
            work=0,
            kind="elt",
        )
    ]


def gdn(l):
    t = f"L{l:02d}.gdn"
    u = []
    u += hc_read(t)
    u.append(
        dict(
            unit=f"{t}.in_proj_qkv [2*16*128+48*128, H]",
            w=mm(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD, H, BF8),
            w_fmt="bf8",
            in_=row(H),
            out=row(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD),
            work=0,
            kind="mm",
            M=ROWS,
            N=2 * GDN_QK * GDN_HD + GDN_V * GDN_HD,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.conv1d (4 tap, depthwise, silu; tap state 3 rows)",
            w=(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD) * CONV_K * BF16,
            w_fmt="bf16",
            in_=row(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD),
            out=row(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD),
            work=0,
            kind="elt",
            state_l1=(2 * GDN_QK * GDN_HD + GDN_V * GDN_HD) * (CONV_K - 1) * BF16,
        )
    )
    u.append(
        dict(
            unit=f"{t}.in_proj_z [48*128, H]",
            w=mm(GDN_V * GDN_HD, H, BF8),
            w_fmt="bf8",
            in_=row(H),
            out=row(GDN_V * GDN_HD),
            work=0,
            kind="mm",
            M=ROWS,
            N=GDN_V * GDN_HD,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.in_proj_a [48, H] -> g = -exp(A_log) * softplus(a + dt_bias)",
            w=mm(GDN_V, H, BF8) + 2 * GDN_V * FP32,
            w_fmt="bf8",
            in_=row(H),
            out=row(GDN_V, FP32),
            work=0,
            kind="mm",
            M=ROWS,
            N=GDN_V,
            K=H,
        )
    )  # checkpoint: in_proj_a (48, 2560), A_log (48,), dt_bias (48,)
    u.append(
        dict(
            unit=f"{t}.in_proj_b [48, H] -> beta = sigmoid",
            w=mm(GDN_V, H, BF8),
            w_fmt="bf8",
            in_=row(H),
            out=row(GDN_V, FP32),
            work=0,
            kind="mm",
            M=ROWS,
            N=GDN_V,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.l2norm q, k (16 heads, no weight) + repeat_interleave x3",
            w=0,
            w_fmt="-",
            in_=row(2 * GDN_QK * GDN_HD),
            out=row(2 * GDN_V * GDN_HD),
            work=0,
            kind="elt",
        )
    )
    u.append(
        dict(
            unit=f"{t}.gated delta rule (48 value heads, S fp32 [128,128] per head per user; 2 rows serial)",
            w=0,
            w_fmt="-",
            in_=row(2 * GDN_V * GDN_HD) + row(GDN_V * GDN_HD) + 2 * row(GDN_V, FP32),
            out=row(GDN_V * GDN_HD, FP32),
            work=GDN_V * GDN_HD * FP32 * 4,
            kind="gdr",
            state_S=GDN_V * GDN_HD * GDN_HD * FP32,
        )
    )
    u.append(
        dict(
            unit=f"{t}.norm (gated): rms over 128 per head x weight[128] x sigmoid(z)",
            w=GDN_HD * BF16,
            w_fmt="bf16",
            in_=row(GDN_V * GDN_HD, FP32) + row(GDN_V * GDN_HD),
            out=row(GDN_V * GDN_HD),
            work=0,
            kind="elt",
        )
    )
    u.append(
        dict(
            unit=f"{t}.out_proj [H, 48*128]",
            w=mm(H, GDN_V * GDN_HD, BF8),
            w_fmt="bf8",
            in_=row(GDN_V * GDN_HD),
            out=row(H),
            work=0,
            kind="mm",
            M=ROWS,
            N=H,
            K=GDN_V * GDN_HD,
        )
    )
    u += hc_write(t)
    return u


def qsa(l):
    t = f"L{l:02d}.qsa"
    u = []
    u += hc_read(t)
    u.append(
        dict(
            unit=f"{t}.indexer qk [4*128+128, H] + q/k norm",
            w=mm(IDX_Q * IDX_HD + IDX_KV * IDX_HD, H, BF8) + 2 * IDX_HD * BF16,
            w_fmt="bf8",
            in_=row(H),
            out=row(IDX_Q * IDX_HD + IDX_KV * IDX_HD),
            work=0,
            kind="mm",
            M=ROWS,
            N=IDX_Q * IDX_HD + IDX_KV * IDX_HD,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.index scoring over ctx/4 blocks, top-{IDX_BUDGET} (ctx 1: 1 block, no selection)",
            w=0,
            w_fmt="-",
            in_=row(IDX_Q * IDX_HD),
            out=0,
            work=0,
            kind="elt",
            kv_index_row=IDX_KV * IDX_HD * BF16 / IDX_COMP,
        )
    )
    u.append(
        dict(
            unit=f"{t}.q+gate proj [2*24*256, H] + q norm",
            w=mm(2 * QSA_Q * QSA_HD, H, BF8) + QSA_HD * BF16,
            w_fmt="bf8",
            in_=row(H),
            out=row(2 * QSA_Q * QSA_HD),
            work=0,
            kind="mm",
            M=ROWS,
            N=2 * QSA_Q * QSA_HD,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.k proj [2*256, H] + k norm; v proj [2*256, H]",
            w=2 * mm(QSA_KV * QSA_HD, H, BF8) + QSA_HD * BF16,
            w_fmt="bf8",
            in_=row(H),
            out=2 * row(QSA_KV * QSA_HD),
            work=0,
            kind="mm",
            M=ROWS,
            N=2 * QSA_KV * QSA_HD,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.partial RoPE (64 of 256) on q, k; KV append (ctx 1: 1 row)",
            w=0,
            w_fmt="-",
            in_=row(QSA_Q * QSA_HD) + row(QSA_KV * QSA_HD),
            out=row(QSA_Q * QSA_HD) + row(QSA_KV * QSA_HD),
            work=0,
            kind="elt",
            kv_row=2 * QSA_KV * QSA_HD * BF16,
        )
    )
    u.append(
        dict(
            unit=f"{t}.SDPA over the selected rows (ctx 1: {ROWS} rows) x sigmoid(gate)",
            w=0,
            w_fmt="-",
            in_=row(QSA_Q * QSA_HD) + 2 * row(QSA_KV * QSA_HD) + row(QSA_Q * QSA_HD),
            out=row(QSA_Q * QSA_HD),
            work=row(QSA_Q, FP32) * ROWS,
            kind="elt",
        )
    )
    u.append(
        dict(
            unit=f"{t}.o_proj [H, 24*256]",
            w=mm(H, QSA_Q * QSA_HD, BF8),
            w_fmt="bf8",
            in_=row(QSA_Q * QSA_HD),
            out=row(H),
            work=0,
            kind="mm",
            M=ROWS,
            N=H,
            K=QSA_Q * QSA_HD,
        )
    )
    u += hc_write(t)
    return u


def moe(l, spec=True):
    t = f"L{l:02d}.moe"
    u = []
    u += hc_read(t)
    u.append(
        dict(
            unit=f"{t}.router [512, H] bf16 -> softmax -> top-10 (scores, indices; norm_topk_prob)",
            w=mm(E, H, BF16),
            w_fmt="bf16",
            in_=row(H),
            out=row(E, FP32),
            work=row(E, FP32),
            kind="mm",
            M=ROWS,
            N=E,
            K=H,
            topk=True,
        )
    )
    if spec:
        u.append(
            dict(
                unit=f"{t}.speculative router (same weight, one block earlier: the experts to prefetch)",
                w=0,
                w_fmt="(router)",
                in_=row(H),
                out=row(E, FP32),
                work=row(E, FP32),
                kind="mm",
                M=ROWS,
                N=E,
                K=H,
                topk=True,
                spec=True,
            )
        )
    u.append(
        dict(
            unit=f"{t}.shared gate_proj [640, H], up_proj [640, H] (silu x up)",
            w=2 * mm(SH_INT, H, BF8),
            w_fmt="bf8",
            in_=row(H),
            out=row(SH_INT),
            work=row(2 * SH_INT, FP32),
            kind="mm",
            M=ROWS,
            N=2 * SH_INT,
            K=H,
        )
    )
    u.append(
        dict(
            unit=f"{t}.shared down_proj [H, 640] x sigmoid(shared_expert_gate [1, H])",
            w=mm(H, SH_INT, BF8) + mm(1, H, BF16),
            w_fmt="bf8",
            in_=row(SH_INT),
            out=row(H, FP32),
            work=0,
            kind="mm",
            M=ROWS,
            N=H,
            K=SH_INT,
        )
    )
    u.append(
        dict(
            unit=f"{t}.routed expert e: gate_up [2*640, H] BF4 (silu(gate) x up), x10 per row (distinct per census)",
            w=mm(2 * E_INT, H, BF4),
            w_fmt="bf4",
            in_=row(H),
            out=row(E_INT),
            work=row(2 * E_INT, FP32),
            kind="mm_expert",
            M=1,
            N=2 * E_INT,
            K=H,
            per_expert=True,
        )
    )
    u.append(
        dict(
            unit=f"{t}.routed expert e: down [H, 640] BF4 x score, index_add into [rows, H]",
            w=mm(H, E_INT, BF4),
            w_fmt="bf4",
            in_=row(E_INT),
            out=row(H, FP32),
            work=row(H, FP32),
            kind="mm_expert",
            M=1,
            N=H,
            K=E_INT,
            per_expert=True,
        )
    )
    u.append(
        dict(
            unit=f"{t}.routed + shared -> block output [rows, H]",
            w=0,
            w_fmt="-",
            in_=2 * row(H, FP32),
            out=row(H),
            work=0,
            kind="elt",
        )
    )
    u += hc_write(t)
    return u


def ple():
    """checkpoint layers.1.ple (config ple_layer_ids[0]-1 = applied before layer 2's blocks): 16 n-gram heads x 160 = 2560
    embedding from the hashed last-2 tokens (128 shard tables x 2,500,012 x 160 bf16 in DRAM), key_proj [4H, 2560] + norm_key
    (4 groups of H), query = norm_query(residual) (4 groups), gate = <key, query>/sqrt(H) per branch -> signed sqrt -> sigmoid,
    value_proj [H, 2560] x gate per branch -> norm_conv -> depthwise conv1d (4 tap, dilation 3: tap state 9 rows x 4H) silu;
    output = gated value + conv -> residual +="""
    t = "L02.ple"
    return [
        dict(
            unit=f"{t}.n-gram embedding lookup (16 heads x 160 from 128 hashed shard tables in DRAM) -> [rows, 2560]",
            w=0,
            w_fmt="table",
            in_=0,
            out=row(PLE_DIM),
            work=0,
            kind="lookup",
            dram_row=PLE_HEADS * PLE_HEAD_DIM * BF16,
        ),
        dict(
            unit=f"{t}.key_proj [4H, 2560] bf16 + norm_key (4 groups of H)",
            w=mm(HCD, PLE_DIM, BF16) + HCD * BF16,
            w_fmt="bf16",
            in_=row(PLE_DIM),
            out=row(HCD),
            work=0,
            kind="mm",
            M=ROWS,
            N=HCD,
            K=PLE_DIM,
        ),
        dict(
            unit=f"{t}.norm_query(residual) (4 groups); gate = <key,query>/sqrt(H) per branch, signed sqrt, sigmoid",
            w=HCD * BF16,
            w_fmt="bf16",
            in_=row(HCD) + row(HCD),
            out=row(HC, FP32),
            work=0,
            kind="elt",
        ),
        dict(
            unit=f"{t}.value_proj [H, 2560] bf16; gated value = sigmoid(gate[b]) x value (4 branches)",
            w=mm(H, PLE_DIM, BF16),
            w_fmt="bf16",
            in_=row(PLE_DIM) + row(HC, FP32),
            out=row(HCD),
            work=0,
            kind="mm",
            M=ROWS,
            N=H,
            K=PLE_DIM,
        ),
        dict(
            unit=f"{t}.norm_conv (4 groups) -> depthwise conv1d (4 tap, dilation 3; tap state 9 rows x 4H) silu; residual += gated value + conv",
            w=HCD * BF16 + HCD * PLE_CONV_K * BF16,
            w_fmt="bf16",
            in_=row(HCD),
            out=row(HCD),
            work=0,
            kind="elt",
            state_l1=HCD * (PLE_CONV_K - 1) * PLE_DIL * BF16,
        ),
    ]


def tail(k=1):
    u = [
        dict(
            unit="final hyper_connection_mixer: gr.norm [4H] bf16 + down [320, 4H] + up [4H, 320] bf8 (dense plan final_mixer) -> [rows, H]",
            w=2 * mm(HC_LOW, HCD, BF8) + HCD * BF16,
            w_fmt="bf8",
            in_=row(HCD),
            out=row(H),
            work=row(HC_LOW, FP32),
            kind="mm",
            M=ROWS,
            N=HC_LOW,
            K=HCD,
        ),
        dict(
            unit=f"lm_head [{V}, H] bf8 -> logits [rows, V]",
            w=mm(V, H, BF8),
            w_fmt="bf8",
            in_=row(H),
            out=row(V, FP32),
            work=row(V, FP32),
            kind="mm",
            M=ROWS,
            N=V,
            K=H,
        ),
        dict(
            unit="argmax per row (greedy) -> token ids; MTP verify: accept the draft row if it matches",
            w=0,
            w_fmt="-",
            in_=row(V, FP32),
            out=ROWS * 4,
            work=0,
            kind="elt",
        ),
    ]
    if k:
        u += [
            dict(
                unit="mtp.hyper_connection_mixer: gr.norm [4H] bf16 + down [320, 4H] + up [4H, 320] bf8 (dense plan final_mixer) on the backbone residual -> [rows, H]; pre_fc_norm_hidden [4H]",
                w=2 * mm(HC_LOW, HCD, BF8) + 2 * HCD * BF16,
                w_fmt="bf8",
                in_=row(HCD),
                out=row(H),
                work=row(HC_LOW, FP32),
                kind="mm",
                M=ROWS,
                N=HC_LOW,
                K=HCD,
            ),
            dict(
                unit="mtp.embed_tokens(draft token) -> pre_fc_norm_embedding [H]; fc_embedding [H, H] + fc_hidden [H, H] bf8 -> hc init (4 copies)",
                w=2 * mm(H, H, BF8) + H * BF16,
                w_fmt="bf8",
                in_=row(H) + row(H),
                out=row(HCD),
                work=0,
                kind="mm",
                M=ROWS,
                N=2 * H,
                K=H,
                dram_row=H * BF16,
            ),
        ]
        u += [dict(u_, unit=u_["unit"].replace("L48.", "mtp.")) for u_ in qsa(48) + moe(48)]
        u += [
            dict(
                unit="mtp.lm_head (shared weight) -> draft logits -> argmax -> draft token for the next pass",
                w=0,
                w_fmt="(lm_head)",
                in_=row(H),
                out=row(V, FP32),
                work=row(V, FP32),
                kind="mm",
                M=ROWS,
                N=V,
                K=H,
            )
        ]
    return u


def all_units():
    u = [
        dict(
            unit="embed_tokens (table in DRAM, [rows, H] bf16) -> hyper-connection init (4 copies)",
            w=0,
            w_fmt="table",
            in_=0,
            out=row(HCD),
            work=0,
            kind="lookup",
            dram_row=H * BF16,
        )
    ]
    for l in range(L):
        if l == PLE_LAYER:
            u += ple()
        u += (gdn(l) if LAYER[l] == "gdn" else qsa(l)) + moe(l)
    u += tail(1)
    return u


if __name__ == "__main__":
    U = all_units()
    dense = sum(x["w"] for x in U if x["w_fmt"] not in ("bf4", "table", "(router)", "(lm_head)"))
    expert = mm(2 * E_INT, H, BF4) + mm(H, E_INT, BF4)
    S_layer = GDN_V * GDN_HD * GDN_HD * FP32
    conv_layer = (2 * GDN_QK * GDN_HD + GDN_V * GDN_HD) * (CONV_K - 1) * BF16
    print(
        f"units {len(U)}; dense weights {dense/MiB:,.1f} MiB ({dense/1e9:.3f} GB); expert {expert/MiB:.2f} MiB x 512 x 49 = {expert*512*49/1e9:.2f} GB"
    )
    print(
        f"per user: GDN state S {36*S_layer/MiB:.1f} MiB fp32 (read+write per pass {2*36*S_layer/MiB:.1f}), conv taps {36*conv_layer/KiB:.0f} KiB, PLE conv {HCD*3*BF16/KiB:.0f} KiB, KV per token {12*(2*QSA_KV*QSA_HD*BF16+IDX_KV*IDX_HD*BF16/IDX_COMP)/KiB:.1f} KiB"
    )

    # one GDN layer, one QSA layer, one MoE block
    def show(units, title):
        print(f"\n### {title}\n")
        print("| unit (HF) | weight [KiB] | fmt | in [KiB] | out [KiB] | work [KiB] | kind | shape M,N,K |")
        print("|---|---|---|---|---|---|---|---|")
        for x in units:
            shp = f"{x['M']},{x['N']},{x['K']}" if "M" in x else "-"
            print(
                f"| `{x['unit']}` | {x['w']/KiB:,.1f} | {x['w_fmt']} | {x['in_']/KiB:.1f} | {x['out']/KiB:.1f} | {x['work']/KiB:.1f} | {x['kind']} | {shp} |"
            )
        print(
            f"| **sum** | **{sum(x['w'] for x in units)/KiB:,.1f}** | | {sum(x['in_'] for x in units)/KiB:.1f} | {sum(x['out'] for x in units)/KiB:.1f} | {sum(x['work'] for x in units)/KiB:.1f} | | |"
        )

    show(gdn(0), "GDN layer (L00), one of 36")
    show(qsa(3), "QSA layer (L03), one of 12")
    show(moe(0), "MoE block (every layer), one of 48 (+ MTP)")
    show(ple(), "PLE (after layer 2)")
    show(tail(1), "tail: final mixer, lm_head, argmax, MTP layer (k=1)")
    json.dump(
        dict(rows=ROWS, units=U, dense_bytes=dense, expert_bytes=expert, S_layer=S_layer, conv_layer=conv_layer),
        open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step0_units.json", "w"),
    )
