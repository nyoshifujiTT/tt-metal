#!/usr/bin/env python
r"""Step 6 of the detailed design (worklog 12.6, third version): every communication of the chosen form
(DP1 x PP4(R2) x 8chip[TP8], b=1, rows 2), enumerated and timed with the packet model of comm_model.py.
Topology: a group = a 2x4 block of the 8x4 torus (rows 2g, 2g+1; the 4 columns form a ring); hand-offs go to the
next block through the inter-group links (2 hops: chip (r, c) -> (r, c) of the next block; the column's two chips
share the inter-group link).
A hyper-connection read is three choices (worklog 12.6 v3):
  (1) the block output y = sum_i y_i (K-split out_proj / o_proj / experts' down): residual split by columns (form A)
      -> reduce-scatter RS_y; residual replicated on the 8 chips (form B) -> all-reduce AR_y (AG+local: every chip
      adds the 8 partials and writes the residual locally).  The partials are produced in WAVES waves during the
      matmul and sent as they are produced: exposure = the last wave's meeting.
  (2) the low-rank D n (n = scale x (1+w) x R, R = R_start + inj (x) y): D K-split (A: partials AR_down, whole
      exposure 2.24 us) / D N-split (B only: chip i computes its 40 rows from the whole row -> AG_low 320 B) / D
      replicated (no meeting, +6.25 MiB per block on the DRAM lane).  In form B the R_start term is known at the
      previous block's start; the y term rides AR_y as f_i = (D diag(1+w) O_i) h_i with the fold weight F_i
      [1280, 768] bf16 1.9 MiB per chip per attention block (the MoE output has no fold: per-expert K-split).
  (3) the gate U (low-rank -> [2, 4H]) and x: U N-split (chip i's columns -> AG_x of x 1/8, consumed chunk by chunk
      by the projections: exposure 0.19-0.58 us) / U replicated (no meeting, +6.25 MiB per block).
The speculative routers (2 per attention block) use their own hc reads off the chain (their meetings go in the
chain's waiting time); the real read's meetings are not shared with them.
Output: the meeting table, the exposures per block for forms A, B, B+D (D replicated), B+D+U, the hand-offs, the
other rows; step6_comm.json for step 7."""
import json
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import comm_model as cm
import mm_cost

MiB = 1024**2
KiB = 1024
H = 2560
HC = 4
HC_LOW = 320
HCD = HC * H
BF16 = 2
FP32 = 4
TP = 8
ROWS = 2
CLK = 1.35e9
mesh = cm.Mesh(2, 4)


def row(n, dt=BF16):
    return ROWS * n * dt


P_down = row(HC * (HC_LOW + HC + 1), FP32)  # per-branch partials: 4 x (320 + 4 + 1) fp32
P_low = row(HC_LOW / TP, FP32)  # chip i's 40 rows of the low-rank (fp32)
P_x = row(H) / TP
P_y = row(H)
P_f = row(HC * HC_LOW)
P_ids = ROWS * 10 * 2
P_tok = ROWS * 4
P_arg = ROWS * 8 * TP
# the matmul times that hide a meeting: read from step 4 (one source for every unit cost, revision 6)
T4 = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step4_item_time.json"))
# correction 43: with step 4p of a form (argv 3), the matmul times are the placed ones (worst harvest) of that form
P4 = json.load(open(sys.argv[3]))["units"] if len(sys.argv) > 3 else None
if P4 is not None:
    T4 = P4


def t4(prefix, field="mm_us"):
    return next(r[field] for r in T4 if r["unit"].startswith(prefix))


MM_PROJ_GDN = t4("L00.gdn.in_proj_qkv")
MM_PROJ_QSA = t4("L03.qsa.indexer qk")
MM_ROUTER = t4("L00.moe.router")
MM_OUT = t4("L00.gdn.out_proj")
MM_DOWN_EXP = t4("L00.moe.routed expert e: down")
MM_LOW = t4("L00.gdn.gr.down")  # the router pool's matmul part only (the top-10 cannot overlap the all-gather: it needs the logits)
TREE = mm_cost.tree_level(ROWS) / CLK * 1e6 * 2
WAVES = 10
out = {}


def meeting(name, kind, P):
    f = cm.all_reduce if kind == "AR" else cm.all_gather if kind == "AG" else cm.reduce_scatter
    t, form, forms = f(mesh, P)
    out[name] = dict(bytes=P, kind=kind, us=t * 1e6, form=form, forms={k: v * 1e6 for k, v in forms.items()})
    return t * 1e6


def ag_exposure(P, t_mm):
    arr = cm.ag_arrivals(mesh, P, (0, 0))
    end = cm.consume_overlap([a * 1e6 for a in arr], t_mm, TREE)
    return max(0.0, end - (t_mm + TREE)), end


def wave_exposure(kind, P, t_mm):
    """a meeting of P bytes whose payload is produced in WAVES waves during a matmul of t_mm us (wave w ready at
    t_mm x w / WAVES): all waves are simulated on the links with their ready times (the fastest form of the whole
    meeting is used wave by wave); the exposure is the completion minus t_mm"""
    f = cm.all_reduce if kind == "AR" else cm.reduce_scatter
    _, form, _ = f(mesh, P)
    fl = []
    for w in range(1, WAVES + 1):
        t0 = t_mm * w / WAVES * 1e-6
        Pw = P / WAVES
        if form.startswith("AG_tree"):
            fl += [(t0, root, mesh.bfs_tree(root), Pw, (root, w)) for root in mesh.chips]
        elif form.startswith("AG_direct") or form == "RS_direct":
            Pd = Pw if form.startswith("AG_direct") else Pw / mesh.n
            k = 0
            for s in mesh.chips:
                for d in mesh.chips:
                    if s != d:
                        fl.append((t0, s, cm.path_tree(mesh, s, d, k), Pd, (s, d, w)))
                        k += 1
        else:  # RS_tree / RS_tree+AG_tree: approximate by the direct forms' wave pipeline (never the fastest here)
            Pd = Pw / mesh.n
            k = 0
            for s in mesh.chips:
                for d in mesh.chips:
                    if s != d:
                        fl.append((t0, s, cm.path_tree(mesh, s, d, k), Pd, (s, d, w)))
                        k += 1
    done = cm.simulate_mc(fl)[0] * 1e6 + cm.ADD_LEVEL * 1e6
    return max(0.0, done - t_mm), form


# ---- compute of the hyper-connection read's matmuls per split (step 4's cost model, mm_cost.py) ----
def mm_c(M, N, K, fmt="bf16"):
    return mm_cost.one(M, N, K, fmt) / CLK * 1e6


def mm_pool(*mms):
    return mm_cost.pool(*[(x[0], x[1], x[2], x[3] if len(x) > 3 else "bf16") for x in mms]) / CLK * 1e6


DOWN_K = mm_pool((ROWS, HC_LOW, HCD / TP), (ROWS, HC, HCD / TP))  # A: D K-split 1/8 (+ inject K-split, same input)
DOWN_N = mm_pool(
    (ROWS, HC_LOW / TP, HCD), (ROWS, HC, HCD)
)  # B: D N-split 40 rows from the whole row (+ inject, replicated 80 KiB: local)
DOWN_R = mm_pool((ROWS, HC_LOW, HCD), (ROWS, HC, HCD))  # D replicated
UP_N = mm_c(ROWS, HCD / TP, HC_LOW)
UP_R = mm_c(ROWS, HCD, HC_LOW)
OUT_PLAIN = mm_c(ROWS, H, 6 * 128, "bf8")
OUT_FOLD = mm_pool(
    (ROWS, H, 6 * 128, "bf8"), (ROWS, HC * HC_LOW, 6 * 128, "bf16")
)  # out_proj K-split 6 heads, with the fold F_i h_i pooled
if P4 is not None:
    # correction 43: the placed times of this form's units (the form decides which split the unit's pool is)
    FORM6 = json.load(open(sys.argv[3]))["form"]
    _down = t4("L00.gdn.gr.down")
    _up = t4("L00.gdn.gr.up")
    _out = t4("L00.gdn.out_proj")
    if FORM6 == "A":
        DOWN_K, UP_N, OUT_PLAIN = _down, _up, _out
    elif FORM6 in ("B",):
        DOWN_K, UP_N, OUT_FOLD = _down, _up, _out
    elif FORM6 == "Bn":
        DOWN_N, UP_N, OUT_FOLD = _down, _up, _out
    elif FORM6 == "B+D":
        DOWN_R, UP_N, OUT_FOLD = _down, _up, _out


def main():
    print(
        "## meetings on the 2x4 block (8 chips, rows 2): payload per chip, fastest form, time [us] (last byte at the farthest chip)\n"
    )
    print("| name | kind | payload/chip [B] | fastest form | time [us] | other forms [us] |")
    print("|---|---|---|---|---|---|")
    for nm, kind, P in (
        ("AR_down", "AR", P_down),
        ("AG_low", "AG", P_low),
        ("AG_x", "AG", P_x),
        ("RS_y", "RS", P_y),
        ("RS_y_wave", "RS", P_y / WAVES),
        ("AR_y", "AR", P_y),
        ("AR_y_wave", "AR", P_y / WAVES),
        ("AR_yf", "AR", P_y + P_f),
        ("AR_yf_wave", "AR", (P_y + P_f) / WAVES),
        ("AG_ids", "AG", P_ids),
        ("AG_argmax", "AG", P_arg),
    ):
        t = meeting(nm, kind, P)
        o = out[nm]
        print(
            f"| {nm} | {kind} | {P:,.0f} | {o['form']} | {t:.3f} | {', '.join(f'{k} {v:.3f}' for k,v in o['forms'].items() if k!=o['form'])} |"
        )
    ag_gdn, e1 = ag_exposure(P_x, MM_PROJ_GDN)
    ag_qsa, e2 = ag_exposure(P_x, MM_PROJ_QSA)
    ag_moe, e3 = ag_exposure(P_x, MM_ROUTER)
    ag_low = out["AG_low"]["us"]
    ar_down_full = out["AR_down"]["us"]
    ar_down, _ = wave_exposure("AR", P_down, DOWN_K)  # the K-split partials leave in waves during the down matmul
    rs_attn, _ = wave_exposure("RS", P_y, OUT_PLAIN)
    rs_moe, _ = wave_exposure("RS", P_y, MM_DOWN_EXP)
    ary_attn, _ = wave_exposure("AR", P_y, OUT_PLAIN)
    aryf_attn, _ = wave_exposure("AR", P_y + P_f, OUT_FOLD)
    ary_moe, _ = wave_exposure("AR", P_y, MM_DOWN_EXP)
    print(
        f"\nwaves ({WAVES}): the output partials are sent wave by wave during the matmul; exposure beyond the matmul: RS_y attn {rs_attn:.2f} / moe {rs_moe:.2f}, AR_y attn {ary_attn:.2f} / moe {ary_moe:.2f}, AR_yf attn {aryf_attn:.2f} (whole meetings: RS_y {out['RS_y']['us']:.2f}, AR_y {out['AR_y']['us']:.2f}, AR_yf {out['AR_yf']['us']:.2f})"
    )
    print(
        f"hc read matmuls on one chip [us]: down K-split 1/8 {DOWN_K:.2f} / N-split 40 rows {DOWN_N:.2f} / replicated {DOWN_R:.2f}; up N-split {UP_N:.2f} / replicated {UP_R:.2f}; out_proj alone {OUT_PLAIN:.2f} / with the fold F_i h_i pooled {OUT_FOLD:.2f}"
    )
    forms = {}
    agx = dict(gdn=ag_gdn, qsa=ag_qsa, moe=ag_moe)

    # per block: the chain segment from the block's output matmul to the next block's projections:
    #   out = output matmul (+ fold) ; meet_out = its meeting's exposure ; down ; meet_low ; up ; meet_x (AG_x exposure, per consumer)
    def block(out_mm, meet_out, down, meet_low, up):
        return dict(out=out_mm, meet_out=meet_out, down=down, meet_low=meet_low, up=up)

    forms["A"] = dict(
        desc="column-split residual: D K-split (AR_down), U N-split (AG_x), output RS_y (waves)",
        extra_MiB=0.0,
        attn=block(OUT_PLAIN, rs_attn, DOWN_K, ar_down, UP_N),
        moe=block(MM_DOWN_EXP, rs_moe, DOWN_K, ar_down, UP_N),
    )
    forms["B"] = dict(
        desc="replicated residual: outputs AR_y (waves); the attention output carries the fold F_i h_i and the MoE read's g_b (D K-split at the attention block's start, off the chain) rides the same all-reduce, so the MoE read's low-rank is on the chain only through that all-reduce; the attention read's D K-split (AR_down waves), U N-split (AG_x)",
        extra_MiB=(HC * HC_LOW * 768 * BF16 - HC_LOW * HCD * BF16 / TP) / MiB,
        attn=block(OUT_FOLD, aryf_attn, 0.0, 0.0, UP_N),
        moe=block(MM_DOWN_EXP, ary_moe, DOWN_K, ar_down, UP_N),
    )
    forms["Bn"] = dict(
        desc="B with the attention read's D N-split (40 rows from the whole row, AG_low) instead of K-split",
        extra_MiB=(HC * HC_LOW * 768 * BF16 - HC_LOW * HCD * BF16 / TP) / MiB,
        attn=block(OUT_FOLD, aryf_attn, 0.0, 0.0, UP_N),
        moe=block(MM_DOWN_EXP, ary_moe, DOWN_N, ag_low, UP_N),
    )
    forms["B0"] = dict(
        desc="B without the fold (both reads D K-split with AR_down waves)",
        extra_MiB=0.0,
        attn=block(OUT_PLAIN, ary_attn, DOWN_K, ar_down, UP_N),
        moe=block(MM_DOWN_EXP, ary_moe, DOWN_K, ar_down, UP_N),
    )
    forms["B+D"] = dict(
        desc="B with the attention read's D replicated (no meeting; down on the whole row; +6.25 MiB per layer per chip)",
        extra_MiB=(HC * HC_LOW * 768 * BF16 - HC_LOW * HCD * BF16 / TP) / MiB
        + (HC_LOW * HCD * BF16 - HC_LOW * HCD * BF16 / TP) / MiB,
        attn=block(OUT_FOLD, aryf_attn, 0.0, 0.0, UP_N),
        moe=block(MM_DOWN_EXP, ary_moe, DOWN_R, 0.0, UP_N),
    )
    forms["B+D+U"] = dict(
        desc="B+D with both reads' U replicated (no AG_x; +12.5 MiB per layer per chip)",
        extra_MiB=(HC * HC_LOW * 768 * BF16 - HC_LOW * HCD * BF16 / TP) / MiB
        + (HC_LOW * HCD * BF16 - HC_LOW * HCD * BF16 / TP) / MiB * 3,
        attn=block(OUT_FOLD, aryf_attn, 0.0, 0.0, UP_R),
        moe=block(MM_DOWN_EXP, ary_moe, DOWN_R, 0.0, UP_R),
    )
    print(
        "\n## the hyper-connection read on the chain, per block [us]: output matmul + its meeting + down + low-rank meeting + up + AG_x exposure (next block's projections); extra DRAM per chip per pass [MiB]\n"
    )
    print(
        "| form | attention block (GDN / QSA layer) | MoE block (next: GDN / QSA) | GDN layer | QSA layer | per pass [us] | extra DRAM/chip/pass [MiB] | what |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for k, f in forms.items():
        a = f["attn"]
        m = f["moe"]
        a_tot = (
            a["out"] + a["meet_out"] + a["down"] + a["meet_low"] + a["up"] + (0.0 if k in ("B+D+U",) else agx["moe"])
        )  # the attention output feeds the MoE read -> AG_x consumed by the router pool
        m_gdn = (
            m["out"] + m["meet_out"] + m["down"] + m["meet_low"] + m["up"] + (0.0 if k in ("B+D+U",) else agx["gdn"])
        )
        m_qsa = (
            m["out"] + m["meet_out"] + m["down"] + m["meet_low"] + m["up"] + (0.0 if k in ("B+D+U",) else agx["qsa"])
        )
        f["gdn"] = a_tot + m_gdn
        f["qsa"] = a_tot + m_qsa
        f["pass_us"] = 36 * f["gdn"] + 13 * f["qsa"]
        f["pass_MiB"] = f["extra_MiB"] * 49 / 4
        f["attn_total"] = a_tot
        f["moe_gdn"] = m_gdn
        f["moe_qsa"] = m_qsa
        f["agx"] = agx
        print(
            f"| {k} | {a['out']:.2f}+{a['meet_out']:.2f}+{a['down']:.2f}+{a['meet_low']:.2f}+{a['up']:.2f}+{0.0 if k=='B+D+U' else agx['moe']:.2f} = {a_tot:.2f} | {m['out']:.2f}+{m['meet_out']:.2f}+{m['down']:.2f}+{m['meet_low']:.2f}+{m['up']:.2f}+{0.0 if k=='B+D+U' else agx['gdn']:.2f}/{0.0 if k=='B+D+U' else agx['qsa']:.2f} = {m_gdn:.2f}/{m_qsa:.2f} | {f['gdn']:.2f} | {f['qsa']:.2f} | {f['pass_us']:.0f} | {f['pass_MiB']:+.1f} | {f['desc']} |"
        )
    base = forms["A"]["pass_us"]
    print(
        "\nagainst form A per pass: "
        + ", ".join(
            f"{k} {f['pass_us']-base:+.0f} us for {f['pass_MiB']:+.1f} MiB" for k, f in forms.items() if k != "A"
        )
    )
    print(
        f"exposures used: AR_down {ar_down:.2f}, AG_low {ag_low:.2f}, AG_x consumed by the K chunks: GDN {ag_gdn:.2f} / QSA {ag_qsa:.2f} / MoE {ag_moe:.2f}, RS_y waves: attn {rs_attn:.2f} / MoE {rs_moe:.2f}, AR_y waves: attn {ary_attn:.2f} / MoE {ary_moe:.2f}, AR_yf waves {aryf_attn:.2f}"
    )
    # hand-offs
    print(
        "\n## hand-offs between groups (chip i -> chip i of the next block, 2 hops; the column's two chips share the inter-group link)\n"
    )
    print("| cut (step 3) | live rows per chip | bytes/chip | time [us] |")
    print("|---|---|---|---|")
    res_cols = row(HCD) / TP
    cuts = {
        0: ("frame 7 -> 0 (pass wrap)", "token ids (2 x 4 B); group 0 reads the embedding rows itself", P_tok),
        1: (
            "L06.gdn.norm (gated) | out_proj",
            "residual columns 1/8 (form A; B: none, the next group holds it after AR_y) + gated-norm rows of the chip's 6 value heads [2,768] bf16 + inject [2,4] fp32",
            res_cols + row(6 * 128) + row(HC, FP32),
        ),
        2: (
            "L13.gdn.gr.inject | in_proj_qkv",
            "residual 1/8 + block input x [2,H] bf16 + inject",
            res_cols + row(H) + row(HC, FP32),
        ),
        3: (
            "L19.moe.shared down | routed gate_up",
            "residual 1/8 + x [2,H] + router result (10 x (score fp32 + id int16) x 2 rows) + shared gate/up rows [2,80]x2 + inject",
            res_cols + row(H) + ROWS * 10 * 6 + row(2 * 80) + row(HC, FP32),
        ),
        4: (
            "L26.moe.gr.down | gr.up",
            "residual 1/8 + low-rank [2,320] fp32 + normalized residual columns 1/8 + inject",
            res_cols + row(HC_LOW, FP32) + res_cols + row(HC, FP32),
        ),
        5: (
            "L33.gdn.conv1d | in_proj_z",
            "residual 1/8 + x [2,H] + conv output rows of the chip's 1280 columns + inject",
            res_cols + row(H) + row(1280) + row(HC, FP32),
        ),
        6: (
            "L39.moe.routed gate_up | routed down",
            "residual 1/8 + expert rows of the chip's 43.9 tiles [2, 43.9x32] bf16 + router scores + inject",
            res_cols + ROWS * 43.87 * 32 * BF16 + ROWS * 10 * 6 + row(HC, FP32),
        ),
        7: (
            "L46.moe.shared gate/up | shared down",
            "residual 1/8 + x [2,H] + shared rows [2,80] + router result + inject",
            res_cols + row(H) + row(80) + ROWS * 10 * 6 + row(HC, FP32),
        ),
    }
    hand = []
    big = cm.Mesh(4, 4)
    for k in range(8):
        nm, desc, b = cuts[k]
        fl = [(0.0, (0, c), cm.path_tree(big, (0, c), (2, c)), b, (0, c)) for c in range(4)] + [
            (0.0, (1, c), cm.path_tree(big, (1, c), (3, c)), b, (1, c)) for c in range(4)
        ]
        t = cm.hand_over(fl, b) * 1e6  # revision 8: delivered to the receiving chip's cores
        hand.append(dict(cut=k, desc=nm, bytes=b, us=t))
        print(f"| {nm} | {desc} | {b:,.0f} | {t:.3f} |")
    res_hop = cm.hand_over([(0.0, (1, 0), cm.path_tree(big, (1, 0), (3, 0)), res_cols, (1, 0))], res_cols) * 1e6
    print(
        f"\nhand-offs per pass: {sum(h['us'] for h in hand):.2f} us (8 cuts); forms B: the next group needs the whole residual [2,4H]: the sender sends its columns 1/8 as in A and the receiving group all-gathers them (AG of 5 KiB per chip: +{cm.all_gather(mesh,res_cols)[0]*1e6:.2f} us per cut)"
    )
    out["hand_B_extra"] = cm.all_gather(mesh, res_cols)[0] * 1e6
    print(
        "\n## other rows\n- speculative routers: 2 hc reads per attention block off the chain (their meetings: form A AR_down "
        + f"{ar_down:.2f} + AG_x {out['AG_x']['us']:.2f}; form B AG_low {ag_low:.2f} + AG_x {out['AG_x']['us']:.2f}) + router 1.33 us; for the first MoE blocks of a frame the previous group sends the residual columns 1/8 (5 KiB, 2 hops {res_hop:.3f} us) to the next (idle) group, which runs the read with its own weights"
    )
    print(
        f"- argmax: local on each chip's V/8 columns, candidates (value, index) x 2 rows all-gathered on the last group: {out['AG_argmax']['us']:.3f} us; token ids to group 0 (pass wrap): {hand[0]['us']:.3f} us"
    )
    print(
        "- no collective: the router (replicated), the kv projection of a QSA layer (replicated on the 4 chips of a kv group), the indexer (replicated), the GDN state S (the chip's own DRAM)"
    )
    out.update(
        hand=hand,
        forms=forms,
        parts=dict(
            ar_down=ar_down,
            ag_low=ag_low,
            ag_gdn=ag_gdn,
            ag_qsa=ag_qsa,
            ag_moe=ag_moe,
            rs_attn=rs_attn,
            rs_moe=rs_moe,
            ary_attn=ary_attn,
            ary_moe=ary_moe,
            aryf_attn=aryf_attn,
            res_hop=res_hop,
            ids_hop=cm.hand_over([(0.0, (1, 0), cm.path_tree(big, (1, 0), (3, 0)), ROWS * 10 * 2, (1, 0))], ROWS * 10 * 2) * 1e6,
            ar_down_full=ar_down_full,
            spec_meet_A=ar_down_full + out["AG_x"]["us"],
            spec_meet_B=ag_low + out["AG_x"]["us"],
            down_k=DOWN_K,
            down_n=DOWN_N,
            down_r=DOWN_R,
            up_n=UP_N,
            up_r=UP_R,
            out_plain=OUT_PLAIN,
            out_fold=OUT_FOLD,
        ),
        waves=WAVES,
    )
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step6_comm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
