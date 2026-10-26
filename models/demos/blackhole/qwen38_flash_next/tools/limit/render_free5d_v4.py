#!/usr/bin/env python
"""Render README section 11 (revision 4) from routing/free5d_v4_results*.json and the census. Every number is computed; deterministic."""
import json
import math
import sys

_argv = list(sys.argv)
sys.argv = ["render"]
sys.argv = ["/data/yoshifuji/qwen38/scripts/free5d_v4.py"]
exec(
    open("/data/yoshifuji/qwen38/scripts/free5d_v4.py")
    .read()
    .replace('if __name__=="__main__": main(*sys.argv[1:])', "")
)
R = json.load(open("/data/yoshifuji/qwen38/routing/free5d_v4_results.json"))
CTXS = (1, 128, 8192, 32768, 262144, 1000000)
RC = {
    c: json.load(open(f"/data/yoshifuji/qwen38/routing/free5d_v4_results_ctx{c}.json")) for c in CTXS
}  # H-26: the context sweep (k=1)
cfgdict_axes = {r["cfg"]: r["axes"] for v in R.values() for r in v}
S = json.load(open("/data/yoshifuji/qwen38/routing/stats.json"))
routing = Routing("/data/yoshifuji/qwen38/routing/stats.json", "/data/yoshifuji/qwen38/routing/stats_stage2.json")
S2 = json.load(open("/data/yoshifuji/qwen38/routing/stats_stage2.json"))
CASES = [
    ("hit", "全ヒット: routed を router の前に読み終える（2 段先読みが全部当たる）"),
    ("census", "census ヒット率: 2 段先読み（2 ブロック早い予測 P2、1 ブロック早い予測 P1、外れは router 後）の実測集合、無駄読み込み"),
    ("miss", "全ミス: router の後に読む（今の runtime の形）"),
    ("all", "全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない"),
]


def rs(case, k):
    return R[f"{case}|k={k}"]


def feas(l):
    return [r for r in l if r["feasible"]]


def best(case, k, key):
    return max(feas(rs(case, k)), key=lambda r: r[key])


def get(case, k, **ax):
    ax = {"DP": 1, "PP": 1, "DP_a": 1, "CP": 1, "n_mb": 1, **ax}
    ax.setdefault("EP", 1)
    ax.setdefault("TP_x", ax["TP_a"] * ax["DP_a"] // ax["EP"])
    return next(r for r in rs(case, k) if all(A(r)[key] == val for key, val in ax.items()))


def A(r):
    return r["axes"]


def impl(r, k):
    a = A(r)
    base = a["PP"] == 1 and a["DP_a"] == 1 and a["TP_a"] in (4, 8) and a["EP"] == a["TP_a"] and a["CP"] == 1
    return base and (a["b"] == 1 if k == 0 else a["b"] * (k + 1) <= 32)


def bestimpl(case, k, key):
    return max([r for r in feas(rs(case, k)) if impl(r, k)], key=lambda r: r[key])


def par(case, k):
    return pareto(rs(case, k))


def bval(c):
    return int(c.split("b=")[1].split()[0])


def rsc(ctx, case, k=1):
    return RC[ctx][f"{case}|k={k}"]


def bestc(ctx, case, key):
    return max([r for r in rsc(ctx, case) if r["feasible"]], key=lambda r: r[key])


def parc(ctx, case):
    return pareto(rsc(ctx, case))


def bestbc(ctx, b, case="hit"):
    l = [r for r in rsc(ctx, case) if r["feasible"] and A(r)["b"] == b]
    return max(l, key=lambda r: r["tsu"]) if l else None


def table_ctx():
    o = [
        "| ctx | tsu 最大（全ヒット） | 構成 | T_tok [ms]; DRAM/pass / 演算 / 通信 | tsu 最大（census / 全ミス） | agg 最大（全ヒット） | 構成 | users | T_pass [ms] | agg 最大（census / 全ミス） | DP1 で容量に入る最大 b |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in CTXS:
        t = bestc(c, "hit", "tsu")
        m = bestc(c, "miss", "tsu")
        cs = bestc(c, "census", "tsu")
        a = bestc(c, "hit", "agg")
        am = bestc(c, "miss", "agg")
        ac = bestc(c, "census", "agg")
        bmax = max(A(r)["b"] for r in rsc(c, "hit") if r["feasible"] and A(r)["DP"] == 1)
        o.append(
            f"| {c:,} | {t['tsu']:,.0f} | `{t['cfg']}` | {t['T_tok_ms']:.3f}; {t['dram_ms']:.3f} / {t['comp_ms']:.3f} / {t['comm_ms']:.3f} | {cs['tsu']:,.0f} / {m['tsu']:,.0f} | {a['agg']:,.0f} | `{a['cfg']}` | {a['users']} | {a['slot_ms']:.1f} | {ac['agg']:,.0f} / {am['agg']:,.0f} | {bmax:,} |"
        )
    return "\n".join(o)


def table_ctx_b():
    bs = (1, 4, 16, 64, 256, 1024)
    o = ["| ctx \\ b | " + " | ".join(str(b) for b in bs) + " |", "|---|" + "---|" * len(bs)]
    for c in CTXS:
        cells = []
        for b in bs:
            x = bestbc(c, b)
            cells.append(f"{x['tsu']:,.0f} `{x['cfg'].split(' b=')[0]}`" if x else "容量超過")
        o.append(f"| {c:,} | " + " | ".join(cells) + " |")
    return "\n".join(o)


def cfgdict(s):
    return dict(cfgdict_axes[s])


def tp_rows(b, case="hit", k=1):
    out = []
    for TP in (4, 8, 16, 32):
        cands = [r for r in feas(rs(case, k)) if A(r)["TP_a"] == TP and A(r)["b"] == b and A(r)["DP_a"] == 1]
        ep1 = [r for r in cands if A(r)["EP"] == 1 and A(r)["CP"] == 1 and A(r)["PP"] == 1][0]
        bt = max(cands, key=lambda r: r["tsu"])
        out.append((TP, ep1, bt))
    return out


def ppb1(PP, TP, R=None):
    if R is None:
        return max(
            [
                r
                for r in feas(rs("hit", 1))
                if A(r)["DP"] == 1 and A(r)["PP"] == PP and A(r)["TP_a"] == TP and A(r)["b"] == 1
            ],
            key=lambda r: r["tsu"],
        )
    # the results keep only the winning round count per configuration: evaluate the fixed-R form directly (all other choices free)
    cfg = dict(DP=1, PP=PP, TP_a=TP, DP_a=1, EP=1, TP_x=TP, CP=1, b=1, n_mb=1)
    S = TP
    best = None
    layouts = [((4, 4), 3), ((8, 2), 1)] if (PP == 2 and S == 16) else [(SUBMESH[S], HAND_HOPS[PP])]
    for sub, hh in layouts:
        keep = SUBMESH[S]
        SUBMESH[S] = sub
        for rep in (0, 1):
            for bal in ("bytes", "chain"):
                for buf in (0, 1, 2, 3):
                    for xf in ("slice", "tiles") if TP >= 8 else ("slice",):
                        for hf in ("A", "B", "Bn", "B+D") if TP > 1 else ("A",):
                            HC_FORM[0] = hf
                            x = evaluate_form(cfg, routing, 192, 1, "hit", rep, bal, hh, buf, xf, R)
                            if best is None or (x["T_tok_ms"], x["dram_ms"]) < (best["T_tok_ms"], best["dram_ms"]):
                                best = x
        SUBMESH[S] = keep
    HC_FORM[0] = "A"
    return best


hb1 = best("hit", 1, "tsu")
hagg = best("hit", 1, "agg")
cb1 = best("census", 1, "tsu")
cagg = best("census", 1, "agg")
mb1 = best("miss", 1, "tsu")
magg = best("miss", 1, "agg")
ab1 = best("all", 1, "tsu")
aagg = best("all", 1, "agg")
hb1_0 = best("hit", 0, "tsu")
hagg_0 = best("hit", 0, "agg")
cur = get("miss", 1, DP=8, TP_a=4, EP=4, b=16)
curh = get("hit", 1, DP=8, TP_a=4, EP=4, b=16)
ia = bestimpl("hit", 1, "agg")
it = bestimpl("hit", 1, "tsu")
ima = bestimpl("miss", 1, "agg")
imt = bestimpl("miss", 1, "tsu")
tp32 = get("hit", 1, TP_a=32, EP=1, b=1)
pp_tp = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)]
ladder = [(PP, TP, ppb1(PP, TP)) for PP, TP in pp_tp]


def with_hop(mult, cfg, case="hit", k=1):
    global HOP
    h0 = HOP
    HOP = h0 * mult
    r = evaluate(cfg, routing, 192, k, case)
    HOP = h0
    return r


sens = {PP: {m: with_hop(m, cfgdict(r["cfg"]))["tsu"] for m in (1.0, 2.0, 4.0)} for PP, TP, r in ladder}


def with_stream(mult, cfg, case="hit", k=1):
    global L1_STREAM
    s0 = L1_STREAM
    L1_STREAM = s0 * mult
    r = evaluate(cfg, routing, 192, k, case)
    L1_STREAM = s0
    return r


sens_stream = [with_stream(m, cfgdict(hb1["cfg"]))["tsu"] for m in (0.0, 2.0)]


def bestsens(m):
    return max(ladder, key=lambda x: sens[x[0]][m])


asym = 32 * BW_DRAM / (L_GDN * S_USER_LAYER * 2 + L_QSA * (kv_read_row(192, 1) + kv_write_row()))


def table_pareto(case, k):
    o = [
        "| agg [t/s] | tsu | users | 構成 | router / expert / hc | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in par(case, k):
        o.append(
            f"| {r['agg']:,.0f} | {r['tsu']:.1f} | {r['users']} | `{r['cfg']}` | {r['router']} / {r['xform']} / {r['hc_form']} | {r['slot_ms']:.3f} | {r['dram_ms']:.3f} | {r['comp_ms']:.3f} | {r['comm_ms']:.3f} | {r['routed_exposed_ms']:.3f} | {r['resident_MB']:.0f} / {r['buf_MB']:.1f} ({r['dense_chip_MB']:.0f}) | {r['busiest']:.1f} | {r['src']} |"
        )
    return "\n".join(o)


def table_tp(b):
    o = [
        "| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for TP, ep1, bt in tp_rows(b):
        o.append(
            f"| {TP} | {ep1['tsu']:.0f} | {ep1['agg']:,.0f} ({ep1['users']}) | `{bt['cfg']}` | {bt['router']} / {bt['xform']} | {bt['tsu']:.0f} | {bt['agg']:,.0f} ({bt['users']}) | {bt['T_tok_ms']:.3f}; {bt['dram_ms']:.3f} / {bt['comp_ms']:.3f} / {bt['comm_ms']:.3f} |"
        )
    return "\n".join(o)


def table_ladder():
    o = [
        "| PP × TP_a（b=1 の最良、EP/DCP/周/hc 形 自由） | 構成 | router / expert / 段の切り方 / 段の形 / hc | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | L1: 常駐 dense / 先読みバッファ / 空き時間の先読み [MB]（段の dense） | hop ×1 / ×2 / ×4 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for PP, TP, r in ladder:
        o.append(
            f"| PP{PP} × TP{TP} | `{r['cfg']}` | {r['router']} / {r['xform']} / {r['balance']} / {r['layout']} / {r['hc_form']} | {r['tsu']:.0f} | {r['T_tok_ms']:.3f} | {r['dram_ms']:.3f} | {r['comp_ms']:.3f} | {r['comm_ms']:.3f} | {r['resident_MB']:.0f} / {r['buf_MB']:.1f} / {r['prefetch_MB']:.0f} ({r['dense_chip_MB']:.0f}) | {sens[PP][1.0]:.0f} / {sens[PP][2.0]:.0f} / {sens[PP][4.0]:.0f} |"
        )
    return "\n".join(o)


def table_best():
    o = ["| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |", "|---|---|---|---|---|---|---|---|---|"]
    for case, _ in CASES:
        for k in (0, 1):
            a = best(case, k, "agg")
            t = best(case, k, "tsu")
            o.append(
                f"| {case} | {k} | {a['agg']:,.0f} | {a['tsu']:.1f} | {a['users']} | `{a['cfg']}` | {t['tsu']:.0f} | {t['agg']:,.0f} | `{t['cfg']}` |"
            )
    return "\n".join(o)


def table_impl():
    o = ["| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |", "|---|---|---|---|---|---|"]
    for case, _ in CASES:
        for k in (0, 1):
            a = bestimpl(case, k, "agg")
            t = bestimpl(case, k, "tsu")
            o.append(
                f"| {case} | {k} | {a['agg']:,.0f} ({a['tsu']:.0f} tsu) | `{a['cfg']}` | {t['tsu']:.0f} ({t['agg']:,.0f} agg) | `{t['cfg']}` |"
            )
    return "\n".join(o)


def bestb(b, case="hit", k=1):
    return max([r for r in feas(rs(case, k)) if A(r)["b"] == b], key=lambda r: r["tsu"])


def first_b_pp1():
    return next(b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) if A(bestb(b))["PP"] == 1)


def first_b_dcp():
    return next(
        b
        for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
        if all(A(r)["CP"] > 1 for r in par("hit", 1) if A(r)["b"] >= b)
    )


def dcpb1(c):
    a = A(hb1)
    return max(
        [
            r
            for r in feas(rs("hit", 1))
            if A(r)["DP"] == a["DP"]
            and A(r)["PP"] == a["PP"]
            and A(r)["TP_a"] == a["TP_a"]
            and A(r)["EP"] == a["EP"]
            and A(r)["b"] == 1
            and A(r)["CP"] == c
        ],
        key=lambda r: r["tsu"],
    )


b1 = S["batches"]["1"]
b16 = S["batches"]["16x2"]
b256 = S["batches"]["256"]
lat = {n: last_tile(diameter(n)) * 1e6 for n in (4, 8, 16, 32)}
ncoll = 6
dense_chip32 = DENSE_TOTAL / 32 / 1e6
l1_total = L1_CHIP / 2**20
l1_stream = L1_STREAM / 2**20
l1_free_b1 = (L1_CHIP - L1_STREAM - 4 * 2 * H * BF16 - 2 * EXPERT / 8) / 2**20


def frames_of(cfgax, ctx=192, case="hit"):
    """the chosen form's winning options and its per-frame numbers (FRAMES_SINK)"""
    global FRAMES_SINK
    a = dict(cfgax)
    R = a.pop("rounds", 1)
    best = None
    S = a["TP_a"] * a["DP_a"]
    PP = a["PP"]
    layouts = [((4, 4), 3), ((8, 2), 1)] if (PP == 2 and S == 16) else [(SUBMESH[S], HAND_HOPS[PP])]
    FRAMES_SINK = None
    for sub, hh in layouts:
        keep = SUBMESH[S]
        SUBMESH[S] = sub
        for rep in (0, 1):
            for bal in ("bytes", "chain") if PP > 1 else ("bytes",):
                for buf in (0,) if case == "miss" else (0, 1, 2, 3):
                    for xf in ("slice", "tiles") if a["TP_x"] >= 8 else ("slice",):
                        for hf in ("A", "B", "Bn", "B+D") if a["TP_a"] > 1 else ("A",):
                            HC_FORM[0] = hf
                            x = evaluate_form(a, routing, ctx, 1, case, rep, bal, hh, buf, xf, R)
                            if best is None or (x["T_tok_ms"], x["dram_ms"]) < (
                                best[0]["T_tok_ms"],
                                best[0]["dram_ms"],
                            ):
                                best = (x, rep, bal, hh, buf, xf, sub, hf)
        SUBMESH[S] = keep
    x, rep, bal, hh, buf, xf, sub, hf = best
    keep = SUBMESH[S]
    SUBMESH[S] = sub
    FRAMES_SINK = {}
    HC_FORM[0] = hf
    evaluate_form(a, routing, ctx, 1, case, rep, bal, hh, buf, xf, R)
    fr = dict(FRAMES_SINK)
    FRAMES_SINK = None
    SUBMESH[S] = keep
    HC_FORM[0] = "A"
    return x, fr, dict(rep=rep, bal=bal, hh=hh, buf=buf, xf=xf, sub=sub, R=R, hf=hf)


T1 = frames_of(dict(A(hb1), rounds=hb1["rounds"]))
T1x, T1f, T1o = T1
AG1 = frames_of(dict(A(hagg), rounds=hagg["rounds"]))
AGx, AGf, AGo = AG1
assert abs(T1x["tsu"] - hb1["tsu"]) < 0.5 and abs(AGx["agg"] - hagg["agg"]) < 1, (
    T1x["tsu"],
    hb1["tsu"],
    AGx["agg"],
    hagg["agg"],
)


def table_frames(fr, PP):
    o = [
        "| 枠 | chip 群 | 層 | dense [MB] | 状態+KV [MB] | routed [MB] | 鎖 [ms] | 占有 [ms] | 空き時間の先読み [MB] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for f, (a, z) in enumerate(fr["cuts"]):
        lay = f"L{a}〜L{min(z,L)-1}" + ("+tail" if z > L else "") if a < L else "tail"
        o.append(
            f"| {f} | {f%PP} | {lay} | {fr['dense'][f]/1e6:.1f} | {fr['state'][f]/1e6:.1f} | {fr['routed'][f]/1e6:.1f} | {fr['chain'][f]*1e3:.3f} | {fr['occ'][f]*1e3:.3f} | {fr['pre'][f]/1e6:.1f} |"
        )
    return "\n".join(o)


def table_units():
    MiB = 2**20
    rows = [
        ("GDN 層（36 層）: in_proj qkv [2·16·128+48·128, H] bf8", W_GDN["qkv"]),
        ("GDN: in_proj z [48·128, H] bf8", W_GDN["z"]),
        ("GDN: out_proj [H, 48·128] bf8", W_GDN["out"]),
        ("GDN: a/b 射影 [48, H]×2 bf8 + conv/norm/A_log/dt_bias", W_GDN["ab"] + W_GDN["small"]),
        ("QSA 層（12 層）: q+gate 射影 [2·24·256, H] bf8", W_QSA["qg"]),
        ("QSA: k/v 射影 [2·256, H]×2 bf8", W_QSA["kv"]),
        ("QSA: o_proj [H, 24·256] bf8", W_QSA["o"]),
        ("QSA: indexer 射影 [4·128+128, H] bf8 + norm", W_QSA["idx"] + W_QSA["small"]),
        ("MoE（全 48 層 + MTP）: shared expert up/gate/down [640, H]×3 bf8 + gate", W_MOE["shared"]),
        ("MoE: router [512, H] bf16", W_MOE["router"]),
        ("MoE: routed expert 1 個 up/gate [2·640, H] + down [H, 640] BF4（× 512 × 49 層）", EXPERT),
        ("hyper-connection（層ごとに 2 ブロック）: down [320, 4H] + up [4H, 320] bf16 + inject + norm", hc_block),
        ("層の残り: hc 2 ブロック + norm 2 本", W_OTHER),
        ("PLE（層 2 の後）: [4H, H] + [H, H] bf16 + 表", PLE),
        ("final mixer [320, 4H]×2 bf8 + norm", FINAL),
        ("lm_head [248320, H] bf8", LM_HEAD),
        ("MTP 層（QSA + MoE + 射影 2 本 + mixer 2 本 + hc）", MTP),
    ]
    o = ["| 単品（HF 定義） | バイト [MiB] |", "|---|---|"] + [f"| {n} | {v/MiB:,.2f} |" for n, v in rows]
    o += [
        f"| **dense 合計（GDN 36 × {(sum(W_GDN.values())+sum(W_MOE.values())+W_OTHER)/MiB:.1f} + QSA 12 × {(sum(W_QSA.values())+sum(W_MOE.values())+W_OTHER)/MiB:.1f} + PLE + final + lm_head + MTP）** | **{DENSE_TOTAL/MiB:,.0f}（{DENSE_TOTAL/GB:.3f} GB）** |",
        f"| **routed expert 合計（512 × 49 層 × {EXPERT/MiB:.2f}）** | **{EXPERTS_TOTAL/MiB:,.0f}（{EXPERTS_TOTAL/GB:.2f} GB）** |",
        f"| ユーザ状態: GDN 状態 S [48,128,128] fp32 × 36 層（+ conv tap 3 行 bf16） | {36*S_USER_LAYER/MiB:.1f}/user 常駐、1 pass 読み+書き {36*S_USER_LAYER*2/MiB:.1f} |",
        f"| ユーザ KV: QSA 1 層 1 token K/V 2 head × 256 × 2 × bf16 + index key 128/4 | {kv_write_row()/1024:.2f} KiB/token/層 = ctx 192 で {12*kv_resident(192)/MiB:.1f}、262k で {12*kv_resident(262144)/MiB:,.0f}、1M で {12*kv_resident(1000000)/MiB:,.0f} MiB/user |",
    ]
    return "\n".join(o)


def table_census():
    o = [
        "| B（行） | 実 distinct expert/層（一様仮定） | 2 段先読みのヒット率 h2 / h12 | 最繁忙 group（EP2 / EP4 / EP8、expert 数） | 最繁忙 chip（tiles 形 TPx8 / TPx16 / TPx32、tile 数; 均等なら distinct×20/TPx） | tiles TPx8 の集合（P2 / P1−P2 / 外れ / 無駄読み、tile） |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("1", "1x2", "16x2", "32x2", "128x2", "256"):
        b = S["batches"][key]
        t = b.get("busiest_tiles", {})
        s2 = S2[key]
        t8 = s2["tiles"]["1x8"]
        o.append(
            f"| {key} | {b['distinct_mean']:.1f}（{b['distinct_uniform']:.1f}） | {s2['hit2']:.2f} / {s2['hit12']:.2f} | {b['busiest']['2']['actual']:.1f} / {b['busiest']['4']['actual']:.1f} / {b['busiest']['8']['actual']:.1f} | {t['1x8']['actual']:.0f} / {t['1x16']['actual']:.0f} / {t['1x32']['actual']:.0f}（{b['distinct_mean']*20/8:.0f} / {b['distinct_mean']*20/16:.0f} / {b['distinct_mean']*20/32:.0f}） | {t8['P2']:.1f} / {t8['P1mP2']:.1f} / {t8['miss']:.1f} / {t8['wasted']:.1f} |"
        )
    return "\n".join(o)


a1 = A(hb1)
S1 = a1["TP_a"] * a1["DP_a"]
aa = A(hagg)
SA = aa["TP_a"] * aa["DP_a"]
P1 = T1x
next_frame = max(T1f["dense"]) / 1e6
routed_layer = T1f["routed"][0] / max(1, T1f["cuts"][0][1] - T1f["cuts"][0][0]) / 1e6
dram_items = None
hand_us = T1f["hand"] * 1e6
comm_gdn = T1f["comm"][0] * 1e6
comm_qsa = T1f["comm"][3] * 1e6
comm_tail = T1f["comm"][48] * 1e6
dram_items = dict(dense=sum(T1f["nonres"]) / 1e6, state=sum(T1f["state"]) / 1e6, routed=sum(T1f["routed"]) / 1e6)
heads_v = GDN_V_HEADS // a1["TP_a"]
heads_qk = GDN_QK_HEADS // a1["TP_a"]
heads_q = QSA_HEADS // a1["TP_a"]
kvrep = max(1, a1["TP_a"] // QSA_KV_HEADS)
agg_dram = dict(dense=sum(AGf["nonres"]) / 1e6, state=sum(AGf["state"]) / 1e6, routed=sum(AGf["routed"]) / 1e6)


def steps():
    """from the served form to the limit, each step's number (ctx 192, k=1)"""
    o = []
    o.append(("今の配備の形（DP8 × TP4、lanes 16、k=1、router 後に expert を読む）の実測", f"{1186.9:,.1f} tok/s（14.6 tok/s/user）"))
    o.append(
        (
            "同じ形の限界（全ミス）: T_pass {:.2f} ms は段の DRAM/pass で、うち router 後の routed 読み {:.2f} ms、1 chip の dense {:,.0f} MB のうち常駐 {:.0f} MB（残りを pass ごとに読む）".format(
                cur["slot_ms"], cur["routed_exposed_ms"], cur["dense_chip_MB"], cur["resident_MB"]
            ),
            f"{cur['agg']:,.0f} tok/s（効率 {1186.9/cur['agg']*100:.1f}%）",
        )
    )
    o.append(("同じ形で routed を先読みしても（全ヒット）", f"{curh['agg']:,.0f}（DRAM/pass 支配で不変）"))
    o.append(
        (
            "今日の実装拘束内（TP∈{4,8}, EP=TP, PP=DCP=1, lanes×2 ≤ 32）の最良: agg / tsu",
            f"{ia['agg']:,.0f}（`{ia['cfg']}`）/ {it['tsu']:,.0f}（`{it['cfg']}`）",
        )
    )
    b16 = max([r for r in feas(rs("hit", 1)) if A(r)["b"] == 16], key=lambda r: r["tsu"])

    def axes_desc(r):
        a = A(r)
        return (
            f"TP{a['TP_a']}"
            + (f" × DPa{a['DP_a']}" if a["DP_a"] > 1 else "")
            + (f" + DCP{a['CP']}" if a["CP"] > 1 else "")
            + (f"、PP{a['PP']}" + (f" の {r['rounds']} 周" if r.get("rounds", 1) > 1 else "") if a["PP"] > 1 else "")
            + f"、expert {r['xform']}（EP{a['EP']} × TPx{a['TP_x']}）、router {'複製' if r['router']=='replicated' else '列分割'}、hc {r['hc_form']}"
        )

    o.append(
        (
            f"拘束を外した b=16（同じ 16 users/instance）: {axes_desc(b16)}",
            f"tsu {b16['tsu']:,.0f}、agg {b16['agg']:,.0f}（`{b16['cfg']}`）",
        )
    )
    o.append((f"b=1 の tsu 最大: {axes_desc(hb1)}、L1 に次の枠を先読み", f"{hb1['tsu']:,.0f}（`{hb1['cfg']}`）"))
    o.append(
        (
            f"agg 最大: DP{A(hagg)['DP']} × {A(hagg)['TP_a']*A(hagg)['DP_a']}chip、{axes_desc(hagg)}、b={A(hagg)['b']}（{hagg['users']:,} users）",
            f"{hagg['agg']:,.0f}（漸近 {asym*1.376:,.0f} の {hagg['agg']/(asym*1.376)*100:.0f}%）",
        )
    )
    return "\n".join(f"| {k} | {v} |" for k, v in o)


def short(cfg):
    return cfg.split(" b=")[0]


def ctx_forms_tsu():
    parts = []
    for c in CTXS:
        r = bestc(c, "hit", "tsu")
        parts.append(f"{c:,} → `{short(r['cfg'])}`（{r['tsu']:,.0f}）")
    return "、".join(parts) + f"。1M は常駐 KV {12*kv_resident(1000000)/1e9:.0f} GB/user が DCP の枚数に分かれないと DRAM に入らない。"


def ctx_forms_agg():
    parts = []
    for c in CTXS:
        r = bestc(c, "hit", "agg")
        parts.append(
            f"{c:,} → `{r['cfg']}`（{r['agg']:,.0f} t/s、DP1 で入る最大 b {max(A(x)['b'] for x in rsc(c,'hit') if x['feasible'] and A(x)['DP']==1):,}）"
        )
    return "、".join(parts) + "。"


def reading_ctx():
    forms = {c: short(bestc(c, "hit", "tsu")["cfg"]) for c in CTXS}
    same = [c for c in CTXS if forms[c].startswith(forms[1].split(" |")[0])]
    dcp = [(c, A(bestc(c, "hit", "tsu"))["CP"]) for c in CTXS]
    bmax = " / ".join(
        "{:,}".format(max(A(r)["b"] for r in rsc(c, "hit") if r["feasible"] and A(r)["DP"] == 1)) for c in CTXS
    )
    missv = " / ".join("{:,.0f}".format(bestc(c, "miss", "tsu")["tsu"]) for c in CTXS)
    dcp1_262 = max(r["tsu"] for r in rsc(262144, "hit") if r["feasible"] and A(r)["b"] == 1 and A(r)["CP"] == 1)
    others = (
        ""
        if len(same) == len(CTXS)
        else "、" + "、".join("{:,} で `{}`".format(c, forms[c]) for c in CTXS if c not in same)
    )
    return "tsu 最大の形は ctx {:,}〜{:,} で `{}` の系（DCP が {} と増える: index block の走査が分かれる; 262k で DCP4 {:,.0f} vs DCP1 {:,.0f}）{}。agg 最大は常駐 KV が b を縛るので系列長とともに落ちる（DP1 で入る最大 b: {}）。全ミス（今の runtime の形）の tsu は系列長に鈍感（{}）。".format(
        min(same),
        max(same),
        forms[1],
        " → ".join(str(d) for c, d in dcp),
        bestbc(262144, 1)["tsu"],
        dcp1_262,
        others,
        bmax,
        missv,
    )


b1s = S["batches"]["1"]
b16s = S["batches"]["16x2"]
C7T = json.load(open("/data/yoshifuji/qwen38/routing/design_step7_chain_hit_A.json"))["T_steady"]
out = f"""## 11. Decode limit on one Blackhole Galaxy: 5 軸（DP, PP, TP, EP, DCP）の限界性能と、そこから採る形

この節は、32 chip の BH Galaxy で本モデルの decode を回すときの**限界性能**（物理値と HF 定義だけから出る 1 token の時間と機械全体の tokens/s）と、その限界を出す**形**（5 軸の分け方、chip の中身、1 token の時間線）を、詳細設計と実装がそのまま使える粒度で書く。並び: §11.1 目標と指標 → §11.2 設計の入力（モデル・routing の実態・機械の物理値・系列長） → §11.3 確定原則（設計規則） → §11.4 解析モデル（時間の式） → §11.5 限界性能の答え（系列長別、ctx 192 の詳細） → §11.6 採る形の詳細設計（tsu 側 / agg 側） → §11.7 今の実装との差と近づける順番 → §11.8 未確定と決着方法 → §11.9 生成・検証・運用。本文の数値・順位は全て生成器 `render_free5d_v4.py` が `free5d_v4.py` の結果と census から決定的に生成する（手書きの数値は無い）。第 4 版は §12 の詳細設計で見つかった 8 点（§12.8）を取り込んだもの: 会合（collective）は packet 模型、routed は 2 段先読み、tail の全単品、受け渡しは生きている行、hyper-connection の形の軸、先読み容量。経緯・指摘・撤回は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §9.4〜§13（追記専用）、生成器と census は同じ共有ディレクトリ（`BH-GLX_Qwen3.8-Flash-Next_free5d_v4_20261026.py`、`..._render_free5d_v4_20261026.py`、`..._design_comm_model_20261026.py`、`..._design_census_stage2_batches_20261026.py`、`..._routing_census_20261023.py`、`..._routing_stats_20261023.py`）。README と食い違ったら新しい方の作業記録が正しい。

### 11.1 目標と指標

- **tsu**（tokens/s/user）= R_eff / T_tok。T_tok は 1 user の 1 token が出る間隔（1 pass の時間）。**agg**（tokens/s、機械全体）= DP × b × R_eff / T_pass。R_eff = 1（k=0）/ {TOKENS_PER_PASS[1]}（MTP k=1: verify 2 行/user で 1 pass に出る token の期待値。本件プロンプト集合 ISL 128 の採択率）。
- **限界** = 物理値と HF 定義だけから出る値（§11.4 の式）。**効率** = 実測 / 同じ形の限界。目標は限界の形へ実装を寄せ、効率を上げること。今の配備（DP8 × TP4、lanes 16、k=1、ISL 128 / OSL 128、128 同時）は実測 {1186.9:,.1f} tok/s で、同じ形の限界 {cur['agg']:,.0f} の **{1186.9/cur['agg']*100:.1f}%**（§11.7）。
- 条件は並列列挙する（代表仮定を置かない）: 系列長 ctx ∈ {{1（KV 無視）, 128, 8,192, 32,768, 262,144（native `max_position_embeddings`）, 1,000,000（YaRN factor 4、モデルカードの最大長）}} と 192（今の配備の計測条件 = ISL 128 + decode の途中、実測との比較用のみ）；routed expert の先読みの当たり方 4 ケース（§11.4.5）；MTP k ∈ {{0, 1}}；batch b は 1〜2,048（agg の漸近まで）。
- この節で決めること: 系列長ごとの tsu 最大・agg 最大とその形（§11.5）、採る形の詳細（§11.6）、今の実装から限界へ近づける順番（§11.7）。決めないこと: 精度（`docs/NUMERICS.md`）、prefill（`docs/PREFILL.md`）、server（`docs/SERVER.md`）。

### 11.2 設計の入力

#### 11.2.1 モデル（HF 定義、配備の数値形式: dense bf8 1,088 B/tile、hc mixer・router・norm bf16、routed expert BF4 576 B/tile）

H = {H}、48 層 = 36 GDN（gated delta net、value head 48 × 128、qk head 16 × 128、状態 fp32）+ 12 QSA（query 24 head × 256、kv 2 head、indexer 4 head × 128 で index block（圧縮 4）を採点し budget 2,048 行を選ぶ sparse attention）、全層 MoE（512 expert top-10 × 640 + shared 640）、hyper-connection 4 枝（残差 [rows, 4H]、低ランク 320）、PLE、vocab 248,320、MTP 1 層。

{table_units()}

#### 11.2.2 routing の実態（census: 本件プロンプト集合 34 item × 256 decode 位置の層別 top-10、CPU oracle）

routed の分子は一様仮定ではなく census の実測: 同時 B 行の実 distinct expert 数、EP 群に丸ごと置いたときの最繁忙群、tile 回し置きのときの最繁忙 chip、2 段先読み（層 l の router を 2 ブロック早い / 1 ブロック早い残差で回す）の集合とヒット率（h2 = P2 のヒット率、h12 = P1 ∪ P2 のヒット率）。rows > 256 は外挿（distinct の減衰則と偏り則）。

{table_census()}

#### 11.2.3 機械の物理値（設計の入力、出典は 27B README §3 と同じ）

- DRAM {BW_DRAM/1e9:.0f} GB/s/chip（読み書きの和）、容量 31.9 GiB の 90% = {DRAM_CHIP/GB:.1f} GB を使う（runtime・trace・CB に 10%）。
- chip 間 link {BW_LINK/1e9:.0f} GB/s × 2 本/方向 × 4 方向（8 × 4 torus、submesh の直径: {'、'.join(f'{SUBMESH[n][0]}×{SUBMESH[n][1]} で {diameter(n)}' for n in (4,8,16,32))} hop）、1 hop {HOP*1e9:.0f} ns（EthernetSsDesignSpec Table 16）+ 蓄積転送 2 KiB/(96 B/cycle) = {saf(PKT)*1e9:.0f} ns、中継 NoC 92 cycle + 32 flit、1 packet 4,096 B + 56 B。
- Tensix 130 core × 1.35 GHz: 行列積 HiFi2 {MM} cycle/32×32 tile-MAC、要素演算 16 cycle/bf16 tile・32 cycle/fp32 tile、gated delta rule 1 組 (value head, user) 4,224 cycle（27B §9）→ 32 列 4 単位に分け 1 単位 256 + 800 × 行 cycle、router top-10 は logit tile ごと 10 pass × 32 cycle → NoC 集約 92 cycle → 10 pass。op 固定費は数えない（27B §9 と同じ）。
- L1 1,536 KiB/core − 固定専有 140 KiB/core（firmware 40 + kernel 100）= {L1_CHIP/2**20:.1f} MiB/chip。ストリームバッファ 1 core {(L1_MM+L1_GDR+L1_KVC)//1024} KiB（行列積 tile 2 面 + 部分和 {L1_MM//1024}、GDR 状態単位 2 面 + 出力 {L1_GDR//1024}、KV/index チャンク 2 面 {L1_KVC//1024}）= {L1_STREAM/2**20:.1f} MiB/chip。

#### 11.2.4 系列長で変わるもの

QSA 1 行 1 層の KV 読み = 選択 K/V min(ctx, 2048) 行 × 2 head × 256 × 2 × bf16 + index block ctx/4 × 128 × bf16（indexer は全 block を採点する）: 1 user 1 pass で ctx 1: 0、128: {12*kv_read_row(128,1)/1e6:.1f} MB、8k: {12*kv_read_row(8192,1)/1e6:.0f}、32k: {12*kv_read_row(32768,1)/1e6:.0f}、262k: {12*kv_read_row(262144,1)/1e6:.0f}、1M: {12*kv_read_row(1000000,1)/1e6:.0f} MB。index 走査の演算（1 行 1 層、1 chip）: 262k {4*math.ceil(262144/4/32)*(128/32)*MM/32/CLK*1e6:.0f} µs、1M {4*math.ceil(1000000/4/32)*(128/32)*MM/32/CLK*1e6:.0f} µs。常駐 KV: 1 user で 262k {12*kv_resident(262144)/1e9:.1f} GB、1M {12*kv_resident(1000000)/1e9:.1f} GB（1 chip の DRAM の {12*kv_resident(1000000)/DRAM_CHIP*100:.0f}% → DCP が容量のためにも要る）。YaRN は RoPE の周波数を変えるだけで decode のバイト・演算を変えない。GDN 状態は系列長に依らない。

### 11.3 確定原則（本件の設計規則。詳細設計と実装はこれに従い、外れるときは本節を先に直す）

**A. 限界と分母**
1. 限界は物理値（§11.2.3）と HF 定義（§11.2.1）だけから立てる。実装の実測値（collective の時間、expert の実効帯域、fabric の α、packet 固定費、τ）を限界に入れない。実測は効率の分子にだけ使う。
2. 単価は出典のあるものだけを使う（行列積 HiFi2 {MM} cycle、GDR 4,224 cycle/組、tile 読み 32 cycle、op 固定費なし）。出典の無い単価は置かない。
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
11. routed expert は router の後にしか読めない（dense だけ先読み可）。先読みは **2 段**: 層 l の router を 2 ブロック早い残差に当てた集合 P2 を層 l−1 の間に、1 ブロック早い集合 P1 の差分を層 l の attention ブロックの間に読み、外れ A − P1 − P2 は router 後（発行レイテンシ {ISSUE_LAT*1e6:.2f} µs）に読む。層の時間 = 前段の鎖 + [発行] + max(expert 段の鎖, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。L00 の expert は確定（embedding 行）だが pass 開始後にしか発行できず、MTP 層の expert は本 pass の argmax 後にしか予測できない（構造的な露出）。
12. routed の分子は census（§11.2.2）。全 expert 読みは群の 512/EP expert を毎 step 読む形で偏り無し。
13. 露出した routed 読みを二重に数えない（占有 = max(演算+通信, 先読み分/BW) + 露出分/BW）。

**D. メモリ**
14. L1 は積み上げで勘定する: 固定専有 → ストリームバッファ → 活性 4 × [rows, H] bf16 + 残差（列分割なら [rows, 4H]/TP_a、複製なら全体）→ expert 片 2 枚 → 最終段の logits chunk → 残りを常駐 dense と routed 先読みバッファに分ける（分け方は 0 / 1 層分 / 2 層分 / 残り全部の 4 通りを列挙）。空き時間の先読みは常駐の残り全部から routed 3 層分を取り置いた分まで（枠の dense は鎖が消費した順に空く）。dense 5.38 GB の 32 分割 160 MiB は L1 に全部は置けない。
15. 候補を「物理的に劣るはず」という理由で列挙から外さない。選択は (時間, pass あたりの DRAM バイト) の辞書順だけで決める。
16. 残差は L1 に入らなければ DRAM に置き、層ごとの往復を DRAM 車線に加える。容量は expert 分 + 段の dense 分 + ユーザの状態/KV ≤ {DRAM_CHIP/GB:.1f} GB/chip（PP > 1 では自段の層分だけ）。

**E. 通信と演算**
17. 通信（会合 = collective 1 回）は packet 模型で数える（`comm_model.py`: link 25 GB/s/方向、hop 280 ns、送り手の蓄積転送 16 ns/2 KiB packet、中継 92 cycle + 32 flit、packet 2 KiB + 56 B、link は FIFO、経路は縦 → 環の短い向き、形は BFS 木の multicast / 直送 / 木の reduce の速いもの、all-reduce は AG+局所 と RS+AG の速い方）。鎖に乗る露出: 入力 x の all-gather は届いた K chunk から次の射影が追いかける（露出 = 最後の chunk + 残りの仕事 − 射影）、部分和の reduce-scatter / all-reduce は 10 波で行列積の間に流す（露出 = 行列積を超えた分）、hc 読みの部分和の all-reduce も down の間に波で流す。投機 router の会合は鎖の外。
18. 層の会合は hyper-connection の形から数え、形は軸として列挙する: **A**（配備 `read_tp4`: 残差を列で TP_a 分割、down K 分割 → 部分和 AR（波）、up N 分割 → x の AG、出力 RS（波））、**B**（残差を複製: 出力 AR（波）に attention 出力の折り畳み F_i h_i（[1280, 768] bf16/chip/ブロック）を相乗りさせ MoE 読みの low-rank を局所に）、**Bn**（B + attention 読みの D を N 分割 + low-rank の AG）、**B+D**（B + D を複製）。B 系は受け渡しで残差の AG が 1 本増える。router は複製（会合無し、dense +{W_MOE['router']*48/1e6:.0f} MB/chip）と列分割（候補の AG 1 本）の両形、DP_a > 1 なら a2a 2 本、DCP > 1 なら merge 1 本（QSA 層）。
19. b=1 の演算は行列積の鎖で決まる: 出力 tile が core 数より少なければ K を分けて部分和を NoC の木で足し（1 段 92 cycle + 64 flit + 32 cycle）、同じ入力を読む行列積は並べて走る（27B §9「和でなく max」）。GDR の単位は core に 1 つずつ乗り、verify 行は再帰で直列。

**F. 条件と結論の出し方**
20. 系列長は今の配備の計測条件（192）を設計条件にせず、§11.1 の 6 点を並べる。
21. agg は b を漸近（ユーザ状態の往復 {asym*1.376:,.0f} t/s）まで振り、容量で落ちる b を明記する。「EP=1・DCP=1 が最速」のような結論を比較せずに前提から出さない。
22. tail は全単品を数える: final mixer（+ 部分和 AR + x の AG）、lm_head、局所 argmax（7.5 tile/core + NoC 集約）+ 候補の AG、MTP の mixer・fc・MTP 層・draft lm_head + argmax。PLE は演算と統計の AR × 2。
23. 群間の受け渡しは枠の境で生きている行全部（残差 1/TP_a + ブロック入力 x + inject）を、列の 2 chip が共有する群間 link に packet 模型で流す。
24. 限界モデルは層の中で DRAM 車線を平滑化する上限（max(鎖, routed/BW)）。時間軸で車線を追う詳細設計（§12）は routed の読みが層境を越えて積み上がる分だけ長くなりうる（全ヒットでは一致、census・全ミスでは §12 が長い）。設計の到達値は §12、§11 は形の比較と上限。

### 11.4 解析モデル（時間の式。実装は `free5d_v4.py`、docstring に全式）

#### 11.4.1 軸と列挙

32 = DP × PP × S、S = TP_a × DP_a = EP × TP_x、DCP | max(1, TP_a/2)、周 R ∈ {{1, 2, 4}}（PP > 1）、n_mb ∈ 1〜min(PP, b)、b ∈ {{1, 2, …, 2048}}。各構成に対し router 2 形 × expert 分割 2 形 × L1 の分け方 4 通り × 段の切り方 2 通り × 段の形（PP2 の 16 chip は 4×4 と 8×2）を評価し、(T_tok, DRAM/pass) の辞書順で速いものを採る。全 {len(rs('hit',1))} 構成。

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） |
|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食い、1 chip の dense 分担が DP 倍で常駐分が減る） | 無し |
| PP(R) | 層（+ lm_head）を PP × R 枠に切り、枠 f を群 f mod PP に | 群は自枠の dense のうち L1 に入る分を常駐、空いている間に次の枠を先読み、残りと routed・状態・KV を窓で流す | 段間受け渡し（残差 [rows, 4H]/TP、段の並びで 1〜3 hop）× PP × R |
| TP_a | 行列を head / 列で TP_a 枚に（残差も列で分ける: `read_tp4` の形） | 1/TP_a | hc 統計 AR × 2 + 入力 AG × 2 + 出力 RS × 2 |
| EP × TPx | expert を群に丸ごと、群内を slice か tiles で | expert は 1 回 | DP_a > 1 のとき dispatch + combine の a2a；最繁忙群が時間を決める |
| DCP | kv-head 複製群の中で系列を分ける | 無関係 | merge 1 本（QSA 層） |

#### 11.4.2 層の時間と段の時間

- 層 = 前段（hc 読み 2 本、GDN または QSA、shared + router の射影、top-10、それらの collective）+ [発行レイテンシ] + max(expert 段の鎖（expert の up/gate → down + shared down、RS）, 先読みできなかった routed/BW)、かつ ≥ 層の routed 全バイト/BW。会合の露出は原則 17（x の AG は K chunk で射影が追いかける、部分和の RS / AR は 10 波で行列積の間に流す）、a2a / merge は link の合計が演算の半分までに隠れ残りが露出。L00 と MTP 層は routed の構造的露出（原則 11）を足す。
- 枠 f の占有 occ_f = max(Σ層 + 受け渡し, 窓 = (常駐しない dense − 空き時間の先読み + micro-batch の routed + 状態/KV + 残差の往復)/BW)。群 g の DRAM/pass = Σ(群の枠) [常駐しない dense × n_mb + 状態/KV + routed × n_mb + 残差往復 × n_mb]/BW。
- **T_pass = max(max_g 群の DRAM/pass, max_g Σ(群の枠) occ_f × n_mb, Σ_f occ_f)**。空き時間の先読みは「T_pass − 群の占有」を枠に配り、常駐の残り − routed 3 層分まで（不動点）。T_tok = T_pass。tail = 原則 22、受け渡し = 原則 23。

#### 11.4.3 L1 と DRAM の勘定（1 chip）

L1 {L1_CHIP/2**20:.1f} MiB − ストリームバッファ {L1_STREAM/2**20:.1f} MiB − 活性（b=1 で {(4*2*H*BF16)/2**10:.0f} KiB + 残差）− expert 片 2 枚 − [最終段: logits chunk 32 行 × V/TP_a × fp32] = 残り → 常駐 dense + routed 先読みバッファ。DRAM 容量: expert {EXPERTS_TOTAL/GB:.1f} GB/(PP × EP × TPx) + 段の dense + user × (状態 {36*S_USER_LAYER/1e9:.3f} GB/PP + KV) ≤ {DRAM_CHIP/GB:.1f} GB。

#### 11.4.4 通信と演算の単価

会合（packet 模型、b=1 の行、submesh 4 / 8 / 16 / 32 chip）: x の AG {meet('AG',4,2*H*BF16/4)*1e6:.2f} / {meet('AG',8,2*H*BF16/8)*1e6:.2f} / {meet('AG',16,2*H*BF16/16)*1e6:.2f} / {meet('AG',32,2*H*BF16/32)*1e6:.2f} µs、hc 部分和の AR（枝ごと [2, 4×325] fp32）{meet('AR',4,2*HC*(HC_LOW+HC+1)*FP32)*1e6:.2f} / {meet('AR',8,2*HC*(HC_LOW+HC+1)*FP32)*1e6:.2f} / {meet('AR',16,2*HC*(HC_LOW+HC+1)*FP32)*1e6:.2f} / {meet('AR',32,2*HC*(HC_LOW+HC+1)*FP32)*1e6:.2f}、出力部分和 [2, H] bf16 の RS {meet('RS',4,2*H*BF16)*1e6:.2f} / {meet('RS',8,2*H*BF16)*1e6:.2f} / {meet('RS',16,2*H*BF16)*1e6:.2f} / {meet('RS',32,2*H*BF16)*1e6:.2f}（鎖に乗る露出はこれより短い: TP8 で AG_x {ag_exposure(8,2*H*BF16/8,mm_c(2,(2*16*128+2*48*128)/8+64,H)/CLK)*1e6:.2f}、RS の波 {wave_exposure('RS',8,2*H*BF16,mm_c(2,H,48*128/8)/CLK)*1e6:.2f}、AR の波 {wave_exposure('AR',8,2*HC*(HC_LOW+HC+1)*FP32,mm_pool((2,HC_LOW,HC_DIM/8),(2,HC,HC_DIM/8))/CLK)*1e6:.2f} µs）。行列積の鎖（1 chip、b=1、K 分割の木込み）: GDN qkv/z/a/b pool TP8 {mm_c(2,(2*16*128+2*48*128)/8+64,H)/CLK*1e6:.2f} µs、TP32 {mm_c(2,(2*16*128+2*48*128)/32+64,H)/CLK*1e6:.2f} µs。GDR 1 単位 {256+800*2:,} cycle（k=1）。tile padding: expert 640 列を TPx 分割 → TPx16 {tile_pad(640,16):.1f} 倍、TPx32 {tile_pad(640,32):.1f} 倍（slice 形のみ）。

#### 11.4.5 routed の 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に読む分 |
|---|---|---|---|
| hit | routed を router の前に読み終える（2 段先読みが全部当たる; L00 は pass 開始後、MTP 層は自分の attention 後） | distinct × 2.76 MB | 0 |
| census | 2 段先読みの実測集合（B=1×2 行: 実 {S2['1x2']['tiles']['1x8']['A']:.1f} tile/層のうち P2 {S2['1x2']['tiles']['1x8']['P2']:.1f}、P1−P2 {S2['1x2']['tiles']['1x8']['P1mP2']:.1f}、外れ {S2['1x2']['tiles']['1x8']['miss']:.1f}、無駄読み {S2['1x2']['tiles']['1x8']['wasted']:.1f}（tiles 形 TPx8 最繁忙 chip）; ヒット率 h12 B=1×2 {S2['1x2']['hit12']:.2f}、B=256 {S2['256']['hit12']:.2f}） | A ∪ P1 ∪ P2 × 2.76 MB/expert | 外れ分 + 発行レイテンシ |
| miss | router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 毎 step 群の 512/EP expert 全部を読む（router に依存しない、偏り無し） | 512/EP × 2.76 MB | 0 |

### 11.5 限界性能の答え

#### 11.5.1 系列長ごとの最大（k=1、全ヒット；census / 全ミスを併記）

{table_ctx()}

b ごとの tsu 最良（全ヒット、k=1）:

{table_ctx_b()}

読み方: {reading_ctx()}

#### 11.5.2 ctx 192（今の配備の計測条件）の詳細

- **agg 最大（全ヒット、k=1）**: **{hagg['agg']:,.0f} t/s** = `{hagg['cfg']}`（{hagg['users']} users、T_pass {hagg['slot_ms']:.1f} ms = 群の DRAM/pass: dense {agg_dram['dense']:.0f} MB + 状態/KV {agg_dram['state']:,.0f} MB + routed {agg_dram['routed']:,.0f} MB per chip；演算 {hagg['comp_ms']:.1f} + 通信 {hagg['comm_ms']:.1f} ms は窓の中；L1 常駐 {hagg['resident_MB']:.0f} / 段の dense {hagg['dense_chip_MB']:.0f} MB、活性 {4*hagg['rows']*H*BF16/2**20:.0f} MiB）。漸近 {asym*1.376:,.0f} の {hagg['agg']/(asym*1.376)*100:.0f}%。census {cagg['agg']:,.0f}、全ミス {magg['agg']:,.0f}、全 expert 読み {aagg['agg']:,.0f}。k=0 {hagg_0['agg']:,.0f}。
- **tsu 最大（全ヒット、k=1）**: **{hb1['tsu']:,.0f} t/s/u** = `{hb1['cfg']}`（T_tok {hb1['T_tok_ms']:.3f} ms = 演算 {hb1['comp_ms']:.3f} + 通信 {hb1['comm_ms']:.3f} + 受け渡し + 窓；群の DRAM/pass {hb1['dram_ms']:.3f} ms；周 R1 {ppb1(a1['PP'],a1['TP_a'],1)['tsu']:,.0f} / R2 {ppb1(a1['PP'],a1['TP_a'],2)['tsu']:,.0f} / R4 {ppb1(a1['PP'],a1['TP_a'],4)['tsu']:,.0f}；DCP1 {dcpb1(1)['tsu']:,.0f} / DCP2 {dcpb1(2)['tsu']:,.0f} / DCP4 {dcpb1(4)['tsu']:,.0f}）。census {cb1['tsu']:,.0f}（`{cb1['cfg']}`）、全ミス {mb1['tsu']:,.0f}（`{mb1['cfg']}`）、全 expert 読み {ab1['tsu']:,.0f}。k=0 {hb1_0['tsu']:,.0f}。
- PP × TP の梯子（b=1、EP/DCP/周は自由）と hop 設計値への感度:

{table_ladder()}

hop 280 ns を 1 / 2 / 4 倍にしたときの最良は PP{bestsens(1.0)[0]}×TP{bestsens(1.0)[1]} / PP{bestsens(2.0)[0]}×TP{bestsens(2.0)[1]} / PP{bestsens(4.0)[0]}×TP{bestsens(4.0)[1]}。ストリームバッファを 0 / 2 倍にすると tsu 最大点は {sens_stream[0]:,.0f} / {sens_stream[1]:,.0f}。

- 各 TP の tsu 最良（b=1）:

{table_tp(1)}

- 各 TP の tsu 最良（b=16; DP が違うと users が違うので agg は同列に比べない）:

{table_tp(16)}

- ケース × k の最大:

{table_best()}

- Pareto 前線（全ヒット、k=1; PP>1 {sum(1 for r in par('hit',1) if A(r)['PP']>1)} 点、EP>1 {sum(1 for r in par('hit',1) if A(r)['EP']>1)} 点、DCP>1 {sum(1 for r in par('hit',1) if A(r)['CP']>1)} 点、全 {len(par('hit',1))} 点）:

{table_pareto('hit',1)}

- Pareto 前線（census、k=1）:

{table_pareto('census',1)}

- Pareto 前線（全ミス = 今の runtime の形、k=1）:

{table_pareto('miss',1)}

- 今日の実装拘束内（TP ∈ {{4,8}}、EP = TP、PP = DCP = 1、lanes × (k+1) ≤ 32）:

{table_impl()}

### 11.6 採る形の詳細設計

#### 11.6.1 tsu 側: `{hb1['cfg']}`（ctx 192、k=1、全ヒット: {hb1['tsu']:,.0f} tsu）

- **chip の配置**: 32 chip を PP{a1['PP']} 群 × {S1} chip（8 × 4 torus 上の {T1o['sub'][0]}×{T1o['sub'][1]} ブロック、群の直径 {diameter(S1)} hop、群間の受け渡し {T1o['hh']} hop）。周 R{T1o['R']}: 層 + tail を {a1['PP']*T1o['R']} 枠に切り（{T1o['bal']} 均等）、枠 f を群 f mod {a1['PP']} に置く。1 token は 8 枠を順に通り、各群は 1 pass に {T1o['R']} 回番が回る。
- **群の中（TP{a1['TP_a']}）**: GDN は value head {heads_v} 本/chip（状態 S もその chip の DRAM）、qk head {heads_qk} 本/chip；QSA は query head {heads_q} 本/chip、kv head 2 本を {kvrep} 枚ずつが複製（DCP{a1['CP']}: {'複製群の内側で系列を '+str(a1['CP'])+' 分割' if a1['CP']>1 else 'ctx 192 では分けない'}）；hc は形 {T1o['hf']}（{'残差 [rows, 4H] を列で '+str(a1['TP_a'])+' 分割、down K 分割・up N 分割' if T1o['hf']=='A' else '残差を複製、attention 出力に折り畳みを相乗り'}）；shared expert は列で {a1['TP_a']} 分割；router は **{T1o['rep'] and '複製' or '列分割'}**（{'collective 無し' if T1o['rep'] else '候補の all-gather 1 本'}）；routed expert は EP{a1['EP']} × TPx{a1['TP_x']} の **{T1o['xf']}** 形（{'20 tile を e mod '+str(a1['TP_x'])+' から回し置き、padding 無し、最繁忙 chip は census で '+f"{hb1['busiest']:.2f}"+' expert 相当/層' if T1o['xf']=='tiles' else 'expert ごとに等分'}）。
- **枠**（dense は群の DRAM にあり、空いている間に次の枠を L1 のバッファへ先読み; routed・状態は窓で）:

{table_frames(T1f, a1['PP'])}

- **1 chip の L1**（{L1_CHIP/2**20:.1f} MiB）: ストリームバッファ {L1_STREAM/2**20:.1f} MiB + 活性 {(4*2*H*BF16+2*HC_DIM/a1['TP_a']*BF16)/2**10:.0f} KiB + expert 片 2 枚 + 残り {T1f['free']/1e6:.1f} MB = 常駐 dense {T1x['resident_MB']:.0f} MB + routed 先読みバッファ {T1f['buf']/1e6:.1f} MB。次の枠の dense 最大 {next_frame:.1f} MB + 1 層の routed {routed_layer:.1f} MB がこのバッファに入る（余り {T1f['buf']/1e6-next_frame-routed_layer:.0f} MB）。
- **1 chip の DRAM**（1 pass に読む量 {sum(dram_items.values()):.0f} MB = dense {dram_items['dense']:.0f} + 状態/KV {dram_items['state']:.0f} + routed {dram_items['routed']:.0f}; {T1x['dram_ms']:.3f} ms = 帯域の {T1x['dram_ms']/T1x['T_tok_ms']*100:.0f}%）。常駐: expert {EXPERTS_TOTAL/(a1['PP']*a1['EP']*a1['TP_x'])/1e9:.2f} GB + 自群の dense + ユーザ状態。
- **1 層の通信**（TP{a1['TP_a']}、hc 形 {T1o['hf']}、packet 模型の露出）: GDN 層 {comm_gdn:.2f} µs、QSA 層 {comm_qsa:.2f} µs、tail {comm_tail:.1f} µs → 1 pass {hb1['comm_ms']:.3f} ms（原則 17/18）。群間受け渡し {hand_us:.2f} µs × {a1['PP']*T1o['R']}（生きている行: 残差 1/{a1['TP_a']} + x + inject）。
- **1 token の時間線**: 演算 {hb1['comp_ms']:.3f} ms（行列積の鎖が大半、GDR {36*(256+800*2)/CLK*1e3:.3f} ms、tail 込み）+ 通信 {hb1['comm_ms']:.3f} + 受け渡し {a1['PP']*T1o['R']*hand_us/1e3:.3f} + 窓の超過 = {hb1['T_tok_ms']:.3f} ms。routed（全ヒット）は 2 ブロック早い予測で前の層の裏に読まれ、露出は L00 と MTP 層の構造的な分だけ。詳細設計（§12.6）の同じ形の値は {C7T:.1f} µs（同じ入力の時間軸模擬、差 {hb1['T_tok_ms']*1e3-C7T:+.1f} µs）。
- **何が床か**: 通信 {hb1['comm_ms']/hb1['T_tok_ms']*100:.0f}%、演算 {hb1['comp_ms']/hb1['T_tok_ms']*100:.0f}%。hop が 2 倍なら PP{bestsens(2.0)[0]}×TP{bestsens(2.0)[1]} に移る。TP32 × PP1 は通信 {tp32['comm_ms']:.3f} ms が床で {tp32['tsu']:,.0f}。

#### 11.6.2 agg 側: `{hagg['cfg']}`（ctx 192、k=1、全ヒット: {hagg['agg']:,.0f} t/s）

- **chip の配置**: DP{aa['DP']} 複製 × {SA} chip（{SUBMESH[SA][0]}×{SUBMESH[SA][1]}）、PP1、各複製が b={aa['b']} users（rows {hagg['rows']}）。dense は TP{aa['TP_a']}（{'複製' if AGo['rep'] else '列分割'} router）、routed は EP{aa['EP']} × TPx{aa['TP_x']}（{AGo['xf']}: TPx{aa['TP_x']} は 20 tile を割り切るので padding 無し；最繁忙群 {hagg['busiest']:.0f} expert/層 = 群の全部、全 expert を毎 step 読む形に近い）、DCP{aa['CP']}（kv-head 複製群 {max(1,aa['TP_a']//2)} 枚で KV を分ける）。
- **1 chip の DRAM/pass** {hagg['slot_ms']:.1f} ms = dense {agg_dram['dense']:.0f} MB（常駐 {hagg['resident_MB']:.0f} MB の残り）+ 状態/KV {agg_dram['state']:,.0f} MB（{aa['b']} users × GDN 状態 {36*S_USER_LAYER*2/1e6:.0f} MB 往復 /{aa['TP_a']} + KV）+ routed {agg_dram['routed']:,.0f} MB。演算 {hagg['comp_ms']:.1f} ms・通信 {hagg['comm_ms']:.1f} ms は DRAM の窓に隠れる。
- **L1**: 活性 {4*hagg['rows']*H*BF16/2**20:.0f} MiB（rows {hagg['rows']} × 4 × H bf16）が常駐 dense を {hagg['resident_MB']:.0f} MB に削る。残差 {hagg['rows']*HC_DIM/aa['TP_a']*BF16/2**20:.0f} MiB は {hagg['residual_in']}。
- **容量**: {hagg['cap_GB']:.1f} GB/chip（expert {EXPERTS_TOTAL/(aa['PP']*aa['EP']*aa['TP_x'])/1e9:.1f} + dense + {aa['b']} users の状態/KV）。b=2048 は ctx 192 まで、8k で DP1、32k で b=512、262k で 128、1M で 32 が容量の上限（§11.5.1）。
- **何が床か**: ユーザ状態の往復（漸近 {asym*1.376:,.0f} t/s）。重みの読みはこの b でほぼ償却されている（{hagg['agg']/(asym*1.376)*100:.0f}%）。

#### 11.6.3 系列長で形がどう変わるか

- b=1 の tsu 側: {ctx_forms_tsu()}
- agg 側は常駐 KV で b が縛られる: {ctx_forms_agg()}

### 11.7 今の実装との差と、限界へ近づける順番（ctx 192、k=1）

| 段階 | 値 |
|---|---|
{steps()}

今の形の効率 {1186.9/cur['agg']*100:.1f}% の差（{cur['agg']:,.0f} − {1186.9:,.0f}）は限界モデルの外（実装の stall・admission・lane の同期・router 後の読みの直列化・dispatch）にあり、作業記録 §9.4 の分解で追う。限界側で形を変える順: (1) router 後の読みを 2 段先読みに（全ミス → census → 全ヒット。同じ形では DRAM/pass 支配で agg は変わらず、tsu 側で効く）、(2) b=1 の tsu 最大の形（`{hb1['cfg']}`）へ: expert {hb1['xform']}、router {'複製' if hb1['router']=='replicated' else '列分割'}、PP{A(hb1)['PP']} の {hb1['rounds']} 周で次の枠を L1 に先読み、(3) agg 側は `{hagg['cfg']}` で b を漸近まで上げる。各段階の限界値は §11.5 の表の該当行。

### 11.8 未確定（決着方法）

- hop {HOP*1e9:.0f} ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §11.5.2 の感度の範囲（×2 で PP{bestsens(2.0)[0]}×TP{bestsens(2.0)[1]}）。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（最大点 `{hagg['cfg']}` を含む）はこの外挿に依存。
- k=1 の R_eff {TOKENS_PER_PASS[1]} は本件プロンプト集合（ISL 128）の採択率。投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖で、device の BF4 連鎖では未測。
- GDR の 1 単位 256 + 800 × 行 cycle は 27B §9 の 4,224 cycle/(head, user) を 4 単位に分けた机上値（単位間の k'S の交換は数えていない）。行列積の K 分割の木の段数・NoC の値は設計値。
- L1 のストリームバッファ 1 core {(L1_MM+L1_GDR+L1_KVC)//1024} KiB と固定専有 140 KiB/core、活性 4 × [rows, H] は設計値（感度: バッファ ×0 / ×2 で tsu 最大点 {sens_stream[0]:,.0f} / {sens_stream[1]:,.0f}）。
- 枠の群への割付は f mod PP の固定 round-robin（群間の量の差が残る近似）。空いている段の chip が隣の段の routed を自分の DRAM から読んで ETH で渡す形（入口 8 link × 25 GB/s = {8*25/512*100:.0f}% の DRAM 相当）は列挙していない。
- 系列長の掃引は k=1 の 4 ケースのみ。長文脈の prefill（chunked prefill、KV の書き込み）は decode の限界の外。1M は YaRN 前提で精度は限界の外。
- DCP の局所 top-k（各 chip が budget/DCP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。

### 11.9 生成・検証・運用

- 生成: `routing_census.py`（CPU oracle、34 item）→ `routing_stats.py` → `routing/stats.json`、`design/census_stage2_batches.py` → `routing/stats_stage2.json`；`run_free5d_v4.sh`（`free5d_v4.py routing/stats.json <ctx> <out> [k1]`: ctx 192 は 4 ケース × k 0/1、他の ctx は k=1、7 ctx 並列）→ `render_free5d_v4.py` → 本節；`install_free5d_v4.sh` が README の §11 を差し替える。
- 検証: 同じ入力から最初から再生成し、stats・stats_stage2・results・design・README が byte 一致（`sha256sum -c routing/V4.sha256`、`scripts/cycle_v4.sh` が census → 列挙 → 設計 → 設置 → 検算を 1 周する）、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。訂正があれば最初から。
- 運用: 作業記録は追記専用・日本語・実時間 UTC、ユーザの発言と答え（誤答も）を 1 往復ごとに記録する。README は「現在の確定状態」だけを書き、誤りは上書きで直す。数値・順位は生成値だけ。限界に実測値を入れない。

"""
sys.stdout.write(out)
