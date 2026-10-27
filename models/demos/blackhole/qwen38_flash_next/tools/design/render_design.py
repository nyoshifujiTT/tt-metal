#!/usr/bin/env python
r"""Render README section 12 (the detailed design of the chosen form at ctx 1, B=1, k=1) from the generated jsons of
steps 0-7 (routing/design_*.json, census_stage2.json).  Every number is computed here; nothing is typed in.
Usage: render_design.py <routing dir> > sec12.md"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import stages as SG
import comm_model as cm

D = sys.argv[1] if len(sys.argv) > 1 else "/data/yoshifuji/qwen38/routing"
MiB = 1024**2
KiB = 1024
BW = 512e9
R_EFF = 1.376


def J(n):
    return json.load(open(f"{D}/{n}"))


S0 = J("design_step0_units.json")
S2 = J("design_step2_account.json")
S4 = J("design_step4_item_time.json")
S6 = J("design_step6_comm.json")
CEN = J("census_stage2.json")["rows2"]
FORMS = ("A", "B", "Bn", "B+D")
C7 = {f"{c}_{f}": J(f"design_step7_chain_{c}_{f}.json") for c in ("hit", "census", "miss") for f in FORMS}
S3F = {f: J(f"design_step3_frames_{f}.json") for f in FORMS}
P3F = {f: J(f"design_step3p_place_{f}.json") for f in FORMS}
P4F = {f: J(f"design_step4p_{f}.json") for f in FORMS}
DRAM_F = {f: sum(S3F[f]["stored"]) for f in FORMS}
# revision 8: the chosen hc form = the lexicographic minimum of (hit T_pass, DRAM per pass) (README 11.3 rule 15)
CHOSEN = min(FORMS, key=lambda f: (round(C7[f"hit_{f}"]["T_steady"], 1), DRAM_F[f]))
S3 = S3F[CHOSEN]
S5 = J(f"design_step5_frame_time_{CHOSEN}.json")
S6 = J(f"design_step6_comm_{CHOSEN}.json")
S6B = J(f"design_step6b_hand_{CHOSEN}.json")
P3 = P3F[CHOSEN]
P4 = P4F[CHOSEN]
U = S2["units"]
U0 = S0["units"]
FR = S3["frames"]
T4 = S4
o = []
P = o.append
dense_total = S0["dense_bytes"]
expert_total = S0["expert_bytes"]
STO = S3["stored"]
chip_dense = sum(STO[i] for i, x in enumerate(U) if x["kind"] != "mm_expert")
chip_routed = sum(STO[i] for i, x in enumerate(U) if x["kind"] == "mm_expert")
chip_S = sum(x["S"] for x in U)
chip_rows = sum(x["in_"] + x["out"] + x["work"] for x in U)
chip_state = sum(x["state_l1"] for x in U)
chip_reads = sum(x["dram_read"] for x in U)
hit = C7[f"hit_{CHOSEN}"]
T = hit["T_steady"]
tsu = 1e6 / T * R_EFF


def tsu_of(c):
    return 1e6 / C7[c]["T_steady"] * R_EFF


P(f"## 12. 詳細設計（ctx 1、B=1、k=1）: `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1]`、hc 形 {CHOSEN} の確定状態\n")
P(
    f"この節は §11.6.1 の形を単品の段（コア・RISC・スレッドの役割、命令列、同期、NoC の経路と飛行）まで降ろして確定した設計（作業記録 §12 の手順 0〜7 と改訂 8 の段の設計 D-2〜D-7）で、実装がそのまま使う: 単品の割付（§12.1）、1 chip の L1 と DRAM の会計（§12.2）、枠（§12.3）、単品の時間 = 段の積み上げ（§12.4）、通信（§12.5）、鎖を追った定常状態 = 到達目標（§12.6）、実装への予算（§12.7）、限界モデル §11 との差（§12.8）、生成と検証（§12.9）。条件: 極短文（ctx 1: KV・index block を無視）、1 user、MTP k=1（verify 2 行）、数値形式は配備のもの。本文の数値は全て生成器 `scripts/design/*.py` の出力（`render_design.py` が本節を組む）。時間は全て仕様（tt-isa-documentation、EthernetSsDesignSpec ほか、§11.2.3）の命令数 × cycle から積み上げ、実測は入れない。文書に段数の無い 4 つの遅延は予算として置き、**この予算の場合には** の値: packer の段数 P_PACK {SG.P_PACK} cycle、unpacker の段数 P_UNP {SG.P_UNP} cycle、NIU の書き要求のペイロード読み N_RD {SG.N_RD} cycle、GDDR の読みの pipeline 0.3 µs（実装の段階でデバイスの合否確認で上限を超えないことを見る）。\n"
)
P(
    f"**到達目標（この形の 100%）: 1 pass {T:.1f} µs = {tsu:,.0f} tsu（全ヒット、hc 形 {CHOSEN}、{S3['R']} 周 {len(FR)} 枠、置き方は 14 通りの harvest の最悪、routed の取り置き 2 層）。** 内訳: 演算 {hit['comp']:.1f} + 会合の露出 {hit['expo']:.1f} + 読み待ち {hit['wait']:.1f} + 受け渡し {hit['hand']:.1f} + 折り返し {T-hit['comp']-hit['expo']-hit['wait']-hit['hand']:.1f} µs。DRAM 車線 {min(hit['busy'])*100:.0f}〜{max(hit['busy'])*100:.0f}%、L1 高水位 {min(hit['l1_hwm'])/MiB:.0f}〜{max(hit['l1_hwm'])/MiB:.0f} MiB / {S3['cap']/MiB:.1f}。census（2 段先読みの census のヒット率）{C7[f'census_{CHOSEN}']['T_steady']:.1f} µs = {tsu_of(f'census_{CHOSEN}'):,.0f} tsu、全ミス（今の runtime の形）{C7[f'miss_{CHOSEN}']['T_steady']:.1f} µs = {tsu_of(f'miss_{CHOSEN}'):,.0f} tsu。\n"
)
P("### 12.1 単品と割付（手順 0〜1）\n")
P(
    f"1 pass の単品は {len(U0)} 個（HF forward の順: embed → L00〜L47（GDN 36 / QSA 12、各層 MoE、L02 の前に PLE）→ final mixer → lm_head → argmax → MTP 層（mixer、fc、QSA、MoE、lm_head））。全体 dense {dense_total/MiB:,.1f} MiB、routed expert {expert_total*512*49/1e9:.2f} GB（49 層 × 512 × {expert_total/MiB:.2f} MiB BF4）、GDN 状態 S {S0['S_layer']*36/MiB:.1f} MiB/user fp32、conv tap {S0['conv_layer']*36/KiB:,.0f} KiB/user。\n"
)
P("| 単品の種類 | TP8 の分け方（1 chip の持ち分） | 会合 |")
P("|---|---|---|")
rows = [
    (
        "hc 読み（gr.norm / down / up / gate / inject）",
        ("残差 [2, 4H] を各 chip に複製（形 Bn）: attention 読みは down を N 分割（low-rank の行 40 本を残差の全体から）→ low-rank の AG、" if CHOSEN == "Bn" else "残差 [2, 4H] を各 chip に複製（形 B）: attention 読みは down を K 分割（部分和 [2, 4×(320+4+1)] fp32、枝ごとなので scale が可換）、")
        + "MoE 読みの low-rank は attention の出力の会合に相乗りした折り畳み F_i h_i で局所; up は N 分割、gate × normalized は局所、inject は down と pool",
        ("attention 読み: AG_low → AG_x（K chunk）" if CHOSEN == "Bn" else "attention 読み: AR_down（波）→ AG_x（K chunk）") + "; MoE 読み: 会合なし（AR_yf に相乗り）",
    ),
    (
        "GDN in_proj qkv / z / a / b",
        "value head 6（key head 2: q,k 2×2×128 + v 6×128 = 1,280 列）、z 6 head、a/b 6 列（tile pad 32）: 同じ入力なので pool",
        "無し",
    ),
    (
        "conv1d / l2norm / GDR / norm (gated)",
        "自分の列・head（GDR は 6 head × 16 単位（8 列ずつ）= 96 単位を 96 core に、S は単位の Dst に fp32 で置き SFPU で更新）、S はその chip の DRAM（b=1: 往復 {:.1f} MiB/pass）".format(
            2 * chip_S / MiB
        ),
        "無し",
    ),
    ("GDN out_proj / QSA o_proj", f"K 分割（6 value head / 3 q head の行）→ 部分和 [2, H] fp32 + 折り畳み F_i h_i [2, 1280]（形 {CHOSEN}）→ bf16 で AR（波）", "AR_yf（波）"),
    ("QSA q+gate / k,v / indexer", "q head 3（q+gate 1,536 列）、kv 1 head を群 4 枚で複製、indexer 複製（局所採点）: pool", "無し"),
    (
        "MoE router / shared",
        "router 複製（512 logit 局所、top-10 局所）、shared expert gate [1, H] も同じ pool、shared gate/up N 分割 1/8（80 列 = 3 tile、gate と up の同じ index の tile を対で 1 コア、owner で silu × up）、shared down K 分割（routed の down と K を連結した 1 つの累算、owner で gate 付きで足す）",
        "無し",
    ),
    (
        "routed expert gate_up / down",
        "tiles 形: expert e の 20 列 tile を e mod 8 から回し置き（最繁忙 chip = census 1x2 行 {:.1f} tile/層）、gate/up は同じ index の tile の対（owner で silu × up × score）、down は持つ tile の K 行 32 本を expert をまたいで K に連結した 1 つの累算（出力 H 列）".format(
            S2["busiest_tiles"]
        ),
        f"AR_yf に相乗り（形 {CHOSEN}: MoE ブロックの出力の会合）",
    ),
    ("PLE（L02 の前）", "key_proj N 分割 1/8、value_proj N 分割 1/8、n-gram 行は各 chip が読む、norm の統計は AR", "AR（統計）"),
    ("lm_head / argmax", "vocab 1/8（31,040 列）、局所 argmax（970 tile を 130 core）→ 候補 AG（8 × 2 行 × 8 B）", "AG_argmax"),
    (
        "投機 router（層 l の router を 1 / 2 ブロック早い残差に）",
        "層 l の router を持つ群で、自前の hc 読み + 会合 + router（鎖の外）。L00 は確定（embedding 行から）、MTP 層は予測不可（本 pass の argmax に依存）",
        f"形 {CHOSEN} の attention 読みと同じ会合（鎖の外）",
    ),
]
for a, b, c in rows:
    P(f"| {a} | {b} | {c} |")
P("")
P(
    "数値形式（配備の `ttnn/decode_matmul.py` の dense plan と同じ）: dense（GDN・QSA・shared・lm_head・final / MTP の mixer（改訂 6 までは bf16 と数えていた誤り）・MTP の fc）bf8（1,088 B/tile）、各層の hc mixer・router・norm・PLE bf16、routed expert BF4（576 B/tile）、活性 bf16、S fp32、行列積の部分和 fp32 DEST、線上の部分和 bf16。行列積の fidelity は配備どおり重みの形式で決まる（bf8 HiFi2、BF4 LoFi、bf16 HiFi4）。合格線は `docs/NUMERICS.md` の acceptance（pinned 表、`json` 96/96）。K 分割の足す順（置き方の木の順: `unit_assign.h`）・hc 統計の reduce の順・GDR の (value head, 8 列) の単位分割は fp32 の足し順を変えるだけで、丸めの回数（bf16 に落とす所）は配備と同じ。精度はデバイスで PCC と発散 index で判定する（K-6）。\n"
)
P("### 12.2 1 chip の L1 と DRAM の会計（手順 2）\n")
P(
    f"1 chip の 1 pass（全層の和。1 chip はこのうち自群の枠の分）: dense {chip_dense/MiB:,.1f} MiB（= 全体/8 + router 複製 + indexer 複製 + kv 射影の群内複製 + 形 {CHOSEN} の折り畳み + 8 列の群と半 tile への詰めの padding: stored）、routed tile {chip_routed/MiB:.1f} MiB（最繁忙 chip、全ヒット）、S {chip_S/MiB:.1f} MiB（往復 {2*chip_S/MiB:.1f}）、行と作業域 {chip_rows/MiB:.1f} MiB、L1 常駐状態（conv tap、PLE tap）{chip_state/KiB:.0f} KiB、DRAM だけの読み（embedding / n-gram 行）{chip_reads/KiB:.1f} KiB。DRAM の動く量 {(chip_dense+chip_routed+2*chip_S+chip_reads)/MiB:,.1f} MiB/pass/chip = 1 chip を直列に見て {(chip_dense+chip_routed+2*chip_S+chip_reads)/BW*1e3:.3f} ms。単品ごとの表は `routing/design_step2_account.md`。\n"
)
P("### 12.3 枠（手順 3）\n")
P(
    f"1 chip の L1 {S3['l1_chip']/MiB:.1f} MiB（130 core × 1,536 KiB − 固定専有 140 KiB/core: firmware 40 + kernel 100 の予算）"
    f"− 受け場所 {S3['landing']/MiB:.2f} MiB（全コアの同じ番地 {S3['landing_core']/KiB:.1f} KiB/core: 旗 64 語、x の受け 2 × rows × K {S3['kmax']:,} × bf16（multicast で受ける pool の最大 K の入力全体、2 面）、"
    f"Stream-K の部分和 {S3['fanin']} 個（全 pool の木の最大 fan-in）、会合の受け 2 × {S3['meet_max']:,.0f} B、受け渡しの受け {S3['hand_max']:,.0f} B; `stages.landing_core`、番地は `dataflow/design/chip_l1.h`）"
    f"− 常駐状態 {S3['state_l1']/MiB:.2f} = **枠の容量 {S3['cap']/MiB:.1f} MiB**（コアごとの枠の領域 {S3['region']/KiB:,.1f} KiB）。"
    f"改訂 8 の訂正 33: §11 の限界模型の「ストリームバッファ 192 KiB/core」は §12 の段の設計に対応する buffer が無い（重みは枠の列に常駐、S は枠に置き単位の作業は Dst、ctx 1 は KV チャンク無し）ので持たない。"
    f"枠の列 = 単品の重み（16 列の半 tile と 8 列の群に詰めた stored、形 {CHOSEN} の折り畳み込み: 訂正 32・41）+ S + 枠の単品の作業域の最大 + 残差（単品の行は実行中だけ受け場所にある: 訂正 42）。"
    f"1 pass の列（1 chip、全層の和）{sum(f['column'] for f in FR)/MiB:,.1f} MiB → **{len(FR)} 枠 = {S3['R']} 周**（全部入る最小の周: 27B の規則）。単品境でバイト均等に切る。"
    f"枠に入るかは chip の和ではなくコアごとの最大で決まる（§12.4.1 の置き方: 14 通りの harvest の全部で、全枠のコアの最大 ≤ 枠の領域）:\n"
)
P("| 枠 | 群 | 単品の範囲 | 列 [MiB] | 重み [MiB] | うち routed [MiB] | S [MiB] | DRAM/pass [MiB] | L1 の余り [MiB] |")
P("|---|---|---|---|---|---|---|---|---|")
for f in FR:
    P(
        f"| {f['frame']} | {f['group']} | {U[f['a']]['unit'].split(' ')[0]} 〜 {U[f['z']-1]['unit'].split(' ')[0]} | {f['column']/MiB:.1f} | {f['weights']/MiB:.1f} | {f['routed']/MiB:.1f} | {f['S']/MiB:.1f} | {f['dram']/MiB:.1f} | {f['left']/MiB:.1f} |"
    )
gd = [sum(f["dram"] for f in FR if f["group"] == g) for g in range(S3["PP"])]
P(
    f"\n群ごとの 1 chip の DRAM/pass {min(gd)/MiB:.1f}〜{max(gd)/MiB:.1f} MiB（偏り {(max(gd)/min(gd)-1)*100:.1f}%）= {min(gd)/BW*1e3:.3f}〜{max(gd)/BW*1e3:.3f} ms。枠の列は dense が鎖に消費された順に空くので、群が空いている間（他 3 群の番）に次の枠の dense を**全量**先読みできる（先読みに使える L1 = 容量 − (列 − dense) = {min(f['l1_free_idle'] for f in S5['frames'])/MiB:.0f}〜{max(f['l1_free_idle'] for f in S5['frames'])/MiB:.0f} MiB/枠、手順 5）。routed tile は投機 router 後にしか読めないので窓の中（routed 2 層分 ≈ {2*S2['busiest_tiles']*(3*640*2560*576/1024)/20/MiB:.1f} MiB を L1 に取り置く: §12.6 で 1〜5 層を比べた最速）。\n"
)
# ---------- 12.4 unit times ----------
P("### 12.4 単品の時間 = 段の積み上げ（手順 4 / 改訂 8 D-2、1 chip、仕様のみ）\n")


def layer_sum(pref):
    return sum(t["t_us"] for i, t in enumerate(T4) if not t["spec"] and U[i]["layer"] == pref)


t_gdn = layer_sum("L00")
t_qsa = layer_sum("L03")
t_ple = sum(t["t_us"] for i, t in enumerate(T4) if ".ple." in U[i]["unit"])
t_tail = sum(t["t_us"] for i, t in enumerate(T4) if not t["spec"] and not U[i]["layer"].startswith("L"))
t_pass = sum(t["t_us"] for t in T4 if not t["spec"])
t_spec = next(t["t_us"] for i, t in enumerate(T4) if U[i]["unit"].startswith("L00.moe.speculative router"))
t_hcr = next(t["t_us"] for i, t in enumerate(T4) if U[i]["unit"].startswith("L00.moe.gr.down")) + next(t["t_us"] for i, t in enumerate(T4) if U[i]["unit"].startswith("L00.moe.gr.up"))
by = {}
for i, t in enumerate(T4):
    if U[i]["layer"] == "L00" and not t["spec"] and t["t_us"] > 0.05:
        by[U[i]["unit"].split(" ")[0]] = t["t_us"]
P(
    f"単品の時間は段の積み上げ（`stages.py`: 仕様の定数に出典を付け、単品の種類ごとに段の cycle を足す）。行列積の pool（同じ入力を読む行列積を並べる）の時間 = **実際に作れるコアの割付**の時間（`pool_plan.py`、改訂 8 の訂正 18: 閉じた式「仕事 / 130 + 一様な木」は K 単位が整数で列ごとに K が違うので作れない）。割付は 3 方式の最速: A（列ごとにコア数 q を貪欲に足し、各コアは K 単位の連続区間、部分和は heap の木で owner へ）、B（線形 Stream-K: (列, K 単位) の列を 130 の等 cycle 区間に単位の境で切る。複数のコアにまたがる列の owner は、その列が自分の区間の最後の列になるコア = またがる最初のコアで、区間が次の列へ続くコアは葉）、C（列を丸ごと、長い順に空いたコアへ）。列ごとの完了は、コアの実際の終わりから木を追って数える: MAC（tile-MAC = max(MVMUL の命令数, 重み tile の SrcA への unpack): MVMUL.md・MatrixUnit.md の 8 行 × 16 × 16 を 1 cycle、fidelity の phase ごとに 1 本、UNPACR_Regular.md の BH 128 B/cycle → 2 行で bf8 HiFi2 8、BF4 LoFi 8、bf16 HiFi4 16 cycle）+ 送り手（SFPU の fp32 finalize {SG.finalize_partial(2)}、pack 2 行 {SG.pack(4)}、NoC 書き {SG.issue()} + N_RD、2 hop の飛行）+ 木の段（poll {SG.poll()}、部分和の Dst への unpack、SFPU の足し込み）+ owner（finalize、pack、書き、後処理）。コアごとの割付は `dataflow/design/unit_assign.h`。MoE（訂正 19〜21）: gate/up は tile の対（同じ index の gate と up を同じコア）で owner が silu × up（× score）、routed の down は K = この chip の expert tile + shared の 80 列を連結した 1 つの累算で owner が routed と gate 付き shared を足す。+ 出力の受け渡し（owner → hub {SG.UP_OWN[40]}/{SG.UP_OWN[80]}/{SG.UP_HOPS} hop（owner を hub の近くに置く。13×10 の格子・14 通りの harvest の最悪）、hub の検知と発行、NoC0 と NoC1 の 2 つの長方形への multicast {SG.MC_HOPS} hop、受け手の検知）。飛行は 5 + 9 × hop + 5 + flit（NoC/README.md）。要素演算は owner の finalize に融合（SFPU の fp32 関数の命令数: silu {SG.ACT['silu']}、sigmoid {SG.ACT['sigmoid']}、rsqrt {SG.ACT['rsqrt']} cycle/群）。GDR は (value head, 8 列) の単位 = S [128, 8] fp32 を Dst に置き 1 行 2 パスを SFPU で（列の和は SFPTRANSP で行レーンに複製）、2 行の直列で {SG.gdr_unit(2, 8)} cycle。router の top-10 は 16 個の router owner がそれぞれ top-10 を取り選択コアで併合 {SG.topk10()} cycle。SDPA（ctx 1）は kv head のコアで 3 q head × 2 行を解き o_proj のコアへ直接 multicast。argmax は各コアの約 8 logit 列 tile を SFPSWAP で併合して 1 hop。1 chip の鎖（step4 の単品の和、hc 形 A の単品で数える; 採る形の鎖は §12.6）: **層 L00（GDN + MoE）{t_gdn:.1f} µs**（"
    + "、".join(f"{k} {v:.2f}" for k, v in by.items())
    + f"）、**層 L03（QSA + MoE）{t_qsa:.1f} µs**、PLE {t_ple:.1f}、tail（mixer + lm_head × 2 + argmax + MTP の fc）{t_tail:.1f} → **1 pass の鎖 {t_pass:.1f} µs**。投機 router {t_spec:.2f} µs/本（鎖の外）。単品ごとの表は `routing/design_step4_item_time.md`。\n"
)
# ---------- 12.5 communication ----------
# ---------- 12.4b placement (corrections 30-44) ----------
P("#### 12.4.1 置き方（コアの物理配置）と、置いた時間（改訂 8 の訂正 30〜44）\n")
P(
    f"単品の行列積の pool の論理コア k を、harvest の場合ごとのラスター順の道 P（x の昇順に列、各列は y 2 → 11）の P[(k + o) mod 130] に置く（`raster.py`、`placement.py`）。"
    f"後の論理コアは前の論理コアから右・下にあるので、子 → 親の部分和は NoC1（左・上）で、木の辺の hop は論理の距離と列の境で決まる。pool ごとの回し o は、枠の全 pool の置いた時間の和を最小にし、"
    f"コアごとの L1（置いた重み + 枠の S・状態・残差・作業域の最大の分担）≤ 枠の領域 {P3['region']/KiB:,.1f} KiB を制約にする（局所探索 → はみ出す枠は min-max の均し → 入る範囲で時間を戻す）。"
    f"pool の時間は、置いた後の実際の hop（木の各辺、owner → hub）で数え直し、方式（列ごと / 線形 Stream-K / 列丸ごと）も置き方の上で選び直す。設計値は 14 通りの harvest の最悪。"
    f"採る形 {CHOSEN} の {len(P3['frames'])} 枠の全部が、14 通りの harvest のすべてで入る: コアの最大 {min(f['max'] for f in P3['frames'])/KiB:,.1f}〜{max(f['max'] for f in P3['frames'])/KiB:,.1f} KiB。"
    f"行列積の時間（1 pass、1 chip の単品の和）: 置いた値 {sum(u['mm_us'] for u in P4['units'] if not u['spec']):.1f} µs、段の模型（hop の楽観の定数）{sum(u['mm_model_us'] for u in P4['units'] if not u['spec']):.1f} µs。"
    f"置き方は `dataflow/design/unit_assign.h` に出す（pool ごとのコアの割付と、harvest ごとの回し）。\n"
)
P("### 12.5 通信（手順 6）\n")
P(
    f"通信の時間は packet 模型（`comm_model.py`: 隣の chip へ 1 方向 2 チャネル × 25 GB/s（チャネルごとに FIFO、packet は先に空くチャネルへ、chunk は最少 packet 数と 2 の倍数の等分の速い方）、hop 306 ns（EthernetSsDesignSpec Table 16 の 400G の 280 ns + 200G の RS-FEC の codeword の組 25.6 ns）、送り手の蓄積転送 96 B/cycle、packet 2 KiB + 56 B、経路は縦 → 環の短い向き、collective は BFS 木の multicast / 直送 / 木の reduce の速い形）。chip 内の段（改訂 8 D-4、会合の種類ごと）: 足し込み系（AR、RS の部分和）の起点 = 部分和を作った owner（Ethernet タイルから {cm.NOC_ETH_CORE} hop 以内にある 91 コア以上の中に置く: core_map の 14 通りの harvest で確認）が hub と Ethernet タイルへ 2 本の書き（2 × (発行 + N_RD)）+ {cm.NOC_ETH_CORE} hop + 送り側の検知と送り {cm.ETH_SEND} cycle = {round(cm.ORIGIN_ADD * cm.CLK)} cycle + flit、足し込みの段 = 受けた Ethernet → owner {cm.NOC_ETH_CORE} hop + 検知 + Dst への unpack + SFPU の足し込み = {round(cm.ADD_LEVEL * cm.CLK)} cycle; 集め（AG）の起点 = 作った pool の hub（Ethernet タイルから {cm.NOC_HUB_ETH} hop の位置に選ぶ）→ Ethernet {cm.NOC_HUB_ETH} hop + 検知と送り = {round(cm.ORIGIN_GATHER * cm.CLK)} cycle + flit、終点 = 受けた Ethernet タイルが 2 本の multicast で全 130 コアへ（最悪 {cm.NOC_ETH_ALL} hop）+ 受け手の検知 = {round(cm.DELIVER_GATHER * cm.CLK)} cycle + flit; 中継 = 受け側の検知 + 同じ側の対の Ethernet タイルへ {cm.NOC_ETH_ETH} hop の NoC 書き + 送り側の検知 + 送り = {cm.RELAY_CYC} cycle + flit; 群間の受け渡し = 足し込み系の起点 + 受けた chip で全コアへ配る段（集めの終点と同じ）。群 = 2×4 ブロック、群間は 2 hop（列の 2 chip が群間 link を共有）。\n"
)
P("| 会合 | 中身 | payload/chip [B] | 速い形 | 時間 [µs] |")
P("|---|---|---|---|---|")
names = {
    "AR_down": "hc 読みの部分和 + 統計（枝ごと [2, 4×325] fp32）",
    "AG_low": "低ランク行 40 本（形 Bn）",
    "AG_x": "ブロック入力 x [2, H/8] bf16",
    "RS_y": "ブロック出力の部分和 [2, H] bf16",
    "AR_y": "同上の all-reduce（複製残差）",
    "AR_yf": "y + 折り畳み f [2, H+1280] bf16",
    "AG_ids": "予測 expert id 40 B",
    "AG_argmax": "argmax 候補 8 × 2 行 × 8 B",
}
for k, v in names.items():
    m = S6[k]
    P(f"| {k} | {v} | {m['bytes']:,.0f} | {m['form']} | {m['us']:.2f} |")
pa = S6["parts"]
P(
    f"\n鎖に乗る露出（会合の全体ではなく、演算と重ねて残る分）: AG_x は届いた K chunk から射影を進めるので露出 GDN {pa['ag_gdn']:.2f} / QSA {pa['ag_qsa']:.2f} / MoE {pa['ag_moe']:.2f} µs；RS_y / AR_down は出力を 10 波で出し波ごとに流すので露出 RS_y {pa['rs_attn']:.2f}（attention）/ {pa['rs_moe']:.2f}（MoE）、AR_down {pa['ar_down']:.2f} µs。投機 router の会合（AR_down {pa['ar_down_full']:.2f} + AG_x {S6['AG_x']['us']:.2f}）は鎖の外。\n"
)
P("hc の形（残差の置き方と down/up の分け方）を列挙して手順 7 で時間比較した:\n")
P("| 形 | 1 層の hc 区間（出力行列積 + 会合 + down + 会合 + up + AG_x 露出、GDN / QSA; 採る形の置いた時間で）[µs] | pass [µs] | DRAM 増 [MiB/chip/pass]（形 A に対する stored の増: 訂正 41） | 中身 |")
P("|---|---|---|---|---|")
for k, f in S6["forms"].items():
    dm = f"{(DRAM_F[k] - DRAM_F['A']) / S3F['A']['PP'] / MiB:+.1f}" if k in DRAM_F else "（§12.6 で比べない）"
    P(f"| {k} | {f['gdn']:.2f} / {f['qsa']:.2f} | {f['pass_us']:.0f} | {dm} | {f['desc']} |")
P("")
P("| 群間の受け渡し（枠の境） | 生きている行 | bytes/chip | [µs] |")
P("|---|---|---|---|")
VALNAME = {"residual": "残差 1/8", "token ids": "token id", "x": "x [2, H]", "gr.inject": "inject", "gr.norm": "正規化残差", "norm": "gated norm の行", "router": "router の結果", "shared_act": "shared の silu × up 行", "expert_act": "expert の行", "conv1d": "conv 出力", "gr.down": "低ランク [2, 320] fp32"}


def valname(k):
    v = k.split(":")[-1]
    return VALNAME.get(v, v)


for h in S6B["hand"]:
    P(f"| {h['before']} \\| {h['after']} | {' + '.join(f'{valname(k)} {b:,.0f}' for k, b in h['alive'].items())} | {h['bytes']:,.0f} | {h['us']:.2f} |")
P(
    f"\n受け渡し合計 {sum(h['us'] for h in S6B['hand']):.1f} µs/pass（step 6b: 枠の切れ目ごとに、切れ目の前で作られ後で使われる値を HF のデータの流れから数える; 最大 {max(h['bytes'] for h in S6B['hand']):,.0f} B は枠の容量が受け場所に仮置きした {S6B['bound']:,.0f} B 以下）（形 B 系は受け手で残差の AG +{S6['hand_B_extra']:.2f} µs/境）。投機 router が群をまたぐ先頭層: 前の群が残差 1/8（5 KiB）を 2 hop {pa['res_hop']:.2f} µs で送り、受け手（空いている群）が自前の重みで読み + router を回す。link の共有: 群内の会合（BFS 木）は環の E/W とブロック内の縦 link だけを使い、群間の縦 link（受け渡し、残差 hop、router と expert が別枠に落ちる切れ目 3・6 の id hop）とは物理的に別で、群間 link に乗る 3 本は時刻が重ならない（受け渡しは枠の最後、残差 hop と id hop は枠の途中）ので予定表は要らない。\n"
)
# ---------- 12.6 steady state ----------
P("### 12.6 鎖を追った定常状態 = 到達目標（手順 7）\n")
P(
    f"`step7_chain.py`（離散事象、12 pass 流して後半 6 pass の平均）: 単品を HF 順に、前の単品の終わり + 会合の露出（+ 枠の境の受け渡し）から始め、読みが届くまで待つ（届いた分から消費、最後の片の後 t/8）。群ごとの DRAM 車線（512 GB/s）は 512 KiB の片を消費順に発行し、routed は投機 router のトリガ後、L1 は積み上げ（先の routed 2 層分の余地を取り置く = 27B 大原則 1 の本件版。取り置き 1〜5 層を比べた最速）。routed の発行時刻: P2 = 層 l−1 の attention 開始 + 自前の hc 読み {t_hcr:.2f} + 会合 + router {t_spec:.2f} + 発行 {SG.routed_issue()/SG.CLK*1e6:.2f}（router の結果 → 8 bank の DRISC へ 8 本の unicast（DRAM タイルへは multicast できない）→ 表引き + DRISC の 1 頁 → GDDR の pipeline 0.3 µs（予算）→ 最初の頁がコアへ; §12.7; 群をまたげば + 残差 hop）、P1 = 同じく層 l、外れ = router 後 + {SG.routed_issue()/SG.CLK*1e6:.2f}。L00 は pass 開始 + {S6['parts']['down_r'] + S6['parts']['up_r'] + t_spec + SG.routed_issue()/SG.CLK*1e6:.1f} µs（確定: 全行の hc 読み + router + 発行）、MTP 層は自分の attention 開始（予測不可）。\n"
)
P("| ケース / hc 形 | T_pass [µs] | tsu | 演算 | 会合の露出 | 読み待ち（dense / routed） | 受け渡し | 車線 | L1 高水位 [MiB] |")
P("|---|---|---|---|---|---|---|---|---|")
lab = {f"hit_{f}": f"全ヒット / {f}" for f in FORMS}
lab.update({f"census_{f}": f"census（2 段先読み）/ {f}" for f in FORMS})
lab.update({f"miss_{f}": f"全ミス（router 後）/ {f}" for f in FORMS})
for c, l in lab.items():
    r = C7[c]
    P(
        f"| {l} | **{r['T_steady']:.1f}** | **{tsu_of(c):,.0f}** | {r['comp']:.1f} | {r['expo']:.1f} | {r['wait']:.1f}（{r['waits']['dense']:.1f} / {r['waits']['routed']:.1f}） | {r['hand']:.1f} | {min(r['busy'])*100:.0f}〜{max(r['busy'])*100:.0f}% | {min(r['l1_hwm'])/MiB:.0f}〜{max(r['l1_hwm'])/MiB:.0f} |"
    )
P("")
P("| 枠 | 群 | 演算 [µs] | 始まり → 終わり [µs]（幅） | 会合の露出 | 読み待ち | 受け渡し |")
P("|---|---|---|---|---|---|---|")
for r in hit["frames"]:
    P(
        f"| {r['frame']} | {r['group']} | {r['comp']:.1f} | {r['start']:.1f} → {r['end']:.1f}（{r['end']-r['start']:.1f}） | {r['expo']:.1f} | {r['wait']:.1f} | {r['hand']:.2f} |"
    )
P(
    f"\n読み: 採る形は全ヒット（効率 100% の分母のケース）の (T_pass, DRAM 増) の辞書順の最小 = **形 {CHOSEN}**（§11.3 原則 15; "
    + "、".join(f"{f} {C7[f'hit_{f}']['T_steady']:.1f} µs / 重み {DRAM_F[f]/MiB:,.1f} MiB/pass（全層の和、1 chip はその 1/{S3F[f]['PP']}）・{S3F[f]['R']} 周" for f in FORMS)
    + f"）。census と全ミスの最速はそれぞれ {min(FORMS, key=lambda f: C7[f'census_{f}']['T_steady'])}（{min(C7[f'census_{f}']['T_steady'] for f in FORMS):.1f} µs）と {min(FORMS, key=lambda f: C7[f'miss_{f}']['T_steady'])}（{min(C7[f'miss_{f}']['T_steady'] for f in FORMS):.1f} µs）。改訂 8 で会合 1 本の段（chip 内の検知・発行・NoC 区間、200G の FEC）が重くなり、残差を複製して会合を減らす B 系の利点が、折り畳み重みと受け手の残差 AG の分を上回った（改訂 7 では A が最良だった）。B+D は D の複製で列が増え、周も増える。本件 b=1 は「DRAM 車線 {min(hit['busy'])*100:.0f}〜{max(hit['busy'])*100:.0f}% + 鎖」の釣り合いの点にある。全ヒットでも残る routed の露出 {hit['waits']['routed']:.1f} µs は L00（pass 開始後にしか発行できない）と MTP 層（argmax 後にしか予測できない）の分で構造的。投機 router の演算 {hit['spec_total']:.0f} µs/pass は鎖の待ち {hit['expo']+hit['wait']:.0f} µs に入る。\n"
)
# ---------- 12.7 budgets ----------
P("### 12.7 実装への予算（効率 100% = §12.6 の値を出す条件）\n")
P(
    f"- 単品の演算: §12.4 の段の積み上げ（`stages.py`）を §12.4.1 の置き方の上で数えた時間（14 通りの harvest の最悪）。1 pass {hit['comp']:.1f} µs。各段の cycle がそのまま実装の予算で、コンパイル時定数のヘッダに固定する（設計の正はヘッダ: §12.9）。\n- 会合: §12.5 の露出（chip 内の段を含む; AG_x は K chunk で射影が追いかける、RS/AR は 10 波で流す、投機の会合は鎖の外）。1 pass {hit['expo']:.1f} µs。\n- 受け渡し: 枠の境で生きている行だけを 2 hop（形 {CHOSEN} は受け手で残差の AG; {hit['hand']:.1f} µs/pass）。\n- DRAM: 各群 {min(gd)/MiB:.0f}〜{max(gd)/MiB:.0f} MiB/pass を 512 GB/s の {min(hit['busy'])*100:.0f}〜{max(hit['busy'])*100:.0f}% で（`dram_plan.h`: 各コアは置いた重みを ≤ 2 本の流れ（bank、端点タイル、NOC1）で読み、bank は枠の 1/8 ずつ、link の使用率 ≤ 0.5 の整数計画; 片 512 KiB、消費順の発行、全コア共通の周期（I-1b' の 99.35% の形）、routed は 2 段先読みと外れのトリガで）。読み待ち {hit['wait']:.1f} µs 以内。\n- L1: 枠の容量 {S3['cap']/MiB:.1f} MiB、routed の取り置き 2 層分、高水位 {max(hit['l1_hwm'])/MiB:.0f} MiB。\n- 投機 router: 層 l の router を持つ群で、attention ブロック開始の残差から 2 本（l、l+1）を鎖の待ち時間に回す。ヒット率は census（h12 {CEN['pass']['hit12_mean']:.3f}、h2 {CEN['pass']['hit2_mean']:.3f}）で、外れ分は router 後に読む（census ケース {C7[f'census_{CHOSEN}']['T_steady']:.1f} µs がその場合の到達値）。\n- 文書に段数の無い遅延の予算（この予算の場合には）: P_PACK {SG.P_PACK}、P_UNP {SG.P_UNP}、N_RD {SG.N_RD} cycle、GDDR の読みの pipeline 0.3 µs。実装の最初に、デバイスでこれらの上限を超えないことを合否で確かめる（測って値を決めるのではない）。\n- routed の発行（`stages.routed_issue`）: router の単品の終わり → 選択コアが併合の直後に id を 8 bank の DRISC へ 8 本の unicast（DRAM タイルへは multicast できない: NoC/README.md:32）→ DRISC の表引きと 1 頁 {SG.DRISC_PAGE} cycle → GDDR の pipeline → 最初の頁（2 KiB、最悪 {SG.DRAM_HOPS} hop）がコアへ = {SG.routed_issue()} cycle（softmax と multicast の後の段と並行）。\n"
)
# ---------- 12.8 diff to the limit ----------
P("### 12.8 限界モデル §11 との整合\n")


def same_form(r):
    """the chosen form PP4 x 8chip[TP8 | EP1 x TPx8] b=1, whatever round count the enumeration picked for the case"""
    a = r["axes"]
    return a["DP"] == 1 and a["PP"] == 4 and a["TP_a"] == 8 and a["EP"] == 1 and a["CP"] == 1 and a["b"] == 1 and a["n_mb"] == 1


RES = json.load(open(f"{D}/free5d_v8_results_ctx1.json"))
v8 = {c: next(r for r in RES[f"{c}|k=1"] if same_form(r)) for c in ("hit", "census", "miss")}
best = {c: RES[f"{c}|k=1"][0] for c in ("hit", "census", "miss")}
best_hit_tsu = max(RES["hit|k=1"], key=lambda r: r.get("tsu", 0))
v8T = v8["hit"]["slot_ms"] * 1e3
P(
    f"§11（改訂 8: `free5d_v8.py`、本節と同じ段の模型 `stages.py` / `comm_model.py` を使う）の同じ軸の形（`{v8['hit']['cfg']}`、ctx 1、全ヒット）は {v8T:.1f} µs（{1e6/v8T*R_EFF:,.0f} tsu）= 演算 {v8['hit']['comp_ms']*1e3:.1f} + 会合 {v8['hit']['comm_ms']*1e3:.1f} + 受け渡し・窓 {v8T-v8['hit']['comp_ms']*1e3-v8['hit']['comm_ms']*1e3:.1f}、本節 {T:.1f} µs（{tsu:,.0f} tsu）: 差 {T-v8T:+.1f} µs。"
)
P(
    f"§11 は層の中で DRAM 車線を平滑化する上限（層 = max(鎖, routed/BW)）で、L1 の取り置きと読み待ちを時間軸で追わない（§11.3 原則 24）。そのため §11 は周数を増やす形・dense を複製する形（hc B+D、周 R4）を速く見積もるが、時間軸で追う本節では、枠の境の受け渡しが倍になり（R4）、複製した dense が L1 の取り置きを圧迫して読み待ちが出る（B+D）ので、どちらも採る形（{CHOSEN}・R2）より遅い（§12.6 の表、R4 は作業記録）。設計の到達値（効率 100% の分母）は本節の値、§11 は形の比較と上限。census と全ミスでも同様（§11 census {v8['census']['slot_ms']*1e3:.1f} / 本節 {C7[f'census_{CHOSEN}']['T_steady']:.1f} µs、全ミス {v8['miss']['slot_ms']*1e3:.1f} / {C7[f'miss_{CHOSEN}']['T_steady']:.1f} µs）。\n"
)
P("### 12.9 生成と検証\n")
P(
    "- 生成: `scripts/design/run_design.sh`（step0 → step2 → step4 → step6 → step3 → step5 → census_stage2 → step7 の各ケース、入力は HF 定義・`routing/stats.json`・`routing/*.npz`・仕様の値だけ; 単品と会合の段は `stages.py`・`pool_plan.py`・`comm_model.py`; step3 は step6 の会合と受け渡しの最大と step4 の pool の fan-in から受け場所を引く）→ `render_design.py` → 本節；`install_design.sh` が README の §12 を差し替える。\n- **設計の正はヘッダ**（`dataflow/design/`、`tools/design/gen_headers.sh` が生成; `dram_plan.h` は harvest・枠・コアごとの DRAM の読みの流れ（bank、端点タイル、NoC、バイト ≤ 2 本: bank 均等、link の使用率 ≤ 0.5 の整数計画）、`unit_assign.h` は (pool, harvest, 回し) ごとの方式・予算・仕事と親）: `core_map.h`（harvest 14 通りの hub・multicast の長方形・owner の順・Ethernet 近傍・中継の対）、`stage_costs.h`（仕様の定数・予算・段の cycle 予算）、`design_plan.h`（1 pass の単品と枠）、`chip_l1.h`（1 コアの L1 の番地）、`coll_routes.h`（2×4 群の BFS 木と群間の受け渡しの経路）、`unit_assign.h`（pool ごとのコアの割付と cycle 予算）。各ヘッダの static_assert と host 検査 `dataflow/design_tests/test_qwen38_design_host.cpp`（デバイス不要）がヘッダの継ぎ目を検査する。カーネルは設計値をこれらのヘッダからコンパイル時に引き、ランタイム引数は実行時の HW の事実（DRAM のバッファの base、harvest の場合番号と自コアの index、Ethernet のチャネル）だけにする。カーネルの値とヘッダの対応（H-3）: 役割・K の範囲・列・部分和の親・fan-in → `unit_assign.h`（harvest と自コアのラスター順の index から引く）; hub と multicast の長方形・owner の順 → `core_map.h`; 単品の形・枠の範囲 → `design_plan.h`; L1 の番地（x の受け・部分和・会合と受け渡しの受け・旗・枠の領域）→ `chip_l1.h`; 会合の木・方向・packet 数と大きさ・群間の受け渡しの経路 → `coll_routes.h`; DRAM の読みの bank・端点・NoC・バイト → `dram_plan.h`; 段の cycle 予算（デバイスでの合否の線）→ `stage_costs.h` と `unit_assign.h` の pool の予算。\n- 検証: 同じ入力から再生成して `routing/design_*.json`・README・ヘッダが byte 一致、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。\n- 作業記録: `BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §12〜（手順ごとの結果と見つけて直した誤り）。\n"
)
print("\n".join(o))
