#!/usr/bin/env python
r"""Render README section 12 (the detailed design of the chosen form at ctx 1, B=1, k=1) from the generated jsons of
steps 0-7 (routing/design_*.json, census_stage2.json).  Every number is computed here; nothing is typed in.
Usage: render_design.py <routing dir> > sec12.md"""
import json
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "/data/yoshifuji/qwen38/routing"
MiB = 1024**2
KiB = 1024
BW = 512e9
R_EFF = 1.376


def J(n):
    return json.load(open(f"{D}/{n}"))


S0 = J("design_step0_units.json")
S2 = J("design_step2_account.json")
S3 = J("design_step3_frames.json")
S4 = J("design_step4_item_time.json")
S5 = J("design_step5_frame_time.json")
S6 = J("design_step6_comm.json")
CEN = J("census_stage2.json")["rows2"]
C7 = {
    c: J(f"design_step7_chain_{c}.json")
    for c in ("hit_A", "census_A", "miss_A", "hit_B", "hit_Bn", "hit_B+D", "hit_A_S")
}
U = S2["units"]
U0 = S0["units"]
FR = S3["frames"]
T4 = S4
o = []
P = o.append
dense_total = S0["dense_bytes"]
expert_total = S0["expert_bytes"]
chip_dense = sum(x["w"] for x in U if x["kind"] != "mm_expert")
chip_routed = sum(x["w"] for x in U if x["kind"] == "mm_expert")
chip_S = sum(x["S"] for x in U)
chip_rows = sum(x["in_"] + x["out"] + x["work"] for x in U)
chip_state = sum(x["state_l1"] for x in U)
chip_reads = sum(x["dram_read"] for x in U)
hit = C7["hit_A"]
T = hit["T_steady"]
tsu = 1e6 / T * R_EFF


def tsu_of(c):
    return 1e6 / C7[c]["T_steady"] * R_EFF


P(f"## 12. 詳細設計（ctx 1、B=1、k=1）: `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1]` の確定状態\n")
P(
    f"この節は §11.6.1 の形を単品の粒まで降ろして確定した設計（作業記録 §12 の手順 0〜7）で、実装がそのまま使う: 単品の割付（§12.1）、1 chip の L1 と DRAM の会計（§12.2）、枠（§12.3）、単品の時間（§12.4）、通信（§12.5）、鎖を追った定常状態 = 到達目標（§12.6）、実装への予算の配り方（§12.7）、限界モデル §11 との差（§12.8）、生成と検証（§12.9）。条件: 極短文（ctx 1: KV・index block を無視）、1 user、MTP k=1（verify 2 行）、数値形式は配備のもの。本文の数値は全て生成器 `scripts/design/*.py` の出力（`render_design.py` が本節を組む）。\n"
)
P(
    f"**到達目標（この形の 100%）: 1 pass {T:.1f} µs = {tsu:,.0f} tsu（全ヒット、hc 形 A）。** 内訳: 演算 {hit['comp']:.1f} + 会合の露出 {hit['expo']:.1f} + 読み待ち {hit['wait']:.1f} + 受け渡し {hit['hand']:.1f} + 折り返し {T-hit['comp']-hit['expo']-hit['wait']-hit['hand']:.1f} µs。DRAM 車線 {min(hit['busy'])*100:.0f}〜{max(hit['busy'])*100:.0f}%、L1 高水位 {min(hit['l1_hwm'])/MiB:.0f}〜{max(hit['l1_hwm'])/MiB:.0f} MiB / {S3['cap']/MiB:.1f}。census（2 段先読みの実測ヒット率）{C7['census_A']['T_steady']:.1f} µs = {tsu_of('census_A'):,.0f} tsu、全ミス（今の runtime の形）{C7['miss_A']['T_steady']:.1f} µs = {tsu_of('miss_A'):,.0f} tsu。\n"
)
P("### 12.1 単品と割付（手順 0〜1）\n")
P(
    f"1 pass の単品は {len(U0)} 個（HF forward の順: embed → L00〜L47（GDN 36 / QSA 12、各層 MoE、L02 の前に PLE）→ final mixer → lm_head → argmax → MTP 層（mixer、fc、QSA、MoE、lm_head））。全体 dense {dense_total/MiB:,.1f} MiB、routed expert {expert_total/1e9:.2f} GB（49 層 × 512 × 2.64 MiB BF4）、GDN 状態 S {S0['S_layer']*36/MiB:.1f} MiB/user fp32、conv tap {S0['conv_layer']*36/KiB:,.0f} KiB/user。\n"
)
P("| 単品の種類 | TP8 の分け方（1 chip の持ち分） | 会合 |")
P("|---|---|---|")
rows = [
    (
        "hc 読み（gr.norm / down / up / gate / inject）",
        "残差 [2, 4H] を列で 1/8（形 A）: norm の統計は局所和、down は K 分割（部分和 [2, 4×(320+4+1)] fp32、枝ごとなので scale が可換）、up は N 分割、gate × normalized は局所、inject は K 分割（down と pool）",
        "AR_down（波）→ AG_x（K chunk）",
    ),
    (
        "GDN in_proj qkv / z / a / b",
        "value head 6（key head 2: q,k 2×2×128 + v 6×128 = 1,280 列）、z 6 head、a/b 6 列（tile pad 32）: 同じ入力なので pool",
        "無し",
    ),
    (
        "conv1d / l2norm / GDR / norm (gated)",
        "自分の列・head（GDR は 6 head × 4 単位 = 24 単位を 24 core に）、S はその chip の DRAM（b=1: 往復 {:.1f} MiB/pass）".format(
            2 * chip_S / MiB
        ),
        "無し",
    ),
    ("GDN out_proj / QSA o_proj", "K 分割（6 value head / 3 q head の行）→ 部分和 [2, H] fp32 → bf16 で RS（波）", "RS_y（波）"),
    ("QSA q+gate / k,v / indexer", "q head 3（q+gate 1,536 列）、kv 1 head を群 4 枚で複製、indexer 複製（局所採点）: pool", "無し"),
    (
        "MoE router / shared",
        "router 複製（512 logit 局所、top-10 局所）、shared gate/up N 分割 1/8（80 列 → pad 96）、shared down K 分割（expert down と pool）",
        "無し",
    ),
    (
        "routed expert gate_up / down",
        "tiles 形: expert e の 20 列 tile を e mod 8 から回し置き（最繁忙 chip = census 1x2 行 {:.1f} tile/層）、down は tile の K 行 32 本".format(
            S2["busiest_tiles"]
        ),
        "RS_y に相乗り",
    ),
    ("PLE（L02 の前）", "key_proj N 分割 1/8、value_proj N 分割 1/8、n-gram 行は各 chip が読む、norm の統計は AR", "AR（統計）"),
    ("lm_head / argmax", "vocab 1/8（31,040 列）、局所 argmax（970 tile を 130 core）→ 候補 AG（8 × 2 行 × 8 B）", "AG_argmax"),
    (
        "投機 router（層 l の router を 1 / 2 ブロック早い残差に）",
        "層 l の router を持つ群で、自前の hc 読み + 会合 + router（鎖の外）。L00 は確定（embedding 行から）、MTP 層は予測不可（本 pass の argmax に依存）",
        "AR_down + AG_x（鎖の外）",
    ),
]
for a, b, c in rows:
    P(f"| {a} | {b} | {c} |")
P("")
P(
    "数値形式: dense bf8（1,088 B/tile）、hc・router・norm・mixer bf16、routed expert BF4（576 B/tile）、活性 bf16、S fp32、行列積の部分和 fp32 DEST、線上の部分和 bf16、行列積 HiFi2。合格線は `docs/NUMERICS.md` の acceptance（pinned 表、`json` 96/96）。K 分割の足す順・hc 統計の reduce の順・GDR の 4 単位分割は bf16 丸めの回数を変えるので、実装時に PCC と発散 index で判定する（机上で決めない）。\n"
)
P("### 12.2 1 chip の L1 と DRAM の会計（手順 2）\n")
P(
    f"1 chip の 1 pass: dense {chip_dense/MiB:,.1f} MiB（= 全体/8 + router 複製 + indexer 複製 + kv 射影の群内複製）、routed tile {chip_routed/MiB:.1f} MiB（最繁忙 chip、全ヒット）、S {chip_S/MiB:.1f} MiB（往復 {2*chip_S/MiB:.1f}）、行と作業域 {chip_rows/MiB:.1f} MiB、L1 常駐状態（conv tap、PLE tap）{chip_state/KiB:.0f} KiB、DRAM だけの読み（embedding / n-gram 行）{chip_reads/KiB:.1f} KiB。DRAM の動く量 {(chip_dense+chip_routed+2*chip_S+chip_reads)/MiB:,.1f} MiB/pass/chip = 1 chip を直列に見て {(chip_dense+chip_routed+2*chip_S+chip_reads)/BW*1e3:.3f} ms。単品ごとの表は `routing/design_step2_account.md`。\n"
)
P("### 12.3 枠（手順 3）\n")
P(
    f"1 chip の L1 {S3['l1_chip']/MiB:.1f} MiB（130 core × 1,536 KiB − 固定専有 140 KiB/core）− ストリームバッファ {S3['stream']/MiB:.1f} MiB（行列積 tile 80 + GDR 48 + KV chunk 64 KiB/core）− 常駐状態 {S3['state_l1']/MiB:.2f} = **枠の容量 {S3['cap']/MiB:.1f} MiB**。1 pass の列（1 chip）{sum(f['column'] for f in FR)/MiB:,.1f} MiB → {sum(f['column'] for f in FR)/S3['cap']:.2f} 枠 → **{len(FR)} 枠 = {S3['R']} 周**（R=1 は 1 枠 277 MiB で不可）。単品境で均等切り:\n"
)
P("| 枠 | 群 | 単品の範囲 | 列 [MiB] | 重み [MiB] | うち routed [MiB] | S [MiB] | DRAM/pass [MiB] | L1 の余り [MiB] |")
P("|---|---|---|---|---|---|---|---|---|")
for f in FR:
    P(
        f"| {f['frame']} | {f['group']} | {U[f['a']]['unit'].split(' ')[0]} 〜 {U[f['z']-1]['unit'].split(' ')[0]} | {f['column']/MiB:.1f} | {f['weights']/MiB:.1f} | {f['routed']/MiB:.1f} | {f['S']/MiB:.1f} | {f['dram']/MiB:.1f} | {f['left']/MiB:.1f} |"
    )
gd = [sum(f["dram"] for f in FR if f["group"] == g) for g in range(S3["PP"])]
P(
    f"\n群ごとの 1 chip の DRAM/pass {min(gd)/MiB:.1f}〜{max(gd)/MiB:.1f} MiB（偏り {(max(gd)/min(gd)-1)*100:.1f}%）= {min(gd)/BW*1e3:.3f}〜{max(gd)/BW*1e3:.3f} ms。枠の列は dense が鎖に消費された順に空くので、群が空いている間（他 3 群の番）に次の枠の dense を**全量**先読みできる（先読みに使える L1 = 容量 − (列 − dense) = {min(f['l1_free_idle'] for f in S5['frames'])/MiB:.0f}〜{max(f['l1_free_idle'] for f in S5['frames'])/MiB:.0f} MiB/枠、手順 5）。routed tile は投機 router 後にしか読めないので窓の中（先読み窓 3 層分 ≈ 18 MiB を L1 に取り置く）。\n"
)
# ---------- 12.4 unit times ----------
P("### 12.4 単品の時間（手順 4、1 chip、物理値のみ）\n")


def layer_sum(pref):
    return sum(t["t_us"] for i, t in enumerate(T4) if not t["spec"] and U[i]["layer"] == pref)


t_gdn = layer_sum("L00")
t_qsa = layer_sum("L03")
t_ple = sum(t["t_us"] for i, t in enumerate(T4) if ".ple." in U[i]["unit"])
t_tail = sum(t["t_us"] for i, t in enumerate(T4) if not t["spec"] and not U[i]["layer"].startswith("L"))
t_pass = sum(t["t_us"] for t in T4 if not t["spec"])
by = {}
for i, t in enumerate(T4):
    if U[i]["layer"] == "L00" and not t["spec"] and t["t_us"] > 0.05:
        by[U[i]["unit"].split(" ")[0]] = t["t_us"]
P(
    f"単価: 行列積 HiFi2 32 cycle/tile-MAC（出力 tile が 130 core より少なければ K を分けて fp32 部分和を NoC の木で足す: 1 段 92 cycle + 64 flit + 32 cycle）、同じ入力を読む行列積は pool、GDR 1 単位 256 + 800 × 行 cycle（core に 1 つ）、top-10 は tile ごと 10 pass × 32 cycle + NoC 集約、要素演算 16 cycle/bf16 tile、argmax は局所（7.5 tile/core）+ 集約。1 chip の鎖: **GDN 層 {t_gdn:.1f} µs**（"
    + "、".join(f"{k} {v:.2f}" for k, v in by.items())
    + f"）、**QSA 層 {t_qsa:.1f} µs**、PLE {t_ple:.1f}、tail（mixer + lm_head × 2 + argmax + MTP の fc）{t_tail:.1f} → **1 pass の鎖 {t_pass:.1f} µs**。投機 router 1.33 µs/本（鎖の外）。単品ごとの表は `routing/design_step4_item_time.md`。\n"
)
# ---------- 12.5 communication ----------
P("### 12.5 通信（手順 6）\n")
P(
    "通信の時間は packet 模型（`comm_model.py`: link 25 GB/s/方向、hop 280 ns、送り手の蓄積転送 16 ns/2 KiB packet、中継 92 cycle + 32 flit、packet 2 KiB + 56 B、link は FIFO、経路は縦 → 環の短い向き、collective は BFS 木の multicast / 直送 / 木の reduce の速い形）。群 = 2×4 ブロック、群間は 2 hop（列の 2 chip が群間 link を共有）。\n"
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
P("| 形 | 1 層の hc 区間（出力行列積 + 会合 + down + 会合 + up + AG_x 露出、GDN / QSA）[µs] | pass [µs] | DRAM 増 [MiB/chip/pass] | 中身 |")
P("|---|---|---|---|---|")
for k, f in S6["forms"].items():
    P(f"| {k} | {f['gdn']:.2f} / {f['qsa']:.2f} | {f['pass_us']:.0f} | {f['pass_MiB']:+.1f} | {f['desc']} |")
P("")
P("| 群間の受け渡し（枠の境） | 生きている行 | bytes/chip | [µs] |")
P("|---|---|---|---|")
live = {
    0: "token id（2 × 4 B）",
    1: "残差 1/8 + gated-norm の行（6 value head）+ inject",
    2: "残差 1/8 + x + inject",
    3: "残差 1/8 + x + router 結果 + shared の行 + inject",
    4: "残差 1/8 + 低ランク [2,320] fp32 + 正規化残差 1/8 + inject",
    5: "残差 1/8 + x + conv 出力（1,280 列）+ inject",
    6: "残差 1/8 + expert の行（43.9 tile）+ score + inject",
    7: "残差 1/8 + x + shared の行 + router 結果 + inject",
}
for h in S6["hand"]:
    P(f"| {h['desc']} | {live[h['cut']]} | {h['bytes']:,.0f} | {h['us']:.2f} |")
P(
    f"\n受け渡し合計 {sum(h['us'] for h in S6['hand']):.1f} µs/pass（形 B 系は受け手で残差の AG +{S6['hand_B_extra']:.2f} µs/境）。投機 router が群をまたぐ先頭層: 前の群が残差 1/8（5 KiB）を 2 hop {pa['res_hop']:.2f} µs で送り、受け手（空いている群）が自前の重みで読み + router を回す。link の共有: 群内の会合（BFS 木）は環の E/W とブロック内の縦 link だけを使い、群間の縦 link（受け渡し、残差 hop、router と expert が別枠に落ちる切れ目 3・6 の id hop）とは物理的に別で、群間 link に乗る 3 本は時刻が重ならない（受け渡しは枠の最後、残差 hop と id hop は枠の途中）ので予定表は要らない。\n"
)
# ---------- 12.6 steady state ----------
P("### 12.6 鎖を追った定常状態 = 到達目標（手順 7）\n")
P(
    "`step7_chain.py`（離散事象、12 pass 流して後半 6 pass の平均）: 単品を HF 順に、前の単品の終わり + 会合の露出（+ 枠の境の受け渡し）から始め、読みが届くまで待つ（届いた分から消費、最後の片の後 t/8）。群ごとの DRAM 車線（512 GB/s）は 512 KiB の片を消費順に発行し、routed は投機 router のトリガ後、L1 は積み上げ（先の routed 3 層分の余地を取り置く = 27B 大原則 1 の本件版）。routed の発行時刻: P2 = 層 l−1 の attention 開始 + 自前の hc 読み 0.88 + 会合 + router 1.33 + 発行 0.16（群をまたげば + 残差 hop）、P1 = 同じく層 l、外れ = router 後 + 0.16。L00 は pass 開始 + 2.7 µs（確定）、MTP 層は自分の attention 開始（予測不可）。\n"
)
P("| ケース / hc 形 | T_pass [µs] | tsu | 演算 | 会合の露出 | 読み待ち（dense / routed） | 受け渡し | 車線 | L1 高水位 [MiB] |")
P("|---|---|---|---|---|---|---|---|---|")
lab = {
    "hit_A": "全ヒット / A",
    "hit_A_S": "全ヒット / A + S 常駐",
    "hit_B": "全ヒット / B",
    "hit_Bn": "全ヒット / Bn",
    "hit_B+D": "全ヒット / B+D",
    "census_A": "census（2 段先読み）/ A",
    "miss_A": "全ミス（router 後）/ A",
}
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
    f"\n読み: **形 A（配備 `read_tp4` の列分割残差）が最良**。B 系は鎖は短い（会合 −32、演算 −27 µs）が、折り畳み重み +13.4 MiB と受け手の残差 AG +11 µs で車線が 95% に張り付き待ちが出る。B+D は +80 MiB で車線律速。S 常駐は車線 −1% で鎖に効かない。本件 b=1 は「DRAM 車線 {min(hit['busy'])*100:.0f}% + 鎖 100%」の釣り合いの点にあり、鎖を短くする形は車線を増やし、車線を減らす形は鎖に効かない。全ヒットでも残る routed の露出 {hit['waits']['routed']:.1f} µs は L00（pass 開始後 2.7 µs にしか発行できない）と MTP 層（argmax 後にしか予測できない）の分で構造的。投機 router の演算 {hit['spec_total']:.0f} µs/pass は鎖の待ち {hit['expo']+hit['wait']:.0f} µs に入る。\n"
)
# ---------- 12.7 budgets ----------
P("### 12.7 実装への予算（効率 100% = §12.6 の値を出す条件）\n")
P(
    f"- 単品の演算: §12.4 の値（行列積は K 分割 + NoC の木で 130 core 全部を使う; GDR は 24 単位を 24 core に; argmax は局所）。1 pass {hit['comp']:.1f} µs。\n- 会合: §12.5 の露出（AG_x は K chunk で射影が追いかける、RS/AR は 10 波で流す、投機の会合は鎖の外）。1 pass {hit['expo']:.1f} µs。\n- 受け渡し: 枠の境で生きている行だけを 2 hop（{hit['hand']:.1f} µs/pass）。\n- DRAM: 各群 {min(gd)/MiB:.0f}〜{max(gd)/MiB:.0f} MiB/pass を 512 GB/s の {min(hit['busy'])*100:.0f}〜{max(hit['busy'])*100:.0f}% で（8 reader、bank 隣接、片 512 KiB、消費順の発行、routed は 3 段のトリガで）。読み待ち {hit['wait']:.1f} µs 以内。\n- L1: 枠の容量 {S3['cap']/MiB:.1f} MiB、routed の取り置き 3 層分、高水位 {max(hit['l1_hwm'])/MiB:.0f} MiB。\n- 投機 router: 層 l の router を持つ群で、attention ブロック開始の残差から 2 本（l、l+1）を鎖の待ち時間に回す。ヒット率は census（h12 {CEN['pass']['hit12_mean']:.3f}、h2 {CEN['pass']['hit2_mean']:.3f}）で、外れ分は router 後に読む（census ケース {C7['census_A']['T_steady']:.1f} µs がその場合の到達値）。\n"
)
# ---------- 12.8 diff to the limit ----------
P("### 12.8 限界モデル §11 との整合\n")
V4 = json.load(open(f"{D}/free5d_v4_results_ctx1.json"))["hit|k=1"]
v4 = next(r for r in V4 if r["cfg"] == "DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1")
V4c = json.load(open(f"{D}/free5d_v4_results_ctx1.json"))["census|k=1"]
v4c = next(r for r in V4c if r["cfg"] == "DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1")
V4m = json.load(open(f"{D}/free5d_v4_results_ctx1.json"))["miss|k=1"]
v4m = next(r for r in V4m if r["cfg"] == "DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1")
v4T = v4["slot_ms"] * 1e3
P(
    f"§11（第 4 版）は本節の 8 つの知見（会合の packet 模型と重なり、2 段先読みと L00 / MTP 層の構造的露出、tail の全単品、受け渡しの生きている行、hc の形の軸、先読み容量、argmax 局所化、投機 router の会合は鎖の外）を取り込んだもので、同じ形（`{v4['cfg']}`、ctx 1）の値は §11 {v4T:.1f} µs（{1e6/v4T*R_EFF:,.0f} tsu）= 演算 {v4['comp_ms']*1e3:.1f} + 会合 {v4['comm_ms']*1e3:.1f} + 受け渡し・窓 {v4T-v4['comp_ms']*1e3-v4['comm_ms']*1e3:.1f}、本節 {T:.1f} µs（{tsu:,.0f} tsu）: 差 {T-v4T:+.1f} µs（{(T/v4T-1)*100:+.1f}%、§11 は会合の payload を 256 B・行列積時間を 50 ns に量子化して表引きする分）。"
)
P(
    f"census と全ミスでは §11 が短い（census §11 {v4c['slot_ms']*1e3:.1f} / 本節 {C7['census_A']['T_steady']:.1f} µs、全ミス {v4m['slot_ms']*1e3:.1f} / {C7['miss_A']['T_steady']:.1f}）: §11 は層の中で DRAM 車線を平滑化する上限（層 = max(鎖, routed/BW)）、本節は車線を時間軸で追うので routed の読みが層境を越えて積み上がる分だけ長い（§11.3 原則 24）。設計の到達値は本節、§11 は形の比較と上限。"
)
P("§11 の全列挙（6,510 構成 × 形）でも b=1 の tsu 最大は本節の形のままで（§11.5.1）、hc 形は A（§11 の列挙でも B 系は DRAM 車線を増やすだけで勝たない）。\n")
P("### 12.9 生成と検証\n")
P(
    "- 生成: `scripts/design/run_design.sh`（step0 → step2 → step3 → step4 → step5 → census_stage2 → step6 → step7 の各ケース、入力は HF 定義・`routing/stats.json`・`routing/*.npz`・物理値だけ）→ `render_design.py` → 本節；`install_design.sh` が README の §12 を差し替える。\n- 検証: 同じ入力から再生成して `routing/design_*.json` と README が byte 一致、`git status` が空、本文を通読して訂正点ゼロになるまで周回する。\n- 作業記録: `BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §12（手順ごとの結果と見つけて直した誤り）。\n"
)
print("\n".join(o))
