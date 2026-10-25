# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-axis limit (DP, PP, TP, EP, CP), revision 3

この文書は、32 chip の BH Galaxy で decode を回すときの DP・PP・TP・EP・CP の全組合せ（6510 構成 × router の置き方 2 形、batch 1〜2048）について、1 token の時間と機械全体の tokens/s を**物理値と HF 定義だけ**から出し、agg tps と tsu（1 ユーザの tokens/s）それぞれの最大とその構成を示す現在の確定状態である。設計原則（DRAM を止めない等）は課さない: 構成は時間だけで比べ、寝ている chip があっても速ければ採る。実装の実測値は入れない（効率の判定にだけ使う）。経緯・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用。revision 2 からの修正は §10.H7〜H13）、生成器は `BH-GLX_Qwen3.8-Flash-Next_free5d_v3_20261024.py`（全式と出典を docstring に持つ）、routing の分子は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）。この文書は `render_free5d_v3.py` が同じ入力から決定的に生成する。仮定は「この場合には」で断り、代表仮定が置けない所（routed の先読みの当たり方、hop の設計値）は並列列挙する。

## 1. 一般論: 5 つの軸は何を分け、何を払うか

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） | agg | tsu |
|---|---|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食う。1 chip の dense 分担が DP 倍になり L1 常駐分が減る） | 無し | 複製数に線形 | 不変 |
| PP | 層（＋lm_head）を連続の段に（バイト均等か鎖均等）; b 人を n_mb 個（1〜min(PP, b)、自由）の micro-batch に分けて段を順に通す | 段の dense 部のうち L1 の残り（177 MiB/chip − ストリームバッファ 24 MiB − 活性 − routed 先読みバッファ）に入る分は常駐（pass を跨いで読み直さない）、入らない分は段が空いている間に先読みバッファへ、残りと routed・状態・KV は占有窓の中で流し読み | 段間受け渡し（残差 [rows, 4H]/TP、段の並びで 1〜3 hop） | 段の周期 = max(段の占有 × n_mb, 段の DRAM/pass) | 1 token = 段の周期（Σ占有 ≤ 周期） |
| TP | 1 層の行列を head/列で TP 枚に（同じ token を全員で）; 残差 [rows, 4H] も列で分ける（配備 `read_tp4` の形） | 互いに素な 1/TP | hyper-connection の統計 all-reduce（行が小さいときは部分和の all-gather + 局所 reduce で 1 本）+ ブロック入力の all-gather + ブロック出力の reduce-scatter = 3 collective × 2 ブロック = 6 本/層（router 列分割なら + 候補の all-gather 1 本） | 1 step が短い | 大きく効くが「最後の tile のレイテンシ × 6 本 × 48 層」が床 |
| EP | routed expert を群に丸ごと（群内は列分割 TPx） | expert は 1 回 | 行を持たない chip に expert の分担があるとき（DP_a > 1、EP に依らず）dispatch + combine の all-to-all; 最繁忙 group が時間を決め他は寝る | 偏り分だけ損 | 列分割の tile padding を避けられる |
| CP | 1 ユーザの**系列**を kv-head 複製群の中で分割（index ブロックと K/V を 1/CP ずつ、局所 top-k、部分 (o, lse) を all-gather して merge） | 無関係 | merge 1 本（QSA 層のみ） | 長文脈で KV 読みが支配する時に効く | 同左 |

TP と CP は切る方向が直交する: TP は head（GDN の value head 48、QSA の query head 24）を分け、CP は 1 つの kv head の系列（時間軸）を分ける。本件の QSA は kv head が 2 本なので TP_a > 2 では TP_a/2 chip が同じ kv head を複製しており、CP はその複製群の中で系列を分ける。GDN の状態は系列長に依らない（O(1)）ので CP の対象外。

## 2. 専門論: 限界性能の立て方

1. 分母は物理値（DRAM 512 GB/s/chip 読み書き和、chip 間 link 25 GB/s × 2/方向 × 4 方向、1 hop 280 ns + 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、Tensix 130 core × 1.35 GHz: 行列積 32 cycle/32×32 tile-MAC（HiFi2; 1 行列積の鎖は出力 tile を 130 core に配り、出力 tile が core より少ないときは K を分けて部分和を NoC の木で足す: 1 段 92 cycle + 64 flit + 32 cycle）、同じ入力を読む行列積は並べて走る（27B §9: 和でなく max）、要素演算 16 cycle/bf16 tile・32 cycle/fp32 tile、gated delta rule は 27B README §9 の 1 組 (value head, user) 4,224 cycle を 32 列 4 単位に分け 1 単位 256 + 800 × 行数 cycle（verify 行は再帰で直列）、router の top-10 は logit tile ごとに 10 pass × 32 cycle → NoC 集約 92 cycle → 10 pass、op 固定費は数えない（27B §9 と同じ）、L1 1,536 KiB/core のうち 140 KiB/core 固定専有、ストリームバッファ 1 core 192 KiB（行列積の tile 2 面 + 部分和 80、gated delta rule の状態単位 2 面 + 出力 48、KV/index チャンク 2 面 64）= 24.4 MiB/chip、活性 4 × [rows, H] bf16 + 残差 [rows, 4H]/TP_a、routed の expert 片 2 枚、残りを常駐 dense と routed 先読みバッファに分ける（バッファ 0 / 1 層分 / 2 層分の 3 通りで速い方、同じ時間なら pass あたりの DRAM バイトが少ない方。残りを全部バッファにして dense を pass ごとに読み直す形は同じ L1 で窓が同じまま DRAM の読みだけ増えるので採らない）、DRAM 31.9 GiB/chip の 90%）と HF 定義（H 2560、48 層 = 36 GDN + 12 QSA、512 expert top-10 × 640、GDN 48 value head × 128 の fp32 状態、QSA 24 query head × 256 / 2 kv head、indexer budget 2048 compress 4、vocab 248,320、hc 4 枝 low-rank 320、MTP 1 層；dense bf8、hc mixer/router/norm bf16、expert BF4 2,764,800 B）だけ。
2. 分子はバイト: dense 5.378 GB（32 分割で 1 chip 160 MiB に対し、L1 177 MiB からストリームバッファと b=1 の活性・expert 片を引いた残りは 152 MiB: **dense は 32 分割でも全部は L1 に置けず**、置けるのは残りから routed 先読みバッファを引いた分（tsu 最大点 `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` で 155 MB / 段の dense 261 MB、先読みバッファ 4.1 MB）。routed 69.4 GB・状態・KV は DRAM）、GDN 状態 1 ユーザ 1 pass 読み+書き 0.231 GB、QSA KV 1 行 ctx 192 で 4.9 MB。routed は census の実 distinct（B=1 10、B=16×2 行 183、B=256 426）と最繁忙 group（EP 分割時）で、一様仮定は使わない。
3. 層の時間 = 前段（hc 読み、GDN/QSA、shared/router、top-10 とその collective）+ [router 後に読む分があれば発行レイテンシ 0.16 µs] + max(expert 段の演算 + 通信, バッファに先読みできなかった routed + router 後に読む routed のバイト/BW)、かつ ≥ その層の routed 全バイト/BW。先読みできるのは全ヒット: 全部、census: 予測集合（外れ (1−h) は router 後）、全 expert 読み: 全部、全ミス: 無し、いずれも L1 の先読みバッファまで。routed は micro-batch の hidden に依存するので段の空き時間には読めず、最良でも 1 層前の裏で読む。通信は tile ごとに流し、層の link 時間の合計はその層の演算の半分までに隠れる（隠れない分 = link の合計 − 隠れた分 + 最後の tile のレイテンシの合計）。all-reduce は RS+AG（レイテンシ 2 本）と「部分和の all-gather + 局所 reduce」（1 本、link (n−1)×T）の速い方。段 = 48 層 + tail（final mixer, lm_head, MTP 層）を連続分割（DRAM バイト均等と鎖の時間均等の速い方）。段の占有 occ_s = max(Σ層の占有 + 受け渡し, 窓: (常駐しない dense − 空き時間に先読みした分 + micro-batch の routed + 状態/KV + 残差の DRAM 往復)/BW)。段の DRAM/pass = 常駐しない dense × n_mb + 状態/KV + routed × n_mb + 残差の DRAM 往復 × n_mb。**T_pass = max(max_s 段の DRAM/pass, max_s occ_s × n_mb, Σ_s occ_s)**。T_tok = T_pass、agg = DP × b × R_eff / T_pass、tsu = R_eff / T_tok。R_eff = 1（k=0）/ 1.376（k=1、本件プロンプト集合の採択率）。
4. 容量: 各 chip の expert 分 + 段の dense 分 + ユーザの状態/KV ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% 残す）。
5. MoE の router 依存は 4 ケースを並列列挙（§4）。router の重み [512, 2560] bf16 は「全 chip 複製（collective 無し、dense +126 MB/chip）」と「列分割（候補の all-gather 1 本、tile padding）」の両形を評価し速い方を採る（表の router 列）。
6. 実装拘束（TP ∈ {4,8}、EP = TP、PP = CP = 1、lanes×(k+1) ≤ 32）は別枠。

## 3. 応用: TT デバイス（BH Galaxy）で効いた物理

- **通信の床**: 最後の tile のレイテンシは 4 chip 0.68 µs、8 chip 1.07、16 chip 1.85、32 chip 2.23（submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）。層 6 本 × 48 層で TP32 は 0.64 ms/token の床（b=1 の TP32×PP1 の隠れない通信 0.643 ms）。link 時間は b=1 の [2, 2560] bf16 の all-gather で 32 chip 0.05 µs（レイテンシの 2%）。
- **L1 の勘定**: 1 chip 177 MiB（130 core × (1,536 − 140) KiB）− ストリームバッファ 24.4 MiB − 活性（b=1 で 40 KiB + 残差）− expert 片 2 枚 = 152 MiB（b=1、TP8）。dense の 32 分割 160 MiB はこれに入らず、PP の段の dense（1/(PP×TP_a) に router/indexer の複製分）も 198〜360 MB で、常駐できるのは残りから routed 先読みバッファ（全ヒットで 1 層分 = 最繁忙 expert × 2.76 MB × pad/TP_x）を引いた分。入らない分は段が空いている間にバッファへ先読み（PP>1）、残りを窓で流し読む。b=1 の TP32×PP1 は dense/chip 360 MB（router 複製）のうち 157 MB 常駐、残りと routed・状態が DRAM/pass 0.643 ms。
- **tile padding（32 列）**: expert の中間 640 列を TPx 分割すると 1 chip 640/TPx 列で、TPx=32 で 20 列 → 32 に padding = 1.6 倍のバイト。同様に shared 640（TP_a=32 で 1.6 倍）、router 512（列分割 TP32 で 2.0 倍）。**EP は expert を丸ごと置くのでこの padding が無く、偏り（census: 8 group で最繁忙/平均 B=16×2 1.23 倍）と引き換えになる** — TP32 で b≥8 から EP8×TPx4 が EP1×TPx32 に勝つ理由（b=1 では EP1: 1513 vs 1484; b=16 では EP8: 545 vs 445）。
- **GDN 状態**: fp32 [48,128,128] × 36 層 = 1 ユーザ 0.115 GB 常駐、毎 pass 読み+書き 0.231 GB。B に比例して DRAM の列に乗り（b=128×2 行で 29.6 GB、dense + routed 62.1 GB の 48%）→ 重みを償却しきった agg の漸近は 32 × 512 GB/s ÷ (0.2309 GB 状態 + 0.0049 GB KV) = 69,480 pass/s（k=1 で × 1.376 = 95,605 t/s）。gated delta rule の演算は b=1 でも 1 単位 1,856 cycle（k=1）が core に 1 つずつ乗り、tsu 最大点の演算 0.305 ms のうち 0.049 ms。
- **PP の段の空き時間**: 段は micro-batch が他の段にいる間、常駐しきれない dense の先頭を routed 先読みバッファの空き（バッファ − 1 層分の routed）へ先読みし、残りを窓で流す（§5.2 の「空き時間の先読み」列）。常駐を削ってこの先読みを増やしても窓は同じで DRAM の読みが増えるだけなので、L1 は常駐を優先する。段の境は「DRAM バイト均等」と「鎖の時間均等」の速い方、PP2 の 16 chip は 4×4 と 8×2（受け渡し 1 hop）の速い方。
- **routed は router の後**: dense は常駐または段が空いている間に L1 へ先読みできるが、層 l の top-10 は router まで分からず、しかも micro-batch の hidden に依存するので段の空き時間には読めない（最良でも 1 層前の演算の裏、先読みバッファの分まで）。先読みの当たり方で 4 ケース。PP の段を分けると、micro-batch が居る段の chip しか routed を読まないので、b ≥ 8 では PP1（全 32 chip の DRAM が同じ layer の routed を分担）が勝つ。
- **1 行タイル 32 行**: B × (k+1) ≤ 32 が今日の lanes 形の上限。限界性能は tile を複数並べる形で b を 2048 まで振る。

## 4. 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に露出する分 |
|---|---|---|---|
| hit | 全ヒット: routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | census ヒット率: 1 ブロック早い投機 routing の実測ヒット率（外れ分は router 後に読み直す）（h: B=1 0.78、B=16×2 0.85、B=256 0.94） | distinct × 2.76 MB × (2−h) | (1−h) 分 + 発行レイテンシ |
| miss | 全ミス: router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない | 512/EP × 2.76 MB（偏り無し） | 0 |

## 5. 結果（ctx 192 = ISL 128 + decode、分子は census 全数）

### 5.1 一行の答え

- **agg tps 最大（この場合には全ヒット、k=1）**: **81,253 t/s** = `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`（router split、4096 users、T_pass 69.4 ms = 段の DRAM/pass（演算 19.9 + 通信 16.6 は窓の中）、L1 常駐 68 / 段の dense 392 MB（活性 80 MiB が L1 を食う）、tsu 19.8）。重みを償却しきった漸近 95,605 の 85%。census 80,680、全ミス 81,253、全 expert 読み 81,253（agg はどのケースも DRAM 支配で差が小さい）。k=0 全ヒット 60,392。
- **tsu 最大（全ヒット、k=1）**: **1,592 t/s/u** = `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1`（router replicated、T_tok 0.864 ms = Σ段の占有: 演算 0.305 + 通信 0.532 + 受け渡し + 窓; 段の DRAM/pass 0.412 ms、L1 は常駐 dense 155 MB + 先読みバッファ 4.1 MB + 空き時間の先読み 0 MB、段の dense 261 MB）。PP × TP の梯子（§5.2）で PP1×TP32 1,513 / PP2×TP16 1,592 / PP4×TP8 1,506 / PP8×TP4 865: 上位 3 段は tsu 最大の 95% 以内で、順位は hop の設計値で入れ替わる（×2 で PP4×TP8、×4 で PP4×TP8）。census 1,520（`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1`）、全ミス 1,408（`DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1`）、全 expert 読み 306（`DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=1`、1 users）。k=0 全ヒット 1,336。
- **5 軸の答え**: agg 側は **`DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`**。tsu 側は **`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1`**。中間の users 数ごとの最良は §5.6 の Pareto 前線（PP>1 は 3 点、EP>1 は 13 点、CP>1 は 10 点、全 13 点）。
- **今日の実装拘束内（TP∈{4,8}, EP=TP, PP=CP=1, lanes×(k+1) ≤ 32）**: 全ヒット agg 9,824（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 604（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）；全ミス agg 9,824、tsu 604。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` は全ミスで 9,824（T_pass 17.93 ms: 段 DRAM/pass 17.93、演算 3.20 + 通信 0.27 + routed 露出 13.59）、全ヒットで 9,824；**実測 1,186.9 t/s は全ミス机上値の 12.1%**。

### 5.2 PP × TP の梯子（b=1、全ヒット、k=1）と hop 設計値への感度

| PP × TP_a（b=1 の最良、EP/CP 自由） | 構成 | router / 段の切り方 / 段の形 | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | L1: 常駐 dense / 先読みバッファ / 空き時間の先読み [MB]（段の dense） | hop ×1 / ×2 / ×4 |
|---|---|---|---|---|---|---|---|---|---|
| PP1 × TP32 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | replicated / bytes / 8x4 | 1513 | 0.909 | 0.643 | 0.243 | 0.643 | 157 / 2.3 / 0 (360) | 1513 / 979 / 574 |
| PP2 × TP16 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | replicated / bytes / 8x2 | 1592 | 0.864 | 0.412 | 0.305 | 0.532 | 155 / 4.1 / 0 (261) | 1592 / 1078 / 654 |
| PP4 × TP8 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP4 b=1` | split / bytes / 2x4 | 1506 | 0.914 | 0.248 | 0.393 | 0.368 | 153 / 6.8 / 0 (199) | 1506 / 1242 / 918 |
| PP8 × TP4 | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=1` | split / bytes / 1x4 | 865 | 1.591 | 0.215 | 0.531 | 0.233 | 147 / 11.4 / 0 (198) | 865 / 865 / 865 |
| PP16 × TP2 | `DP1xPP16x(TP2xDPa1|EP1xTPx2)xCP1 b=1` | split / chain / 2x1 | 370 | 3.723 | 0.542 | 0.860 | 0.099 | 157 / 0.0 / 0 (396) | 370 / 369 / 369 |
| PP32 × TP1 | `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1` | split / chain / 1x1 | 175 | 7.849 | 1.396 | 1.585 | 0.000 | 155 / 0.0 / 0 (792) | 175 / 175 / 175 |

hop 280 ns を 1 / 2 / 4 倍にしたときの最良は PP2×TP16 / PP4×TP8 / PP4×TP8。「`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` が tsu 最大」は hop が設計値（280 ns、Table 16）に近い場合に限る。

### 5.3 各 TP の tsu 最良（b=1、全ヒット、k=1）

| TP_a | PP1, EP1×TPx=TP（列等分）, CP1 tsu | agg (users) | tsu 最良の構成（PP/EP/CP/n_mb 自由） | router | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 383 | 3,066 (8) | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=1` | split | 865 | 865 (1) | 1.591; 0.215 / 0.531 / 0.233 |
| 8 | 764 | 3,055 (4) | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP4 b=1` | split | 1506 | 1,506 (1) | 0.914; 0.248 / 0.393 / 0.368 |
| 16 | 1419 | 2,838 (2) | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | replicated | 1592 | 1,592 (1) | 0.864; 0.412 / 0.305 / 0.532 |
| 32 | 1513 | 1,513 (1) | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | replicated | 1513 | 1,513 (1) | 0.909; 0.643 / 0.243 / 0.643 |

### 5.4 各 TP の tsu 最良（b=16、全ヒット、k=1; DP が違うと users が違うので agg は同列に比べない）

| TP_a | PP1, EP1×TPx=TP（列等分）, CP1 tsu | agg (users) | tsu 最良の構成（PP/EP/CP/n_mb 自由） | router | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 84 | 10,700 (128) | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=16 mb=8` | replicated | 454 | 7,265 (16) | 3.030; 3.030 / 0.689 / 0.200 |
| 8 | 146 | 9,356 (64) | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=16 mb=4` | split | 481 | 7,697 (16) | 2.860; 2.860 / 0.666 / 0.368 |
| 16 | 233 | 7,446 (32) | `DP1xPP2x(TP16xDPa1|EP4xTPx4)xCP8 b=16 mb=2` | split | 528 | 8,441 (16) | 2.608; 2.608 / 0.646 / 0.722 |
| 32 | 445 | 7,122 (16) | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 577 | 9,234 (16) | 2.384; 2.384 / 0.631 / 1.053 |

### 5.5 ケース × k の最大

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 60,392 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1336 | 1,336 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| hit | 1 | 81,253 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1592 | 1,592 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| census | 0 | 59,965 | 14.6 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1297 | 2,594 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=2 mb=2` |
| census | 1 | 80,680 | 19.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1520 | 1,520 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` |
| miss | 0 | 60,392 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1144 | 1,144 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| miss | 1 | 81,253 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1408 | 1,408 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| all | 0 | 60,392 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 227 | 227 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=1` |
| all | 1 | 81,253 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 306 | 306 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=1` |

### 5.6 Pareto 前線（全ヒット、k=1）

前線に乗る構成のうち PP>1 は 3 点、EP>1 は 13 点、CP>1 は 10 点（全 13 点）。

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.364 | 69.364 | 19.896 | 16.646 | 0.000 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 34.912 | 34.912 | 9.384 | 20.504 | 0.000 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | split | 19.803 | 19.803 | 4.737 | 10.742 | 0.000 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | split | 12.248 | 12.248 | 2.825 | 5.655 | 0.000 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | split | 8.352 | 8.352 | 1.866 | 3.113 | 0.000 | 151 / 0.0 (392) | 62.2 | fitted |
| 28,960 | 226.2 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | split | 6.082 | 6.082 | 1.353 | 1.859 | 0.000 | 154 / 0.0 (392) | 56.4 | census |
| 18,572 | 290.2 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | split | 4.742 | 4.742 | 1.071 | 1.343 | 0.000 | 121 / 34.2 (392) | 49.4 | census |
| 12,800 | 400.0 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | split | 3.440 | 3.440 | 0.869 | 1.201 | 0.000 | 129 / 27.1 (242) | 39.3 | census |
| 9,234 | 577.1 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 2.384 | 2.384 | 0.631 | 1.053 | 0.000 | 138 / 19.5 (242) | 28.3 | census |
| 7,053 | 881.6 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | split | 1.561 | 1.561 | 0.471 | 0.988 | 0.000 | 145 / 12.7 (242) | 18.4 | census |
| 5,330 | 1332.6 | 4 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=4 mb=2` | split | 1.033 | 1.019 | 0.370 | 0.620 | 0.000 | 152 / 7.5 (213) | 18.0 | census |
| 3,152 | 1576.2 | 2 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=2 mb=2` | replicated | 0.873 | 0.871 | 0.305 | 0.532 | 0.000 | 155 / 4.1 (269) | 9.8 | census |
| 1,592 | 1592.5 | 1 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | replicated | 0.864 | 0.412 | 0.305 | 0.532 | 0.000 | 155 / 4.1 (261) | 9.8 | census |

### 5.7 Pareto 前線（census ヒット率、k=1）

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80,680 | 19.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.857 | 69.857 | 19.896 | 16.646 | 0.501 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,153 | 39.1 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 35.159 | 35.159 | 9.384 | 20.504 | 0.254 | 112 / 0.0 (392) | 64.0 | fitted |
| 70,278 | 68.6 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | split | 20.049 | 20.049 | 4.737 | 10.742 | 0.254 | 135 / 0.0 (392) | 64.0 | fitted |
| 56,386 | 110.1 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | split | 12.494 | 12.494 | 2.825 | 5.655 | 0.254 | 146 / 0.0 (392) | 64.0 | fitted |
| 40,998 | 160.1 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | split | 8.592 | 8.592 | 1.866 | 3.113 | 0.247 | 151 / 0.0 (392) | 62.2 | fitted |
| 27,921 | 218.1 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | split | 6.308 | 6.308 | 1.353 | 1.859 | 0.234 | 154 / 0.0 (392) | 56.4 | census |
| 17,492 | 273.3 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | split | 5.034 | 5.034 | 1.071 | 1.343 | 0.301 | 121 / 34.2 (392) | 49.4 | census |
| 11,720 | 366.3 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | split | 3.757 | 3.757 | 0.869 | 1.201 | 0.325 | 129 / 27.1 (242) | 39.3 | census |
| 8,236 | 514.7 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 2.673 | 2.673 | 0.631 | 1.053 | 0.297 | 138 / 19.5 (242) | 28.3 | census |
| 6,190 | 773.8 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | split | 1.778 | 1.778 | 0.471 | 0.988 | 0.225 | 145 / 12.7 (242) | 18.4 | census |
| 4,835 | 1208.9 | 4 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP8 b=4 mb=2` | split | 1.138 | 1.138 | 0.369 | 0.633 | 0.293 | 152 / 7.5 (213) | 18.0 | census |
| 2,930 | 1465.2 | 2 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP8 b=2 mb=2` | replicated | 0.939 | 0.939 | 0.305 | 0.545 | 0.172 | 155 / 4.1 (269) | 9.8 | census |
| 1,520 | 1519.6 | 1 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` | replicated | 0.905 | 0.452 | 0.305 | 0.535 | 0.089 | 155 / 4.1 (261) | 9.8 | census |

### 5.8 Pareto 前線（全ミス = 今の形、k=1）

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.364 | 69.364 | 19.896 | 16.646 | 8.475 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 34.912 | 34.912 | 9.384 | 20.504 | 4.241 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | split | 19.803 | 19.803 | 4.737 | 10.742 | 4.241 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | split | 12.248 | 12.248 | 2.825 | 5.655 | 4.241 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | split | 8.352 | 8.352 | 1.866 | 3.113 | 4.123 | 151 / 0.0 (392) | 62.2 | fitted |
| 28,960 | 226.2 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | split | 6.082 | 6.082 | 1.353 | 1.859 | 3.741 | 154 / 0.0 (392) | 56.4 | census |
| 18,183 | 284.1 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP2 b=64` | split | 4.843 | 4.789 | 1.071 | 1.341 | 3.279 | 156 / 0.0 (392) | 49.4 | census |
| 11,702 | 365.7 | 32 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=32` | replicated | 3.763 | 3.761 | 0.806 | 1.039 | 2.606 | 156 / 0.0 (510) | 39.3 | census |
| 7,775 | 485.9 | 16 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP2 b=16` | replicated | 2.832 | 2.824 | 0.585 | 0.890 | 1.878 | 157 / 0.0 (510) | 28.3 | census |
| 5,933 | 741.6 | 8 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP1 b=8 mb=4` | replicated | 1.855 | 1.738 | 0.478 | 0.308 | 4.806 | 159 / 0.0 (234) | 18.0 | census |
| 4,305 | 1076.3 | 4 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP1 b=4 mb=4` | replicated | 1.278 | 1.195 | 0.392 | 0.308 | 2.633 | 159 / 0.0 (234) | 9.8 | census |
| 2,481 | 1240.7 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP8 b=1` | replicated | 1.109 | 1.109 | 0.305 | 0.545 | 0.398 | 157 / 0.0 (510) | 9.8 | census |
| 1,408 | 1408.5 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | replicated | 0.977 | 0.639 | 0.243 | 0.643 | 0.226 | 159 / 0.0 (360) | 16.5 | census |

### 5.9 今日の実装拘束内の最大

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,263 (283 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 525 (2,101 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| hit | 1 | 9,824 (77 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 604 (2,417 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 0 | 2,117 (265 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 481 (1,925 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 1 | 8,795 (69 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 548 (2,194 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 0 | 2,263 (283 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 525 (2,101 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 1 | 9,824 (77 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 604 (2,417 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 0 | 226 (56 tsu) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` | 56 (226 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 1 | 4,615 (72 tsu) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=16` | 76 (304 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |

## 6. 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §5.2 の感度の範囲。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（agg 最大点 `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` を含む）はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL128）の採択率。
- 投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖。device の BF4 連鎖では未測（queue 済）。
- gated delta rule の 1 単位 256 + 800 × 行 cycle は 27B §9 の 4,224 cycle/(head, user) を 4 単位に分けた机上値（単位間の k'S の交換は NoC 1 回で 1 単位の 1/10 未満、数えていない）。
- b=1 の演算は「行列積の鎖」（K 分割の部分和の木）で決まり、27B §9 が「1 µs 未満」として省いた core 間の受け渡しを本件では数えている（tsu 最大点の演算 0.305 ms のうち行列積の鎖が大半）。K 分割の木の段数・NoC の値は設計値。
- L1 のストリームバッファ 1 core 192 KiB（24.4 MiB/chip）と固定専有 140 KiB/core は設計値（27B §3/§7 の枠の入出力・作業域 3〜10 MiB に準ずる、多い側）。バッファを 0 / 2 倍にしたときの tsu 最大点: 1,594 / 1,493（1 倍 1,592）。
- 活性の L1 占有は micro-batch の 4 × [rows, H] bf16 + 残差 [rows, 4H]/TP_a を設計値として置いた（27B §7 の枠の入出力 3〜10 MiB に準ずる）。agg 最大点 `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` では 80 MiB で常駐 dense を 68 MB に削る。
- 空いている段の chip が隣の段の routed を自分の DRAM から読んで ETH で渡す形（1 chip の入口 8 link × 25 GB/s = 200 GB/s、DRAM 512 GB/s の 39%）は列挙していない。
- CP の局所 top-k（各 chip が budget/CP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。
