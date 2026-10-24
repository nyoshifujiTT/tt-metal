# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-axis limit (DP, PP, TP, EP, CP), revision 3

この文書は、32 chip の BH Galaxy で decode を回すときの DP・PP・TP・EP・CP の全組合せ（6510 構成 × router の置き方 2 形、batch 1〜2048）について、1 token の時間と機械全体の tokens/s を**物理値と HF 定義だけ**から出し、agg tps と tsu（1 ユーザの tokens/s）それぞれの最大とその構成を示す現在の確定状態である。設計原則（DRAM を止めない等）は課さない: 構成は時間だけで比べ、寝ている chip があっても速ければ採る。実装の実測値は入れない（効率の判定にだけ使う）。経緯・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用。revision 2 からの修正は §10.H7〜H9）、生成器は `BH-GLX_Qwen3.8-Flash-Next_free5d_v3_20261024.py`（全式と出典を docstring に持つ）、routing の分子は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）。この文書は `render_free5d_v3.py` が同じ入力から決定的に生成する。仮定は「この場合には」で断り、代表仮定が置けない所（routed の先読みの当たり方、hop の設計値）は並列列挙する。

## 1. 一般論: 5 つの軸は何を分け、何を払うか

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） | agg | tsu |
|---|---|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食う。1 chip の dense 分担が DP 倍になり L1 常駐分が減る） | 無し | 複製数に線形 | 不変 |
| PP | 層（＋lm_head）をバイト均等の段に; b 人を n_mb 個（1〜min(PP, b)、自由）の micro-batch に分けて段を順に通す | 段の dense 部のうち L1 予算（161 MiB/chip − 活性）に入る分は常駐（pass を跨いで読み直さない）、入らない分と routed・状態・KV は占有窓の中で流し読み | 段間受け渡し（残差 [rows, 4H]/TP、1 hop） | 段の周期 = max(段の占有 × n_mb, 段の DRAM/pass) | 1 token = 段の周期（Σ占有 ≤ 周期） |
| TP | 1 層の行列を head/列で TP 枚に（同じ token を全員で）; 残差 [rows, 4H] も列で分ける | 互いに素な 1/TP | hyper-connection の統計 all-reduce + ブロック入力の all-gather + ブロック出力の reduce-scatter = 3 collective × 2 ブロック = 6 本 | 1 step が短い | 大きく効くが「最後の tile のレイテンシ × 6 本 × 48 層」が床 |
| EP | routed expert を群に丸ごと（群内は列分割 TPx） | expert は 1 回 | 行を持たない chip に expert の分担があるとき（DP_a > 1、EP に依らず）dispatch + combine の all-to-all; 最繁忙 group が時間を決め他は寝る | 偏り分だけ損 | 列分割の tile padding を避けられる |
| CP | 1 ユーザの**系列**を kv-head 複製群の中で分割（index ブロックと K/V を 1/CP ずつ、局所 top-k、部分 (o, lse) を all-gather して merge） | 無関係 | merge 1 本（QSA 層のみ） | 長文脈で KV 読みが支配する時に効く | 同左 |

TP と CP は切る方向が直交する: TP は head（GDN の value head 48、QSA の query head 24）を分け、CP は 1 つの kv head の系列（時間軸）を分ける。本件の QSA は kv head が 2 本なので TP_a > 2 では TP_a/2 chip が同じ kv head を複製しており、CP はその複製群の中で系列を分ける。GDN の状態は系列長に依らない（O(1)）ので CP の対象外。

## 2. 専門論: 限界性能の立て方

1. 分母は物理値（DRAM 512 GB/s/chip 読み書き和、chip 間 link 25 GB/s × 2/方向 × 4 方向、1 hop 280 ns + 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、Tensix 130 core × 1.35 GHz: 行列積 32 cycle/32×32 tile-MAC（HiFi2）、要素演算 16 cycle/bf16 tile・32 cycle/fp32 tile、gated delta rule は 27B README §9 の 1 組 (value head, user) 4,224 cycle を 32 列 4 単位に分け 1 単位 256 + 800 × 行数 cycle（verify 行は再帰で直列）、router の top-10 は logit tile ごとに 10 pass × 32 cycle → NoC 集約 92 cycle → 10 pass、op 固定費は数えない（27B §9 と同じ）、L1 1,536 KiB/core のうち 140 KiB/core 固定専有 + 作業域 16 MiB/chip、DRAM 31.9 GiB/chip の 90%）と HF 定義（H 2560、48 層 = 36 GDN + 12 QSA、512 expert top-10 × 640、GDN 48 value head × 128 の fp32 状態、QSA 24 query head × 256 / 2 kv head、indexer budget 2048 compress 4、vocab 248,320、hc 4 枝 low-rank 320、MTP 1 層；dense bf8、hc mixer/router/norm bf16、expert BF4 2,764,800 B）だけ。
2. 分子はバイト: dense 5.378 GB（32 分割で 1 chip 160 MiB: L1 予算 161 MiB に対し DP1・router 列分割なら全部常駐できる。「今回に限っては全部 SRAM に置ける」のは dense だけで、routed 69.4 GB・状態・KV は DRAM）、GDN 状態 1 ユーザ 1 pass 読み+書き 0.231 GB、QSA KV 1 行 ctx 192 で 4.9 MB。routed は census の実 distinct（B=1 10、B=16×2 行 183、B=256 426）と最繁忙 group（EP 分割時）で、一様仮定は使わない。
3. 層の占有 = max(演算 + 隠れない通信, 先読みできる routed 読み/BW) + router 後に露出する routed 読み（全ミス: 全部 + 発行レイテンシ 0.16 µs、census: (1−h) + 発行レイテンシ、全ヒット/全 expert 読み: 0。routed は micro-batch の hidden に依存するので段の空き時間には読めず、最良でも 1 層前の裏で読む）。通信は tile ごとに流し、層の link 時間の合計はその層の演算の半分までに隠れる（隠れない分 = link の合計 − 隠れた分 + 最後の tile のレイテンシの合計）。all-reduce は RS+AG（レイテンシ 2 本）と「部分和の all-gather + 局所 reduce」（1 本、link (n−1)×T）の速い方。段 = 48 層 + tail（final mixer, lm_head, MTP 層）をバイト均等で連続分割。段の占有 occ_s = max(Σ層の占有 + 受け渡し, 窓: (常駐しない dense + micro-batch の routed + 状態/KV + 残差の DRAM 往復)/BW)。段の DRAM/pass = 常駐しない dense × n_mb + 状態/KV + routed × n_mb。**T_pass = max(max_s 段の DRAM/pass, max_s occ_s × n_mb, Σ_s occ_s)**。T_tok = T_pass、agg = DP × b × R_eff / T_pass、tsu = R_eff / T_tok。R_eff = 1（k=0）/ 1.376（k=1、本件プロンプト集合の採択率）。
4. 容量: 各 chip の expert 分 + 段の dense 分 + ユーザの状態/KV ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% 残す）。
5. MoE の router 依存は 4 ケースを並列列挙（§4）。router の重み [512, 2560] bf16 は「全 chip 複製（collective 無し、dense +126 MB/chip）」と「列分割（候補の all-gather 1 本、tile padding）」の両形を評価し速い方を採る（表の router 列）。
6. 実装拘束（TP ∈ {4,8}、EP = TP、PP = CP = 1、lanes×(k+1) ≤ 32）は別枠。

## 3. 応用: TT デバイス（BH Galaxy）で効いた物理

- **通信の床**: 最後の tile のレイテンシは 4 chip 0.68 µs、8 chip 1.07、16 chip 1.85、32 chip 2.23（submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）。層 6 本 × 48 層で TP32 は 0.64 ms/token の床（b=1 の TP32×PP1 の隠れない通信 0.644 ms）。link 時間は b=1 で無視できる。
- **L1 常駐**: dense を 32 分割すると 1 chip 168 MB = 160 MiB で、L1 予算 161 MiB（130 core × (1,536 − 140) KiB − 作業域 16 MiB）から活性を引いた分に入る（b=1 の活性は 41 KiB）。PP の段は自段の dense（1/(PP×TP_a)）を常駐させ、入らない分（router 複製時、DP≥2、EP で TP_x が小さいとき等）だけを micro-batch ごとに流し読む。b=1 の TP32×PP1 は dense/chip 340 MB（router 複製）のうち 169 MB 常駐、残りと routed・状態が DRAM/pass 0.580 ms。
- **tile padding（32 列）**: expert の中間 640 列を TPx 分割すると 1 chip 640/TPx 列で、TPx=32 で 20 列 → 32 に padding = 1.6 倍のバイト。同様に shared 640（TP_a=32 で 1.6 倍）、router 512（列分割 TP32 で 2.0 倍）。**EP は expert を丸ごと置くのでこの padding が無く、偏り（census: 8 group で最繁忙/平均 B=16×2 1.23 倍）と引き換えになる** — TP32 で b≥4 から EP8×TPx4 が EP1×TPx32 に勝つ理由（b=1 では EP1: 1701 vs 1681; b=16 では EP8: 567 vs 461）。
- **GDN 状態**: fp32 [48,128,128] × 36 層 = 1 ユーザ 0.115 GB 常駐、毎 pass 読み+書き 0.231 GB。B に比例して DRAM の列に乗り（b=128×2 行で 29.6 GB、dense + routed 62.1 GB の 48%）→ 重みを償却しきった agg の漸近は 32 × 512 GB/s ÷ (0.2309 GB 状態 + 0.0049 GB KV) = 69,480 pass/s（k=1 で × 1.376 = 95,605 t/s）。gated delta rule の演算は b=1 でも 1 単位 1,856 cycle（k=1）が core に 1 つずつ乗り、tsu 最大点の演算 0.186 ms のうち 0.049 ms。
- **routed は router の後**: dense は常駐または段が空いている間に L1 へ先読みできるが、層 l の top-10 は router まで分からず、しかも micro-batch の hidden に依存するので段の空き時間には読めない（最良でも 1 層前の演算の裏）。先読みの当たり方で 4 ケース。PP の段を分けると、micro-batch が居る段の chip しか routed を読まないので、b ≥ 16 では PP1（全 32 chip の DRAM が同じ layer の routed を分担）が勝つ。
- **1 行タイル 32 行**: B × (k+1) ≤ 32 が今日の lanes 形の上限。限界性能は tile を複数並べる形で b を 2048 まで振る。

## 4. 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に露出する分 |
|---|---|---|---|
| hit | 全ヒット: routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | census ヒット率: 1 ブロック早い投機 routing の実測ヒット率（外れ分は router 後に読み直す）（h: B=1 0.78、B=16×2 0.85、B=256 0.94） | distinct × 2.76 MB × (2−h) | (1−h) 分 + 発行レイテンシ |
| miss | 全ミス: router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない | 512 × 2.76 MB | 0 |

## 5. 結果（ctx 192 = ISL 128 + decode、分子は census 全数）

### 5.1 一行の答え

- **agg tps 最大（この場合には全ヒット、k=1）**: **81,326 t/s** = `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`（router split、4096 users、T_pass 69.3 ms = 段の DRAM/pass（演算 19.2 + 通信 17.0 は窓の中）、常駐 80/373 MB、tsu 19.9）。重みを償却しきった漸近 95,605 の 85%。census 80,752、全ミス 81,326、全 expert 読み 81,119（agg はどのケースも DRAM 支配で差が小さい）。k=0 全ヒット 60,447。
- **tsu 最大（全ヒット、k=1）**: **1,836 t/s/u** = `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1`（router replicated、T_tok 0.749 ms = Σ段の占有: 演算 0.186 + 通信 0.535 + 受け渡し; 段の DRAM/pass 0.364 ms は空き時間に収まる、常駐 169/251 MB）。PP × TP の梯子（§5.2）で PP1×TP32 1,701 / PP2×TP16 1,836 / PP4×TP8 1,798 / PP8×TP4 1,089: 上位 3 段は tsu 最大の 93% 以内で、順位は hop の設計値で入れ替わる（×2 で PP4×TP8、×4 で PP8×TP4）。census 1,650（`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1`）、全ミス 1,330（`DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1`）、全 expert 読み 244（`DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16`、16 users）。k=0 全ヒット 1,711。
- **5 軸の答え**: agg 側は **`DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`**。tsu 側は **`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1`**。中間の users 数ごとの最良は §5.5 の Pareto 前線（PP>1 は 4 点、EP>1 は 13 点、CP>1 は 12 点、全 13 点）。
- **今日の実装拘束内（TP∈{4,8}, EP=TP, PP=CP=1, lanes×(k+1) ≤ 32）**: 全ヒット agg 9,865（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 624（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）；全ミス agg 9,865、tsu 624。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` は全ミスで 9,865（T_pass 17.85 ms: 段 DRAM/pass 17.85、演算 2.66 + 通信 0.23 + routed 露出 13.31）、全ヒットで 9,865；**実測 1,186.9 t/s は全ミス机上値の 12.0%**。

### 5.2 PP × TP の梯子（b=1、全ヒット、k=1）と hop 設計値への感度

| PP × TP_a（b=1 の最良、EP/CP 自由） | 構成 | router | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | 常駐/段の dense [MB] | hop ×1 / ×2 / ×4 |
|---|---|---|---|---|---|---|---|---|---|
| PP1 × TP32 | `DP1xPP1x(TP32xDPa1|EP2xTPx16)xCP1 b=1` | replicated | 1701 | 0.809 | 0.622 | 0.144 | 0.644 | 169/340 | 1701 / 1055 / 599 |
| PP2 × TP16 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` | replicated | 1836 | 0.749 | 0.364 | 0.186 | 0.535 | 169/251 | 1836 / 1184 / 690 |
| PP4 × TP8 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1` | split | 1798 | 0.765 | 0.205 | 0.276 | 0.368 | 169/196 | 1798 / 1435 / 992 |
| PP8 × TP4 | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=1` | split | 1089 | 1.264 | 0.175 | 0.429 | 0.233 | 169/198 | 1089 / 1089 / 1038 |
| PP16 × TP2 | `DP1xPP16x(TP2xDPa1|EP1xTPx2)xCP1 b=1` | split | 336 | 4.094 | 0.488 | 0.771 | 0.099 | 169/396 | 336 / 336 / 336 |
| PP32 × TP1 | `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1` | split | 117 | 11.725 | 1.306 | 1.498 | 0.000 | 169/792 | 117 / 117 / 117 |

hop 280 ns を 1 / 2 / 4 倍にしたときの最良は PP2×TP16 / PP4×TP8 / PP8×TP4。「`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` が tsu 最大」は hop が設計値（280 ns、Table 16）に近い場合に限る。

### 5.3 各 TP の tsu 最良（b=1、全ヒット、k=1）

| TP_a | PP1, EP1×TPx=TP（列等分）, CP1 tsu | agg (users) | tsu 最良の構成（PP/EP/CP/n_mb 自由） | router | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 390 | 3,124 (8) | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=1` | split | 1089 | 1,089 (1) | 1.264; 0.175 / 0.429 / 0.233 |
| 8 | 791 | 3,165 (4) | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1` | split | 1798 | 1,798 (1) | 0.765; 0.205 / 0.276 / 0.368 |
| 16 | 1579 | 3,158 (2) | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` | replicated | 1836 | 1,836 (1) | 0.749; 0.364 / 0.186 / 0.535 |
| 32 | 1701 | 1,701 (1) | `DP1xPP1x(TP32xDPa1|EP2xTPx16)xCP1 b=1` | replicated | 1701 | 1,701 (1) | 0.809; 0.622 / 0.144 / 0.644 |

### 5.4 各 TP の tsu 最良（b=16、全ヒット、k=1; DP が違うと users が違うので agg は同列に比べない）

| TP_a | PP1, EP1×TPx=TP（列等分）, CP1 tsu | agg (users) | tsu 最良の構成（PP/EP/CP/n_mb 自由） | router | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 84 | 10,743 (128) | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=16 mb=8` | split | 516 | 8,258 (16) | 2.666; 2.666 / 0.590 / 0.233 |
| 8 | 147 | 9,418 (64) | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=16 mb=4` | split | 532 | 8,511 (16) | 2.587; 2.587 / 0.521 / 0.368 |
| 16 | 235 | 7,523 (32) | `DP1xPP2x(TP16xDPa1|EP4xTPx4)xCP8 b=16 mb=2` | split | 565 | 9,039 (16) | 2.436; 2.436 / 0.469 / 0.641 |
| 32 | 461 | 7,381 (16) | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 602 | 9,636 (16) | 2.285; 2.285 / 0.407 / 0.999 |

### 5.5 ケース × k の最大

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 60,447 | 14.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1711 | 1,711 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| hit | 1 | 81,326 | 19.9 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1836 | 1,836 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` |
| census | 0 | 60,019 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1451 | 2,902 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=2 mb=2` |
| census | 1 | 80,752 | 19.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1650 | 1,650 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` |
| miss | 0 | 60,447 | 14.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1117 | 1,117 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| miss | 1 | 81,326 | 19.9 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1330 | 1,330 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| all | 0 | 60,183 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 173 | 5,545 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` |
| all | 1 | 81,119 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 244 | 3,905 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` |

### 5.6 Pareto 前線（全ヒット、k=1）

前線に乗る構成のうち PP>1 は 4 点、EP>1 は 13 点、CP>1 は 12 点（全 13 点）。

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 常駐/段の dense [MB] | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,326 | 19.9 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.302 | 69.302 | 19.223 | 16.982 | 0.000 | 80/373 | 128.0 | fitted |
| 80,862 | 39.5 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 34.850 | 34.850 | 8.958 | 20.717 | 0.000 | 124/373 | 64.0 | fitted |
| 71,377 | 69.7 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | split | 19.741 | 19.741 | 4.523 | 10.849 | 0.000 | 147/373 | 64.0 | fitted |
| 57,813 | 112.9 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | split | 12.186 | 12.186 | 2.616 | 5.760 | 0.000 | 158/373 | 64.0 | fitted |
| 42,489 | 166.0 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | split | 8.290 | 8.290 | 1.644 | 3.224 | 0.000 | 163/373 | 62.2 | fitted |
| 29,258 | 228.6 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | split | 6.020 | 6.020 | 1.107 | 1.982 | 0.000 | 166/373 | 56.4 | census |
| 19,091 | 298.3 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | split | 4.613 | 4.613 | 0.795 | 1.383 | 0.000 | 168/373 | 49.4 | census |
| 13,240 | 413.7 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | split | 3.326 | 3.326 | 0.596 | 1.119 | 0.000 | 168/222 | 39.3 | census |
| 9,636 | 602.2 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 2.285 | 2.285 | 0.407 | 0.999 | 0.000 | 168/222 | 28.3 | census |
| 7,497 | 937.2 | 8 | `DP1xPP2x(TP16xDPa1|EP4xTPx4)xCP8 b=8 mb=2` | split | 1.468 | 1.468 | 0.330 | 0.633 | 0.000 | 169/203 | 18.9 | census |
| 6,137 | 1534.3 | 4 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=4 mb=2` | split | 0.897 | 0.897 | 0.233 | 0.624 | 0.000 | 169/203 | 18.0 | census |
| 3,530 | 1764.9 | 2 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=2 mb=2` | replicated | 0.780 | 0.732 | 0.186 | 0.532 | 0.000 | 169/251 | 9.8 | census |
| 1,836 | 1836.3 | 1 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` | replicated | 0.749 | 0.364 | 0.186 | 0.535 | 0.000 | 169/251 | 9.8 | census |

### 5.7 Pareto 前線（census ヒット率、k=1）

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 常駐/段の dense [MB] | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80,752 | 19.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.795 | 69.795 | 19.223 | 16.982 | 0.483 | 80/373 | 128.0 | fitted |
| 80,294 | 39.2 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 35.097 | 35.097 | 8.958 | 20.717 | 0.241 | 124/373 | 64.0 | fitted |
| 70,496 | 68.8 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | split | 19.987 | 19.987 | 4.523 | 10.849 | 0.241 | 147/373 | 64.0 | fitted |
| 56,667 | 110.7 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | split | 12.432 | 12.432 | 2.616 | 5.760 | 0.241 | 158/373 | 64.0 | fitted |
| 41,296 | 161.3 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | split | 8.530 | 8.530 | 1.644 | 3.224 | 0.235 | 163/373 | 62.2 | fitted |
| 28,198 | 220.3 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | split | 6.246 | 6.246 | 1.107 | 1.982 | 0.222 | 166/373 | 56.4 | census |
| 17,952 | 280.5 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | split | 4.906 | 4.906 | 0.795 | 1.383 | 0.287 | 168/373 | 49.4 | census |
| 12,088 | 377.8 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | split | 3.643 | 3.643 | 0.596 | 1.119 | 0.310 | 168/222 | 39.3 | census |
| 8,554 | 534.6 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | split | 2.574 | 2.574 | 0.407 | 0.999 | 0.283 | 168/222 | 28.3 | census |
| 6,524 | 815.5 | 8 | `DP1xPP2x(TP16xDPa1|EP4xTPx4)xCP8 b=8 mb=2` | split | 1.687 | 1.687 | 0.330 | 0.633 | 0.438 | 169/203 | 18.9 | census |
| 5,282 | 1320.4 | 4 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=4 mb=2` | split | 1.042 | 1.042 | 0.233 | 0.620 | 0.272 | 169/203 | 18.0 | census |
| 3,264 | 1631.9 | 2 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=2 mb=2` | replicated | 0.843 | 0.843 | 0.186 | 0.535 | 0.153 | 169/259 | 9.8 | census |
| 1,650 | 1649.6 | 1 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP2 b=1` | replicated | 0.834 | 0.406 | 0.186 | 0.535 | 0.079 | 169/251 | 9.8 | census |

### 5.8 Pareto 前線（全ミス = 今の形、k=1）

| agg [t/s] | tsu | users | 構成 | router | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 常駐/段の dense [MB] | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,326 | 19.9 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | split | 69.302 | 69.302 | 19.223 | 16.982 | 8.294 | 80/373 | 128.0 | fitted |
| 80,862 | 39.5 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | split | 34.850 | 34.850 | 8.958 | 20.717 | 4.147 | 124/373 | 64.0 | fitted |
| 71,377 | 69.7 | 1024 | `DP1xPP1x(TP16xDPa2|EP16xTPx2)xCP8 b=1024` | split | 19.741 | 19.741 | 4.523 | 8.849 | 4.147 | 147/373 | 32.0 | fitted |
| 57,813 | 112.9 | 512 | `DP1xPP1x(TP16xDPa2|EP16xTPx2)xCP8 b=512` | split | 12.186 | 12.186 | 2.616 | 4.760 | 4.147 | 158/373 | 32.0 | fitted |
| 41,892 | 163.6 | 256 | `DP1xPP1x(TP16xDPa2|EP32xTPx1)xCP8 b=256` | split | 8.409 | 8.409 | 1.663 | 2.325 | 4.147 | 163/373 | 16.0 | fitted |
| 26,535 | 207.3 | 128 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=128` | split | 6.637 | 6.637 | 1.100 | 1.689 | 3.657 | 168/691 | 56.4 | census |
| 16,830 | 263.0 | 64 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=64` | split | 5.233 | 5.233 | 0.792 | 1.066 | 3.204 | 168/691 | 49.4 | census |
| 10,769 | 336.5 | 32 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=32` | split | 4.089 | 4.089 | 0.635 | 0.724 | 2.545 | 169/691 | 39.3 | census |
| 7,046 | 440.4 | 16 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=16` | split | 3.125 | 3.125 | 0.470 | 0.590 | 1.832 | 169/691 | 28.3 | census |
| 5,554 | 694.3 | 8 | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP1 b=8 mb=8` | replicated | 1.982 | 1.813 | 0.426 | 0.197 | 8.538 | 169/249 | 16.5 | census |
| 4,153 | 1038.3 | 4 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=4 mb=4` | replicated | 1.325 | 1.224 | 0.264 | 0.308 | 2.562 | 169/244 | 16.5 | census |
| 2,417 | 1208.4 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | replicated | 1.139 | 1.056 | 0.186 | 0.532 | 0.382 | 169/491 | 9.8 | census |
| 1,330 | 1329.7 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | replicated | 1.035 | 0.580 | 0.144 | 0.644 | 0.213 | 169/340 | 16.5 | census |

### 5.9 今日の実装拘束内の最大

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,311 (289 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 546 (2,183 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| hit | 1 | 9,865 (77 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 624 (2,495 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 0 | 2,159 (270 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 498 (1,993 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 1 | 8,827 (69 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 564 (2,258 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 0 | 2,311 (289 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 546 (2,183 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 1 | 9,865 (77 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 624 (2,495 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 0 | 138 (17 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 24 (98 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 1 | 4,172 (33 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 60 (3,835 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=16` |

## 6. 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §5.2 の感度の範囲。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（agg 最大点 `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` を含む）はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL128）の採択率。
- 投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖。device の BF4 連鎖では未測（queue 済）。
- gated delta rule の 1 単位 256 + 800 × 行 cycle は 27B §9 の 4,224 cycle/(head, user) を 4 単位に分けた机上値（単位間の k'S の交換は NoC 1 回で 1 単位の 1/10 未満、数えていない）。
- L1 の作業域 16 MiB/chip と固定専有 140 KiB/core は設計値（27B §3/§7 に準ずる）。常駐の境界（dense/chip が予算を僅かに超える構成）はこの値で動く。
- CP の局所 top-k（各 chip が budget/CP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。
