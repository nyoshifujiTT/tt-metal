# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the unconstrained five-axis limit (DP, PP, TP, EP, CP)

この文書は、32 chip の BH Galaxy で decode を回すときの DP・PP・TP・EP・CP の全組合せ（3220 構成 × batch）について、1 token の時間と機械全体の tokens/s を**物理値と HF 定義だけ**から出し、agg tps と tsu（1 ユーザの tokens/s）それぞれの最大とその構成を示す現在の確定状態である。設計原則（DRAM を止めない等）は課さない: 構成は時間だけで比べ、寝ている chip があっても速ければ採る。実装の実測値は入れない（効率の判定にだけ使う）。経緯・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用）、生成器は `BH-GLX_Qwen3.8-Flash-Next_free5d_20261024.py`（全式と出典を docstring に持つ）、routing の分子は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）。この文書は `render_free5d.py` が同じ入力から決定的に生成する。仮定は「この場合には」で断り、代表仮定が置けない所（routed の先読みの当たり方、collective の算法）は並列列挙する。

## 1. 一般論: 5 つの軸は何を分け、何を払うか

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） | agg | tsu |
|---|---|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食う） | 無し | 複製数に線形 | 不変 |
| PP | 層（＋lm_head）をバイト均等の段に; 段ごとに別の micro-batch | 段の分だけ、micro-batch ごとに | 段間受け渡し（残差 [rows, 4H]） | 全段が忙しければ TP と同じ償却 | 1 token は全段を順に通る: T_tok = PP × 最遅段 |
| TP | 1 層の行列を head/列で TP 枚に（同じ token を全員で） | 互いに素な 1/TP | all-reduce（RS → 残差 → AG）× 2 本 | 1 step が短い | 大きく効くが「最後の tile のレイテンシ × 本数 × 層数」が床 |
| EP | routed expert を群に丸ごと（群内は列分割 TPx） | expert は 1 回 | EP>1 で dispatch + combine の all-to-all; 最繁忙 group が時間を決め他は寝る | 偏り分だけ損 | 列分割の tile padding を避けられる |
| CP | 1 ユーザの**系列**を kv-head 複製群の中で分割（index ブロックと K/V を 1/CP ずつ、局所 top-k、部分 (o, lse) を merge） | 無関係 | merge 1 本（QSA 層のみ） | 長文脈で KV 読みが支配する時に効く | 同左 |

TP と CP は切る方向が直交する: TP は head（GDN の value head 48、QSA の query head 24）を分け、CP は 1 つの kv head の系列（時間軸）を分ける。本件の QSA は kv head が 2 本なので TP_a > 2 では TP_a/2 chip が同じ kv head を複製しており、CP はその複製群の中で系列を分ける。GDN の状態は系列長に依らない（O(1)）ので CP の対象外。

## 2. 専門論: 限界性能の立て方

1. 分母は物理値（DRAM 512 GB/s/chip 読み書き和、chip 間 link 25 GB/s × 2/方向 × 4 方向、1 hop 280 ns + 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、Tensix 130 core × 1.35 GHz: 行列積 33 cycle/32×32 tile-MAC（HiFi2）、要素演算 16 cycle/tile、gated delta rule 400 cycle/(value head, user, S の 32 列) + 1,100/層、op 固定 300 cycle × 層 20 単品、L1 1,536 KiB/core、DRAM 31.9 GiB/chip の 90%）と HF 定義（H 2560、48 層 = 36 GDN + 12 QSA、512 expert top-10 × 640、GDN 48 value head × 128 の fp32 状態、QSA 24 query head × 256 / 2 kv head、indexer budget 2048 compress 4、vocab 248,320、hc 4 枝 low-rank 320、MTP 1 層；dense bf8、hc mixer/router/norm bf16、expert BF4 2,764,800 B）だけ。
2. 分子はバイト: dense 5.378 GB、expert 全体 69.36 GB、GDN 状態 1 ユーザ 1 pass 読み+書き 0.231 GB、QSA KV 1 行 ctx 192 で 4.9 MB。routed は census の実 distinct（B=1 10、B=16×2 行 183、B=256 426）と最繁忙 group（EP 分割時）で、一様仮定は使わない。
3. 層時間 = max(DRAM 車線, 演算 + 隠れない通信 + router 後に露出する routed 読み)。通信は tile ごとに流し、各 all-reduce は演算の半分までに隠れる（隠れない分 = link − 隠れた分 + 最後の tile のレイテンシ）。段 = 48 層 + tail（final mixer, lm_head, MTP 層）をバイト均等で連続分割、slot = 最遅段 + 受け渡し、T_tok = PP × slot、agg = DP × b × R_eff / slot、tsu = R_eff / T_tok。R_eff = 1（k=0）/ 1.376（k=1、本件プロンプト集合の採択率）。
4. 容量: 各 chip の expert 分 + dense 分 + ユーザの状態/KV ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% 残す）。
5. MoE の router 依存は 4 ケースを並列列挙（§4）。
6. 実装拘束（TP ∈ {4,8}、EP = TP、PP = CP = 1、lanes×(k+1) ≤ 32）は別枠。

## 3. 応用: TT デバイス（BH Galaxy）で効いた物理

- **通信の床**: 1 行 [1,2560] bf16 の all-reduce（RS+AG を 1 本）の最後の tile のレイテンシは TP4 1.37 µs、TP8 2.14、TP16 3.69、TP32 4.47（submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）。層 2 本 + hc 統計 2 本 × 48 層で TP32 は 0.86 ms/token の床。link 時間は B=1 で無視できる。
- **tile padding（32 列）**: expert の中間 640 列を TPx 分割すると 1 chip 640/TPx 列で、TPx=32 で 20 列 → 32 に padding = 1.6 倍のバイト。同様に shared 640（TP_a=32 で 1.6 倍）、hc low-rank 320（3.2 倍）。**EP は expert を丸ごと置くのでこの padding が無く、偏り（census: 8 group で最繁忙/平均 B=16×2 1.23 倍）と引き換えになる** — TP32 では EP8×TPx4 が EP1×TPx32 に勝つ理由。
- **GDN 状態**: fp32 [48,128,128] × 36 層 = 1 ユーザ 0.115 GB 常駐、毎 pass 読み+書き 0.231 GB。B に比例して DRAM の列に乗り、B ≥ 64 で dense + routed を超える → agg の漸近は 32 × 512 GB/s ÷ 0.2309 GB = 70,952 pass/s。
- **routed は router の後**: dense は先読みできるが、層 l の top-10 は router まで分からない。先読みの当たり方で 4 ケース。
- **1 行タイル 32 行**: B × (k+1) ≤ 32 が今日の lanes 形の上限。限界性能は tile を複数並べる形で b を 512 まで振る。

## 4. 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に露出する分 |
|---|---|---|---|
| hit | 全ヒット: routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | census ヒット率: 1 ブロック早い投機 routing の実測ヒット率（外れ分は router 後に読み直す）（h: B=1 0.78、B=16×2 0.85、B=256 0.94） | distinct × 2.76 MB × (2−h) | (1−h) 分 |
| miss | 全ミス: router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 |
| all | 全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない | 512 × 2.76 MB | 0 |

## 5. 結果（ctx 192 = ISL 128 + decode、分子は census 全数）

### 5.1 一行の答え

- **agg tps 最大（この場合には全ヒット、k=1）**: **58,385 t/s** = `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512`（2048 users、slot 48.3 ms、DRAM 車線 48.2 ms が支配、tsu 28.5）。census 57,241、全ミス 56,129、全 expert 読み 57,189（agg はどのケースも DRAM 支配で差が小さい）。k=0 全ヒット 44,008。
- **tsu 最大（全ヒット、k=1）**: **1,393 t/s/u** = `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1`（T_tok 0.988 ms: DRAM 0.847 / 演算 0.077 + 通信 0.858 — DRAM と通信レイテンシが同じ桁で両方が床）。census 1,332（`DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1`）、全ミス 1,146、全 expert 読み 224（`DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16`、16 users）。k=0 全ヒット 1,039。
- **5 軸の答え**: agg 側は **`DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512`**。tsu 側は **`DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1`**（32 chip の DRAM を 1 token に全部充てる）。中間の users 数ごとの最良は §5.5 の Pareto 前線。PP>1 の構成が前線に乗るか、EP/CP がどこで乗るかは前線の表のとおり（固定文では書かない）。
- **今日の実装拘束内（TP∈{4,8}, EP=TP, PP=CP=1, lanes×(k+1) ≤ 32）**: 全ヒット agg 9,621（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 504（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）；全ミス agg 9,621、tsu 504。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` は全ミスで 9,621（DRAM 18.31 ms vs 依存 16.11 ms）、全ヒットで 9,621；**実測 1,186.9 t/s は全ミス机上値の 12.3%**。

### 5.2 各 TP の最良（b=1、全ヒット、k=1）と hop 設計値への感度

| TP_a | EP1×TPx=TP（列等分）tsu | agg | 最良の EP 構成 | tsu | agg | T_tok [ms] = DRAM / 演算 / 通信 |
|---|---|---|---|---|---|---|
| 4 | 346 | 2,767 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP2 b=1` | 346 | 2,771 | 3.973 = 3.973 / 0.368 / 0.270 |
| 8 | 609 | 2,434 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1` | 611 | 2,445 | 2.251 = 2.251 / 0.213 / 0.531 |
| 16 | 1056 | 2,111 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP8 b=1` | 1101 | 2,202 | 1.250 = 1.250 / 0.118 / 0.912 |
| 32 | 1393 | 1,393 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | 1393 | 1,393 | 0.988 = 0.847 / 0.077 / 0.858 |

hop 280 ns を 1 / 2 / 4 倍にしたときの tsu（各 TP の最良構成）: TP4 346 / 346 / 346、TP8 611 / 611 / 600、TP16 1101 / 753 / 428、TP32 1393 / 843 / 471 → 最良は 1 倍で TP32、2 倍で TP32、4 倍で TP8。「TP32 が tsu 最大」は hop が設計値（280 ns、Table 16）に近い場合に限る。

### 5.3 各 TP の最良（b=16、全ヒット、k=1）

| TP_a | EP1×TPx=TP（列等分）tsu | agg | 最良の EP 構成 | tsu | agg | T_tok [ms] = DRAM / 演算 / 通信 |
|---|---|---|---|---|---|---|
| 4 | 82 | 10,454 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP2 b=16` | 82 | 10,501 | 16.772 = 16.772 / 2.235 / 0.270 |
| 8 | 139 | 8,921 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=16` | 156 | 9,975 | 8.828 = 8.828 / 1.175 / 0.531 |
| 16 | 219 | 7,007 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=16` | 293 | 9,375 | 4.697 = 4.697 / 0.632 / 0.912 |
| 32 | 396 | 6,328 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 495 | 7,917 | 2.781 = 2.781 / 0.357 / 1.123 |

### 5.4 ケース × k の最大

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 44,008 | 10.7 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 1039 | 1,039 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| hit | 1 | 58,385 | 28.5 | 2048 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512` | 1393 | 1,393 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| census | 0 | 43,178 | 10.5 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 1009 | 1,009 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| census | 1 | 57,241 | 27.9 | 2048 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512` | 1332 | 1,332 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| miss | 0 | 43,534 | 10.6 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 916 | 916 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| miss | 1 | 56,129 | 13.7 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 1146 | 1,146 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| all | 0 | 42,466 | 10.4 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 160 | 5,106 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` |
| all | 1 | 57,189 | 27.9 | 2048 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512` | 224 | 3,589 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` |

### 5.5 Pareto 前線（全ヒット、k=1）

前線に乗る構成のうち PP>1 は 0 点、EP>1 は 10 点、CP>1 は 11 点（全 12 点）。

| agg [t/s] | tsu | users | 構成 | slot [ms] | DRAM | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 58,385 | 28.5 | 2048 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512` | 48.267 | 48.206 | 7.003 | 7.884 | 0.000 | 250.3 | fitted |
| 57,893 | 56.5 | 1024 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=512` | 24.338 | 24.303 | 3.762 | 5.767 | 0.000 | 127.1 | fitted |
| 55,929 | 109.2 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.597 | 12.597 | 1.887 | 3.374 | 0.000 | 64.0 | fitted |
| 40,434 | 157.9 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.712 | 8.712 | 1.272 | 1.997 | 0.000 | 62.2 | fitted |
| 27,320 | 213.4 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 6.447 | 6.447 | 0.911 | 1.336 | 0.000 | 56.4 | census |
| 17,464 | 272.9 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | 5.043 | 5.043 | 0.686 | 1.027 | 0.000 | 49.4 | census |
| 11,525 | 360.2 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | 3.821 | 3.821 | 0.525 | 1.224 | 0.000 | 39.3 | census |
| 7,917 | 494.8 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 2.781 | 2.781 | 0.357 | 1.123 | 0.000 | 28.3 | census |
| 5,584 | 698.0 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | 1.971 | 1.971 | 0.245 | 1.117 | 0.000 | 18.4 | census |
| 3,849 | 962.2 | 4 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP8 b=4` | 1.430 | 1.430 | 0.166 | 1.098 | 0.000 | 11.3 | census |
| 2,588 | 1294.1 | 2 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP4 b=2` | 1.063 | 1.063 | 0.111 | 0.874 | 0.000 | 31.8 | census |
| 1,393 | 1392.9 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | 0.988 | 0.847 | 0.077 | 0.858 | 0.000 | 16.5 | census |

### 5.6 Pareto 前線（census ヒット率、k=1）

| agg [t/s] | tsu | users | 構成 | slot [ms] | DRAM | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 57,241 | 27.9 | 2048 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=512` | 49.231 | 49.170 | 7.003 | 7.884 | 0.944 | 250.3 | fitted |
| 56,752 | 55.4 | 1024 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=512` | 24.828 | 24.792 | 3.762 | 5.767 | 0.479 | 127.1 | fitted |
| 54,855 | 107.1 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.843 | 12.843 | 1.887 | 3.374 | 0.241 | 64.0 | fitted |
| 39,351 | 153.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.952 | 8.952 | 1.272 | 1.997 | 0.235 | 62.2 | fitted |
| 26,394 | 206.2 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 6.673 | 6.673 | 0.911 | 1.336 | 0.222 | 56.4 | census |
| 16,506 | 257.9 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | 5.335 | 5.335 | 0.686 | 1.027 | 0.287 | 49.4 | census |
| 10,642 | 332.6 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | 4.137 | 4.137 | 0.525 | 1.224 | 0.310 | 39.3 | census |
| 7,171 | 448.2 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 3.070 | 3.070 | 0.357 | 1.123 | 0.283 | 28.3 | census |
| 5,029 | 628.6 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | 2.189 | 2.189 | 0.245 | 1.117 | 0.213 | 18.4 | census |
| 3,494 | 873.6 | 4 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP8 b=4` | 1.575 | 1.575 | 0.166 | 1.098 | 0.142 | 11.3 | census |
| 2,399 | 1199.3 | 2 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP4 b=2` | 1.147 | 1.147 | 0.111 | 0.874 | 0.082 | 31.8 | census |
| 1,332 | 1332.2 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | 1.033 | 0.892 | 0.077 | 0.858 | 0.044 | 16.5 | census |

### 5.7 Pareto 前線（全ミス = 今の形、k=1）

| agg [t/s] | tsu | users | 構成 | slot [ms] | DRAM | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 56,129 | 13.7 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 50.207 | 48.366 | 13.286 | 2.912 | 32.087 | 495.2 | fitted |
| 55,437 | 27.1 | 2048 | `DP4xPP1x(TP4xDPa2|EP2xTPx4)xCP1 b=512` | 50.833 | 50.635 | 6.685 | 5.149 | 16.220 | 250.3 | fitted |
| 53,937 | 52.7 | 1024 | `DP2xPP1x(TP8xDPa2|EP4xTPx4)xCP4 b=512` | 26.123 | 25.002 | 3.523 | 3.739 | 8.235 | 127.1 | fitted |
| 52,052 | 101.7 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 13.535 | 12.597 | 1.887 | 3.374 | 4.147 | 64.0 | fitted |
| 38,226 | 149.3 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 9.215 | 8.712 | 1.272 | 1.997 | 4.031 | 62.2 | fitted |
| 26,077 | 203.7 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 6.754 | 6.447 | 0.911 | 1.336 | 3.657 | 56.4 | census |
| 16,801 | 262.5 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP2 b=64` | 5.242 | 5.157 | 0.687 | 1.022 | 3.204 | 49.4 | census |
| 10,640 | 332.5 | 32 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP2 b=32` | 4.138 | 3.956 | 0.502 | 0.936 | 2.545 | 39.3 | census |
| 6,741 | 421.3 | 16 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP1 b=16` | 3.266 | 3.002 | 0.375 | 0.923 | 1.832 | 28.3 | census |
| 4,403 | 550.4 | 8 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP1 b=8` | 2.500 | 2.197 | 0.264 | 0.923 | 1.191 | 18.4 | census |
| 3,039 | 759.8 | 4 | `DP1xPP1x(TP16xDPa2|EP1xTPx32)xCP2 b=4` | 1.811 | 1.696 | 0.193 | 0.716 | 0.783 | 60.5 | census |
| 2,007 | 1003.4 | 2 | `DP1xPP1x(TP16xDPa2|EP1xTPx32)xCP2 b=2` | 1.371 | 1.283 | 0.130 | 0.716 | 0.412 | 31.8 | census |
| 1,146 | 1145.8 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | 1.201 | 0.847 | 0.077 | 0.858 | 0.213 | 16.5 | census |

### 5.8 今日の実装拘束内の最大

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,043 (255 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 425 (1,699 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| hit | 1 | 9,621 (75 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 504 (2,018 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 0 | 1,924 (240 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 395 (1,582 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 1 | 8,632 (67 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 465 (1,860 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 0 | 2,043 (255 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 425 (1,699 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 1 | 9,621 (75 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 504 (2,018 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 0 | 137 (17 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 24 (97 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 1 | 4,128 (32 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 59 (3,749 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=16` |

## 6. 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §5.2 の感度の範囲。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b=512 の agg はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL128）の採択率。
- 投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖。device の BF4 連鎖では未測（queue 済）。
- op 固定 300 cycle × 20 単品/層は 27B の単品分割を借りた（演算の 1% 未満、順位に影響しない）。
- CP の局所 top-k（各 chip が budget/CP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。
