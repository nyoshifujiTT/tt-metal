# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-axis limit (DP, PP, TP, EP, CP), revision 2

この文書は、32 chip の BH Galaxy で decode を回すときの DP・PP・TP・EP・CP の全組合せ（3841 構成、batch 1〜2048）について、1 token の時間と機械全体の tokens/s を**物理値と HF 定義だけ**から出し、agg tps と tsu（1 ユーザの tokens/s）それぞれの最大とその構成を示す現在の確定状態である。設計原則（DRAM を止めない等）は課さない: 構成は時間だけで比べ、寝ている chip があっても速ければ採る。実装の実測値は入れない（効率の判定にだけ使う）。経緯・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用）、生成器は `BH-GLX_Qwen3.8-Flash-Next_free5d_v2_20261024.py`（全式と出典を docstring に持つ）、routing の分子は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）。この文書は `render_free5d.py` が同じ入力から決定的に生成する。仮定は「この場合には」で断り、代表仮定が置けない所（routed の先読みの当たり方、collective の算法）は並列列挙する。

## 1. 一般論: 5 つの軸は何を分け、何を払うか

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） | agg | tsu |
|---|---|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食う） | 無し | 複製数に線形 | 不変 |
| PP | 層（＋lm_head）をバイト均等の段に; b 人を PP 個の micro-batch に分けて段を順に通す | 段の dense 部は段が空いている間に L1 へ先読み（L1 177 MiB/chip に入る分）、入らない分と routed は占有窓の中で流し読み | 段間受け渡し（残差 [rows, 4H]/TP） | 段の周期 = max(段の占有 × micro-batch 数, 段の DRAM/pass) | 1 token = 段の周期（Σ占有 ≤ 周期） |
| TP | 1 層の行列を head/列で TP 枚に（同じ token を全員で） | 互いに素な 1/TP | all-reduce（RS → 残差 → AG）× 2 本 | 1 step が短い | 大きく効くが「最後の tile のレイテンシ × 本数 × 層数」が床 |
| EP | routed expert を群に丸ごと（群内は列分割 TPx） | expert は 1 回 | expert 群が行を持たない chip に跨る（DP_a > 1）ときだけ dispatch + combine の all-to-all（同じ TP 群の中なら行は all-gather 済みで combine は MLP の all-reduce に畳める）; 最繁忙 group が時間を決め他は寝る | 偏り分だけ損 | 列分割の tile padding を避けられる |
| CP | 1 ユーザの**系列**を kv-head 複製群の中で分割（index ブロックと K/V を 1/CP ずつ、局所 top-k、部分 (o, lse) を merge） | 無関係 | merge 1 本（QSA 層のみ） | 長文脈で KV 読みが支配する時に効く | 同左 |

TP と CP は切る方向が直交する: TP は head（GDN の value head 48、QSA の query head 24）を分け、CP は 1 つの kv head の系列（時間軸）を分ける。本件の QSA は kv head が 2 本なので TP_a > 2 では TP_a/2 chip が同じ kv head を複製しており、CP はその複製群の中で系列を分ける。GDN の状態は系列長に依らない（O(1)）ので CP の対象外。

## 2. 専門論: 限界性能の立て方

1. 分母は物理値（DRAM 512 GB/s/chip 読み書き和、chip 間 link 25 GB/s × 2/方向 × 4 方向、1 hop 280 ns + 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、Tensix 130 core × 1.35 GHz: 行列積 33 cycle/32×32 tile-MAC（HiFi2）、要素演算 16 cycle/tile、gated delta rule 400 cycle/(value head, user, S の 32 列) + 1,100/層、op 固定 300 cycle × 層 20 単品、L1 1,536 KiB/core、DRAM 31.9 GiB/chip の 90%）と HF 定義（H 2560、48 層 = 36 GDN + 12 QSA、512 expert top-10 × 640、GDN 48 value head × 128 の fp32 状態、QSA 24 query head × 256 / 2 kv head、indexer budget 2048 compress 4、vocab 248,320、hc 4 枝 low-rank 320、MTP 1 層；dense bf8、hc mixer/router/norm bf16、expert BF4 2,764,800 B）だけ。
2. 分子はバイト: dense 5.378 GB、expert 全体 69.36 GB、GDN 状態 1 ユーザ 1 pass 読み+書き 0.231 GB、QSA KV 1 行 ctx 192 で 4.9 MB。routed は census の実 distinct（B=1 10、B=16×2 行 183、B=256 426）と最繁忙 group（EP 分割時）で、一様仮定は使わない。
3. 層の占有 = max(演算 + 隠れない通信, routed 読み/BW) + router 後に露出する routed 読み（全ミス: 全部、census: (1−h)、全ヒット/全 expert 読み: 0。routed は micro-batch の hidden に依存するので段の空き時間には読めず、最良でも 1 層前の裏で読む）。通信は tile ごとに流し、層の link 時間の合計はその層の演算の半分までに隠れる（隠れない分 = link の合計 − 隠れた分 + 最後の tile のレイテンシの合計）。段 = 48 層 + tail（final mixer, lm_head, MTP 層）をバイト均等で連続分割。段の占有 occ_s = Σ層の占有 + 受け渡し、ただし L1 に入らない dense 部と routed は占有窓の中で読むので occ_s ≥ それらのバイト/BW。段の DRAM/pass = dense（L1 に入れば pass に 1 回、入らなければ micro-batch ごと）+ state/KV + routed × micro-batch 数。**T_pass = max(max_s 段の DRAM/pass, max_s occ_s × n_mb, Σ_s occ_s)**、n_mb = min(PP, b)。T_tok = T_pass、agg = DP × b × R_eff / T_pass、tsu = R_eff / T_tok。R_eff = 1（k=0）/ 1.376（k=1、本件プロンプト集合の採択率）。
4. 容量: 各 chip の expert 分 + dense 分 + ユーザの状態/KV ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% 残す）。
5. MoE の router 依存は 4 ケースを並列列挙（§4）。
6. 実装拘束（TP ∈ {4,8}、EP = TP、PP = CP = 1、lanes×(k+1) ≤ 32）は別枠。

## 3. 応用: TT デバイス（BH Galaxy）で効いた物理

- **通信の床**: 1 行 [1,2560] bf16 の all-reduce（RS+AG を 1 本）の最後の tile のレイテンシは TP4 1.37 µs、TP8 2.14、TP16 3.69、TP32 4.47（submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）。層 2 本 + hc 統計 2 本 × 48 層で TP32 は 0.86 ms/token の床。link 時間は B=1 で無視できる。
- **tile padding（32 列）**: expert の中間 640 列を TPx 分割すると 1 chip 640/TPx 列で、TPx=32 で 20 列 → 32 に padding = 1.6 倍のバイト。同様に shared 640（TP_a=32 で 1.6 倍）、hc low-rank 320（3.2 倍）。**EP は expert を丸ごと置くのでこの padding が無く、偏り（census: 8 group で最繁忙/平均 B=16×2 1.23 倍）と引き換えになる** — TP32 で b≥4 から EP8×TPx4 が EP1×TPx32 に勝つ理由（b=1 では EP1: 1457 vs 1441; b=16 では EP8: 471 vs 396）。
- **GDN 状態**: fp32 [48,128,128] × 36 層 = 1 ユーザ 0.115 GB 常駐、毎 pass 読み+書き 0.231 GB。B に比例して DRAM の列に乗り（b=128×2 行で 29.6 GB、dense + routed 62.1 GB の 48%）→ 重みを償却しきった agg の漸近は 32 × 512 GB/s ÷ (0.2309 GB 状態 + 0.0049 GB KV) = 69,480 pass/s（k=1 で × 1.376 = 95,605 t/s）。
- **routed は router の後**: dense は段が空いている間に L1 へ先読みできるが、層 l の top-10 は router まで分からず、しかも micro-batch の hidden に依存するので段の空き時間には読めない（最良でも 1 層前の演算の裏）。先読みの当たり方で 4 ケース。
- **PP の先読み**: 段の chip は段が空いている間に自分の段の dense 部を L1（177 MiB）へ先読みする。b=1 の PP4×TP8 では段あたり dense + routed 300 MB/chip のうち dense の L1 分が先読みされ、残りと routed が占有窓で流れる — これが TP32 単独（空き時間が無い）に対する PP の利点で、通信も TP8 の 2.14 µs/本 で済む。
- **1 行タイル 32 行**: B × (k+1) ≤ 32 が今日の lanes 形の上限。限界性能は tile を複数並べる形で b を 2048 まで振る。

## 4. 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に露出する分 |
|---|---|---|---|
| hit | 全ヒット: routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | census ヒット率: 1 ブロック早い投機 routing の実測ヒット率（外れ分は router 後に読み直す）（h: B=1 0.78、B=16×2 0.85、B=256 0.94） | distinct × 2.76 MB × (2−h) | (1−h) 分 |
| miss | 全ミス: router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 |
| all | 全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない | 512 × 2.76 MB | 0 |

## 5. 結果（ctx 192 = ISL 128 + decode、分子は census 全数）

### 5.1 一行の答え

- **agg tps 最大（この場合には全ヒット、k=1）**: **81,024 t/s** = `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`（4096 users、T_pass 69.6 ms = 段の占有 × micro-batch（演算 13.7 + 通信 14.5 + 流し読み）、段 DRAM/pass 61.1 ms、tsu 19.8）。重みを償却しきった漸近 95,605 の 85%。census 80,454、全ミス 81,024、全 expert 読み 80,819（agg はどのケースも DRAM 支配で差が小さい）。k=0 全ヒット 60,140。
- **tsu 最大（全ヒット、k=1）**: **1,722 t/s/u** = `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1`（T_tok 0.799 ms = Σ段の占有: 演算 0.213 + 通信 0.428 + L1 に入らない重みと routed の流し読み; 段の DRAM/pass 0.453 ms は空き時間に収まる）。TP32×PP1 b=1 は 1,457（通信 0.858 ms が床）。census 1,492（`DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1`）、全ミス 1,184、全 expert 読み 224（`DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16`、16 users）。k=0 全ヒット 1,651。
- **5 軸の答え**: agg 側は **`DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048`**。tsu 側は **`DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1`**。中間の users 数ごとの最良は §5.5 の Pareto 前線（PP>1 は 1 点、EP>1 は 12 点、CP>1 は 13 点、全 13 点）。
- **今日の実装拘束内（TP∈{4,8}, EP=TP, PP=CP=1, lanes×(k+1) ≤ 32）**: 全ヒット agg 9,621（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 504（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）；全ミス agg 6,531、tsu 504。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` は全ミスで 6,531（T_pass 26.97 ms: 段 DRAM/pass 4.72、演算 2.47 + 通信 0.26 + routed 露出 13.31）、全ヒットで 9,621；**実測 1,186.9 t/s は全ミス机上値の 18.2%**。

### 5.2 各 TP の最良（b=1、全ヒット、k=1）と hop 設計値への感度

| TP_a | PP1, EP1×TPx=TP（列等分）tsu | agg | 最良の構成（PP/EP/CP 自由） | tsu | agg | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|
| 4 | 346 | 2,767 | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP1 b=1` | 1232 | 1,232 | 1.117; 0.387 / 0.368 / 0.262 |
| 8 | 609 | 2,434 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1` | 1722 | 1,722 | 0.799; 0.453 / 0.213 / 0.428 |
| 16 | 1056 | 2,111 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 1635 | 1,635 | 0.841; 0.448 / 0.118 / 0.709 |
| 32 | 1457 | 1,457 | `DP1xPP1x(TP32xDPa1|EP2xTPx16)xCP1 b=1` | 1457 | 1,457 | 0.944; 0.629 / 0.077 / 0.858 |

hop 280 ns を 1 / 2 / 4 倍にしたときの tsu（各 TP の最良構成）: TP4 1232 / 1230 / 1034、TP8 1722 / 1353 / 819、TP16 1635 / 995 / 558、TP32 1457 / 864 / 476 → 最良は 1 倍で TP8、2 倍で TP8、4 倍で TP4。「TP8 が tsu 最大」は hop が設計値（280 ns、Table 16）に近い場合に限る。

### 5.3 各 TP の最良（b=16、全ヒット、k=1）

| TP_a | PP1, EP1×TPx=TP（列等分）tsu | agg | 最良の構成（PP/EP/CP 自由） | tsu | agg | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|
| 4 | 82 | 10,454 | `DP1xPP8x(TP4xDPa1|EP1xTPx4)xCP2 b=16` | 278 | 4,443 | 4.955; 3.925 / 0.538 / 0.270 |
| 8 | 139 | 8,921 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=16` | 341 | 5,452 | 4.038; 2.070 / 0.471 / 0.428 |
| 16 | 219 | 7,007 | `DP1xPP2x(TP16xDPa1|EP4xTPx4)xCP8 b=16` | 431 | 6,896 | 3.193; 1.113 / 0.416 / 0.735 |
| 32 | 396 | 6,328 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 495 | 7,917 | 2.781; 0.911 / 0.357 / 0.902 |

### 5.4 ケース × k の最大

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 60,140 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1651 | 1,651 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| hit | 1 | 81,024 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1722 | 1,722 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1` |
| census | 0 | 59,717 | 14.6 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1443 | 1,443 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| census | 1 | 80,454 | 19.6 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1492 | 1,492 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| miss | 0 | 60,140 | 14.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1018 | 1,018 | `DP1xPP4x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| miss | 1 | 81,024 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 1184 | 1,184 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| all | 0 | 59,879 | 14.6 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 160 | 5,106 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` |
| all | 1 | 80,819 | 19.7 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 224 | 3,589 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` |

### 5.5 Pareto 前線（全ヒット、k=1）

前線に乗る構成のうち PP>1 は 1 点、EP>1 は 12 点、CP>1 は 13 点（全 13 点）。

| agg [t/s] | tsu | users | 構成 | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 81,024 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 69.561 | 61.093 | 13.697 | 14.453 | 0.000 | 128.0 | fitted |
| 80,069 | 39.1 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | 35.195 | 30.962 | 6.156 | 11.343 | 0.000 | 64.0 | fitted |
| 69,998 | 68.4 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 20.129 | 15.896 | 3.079 | 6.146 | 0.000 | 64.0 | fitted |
| 55,929 | 109.2 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.597 | 8.363 | 1.887 | 3.374 | 0.000 | 64.0 | fitted |
| 40,434 | 157.9 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.712 | 4.596 | 1.272 | 1.997 | 0.000 | 62.2 | fitted |
| 27,320 | 213.4 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 6.447 | 2.713 | 0.911 | 1.336 | 0.000 | 56.4 | census |
| 17,464 | 272.9 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | 5.043 | 1.772 | 0.686 | 1.027 | 0.000 | 49.4 | census |
| 11,525 | 360.2 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | 3.821 | 1.222 | 0.525 | 0.960 | 0.000 | 39.3 | census |
| 7,917 | 494.8 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 2.781 | 0.911 | 0.357 | 0.902 | 0.000 | 28.3 | census |
| 5,584 | 698.0 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | 1.971 | 0.756 | 0.245 | 0.902 | 0.000 | 18.4 | census |
| 3,855 | 963.8 | 4 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=4` | 1.428 | 0.678 | 0.166 | 0.902 | 0.000 | 11.3 | census |
| 2,597 | 1298.5 | 2 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP16 b=2` | 1.060 | 0.639 | 0.111 | 0.902 | 0.000 | 31.8 | census |
| 1,722 | 1722.1 | 1 | `DP1xPP4x(TP8xDPa1|EP2xTPx4)xCP4 b=1` | 0.799 | 0.453 | 0.213 | 0.428 | 0.000 | 9.8 | census |

### 5.6 Pareto 前線（census ヒット率、k=1）

| agg [t/s] | tsu | users | 構成 | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 80,454 | 19.6 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 70.054 | 61.093 | 13.697 | 14.453 | 0.483 | 128.0 | fitted |
| 79,512 | 38.8 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | 35.442 | 30.962 | 6.156 | 11.343 | 0.241 | 64.0 | fitted |
| 69,151 | 67.5 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 20.376 | 15.896 | 3.079 | 6.146 | 0.241 | 64.0 | fitted |
| 54,855 | 107.1 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.843 | 8.363 | 1.887 | 3.374 | 0.241 | 64.0 | fitted |
| 39,351 | 153.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.952 | 4.596 | 1.272 | 1.997 | 0.235 | 62.2 | fitted |
| 26,394 | 206.2 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 6.673 | 2.713 | 0.911 | 1.336 | 0.222 | 56.4 | census |
| 16,506 | 257.9 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=64` | 5.335 | 1.772 | 0.686 | 1.027 | 0.287 | 49.4 | census |
| 10,642 | 332.6 | 32 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=32` | 4.137 | 1.222 | 0.525 | 0.960 | 0.310 | 39.3 | census |
| 7,171 | 448.2 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=16` | 3.070 | 0.911 | 0.357 | 0.902 | 0.283 | 28.3 | census |
| 5,029 | 628.6 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=8` | 2.189 | 0.756 | 0.245 | 0.902 | 0.213 | 18.4 | census |
| 3,500 | 874.9 | 4 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP16 b=4` | 1.573 | 0.678 | 0.166 | 0.902 | 0.142 | 11.3 | census |
| 2,406 | 1203.0 | 2 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP16 b=2` | 1.144 | 0.639 | 0.111 | 0.902 | 0.082 | 31.8 | census |
| 1,492 | 1492.2 | 1 | `DP1xPP2x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 0.922 | 0.448 | 0.118 | 0.709 | 0.079 | 9.8 | census |

### 5.7 Pareto 前線（全ミス = 今の形、k=1）

| agg [t/s] | tsu | users | 構成 | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | routed 露出 | 最繁忙 expert/層 | routing |
|---|---|---|---|---|---|---|---|---|---|---|
| 81,024 | 19.8 | 4096 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=2048` | 69.561 | 61.093 | 13.697 | 14.453 | 8.294 | 128.0 | fitted |
| 80,069 | 39.1 | 2048 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=2048` | 35.195 | 30.962 | 6.156 | 11.343 | 4.147 | 64.0 | fitted |
| 69,998 | 68.4 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 20.129 | 15.896 | 3.079 | 6.146 | 4.147 | 64.0 | fitted |
| 55,929 | 109.2 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.597 | 8.363 | 1.887 | 3.374 | 4.147 | 64.0 | fitted |
| 40,434 | 157.9 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.712 | 4.596 | 1.272 | 1.997 | 4.031 | 62.2 | fitted |
| 23,676 | 185.0 | 128 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=128` | 7.439 | 3.423 | 0.881 | 0.814 | 3.657 | 56.4 | census |
| 13,538 | 211.5 | 64 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=64` | 6.505 | 2.482 | 0.671 | 0.653 | 3.204 | 49.4 | census |
| 8,526 | 266.4 | 32 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=32` | 5.165 | 1.301 | 0.502 | 0.949 | 2.545 | 39.3 | census |
| 5,924 | 370.2 | 16 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP8 b=16` | 3.717 | 0.921 | 0.357 | 0.884 | 1.832 | 28.3 | census |
| 4,550 | 568.7 | 8 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP4 b=8` | 2.419 | 0.770 | 0.245 | 0.874 | 1.191 | 18.4 | census |
| 3,411 | 852.8 | 4 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP2 b=2` | 1.613 | 0.896 | 0.167 | 0.716 | 0.702 | 18.0 | census |
| 2,200 | 1100.0 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP4 b=1` | 1.251 | 0.861 | 0.118 | 0.725 | 0.382 | 9.8 | census |
| 1,184 | 1183.9 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` | 1.162 | 0.629 | 0.077 | 0.858 | 0.213 | 16.5 | census |

### 5.8 今日の実装拘束内の最大

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,043 (255 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 425 (1,699 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| hit | 1 | 9,621 (75 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 504 (2,018 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 0 | 1,924 (240 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 395 (1,582 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| census | 1 | 8,632 (67 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 465 (1,860 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 0 | 2,043 (255 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 425 (1,699 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| miss | 1 | 6,531 (51 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 504 (2,018 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 0 | 137 (17 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 24 (97 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| all | 1 | 4,128 (32 tsu) | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 59 (3,749 agg) | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=16` |

## 6. 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §5.2 の感度の範囲。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b=512 の agg はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL128）の採択率。
- 投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖。device の BF4 連鎖では未測（queue 済）。
- op 固定 300 cycle × 20 単品/層は 27B の単品分割を借りた（1 token で 0.002 ms = b=1 TP32 の演算 0.077 ms の 2.1%、b≥16 では 0.1% 以下）。
- CP の局所 top-k（各 chip が budget/CP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。
