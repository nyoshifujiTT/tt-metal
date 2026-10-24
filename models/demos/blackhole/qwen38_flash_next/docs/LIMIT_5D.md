# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-dimensional parallel limit (DP x PP x TP x EP x CP)

この文書は「32 chip の BH Galaxy で decode を回すとき、DP・PP・TP・EP・CP の 5 次元をどう切れば agg tps（機械全体の tokens/s）と tsu（1 ユーザあたりの tokens/s）がそれぞれ最大になるか」を、物理値と HF 定義からの机上モデルで出した**現在の確定状態**を書く。経緯・実測・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用）にあり、この文書と食い違ったら新しい方の作業記録が正しい。生成器は `BH-GLX_Qwen3.8-Flash-Next_limit_perf_5d_20261024.py`（同ディレクトリ、全式と出典を docstring に持つ）、分子の routing 統計は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）、表は `..._limit_5d_tables_20261024.py`、この文書は `render_readme_5d.py` が同じ入力から決定的に生成する。

仮定は「この場合には」で断る。代表仮定が置けない所（collective の算法、expert ストリームの帯域、非 DRAM 時間 τ、先読みのヒット率）は**シナリオとして並列列挙**し（§5）、一本化しない。

## 1. 一般論: 5 つの並列軸は decode で何を分け、何を払うか

decode の 1 step は「全層の重みと、ユーザごとの状態（GDN の再帰状態、QSA の KV）を DRAM から読み、行列積と再帰更新を行い、次の token を出す」の繰り返しで、重み 1 回の読みを何行（ユーザ×verify 行）で償却できるかが agg を、1 step の長さが tsu を決める。

| 軸 | 何を分けるか | 重みの読み | 払うもの（1 層あたり） | agg への効き | tsu への効き |
|---|---|---|---|---|---|
| DP（data） | ユーザ集合を複製モデルに分ける | 複製ごとに全部（DRAM 容量を複製分食う） | 無し（複製間通信ゼロ） | 複製数に線形 | 不変 |
| PP（pipeline） | 層を段に分け、段ごとに別の micro-batch を同時処理 | 段の重みだけ、micro-batch ごとに読み直す | 段間 hop 1 回（活性 rows×H） | 全段が常に忙しければ TP と同じ償却 | 1 token が全段を巡るので T_tok = Σ段 + PP×hop（≈ 1/PP） |
| TP（tensor） | 1 層の行列を TP 枚に分け全員で 1 行列を解く | 互いに素な 1/TP ずつ 1 回 | all-reduce（= reduce-scatter + all-gather）×ブロック数: 2(TP−1)/TP·M/BW + hop×α | 重みを TP 枚で分担（1 step が短くなる） | 大きく効くが α×hop が床になる |
| EP（expert） | routed expert を所有グループに分ける | expert は 1 回だけ（複製しない） | DP_a>1 のとき dispatch+combine の all-to-all（rows×k×H） | distinct expert の償却が instance 全体で効く | router 後にしか読めない露出が残る |
| CP（context） | 1 ユーザの KV を系列方向に分ける | 無関係（KV 読みを CP で割る） | 部分 attention の merge（o, lse） | 長文脈で KV 読みが支配する時に効く | 同左 |

教科書的な式（Megatron-LM、DeepSpeed-Inference、Pope et al. 2022「Efficiently Scaling Transformer Inference」、DeepSeek-V3 の EP320 配備）: ring all-reduce T = 2(N−1)/N·M/B_link + 2(N−1)·α；PP の効率 = M/(P+M−1)（M micro-batch、P 段）；最適バッチ b* は「重み読み時間 = 演算時間」となる点；per-chip latency の床は collective の α 項。CP は KV が系列長で伸びる full-attention にだけ効き、系列長 O(1) の再帰状態（GDN）には効かない（KDA/vLLM の実装でも再帰状態は DCP で分割せず複製）。

## 2. 専門論: パフォーマンスエンジニアリングとしての扱い（本件の指針 §9.3 の 5D 版）

1. 限界性能の分母は物理値（DRAM 帯域、link 帯域、hop レイテンシ、Tensix の FLOP/cycle、L1/DRAM 容量）と HF 定義（形状・dtype）だけから出し、実装の実測値は「校正シナリオ」として別行に置く（混ぜない）。
2. 分子はバイト（重み・状態・KV・routed expert の実 distinct 数）で、routing は一様仮定ではなく実測（census）を使う。一様仮定は distinct を過大に見積もる（B=16: 139 vs 実測 124、B=256: 509 vs 426）。
3. 5 軸は独立ではない: chip 数 32 = DP × PP × S、段の S chip は dense 側で TP_a × DP_a、routed 側で EP × TP_x に**同時に**分かれる。候補は全列挙し、Pareto 前線（agg 降順で tsu が単調増加する点列）で示す。「最大 agg」と「最大 tsu」は別の構成になるのが普通で、1 点だけを答えにしない。
4. 時間は層ごとに「DRAM 車線（prefetch 可能な連続ストリーム）」と「依存車線（router → expert 読み → 演算 → collective）」に分けて max を取る。op を順に足す今日の実行形（直列）と、DRAM を止めない dataflow-first（重ね合わせ）の両方を出す。
5. MoE は「全ヒット／全ミス／実測ヒット率」の 3 ケース（§9.4.9）を 5D の各点で持つ。dense の連続ストリーム設計は dense 部分にだけ適用する。
6. 容量制約（DRAM 32 GB/chip）を必ず同時に判定する。重みは EP×TP_x×PP で 1 回だけ置くのが原則で、DP で複製するたびに「ユーザ状態の置き場」が減る。
7. 実装拘束（今日の runtime: TP∈{4,8}、EP=TP、1 行タイル 32 行、PP/CP 未実装、EP>1 の collective combine 未実装）は別枠で明記し、拘束なしの限界と拘束下の限界の両方を出す。

## 3. 応用: TT デバイス（BH Galaxy）で 5 軸はどう変わるか

- **トポロジ**: 32 chip は 8×4 の 2D torus（`single_bh_galaxy_torus_xy_graph_descriptor.textproto`: dims [8,4] RING/RING、channels 2）。実機 cluster descriptor では「8 chip のリング 4 本、各リングが 4 chip の線 2 本」、全 chip の ETH 次数 4。1 方向 2 link × 25.0 GB/s（`ccl_common.cpp lookup_fabric_link_bw`: 400 Gbps=50 GB/s の半分に制限中）、4 方向で egress 200 GB/s/chip。1 hop 片道 515 ns（1D fabric）/ 619 ns（2D）、p150_x4 実測（`ccl_common.cpp lookup_fabric_hop_latency_ns`）。
- **collective の算法が二通り**: 今日の ttnn Ring/Linear CCL は 1 次元リング（hop = N−1、1 方向 2 link）。torus を使う算法（hop = 直径 = ⌊r/2⌋+⌊c/2⌋、4 方向 8 link）は未実装。両方をシナリオに置く。TP16 の直径 2+1 = 3 hop に対しリングは 15 hop。
- **パケット固定費**: fabric gather の実測（山口 job132、`gr_fold.gather_line`）は 1 パケット 4 KiB あたり 0.64〜0.95 µs、= 線上時間の 2〜4 倍。TP の全 all-reduce がこの固定費を払うので「+0.8 µs/packet」をシナリオ C 以降に入れる。
- **DRAM は読み書き和 512 GB/s/chip**（GDDR6 8 ch）。GDN の fp32 状態（36 層 × [48,128,128]、1 ユーザ 0.231 GB/pass 読み+書き）は dense 重みと同じ DRAM ストリームの一部で、ユーザ数に比例して増える → **agg の漸近は重みではなくユーザ状態の往復**（32 × 512 GB/s ÷ 0.2358 GB/user/pass = 69,480 pass/s）。
- **routed expert のストリーム帯域は spec の 46%**（job128: MoECompute 2.35 ms/pass ÷ 201 experts → 236 GB/s、ring-8 の各 core が 1 bank に紐付く 36 GB/s×8）。spec 512 と校正 236 を並記。
- **1 行タイル 32 行**: decode の行列積は M=32 行の tile 1 行で、B × (k+1) ≤ 32 が今日の lanes 形の上限。gemma-4-31B は 32 users、本件は 16 lanes × k=1。tile を超える行は複数 tile で扱えば限界性能上は制約でない（本モデルは b を 1024 まで振る）。
- **GDN 状態の TP 分割は通信不要**: 48 value head を TP_a で分け、head 内で閉じる（TP_a ≤ 48、16 を超えると 48/TP_a が整数でなくなり padding）。QSA は 24 query head、2 kv head（TP_a > 2 で kv head は複製、CP はその複製内で KV を分ける）。TP_a > 8 で query head、> 16 で value head の端数 padding をバイトに加算。
- **EP > 1 と DP_a**: expert を所有グループに分けると、dense 側の TP_a グループより expert が広い（DP_a > 1）時に dispatch/combine の all-to-all が要る。今日の runtime は EP = TP_a（1×TP 線で expert を TP chip に分配）だけで、EP×TP の 2D は moe_compute の collective combine 未実装（commit f6165e058a）。
- **MTP**: verify 行 k+1 を 1 ユーザが出すので rows = b×(k+1)、採択 R_eff tokens/pass（k=1 で 1.376 実測、k=3 で 2.46、k=4 で 3.02 は 4×p150 の自然文記録 = 仮定）。agg と tsu の両方に R_eff が掛かる。

## 4. 本モデルの式（生成器 `limit_perf_5d.py` の docstring と同じ）

- chip: N = 32 = DP × PP × S、S = TP_a × DP_a = EP × TP_x、CP | max(1, TP_a/2)。
- ユーザ: 1 pipeline に PP micro-batch × b ユーザ、rows = b × (k+1)、R_eff tokens/pass。
- 層 l の chip あたりバイト: dense_l/TP_a + (b/DP_a)·state_l/TP_a + (rows/DP_a)·kv_l(ctx)·h_chip/2/CP + X_l（最繁忙 EP group の distinct expert 数 × 2.7648 MB × pad/TP_x）。
- 層時間: t_l = max(bytes_l/BW_dram, FLOP_l/peak_HiFi2, FLOP_state/peak_SFPU, τ) ∨ (D/BW + λ + X_exposed/BW_x) + 2·AR(TP_a, rows/DP_a×H×2B) + 2·AR(TP_a, hc 統計) + [DP_a>1] 2·A2A(S, rows×k×H×2B/S) + [CP>1, QSA] merge(CP)。
  - AR(n, M) = 2·hop(n)·α + packets·α_pkt + 2(n−1)/n·M/(links·25 GB/s)、torus: hop = 直径, links = 8；ring1d: hop = n−1, links = 2。
  - X_exposed: 全ヒット 0 / 全ミス X_l / 実測 = census の pred1 miss 集合。λ = router → 読み発行 5 µs（仮定）。
- 段: 48 層 + tail（final mixer, lm_head 0.675 GB bf8, MTP 層）を最大段最小で連続分割。t_slot = max_s t_stage + hop、agg = DP × b × R_eff / t_slot、T_tok = PP × t_slot（PP=1 なら t_stage）、tsu = R_eff / T_tok。
- dataflow-first（A°, D°）: 段時間 = max(DRAM 車線の和, 依存車線の和)。
- 容量: experts 69.4 GB/(PP×EP×TP_x) + dense 5.38 GB/(PP×TP_a) + b/DP_a × (状態 0.1155 GB + KV) ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% を残す）。
- 定数: H 2560, L 48 (36 GDN + 12 QSA), E 512 top-10, expert 640, GDN 48×128 fp32 状態, QSA 24 head×256 / 2 kv head, indexer budget 2048 compress 4, vocab 248,320, hc 4×2560 low-rank 320, dense bf8（hc mixer bf16）, expert BF4 2,764,800 B（staged cache 実測 2,764,804 B と一致）, Tensix 130 core × 4096 FLOP/cycle/phase × 1.35 GHz（HiFi2 359 TF/s, LoFi 719, fp32 SFPU 11.2）。
- routing の分子: census（34 item、decode 8,192 位置、CPU oracle、bf16 expert、greedy 連鎖）から rows ∈ {1,2,4,8,16,32,64,128,256} と MTP 2 行 {1..128}×2 の最繁忙 group を取る。rows > 256 は「fitted」（未使用 expert 数が rows 倍増ごとに census の 128→256 比 0.591 で減衰、偏り 1+0.6/√mean、miss 率 = census B=256 の 0.058）と明記。

routing census の要点（分子の実測値）: h1（前 token の top-10 再利用）= 0.356；1 ブロック早い投機 routing（層 l の router を層 l−1 の出力に適用）のヒット率 B=1 0.775 / B=16×2 行 0.845 / B=256 0.942；実 distinct expert/層 B=16 124.2（一様 138.6）、B=32 198.9（239.6）、B=256 426.2（508.7）；最繁忙 4 group の expert 数 B=16×2: 51.3（平均の 1.12 倍）。

## 5. シナリオ（代表仮定が置けないので並列列挙）

| 記号 | collective | packet 固定費 | expert ストリーム | τ（層あたり非 DRAM） | 先読み | 実行形 |
|---|---|---|---|---|---|---|
| A° | torus（直径 hop, 8 link） | 0 | 512 GB/s | 0 | 全ミス（λ 5 µs） | dataflow-first（DRAM 止めない） |
| A | torus | 0 | 512 | 0 | 全ミス | op 直列 |
| A' | torus | 0 | 512 | 0 | 全ヒット | op 直列 |
| B | ring1d（今日の CCL） | 0 | 512 | 0 | 全ミス | op 直列 |
| C | torus | 0.8 µs/4 KiB | 512 | 0 | 全ミス | op 直列 |
| D | torus | 0.8 | 236（校正） | 0 | 全ミス | op 直列 |
| D° | torus | 0.8 | 236 | 0 | 全ミス | dataflow-first |
| E / E' / E'' | torus | 0.8 | 236 | 200 µs（目標値） | 全ミス / 全ヒット / 実測 pred1 | op 直列 |
| F | ring1d | 0.8 | 236 | **校正 620 µs**（DP8×TP4 b=16 k=1 の実測 pass 63.36 ms を再現） | 全ミス | op 直列 |

## 6. 結果（ctx 192 = ISL 128 + decode。分子は census 全数）

### 6.1 一行の答え

- **agg tps 最大（この場合には A°: spec 物理値・torus 算法・DRAM を止めない実行形・MTP k=1）**: **71,688 t/s** = `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024`（4096 users、t_step 78.6 ms、tsu 17.5）。k=0 では 53,433（`DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024`）、k=4（R_eff 3.02 仮定）では 150,324。漸近（ユーザ状態の往復だけ）は 69,480 pass/s × R_eff = k=1 で 95,600 t/s、A° はその 75%。
- **tsu 最大（同 A°, k=1）**: **749 t/s/u** = `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1`（T_tok 1.837 ms、agg 1,498）。k=0 では 592、k=4 では 1335。TP32（`DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=1`）は 540 で TP16 に負ける（query head 24 の padding と直径）。
- **PP32**（1 chip 1.5 層、`DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1`）: tsu 26.3、agg 840（lm_head 0.675 GB を持つ段 1.32 ms が slot を決め 32 hop）；b=32 で agg 10,683、tsu 10.4。PP は collective ゼロで agg を取る形だが、本モデルの Pareto 前線には乗らない（容量上 DP を取れず、段配置で slot が決まる）。F（τ 620 µs）でだけ PP2〜4 が agg 最大に乗る（τ を段数で分散できるため）。
- **EP32（expert を 32 chip に 1 回だけ置き dense を TP4×DP_a8）**: `DP1xPP1x(TP4xDPa8|EP32xTPx1)xCP1 b=256` agg 32,326 / tsu 126.3。agg 最大構成（EP2×TP_x4）より低いのは all-to-all（直径 5 hop × 2）を 48 層で払うため。
- **今日の実装拘束内（PP1, TP∈{4,8}, EP=TP, DP_a=1, CP1, lanes×(k+1) ≤ 32）, A°, k=1**: agg 9,630（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 510（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` は A° で 9,630 / 75.2、F（校正）で 2,780 / 21.7；**実測 1,186.9 t/s は F の 43%、A° の 12%**（F との差 = stall 1.14 倍・admission・ignore_eos 以外の pass 内オーバーヘッド）。

### 6.2 構造的な読み

1. agg はユーザ状態（GDN fp32 0.231 GB/user/pass の読み+書き）の DRAM 往復が漸近で、重みは EP×TP_x で 1 回読めば償却できる。DP で複製すると容量を失い b が伸びない → agg 最大は DP4 以下、TP8〜16 × EP2〜8 × CP4〜8、b ≥ 512。
2. tsu は collective のレイテンシ（hop × α × 層数 × 回数）と expert の直列露出で決まる。TP16（直径 3 hop）が TP32（直径 6 hop + padding）に勝ち、ring1d（15 hop）だと TP8 が最適に落ちる（B: `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` 353）。
3. 先読み（全ヒット）の価値は τ が小さい時だけ: A vs A' は tsu 566 → 628、agg は変わらない。τ=200 µs（E）では tsu が 1/5 に落ち、先読み（E'）で 129、実測ヒット率（E''）で 122。
4. expert ストリーム 236 GB/s（D）は agg を A の 62% に落とす最大の単独因子。bank あたり 36→64 GB/s（2 reader/bank か bank 跨ぎレイアウト）が先。
5. CP は ctx 192 でも Pareto 前線に乗る（KV 4.9 MB/row の分割）が寄与は小さい。ctx 32k（KV 75.5 MB/row）では CP8〜16 が必須になる（別表、`limit_perf_5d.py routing/stats.json 32768`）。

### 6.3 表

### 表 A: シナリオ別の最大 agg と最大 tsu（ctx 192、分子は census 全数）

| シナリオ | k | 最大 agg [t/s] | その tsu | users | 構成 | 最大 tsu | その agg | users | 構成 |
|---|---|---|---|---|---|---|---|---|---|
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 591.7 | 1,183 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 748.9 | 1,498 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 3 | 124,296 | 30.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1076.9 | 2,154 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 4 | 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1334.7 | 2,669 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 0 | 49,471 | 12.1 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 442.2 | 884 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 1 | 62,831 | 15.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 565.6 | 1,131 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 3 | 99,359 | 24.3 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 877.3 | 877 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 4 | 115,703 | 28.2 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 1101.3 | 1,101 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 0 | 49,615 | 12.1 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 494.7 | 989 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 1 | 62,999 | 15.4 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 627.6 | 1,255 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 3 | 99,595 | 24.3 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 959.4 | 959 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 4 | 115,963 | 28.3 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 1207.0 | 1,207 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 0 | 42,540 | 10.4 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 278.5 | 1,114 | 4 | `DP4xPP1x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 1 | 49,717 | 12.1 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 352.7 | 1,411 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 3 | 72,442 | 17.7 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 522.9 | 2,092 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 4 | 82,150 | 20.1 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 622.7 | 2,491 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 0 | 39,650 | 9.7 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 413.8 | 828 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 1 | 45,074 | 11.0 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 531.7 | 1,063 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 3 | 61,597 | 15.0 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP2 b=512` | 809.8 | 810 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 4 | 70,378 | 17.2 | 4096 | `DP4xPP4x(TP2xDPa1|EP1xTPx2)xCP1 b=256` | 1014.9 | 1,015 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 0 | 33,499 | 8.2 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 370.4 | 741 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 1 | 38,911 | 9.5 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 472.3 | 472 | 1 | `DP1xPP1x(TP32xDPa1|EP1xTPx32)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 3 | 54,672 | 13.3 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 679.3 | 679 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 4 | 60,596 | 14.8 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 868.8 | 869 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 0 | 42,833 | 10.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 469.8 | 940 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 1 | 55,166 | 13.5 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 564.0 | 1,128 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 3 | 77,465 | 18.9 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 759.3 | 759 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 4 | 80,623 | 19.7 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 976.1 | 976 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 0 | 33,462 | 8.2 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 85.6 | 171 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 1 | 38,911 | 9.5 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 114.7 | 229 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 3 | 54,672 | 13.3 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 192.3 | 192 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 4 | 60,596 | 14.8 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 238.8 | 239 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 0 | 33,565 | 8.2 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 94.6 | 757 | 8 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 1 | 38,975 | 9.5 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 128.9 | 1,031 | 8 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 3 | 54,743 | 13.4 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 228.9 | 1,831 | 8 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 4 | 60,667 | 14.8 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 278.7 | 1,115 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 0 | 33,565 | 8.2 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 90.0 | 360 | 4 | `DP4xPP1x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 1 | 38,975 | 9.5 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 122.4 | 490 | 4 | `DP4xPP1x(TP8xDPa1|EP1xTPx8)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 3 | 54,743 | 13.4 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 210.5 | 421 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 4 | 60,667 | 14.8 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 268.0 | 1,072 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 0 | 22,362 | 5.5 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 30.5 | 244 | 8 | `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 1 | 23,518 | 5.7 | 4096 | `DP4xPP2x(TP4xDPa1|EP1xTPx4)xCP1 b=512` | 40.8 | 163 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 3 | 33,850 | 8.3 | 4096 | `DP4xPP4x(TP2xDPa1|EP1xTPx2)xCP1 b=256` | 69.2 | 277 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 4 | 38,923 | 9.5 | 4096 | `DP4xPP4x(TP2xDPa1|EP1xTPx2)xCP1 b=256` | 84.1 | 337 | 4 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` |

### 表 P-1: Pareto 前線（A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops), k=1）

| agg [t/s] | tsu | users | 構成 | t_step [ms] | T_tok [ms] | routing |
|---|---|---|---|---|---|---|
| 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 78.620 | 78.620 | fitted |
| 71,617 | 35.0 | 2048 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=1024` | 39.349 | 39.349 | fitted |
| 70,200 | 68.6 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 20.072 | 20.072 | fitted |
| 56,137 | 109.6 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.550 | 12.550 | fitted |
| 40,626 | 158.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.671 | 8.671 | fitted |
| 26,641 | 208.1 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP4 b=128` | 6.611 | 6.611 | census |
| 15,850 | 247.7 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP1 b=64` | 5.556 | 5.556 | census |
| 9,615 | 300.5 | 32 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=32` | 4.579 | 4.579 | census |
| 6,089 | 380.6 | 16 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=16` | 3.616 | 3.616 | census |
| 4,044 | 505.5 | 8 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=4` | 2.722 | 2.722 | census |
| 2,537 | 634.1 | 4 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=2` | 2.170 | 2.170 | census |
| 1,498 | 748.9 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 1.837 | 1.837 | census |

### 表 P-0: Pareto 前線（A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops), k=0）

| agg [t/s] | tsu | users | 構成 | t_step [ms] | T_tok [ms] | routing |
|---|---|---|---|---|---|---|
| 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 76.656 | 76.656 | fitted |
| 53,206 | 26.0 | 2048 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=1024` | 38.492 | 38.492 | fitted |
| 52,069 | 50.8 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 19.666 | 19.666 | fitted |
| 42,031 | 82.1 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.181 | 12.181 | fitted |
| 31,327 | 122.4 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.172 | 8.172 | census |
| 21,651 | 169.2 | 128 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=128` | 5.912 | 5.912 | census |
| 13,484 | 210.7 | 64 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP1 b=64` | 4.746 | 4.746 | census |
| 8,232 | 257.3 | 32 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=16` | 3.887 | 3.887 | census |
| 5,511 | 344.4 | 16 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=8` | 2.903 | 2.903 | census |
| 3,502 | 437.8 | 8 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=4` | 2.284 | 2.284 | census |
| 2,102 | 525.4 | 4 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=2` | 1.903 | 1.903 | census |
| 1,183 | 591.7 | 2 | `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 1.690 | 1.690 | census |

### 表 P-4: Pareto 前線（A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops), k=4）

| agg [t/s] | tsu | users | 構成 | t_step [ms] | T_tok [ms] | routing |
|---|---|---|---|---|---|---|
| 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 82.289 | 82.289 | fitted |
| 147,613 | 72.1 | 2048 | `DP2xPP1x(TP8xDPa2|EP4xTPx4)xCP4 b=1024` | 41.900 | 41.900 | fitted |
| 142,030 | 138.7 | 1024 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=1024` | 21.773 | 21.773 | fitted |
| 112,424 | 219.6 | 512 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=512` | 13.754 | 13.754 | fitted |
| 79,345 | 309.9 | 256 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=256` | 9.744 | 9.744 | fitted |
| 50,270 | 392.7 | 128 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=128` | 7.690 | 7.690 | fitted |
| 30,023 | 469.1 | 64 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=64` | 6.438 | 6.438 | fitted |
| 16,826 | 525.8 | 32 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=32` | 5.744 | 5.744 | fitted |
| 9,881 | 617.5 | 16 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=16` | 4.890 | 4.890 | fitted |
| 6,309 | 788.7 | 8 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP2 b=8` | 3.829 | 3.829 | fitted |
| 4,121 | 1030.2 | 4 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP1 b=4` | 2.931 | 2.931 | fitted |
| 2,669 | 1334.7 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` | 2.263 | 2.263 | fitted |

### 表 I: 今日の実装拘束の中（PP1, TP∈{4,8}, EP=TP, DP_a=1, CP1, lanes×(k+1) ≤ 32; k=0 は 1 stream）での最大

| シナリオ | k | 最大 agg [t/s] | その tsu | 構成 | 最大 tsu | その agg | 構成 |
|---|---|---|---|---|---|---|---|
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 3 | 8,533 | 133.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A° spec, torus, all-miss, DATAFLOW-FIRST overlap (DRAM never stops) | 4 | 7,092 | 221.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 0 | 1,731 | 216.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 304.1 | 1,216 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 1 | 9,206 | 71.9 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 375.3 | 1,501 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 3 | 8,160 | 127.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 529.4 | 2,117 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A  spec, torus collectives, all-miss (lambda 5 us) | 4 | 6,701 | 209.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 704.9 | 2,819 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 0 | 1,825 | 228.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 328.0 | 1,312 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 1 | 9,323 | 72.8 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 401.6 | 1,606 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 3 | 8,263 | 129.1 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 558.2 | 2,233 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| A' spec, torus collectives, all-hit (perfect expert prefetch) | 4 | 6,814 | 212.9 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 746.7 | 2,987 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 0 | 1,683 | 210.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 251.3 | 1,005 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 1 | 8,978 | 70.1 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 314.8 | 1,259 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 3 | 7,960 | 124.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 457.4 | 1,829 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| B  spec, ring1d collectives (today's ttnn Ring/Linear CCL) | 4 | 6,543 | 204.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 600.5 | 2,402 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 0 | 1,674 | 209.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 290.4 | 1,162 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 1 | 8,879 | 69.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 360.0 | 1,440 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 3 | 7,873 | 123.0 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 504.0 | 2,016 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| C  spec + 0.8 us/packet (job132 fit), torus | 4 | 6,490 | 202.8 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 668.4 | 2,674 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 0 | 1,323 | 165.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 229.7 | 919 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 1 | 4,931 | 38.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 268.1 | 1,072 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 3 | 4,241 | 66.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 336.2 | 1,345 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D  C + expert stream 236 GB/s (job128) | 4 | 3,583 | 112.0 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 463.3 | 1,853 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 0 | 1,548 | 193.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 309.2 | 1,237 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 1 | 5,154 | 40.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 343.3 | 1,373 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 3 | 4,425 | 69.1 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 403.0 | 1,612 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| D° D with DATAFLOW-FIRST overlap | 4 | 3,759 | 117.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 569.5 | 2,278 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 0 | 608 | 75.9 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 79.3 | 317 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 1 | 4,296 | 33.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 103.1 | 412 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 3 | 3,635 | 56.8 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 158.5 | 634 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E  D + tau 200 us/layer, all-miss | 4 | 2,871 | 89.7 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 205.1 | 821 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 0 | 757 | 94.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 94.6 | 757 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 1 | 4,964 | 38.8 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 128.6 | 1,028 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 3 | 4,268 | 66.7 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 227.4 | 1,819 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` |
| E' D + tau 200 us/layer, all-hit | 4 | 3,615 | 113.0 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 278.5 | 1,114 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 0 | 691 | 86.4 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 86.4 | 346 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 1 | 4,964 | 38.8 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 116.7 | 467 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 3 | 4,268 | 66.7 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 196.2 | 785 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| E" D + tau 200 us/layer, measured pred1 hit rate (census) | 4 | 3,615 | 113.0 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 267.3 | 1,069 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 0 | 237 | 29.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 29.6 | 237 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 1 | 2,780 | 21.7 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 39.6 | 158 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 3 | 2,398 | 37.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 66.5 | 266 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| F  D + ring1d + tau CALIBRATED to the served 63.36 ms pass, all-miss | 4 | 1,749 | 54.7 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 83.2 | 333 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |

### 表 R: 参照構成（A° と F、k=1）

| 構成 | A° agg | A° tsu | A° t_step [ms] | A° T_tok [ms] | F agg | F tsu | F T_tok [ms] | DRAM/chip [GB] |
|---|---|---|---|---|---|---|---|---|
| `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 9,630 | 75.2 | 18.290 | 18.290 | 2,780 | 21.7 | 63.360 | 19.2 |
| `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=32` | 12,568 | 98.2 | 14.014 | 14.014 | 2,895 | 22.6 | 60.831 | 9.9 |
| `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1` | 840 | 26.3 | 1.638 | 52.404 | 788 | 24.6 | 55.861 | 2.5 |
| `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=32` | 10,683 | 10.4 | 4.122 | 131.898 | 5,692 | 5.6 | 247.562 | 6.2 |
| `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=1` | 540 | 539.9 | 2.548 | 2.548 | 36 | 35.8 | 38.489 | 2.3 |
| `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=32` | 7,963 | 248.8 | 5.530 | 5.530 | 859 | 26.9 | 51.236 | 2.5 |
| `DP1xPP1x(TP4xDPa8|EP32xTPx1)xCP1 b=32` | 6,749 | 210.9 | 6.524 | 6.524 | 1,055 | 33.0 | 41.743 | 3.6 |
| `DP1xPP4x(TP8xDPa1|EP8xTPx1)xCP1 b=32` | 11,854 | 92.6 | 3.715 | 14.858 | 2,758 | 21.5 | 63.859 | 2.9 |
| `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 1,498 | 748.9 | 1.837 | 1.837 | 79 | 39.6 | 34.785 | 4.7 |
| `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 71,688 | 17.5 | 78.620 | 78.620 | 16,958 | 4.1 | 332.354 | 24.7 |


## 7. 実装拘束と、限界に近づく順序（この文書の結論を実装に移すときの順）

1. τ（層あたり 620 µs の非 DRAM 時間、pass の約半分）を削る。device profiler で router→moe_compute、collective、dispatch を同一 run で分解する（§9.3-14）。
2. expert ストリームを 236 → 512 GB/s へ（bank あたり 2 reader、または bank 跨ぎレイアウト）。
3. collective を ring1d から torus 直径算法へ（TP16 で 15 hop → 3 hop）。
4. EP×TP_x の 2D（moe_compute の collective combine）と、32 行を超える rows（複数 tile）の lanes。
5. 先読み（1 ブロック早い投機 routing、ヒット 0.78〜0.94）は 1〜3 の後。
6. CP は長文脈（≥ 32k）で。

## 8. 未確定（決着方法）

- λ（router → expert 読み発行）5 µs と τ の内訳: device profiler で直接観測。
- k=3/4 の R_eff（2.46/3.02）は 4×p150 の自然文記録で、BH Galaxy 上・本 census の prompt 集合では未測。
- rows > 256 の routing は fitted（census は 256 まで）。b=512/1024 の agg はこの外挿に依存する。
- torus 算法の collective（直径 hop、8 link egress）は ttnn に無く、机上値。
- 1 chip あたり DRAM 予約 10% は仮定。
