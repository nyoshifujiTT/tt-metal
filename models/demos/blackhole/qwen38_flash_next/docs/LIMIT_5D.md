# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-dimensional parallel limit (DP x PP x TP x EP x CP)

この文書は「32 chip の BH Galaxy で decode を回すとき、DP・PP・TP・EP・CP の 5 次元をどう切れば agg tps（機械全体の tokens/s）と tsu（1 ユーザあたりの tokens/s）がそれぞれ最大になるか」を、**物理値と HF 定義だけ**から机上で出した現在の確定状態を書く（27B README §1-3 と同じ規則: 現行実装の実測値は設計の入力にしない。実測は合格判定にだけ使う）。経緯・実測・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用）にあり、この文書と食い違ったら新しい方の作業記録が正しい。生成器は `BH-GLX_Qwen3.8-Flash-Next_limit_perf_5d_phys_20261024.py`（同ディレクトリ、全式と出典を docstring に持つ）、分子の routing 統計は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`）、表は `..._limit_5d_phys_tables_20261024.py`、この文書は `..._render_readme_5d_phys_20261024.py` が同じ入力から決定的に生成する。

仮定は「この場合には」で断る。代表仮定が置けない所（collective の算法、expert 先読みのヒット率）は**シナリオとして並列列挙**し（§5）、一本化しない。設計値 2 つ（hop レイテンシ 280 ns、行列積 33 cycle/tile-MAC）に tsu の答えがどれだけ依存するかは §6.2 に感度として書く。

## 1. 一般論: 5 つの並列軸は decode で何を分け、何を払うか

decode の 1 step は「全層の重みと、ユーザごとの状態（GDN の再帰状態、QSA の KV）を DRAM から読み、行列積と再帰更新を行い、次の token を出す」の繰り返しで、重み 1 回の読みを何行（ユーザ×verify 行）で償却できるかが agg を、1 step の長さが tsu を決める。

| 軸 | 何を分けるか | 重みの読み | 払うもの（1 層あたり） | agg への効き | tsu への効き |
|---|---|---|---|---|---|
| DP（data） | ユーザ集合を複製モデルに分ける | 複製ごとに全部（DRAM 容量を複製分食う） | 無し（複製間通信ゼロ） | 複製数に線形 | 不変 |
| PP（pipeline） | 層を段に分け、段ごとに別の micro-batch を同時処理 | 段の重みだけ、micro-batch ごとに読み直す | 段間の受け渡し 1 回（活性 rows×4H） | 全段が常に忙しければ TP と同じ償却 | 1 token が全段を巡るので T_tok = PP × (最遅段 + 受け渡し)（tsu ≈ 1/PP） |
| TP（tensor） | 1 層の行列を TP 枚に分け全員で 1 行列を解く | 互いに素な 1/TP ずつ 1 回 | all-reduce（= reduce-scatter → 残差 → all-gather を 1 本）×ブロック数: link (TP−1)/TP·T/帯域 + 最後の tile のレイテンシ | 重みを TP 枚で分担（1 step が短くなる） | 大きく効くが最後の tile のレイテンシ × 本数 × 層数が床 |
| EP（expert） | routed expert を所有グループに分ける | expert は 1 回だけ（複製しない） | DP_a>1 のとき dispatch+combine の all-to-all（rows×k×H） | distinct expert の償却が instance 全体で効く | router 後にしか読めない露出が残る |
| CP（context） | 1 ユーザの KV を系列方向に分ける | 無関係（KV 読みを CP で割る） | 部分 attention の merge（o, lse） | 長文脈で KV 読みが支配する時に効く | 同左 |

教科書的な式（Megatron-LM、DeepSpeed-Inference、Pope et al. 2022「Efficiently Scaling Transformer Inference」、DeepSeek-V3 の EP320 配備）: ring all-reduce T = 2(N−1)/N·M/B_link + 2(N−1)·α；PP の効率 = M/(P+M−1)（M micro-batch、P 段）；最適バッチ b* は「重み読み時間 = 演算時間」となる点；per-chip latency の床は collective の α 項。CP は KV が系列長で伸びる full-attention にだけ効き、系列長 O(1) の再帰状態（GDN）には効かない（KDA/vLLM の実装でも再帰状態は DCP で分割せず複製）。

## 2. 専門論: パフォーマンスエンジニアリングとしての扱い（本件の指針 §9.3 の 5D 版）

1. 限界性能の分母は物理値（DRAM 帯域、link 帯域、hop レイテンシ、蓄積転送、NoC、Tensix の cycle 単価、L1/DRAM 容量）と HF 定義（形状・dtype）だけから出す。**実装の実測値（collective の所要時間、expert ストリームの実効帯域、op 間の隙間、パケット固定費の当てはめ）は限界性能に入れない**。それらは合格判定（実測 ÷ 限界 = 効率）と「どこで失っているか」の分解にだけ使う（§9.4）。
2. 分子はバイト（重み・状態・KV・routed expert の実 distinct 数）で、routing は一様仮定ではなく実測（census）を使う。一様仮定は distinct を過大に見積もる（B=16: 139 vs 実測 124、B=256: 509 vs 426）。routing は実装の性質ではなくモデルの性質なので分子に入れてよい。
3. 5 軸は独立ではない: chip 数 32 = DP × PP × S、段の S chip は dense 側で TP_a × DP_a、routed 側で EP × TP_x に**同時に**分かれる。候補は全列挙し、Pareto 前線（agg 降順で tsu が単調増加する点列）で示す。「最大 agg」と「最大 tsu」は別の構成になるのが普通で、1 点だけを答えにしない。
4. 時間は層ごとに「DRAM 車線（prefetch 可能な連続ストリーム、止まらない）」と「依存車線（演算 → 隠れない通信 → router 後の expert 読み）」に分けて max を取る。通信は tile（2 KiB）ごとに流して演算と重ね、隠れない分 = link 時間 − 重なった演算 + 最後の tile のレイテンシ。
5. MoE は「全ヒット／全ミス／実測ヒット率」の 3 ケース（§9.4.9）を 5D の各点で持つ。dense の連続ストリーム設計は dense 部分にだけ適用する。
6. 容量制約（DRAM 32 GB/chip）を必ず同時に判定する。重みは EP×TP_x×PP で 1 回だけ置くのが原則で、DP で複製するたびに「ユーザ状態の置き場」が減る。
7. 実装拘束（今日の runtime: TP∈{4,8}、EP=TP、1 行タイル 32 行、PP/CP 未実装、EP>1 の collective combine 未実装）は別枠で明記し、拘束なしの限界と拘束下の限界の両方を出す。
8. 答えが設計値 1 つに乗っているときは、その設計値を 2 倍・4 倍にしたときの答えを並べる（§6.2）。

## 3. 応用: TT デバイス（BH Galaxy）で 5 軸はどう変わるか（物理値と出典）

- **トポロジ**: 32 chip は 8×4 の 2D torus（`single_bh_galaxy_torus_xy_graph_descriptor.textproto`: dims [8,4] RING/RING、channels 2）。実機 cluster descriptor では「8 chip のリング 4 本、各リングが 4 chip の線 2 本」、全 chip の ETH 次数 4。1 方向 2 link × 25.0 GB/s（`ccl_common.cpp lookup_fabric_link_bw`: 400 Gbps = 50 GB/s の半分に制限中）、4 方向で egress 200 GB/s/chip。
- **hop のレイテンシと蓄積転送（設計値）**: 1 hop 片道 280 ns（EthernetSsDesignSpec Table 16、L1→相手 L1）。送り側は蓄積転送でパケット（2 KiB tile）を L1 から 96 B/cycle で読んでから送るので hop ごとに 16 ns が乗る。線上は 1 パケットにつき 56 B 増える（4,096 B payload に対し 1.4%）。中継 chip の NoC は 64 B/cycle、最遠 8 ルータ hop で 92 cycle + flit。受け側は cut-through。
- **collective の算法が二通り**: 今日の ttnn Ring/Linear CCL は 1 次元リング（両方向で hops = ⌈(n−1)/2⌉、1 方向対の 2 link）。torus を使う算法（hops = submesh の直径: 完全リングは n/2、部分列は n−1；経路 = submesh 内の方向数 × 2 link、最大 8）は未実装。両方をシナリオに置く。1 行 [1,2560] bf16 の all-reduce 1 本（RS+AG）の物理時間: 4 chip torus link 0.08 µs + 最後の tile 1.37 µs、8 chip 0.06 + 2.14、16 chip 0.05 + 3.69、32 chip 0.05 + 4.47（ring1d: 1.37 / 2.92 / 6.02 / 12.22 µs）。層に all-reduce 2 本 + hc 統計 2 本 × 48 層なので、B=1 の TP の床は torus で 0.28 / 0.42 / 0.72 / 0.87 ms（TP 4/8/16/32）。
- **DRAM は読み書き和 512 GB/s/chip**（GDDR6 8 ch）。GDN の fp32 状態（36 層 × [48,128,128]、1 ユーザ 0.231 GB/pass 読み+書き）は dense 重みと同じ DRAM ストリームの一部で、ユーザ数に比例して増える → **agg の漸近は重みではなくユーザ状態の往復**（32 × 512 GB/s ÷ (0.2309 GB 状態 + 0.0049 GB KV) = 69,480 pass/s）。
- **Tensix の単価（27B README §3、tt-isa-documentation と tt-llk から）**: 行列積 33 cycle/32×32 tile-MAC（HiFi2）、unpack/pack 16 cycle/tile、op 固定 300 cycle、gated delta rule 400 cycle/(value head, user, S の 32 列) + 1,100 cycle/層、130 worker、1.35 GHz。B=1 では演算が 1 pass に 0.52 ms（TP32）〜0.91 ms（TP4）入り、通信レイテンシと同じ桁になる。
- **routed expert**: 512 expert × 49 層 BF4 = 69.4 GB、1 expert 2,764,800 B。router（top-10）が終わるまでその層の expert は読めない（露出）。
- **1 行タイル 32 行**: decode の行列積は M=32 行の tile 1 行で、B × (k+1) ≤ 32 が今日の lanes 形の上限。tile を超える行は複数 tile で扱えば限界性能上は制約でない（本モデルは b を 1024 まで振る）。
- **GDN 状態の TP 分割は通信不要**: 48 value head を TP_a で分け、head 内で閉じる（TP_a > 16 で最繁忙 chip が ⌈48/TP_a⌉ head を読む）。QSA は 24 query head、2 kv head（TP_a > 2 で kv head は複製、CP はその複製内で KV を分ける）。
- **EP > 1 と DP_a**: dense 側の TP_a グループより expert が広い（DP_a > 1）時に dispatch/combine の all-to-all が要る。今日の runtime は EP = TP_a のみ（EP×TP の 2D は moe_compute の collective combine 未実装、commit f6165e058a）。
- **MTP**: verify 行 k+1 を 1 ユーザが出すので rows = b×(k+1)、採択 R_eff tokens/pass（k=1 で 1.376 = 本件 ISL128 プロンプト集合の採択率、k=3 で 2.46、k=4 で 3.02 は 4×p150 の自然文記録 = 仮定。採択率はモデルとプロンプトの性質で、実装の性質ではない）。

## 4. 本モデルの式（生成器の docstring と同じ）

- chip: N = 32 = DP × PP × S、S = TP_a × DP_a = EP × TP_x、CP | max(1, TP_a/2)。
- ユーザ: 1 pipeline に PP micro-batch × b ユーザ、rows = b × (k+1)、R_eff tokens/pass。
- 層 l の chip あたり DRAM 車線: D_l = dense_l/TP_a + state_l·(b/DP_a)/TP_a + kv_l(ctx)·(rows/DP_a)·h/2/CP、X_l = 最繁忙 EP group の distinct expert 数（census）× 2,764,800 B × pad / TP_x。t_dram = (D_l + X_l)/512 GB/s。
- 依存車線: 演算（27B §3 の単価で rows/DP_a 行、dense は /TP_a、expert は /TP_x）+ 隠れない通信 + router 後に露出する expert 読み（全ヒット 0 / 全ミス X_l / 実測 = census の pred1 miss 集合）/512 GB/s。
- 通信: ring AG/RS の link = (n−1)/n·T·(1+56/4096)/(paths × 25 GB/s)；最後の tile のレイテンシ = hops × (280 ns + 2,048 B/(96 B/cycle)) + (hops−1) × NoC 中継(92 cycle + 32 flit)；all-reduce = RS→残差→AG を 1 本（link は両方、レイテンシは最後の tile が両経路を通るので 2 倍）；層に all-reduce 2 本 + hc 統計 2 本；DP_a>1 で all-to-all 2 本（dispatch, combine）；CP>1 の QSA 層で merge 1 本。隠れない分 = Σlink − min(Σlink, 演算) + Σレイテンシ。
- 層時間 t_l = max(t_dram, 演算 + 隠れない通信 + 露出 expert)。段: 48 層 + tail（final mixer, lm_head 0.675 GB bf8, MTP 層）を最大段最小で連続分割。t_slot = max_s t_stage + 受け渡し（PP>1）、agg = DP × b × R_eff / t_slot、T_tok = PP × t_slot（PP=1 なら t_stage）、tsu = R_eff / T_tok。
- 容量: experts 69.4 GB/(PP×EP×TP_x) + dense 5.38 GB/(PP×TP_a) + b/DP_a × (状態 0.1155 GB + KV) ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% を残す）。
- routing の分子: census（34 item、うち decode 位置を持つ 32 item × 256 = 8,192 位置、CPU oracle、bf16 expert、greedy 連鎖）から rows ∈ {1,2,4,8,16,32,64,128,256} と MTP 2 行 {1..128}×2 の最繁忙 group を取る。rows > 256 は「fitted」（未使用 expert 数が rows 倍増ごとに census の 128→256 比 0.591 で減衰、偏り 1+0.6/√mean、miss 率 = census B=256 の 0.058）と明記。

routing census の要点: h1（前 token の top-10 再利用）= 0.356；1 ブロック早い投機 routing のヒット率 B=1 0.775 / B=16×2 行 0.845 / B=256 0.942；実 distinct expert/層 B=16 124.2（一様 138.6）、B=32 198.9（239.6）、B=256 426.2（508.7）；最繁忙 4 group の expert 数 B=16×2: 51.3（平均の 1.12 倍）。

## 5. シナリオ（代表仮定が置けないので並列列挙。全部物理値、校正値は無い）

| 記号 | collective の算法 | expert 先読み |
|---|---|---|
| P1 | torus（submesh の直径、方向数×2 link） | 全ミス（router 後に全部読む） |
| P2 | torus | 全ヒット（router 前に読み終える） |
| P3 | torus | 実測ヒット率（census の 1 ブロック早い投機 routing の miss 集合だけ露出） |
| P4 | ring1d（今日の ttnn Ring/Linear CCL） | 全ミス |
| P5 | ring1d | 全ヒット |

## 6. 結果（ctx 192 = ISL 128 + decode。分子は census 全数）

### 6.1 一行の答え

- **agg tps 最大（この場合には P1: torus・全ミス・MTP k=1）**: **71,688 t/s** = `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024`（4096 users、t_slot 78.6 ms = DRAM 車線 78.6 / 依存車線 38.8、tsu 17.5）。k=0 では 53,433（`DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024`）、k=4（R_eff 3.02 仮定）では 150,324。漸近（ユーザ状態の往復だけ）は 69,480 pass/s × R_eff = k=1 で 95,605 t/s、P1 はその 75%。**agg の答えは通信の算法（P1〜P5）に依らない**（DRAM 車線が依存車線の 2 倍で、通信は演算に隠れる）。
- **tsu 最大（P1, k=1）**: **831 t/s/u** = `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1`（T_tok 1.655 ms = 演算 0.498 + 通信レイテンシ 17.9 µs/層 × 48 + 露出 expert；DRAM 車線は 0.875 ms で支配していない）。b=1 の各 TP で最良の構成は TP16 `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` 796、TP8 `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1` 620、TP4 `DP8xPP1x(TP4xDPa1|EP1xTPx4)xCP2 b=1` 348。全ヒット（P2）なら 1046（`DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1`）、実測ヒット率（P3）958、ring1d（P4）632（`DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1`）。k=0 では 664、k=4 では 1628。
- **PP32**（48 層 + tail を 32 段、`DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1`）: tsu 26.3、agg 840（tail 段 = final mixer + lm_head + MTP 層の DRAM 1.64 ms が slot を決め、1 token は 32 slot）；b=32 で agg 10,682、tsu 10.4。PP は通信を段間受け渡し 1 本にする形だが、Pareto 前線には乗らない（容量上 DP を取れず、段配置で slot が決まる）。
- **EP32（expert を 32 chip に 1 回だけ置き dense を TP4×DP_a8）**: `DP1xPP1x(TP4xDPa8|EP32xTPx1)xCP1 b=256` agg 32,326 / tsu 126.3。agg 最大構成（EP2×TP_x4）より低いのは all-to-all 2 本（link 18.4 + レイテンシ 9.9 µs/層）を 48 層で払うため。
- **今日の実装拘束内（PP1, TP∈{4,8}, EP=TP, DP_a=1, CP1, lanes×(k+1) ≤ 32）, P1, k=1**: agg 9,630（`DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16`）、tsu 510（`DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1`）。現行形 `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` の限界は 9,630 t/s（DRAM 車線 18.29 ms、依存車線 17.84 ms）；**実測 1,186.9 t/s はその 12.3%（効率）**。差の分解は §9.4（実測の分解）側にあり、限界性能には入れない。

### 6.2 構造的な読みと感度

1. agg はユーザ状態（GDN fp32 0.231 GB/user/pass の読み+書き）の DRAM 往復が漸近で、重みは EP×TP_x で 1 回読めば償却できる。DP で複製すると容量を失い b が伸びない → agg 最大は DP4 以下、TP8〜16 × EP2〜8 × CP4〜8、b ≥ 512。通信は演算に隠れ、答えは算法に依らない。
2. tsu（B=1）は「最後の tile のレイテンシ × 4 本 × 48 層」と「演算 0.5 ms」の和で決まり、DRAM 車線はもう支配していない。TP を上げるほど演算と DRAM は減るがレイテンシの床（TP4 0.28 → TP32 0.87 ms）が増え、P1 では TP32 がわずかに勝つ。**この順位は hop レイテンシの設計値 1 つに乗っている**（各 TP の b=1 最良構成で、hop 280 ns を 1 倍 / 2 倍 / 4 倍にしたときの tsu）: TP32 831 / 597 / 382、TP16 796 / 606 / 411、TP8 620 / 620 / 480、TP4 348 / 348 / 348 → 最良の TP は 1 倍で TP32、2 倍で TP8、4 倍で TP8。「TP32 が最適」は hop 280 ns と torus 算法の両方が成り立つ場合に限る、と読むこと。ring1d（今日の CCL）では TP16 の 632 が最大で TP32 は 391。
3. 先読みの価値（P1→P2）は tsu で 831 → 1046（実測ヒット率 P3 で 958）、agg では無い。
4. CP は ctx 192 でも Pareto 前線に乗る（KV 4.9 MB/row の分割）が寄与は小さい。ctx 32k（KV 75.5 MB/row）では CP8〜16 が必須になる（`limit_perf_5d_phys.py routing/stats.json 32768`）。

### 6.3 表

### 表 A: シナリオ別の最大 agg と最大 tsu（ctx 192、分子は census 全数）

| シナリオ | k | 最大 agg [t/s] | その tsu | users | 構成 | 最大 tsu | その agg | users | 構成 |
|---|---|---|---|---|---|---|---|---|---|
| P1 torus collectives, all-miss | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 664.3 | 664 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` |
| P1 torus collectives, all-miss | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 831.4 | 831 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| P1 torus collectives, all-miss | 3 | 124,296 | 30.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1220.9 | 1,221 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| P1 torus collectives, all-miss | 4 | 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1628.1 | 1,628 | 1 | `DP1xPP1x(TP32xDPa1|EP16xTPx2)xCP1 b=1` |
| P2 torus collectives, all-hit | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 811.5 | 1,623 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` |
| P2 torus collectives, all-hit | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1046.0 | 2,092 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| P2 torus collectives, all-hit | 3 | 124,296 | 30.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1676.4 | 1,676 | 1 | `DP1xPP1x(TP32xDPa1|EP16xTPx2)xCP1 b=1` |
| P2 torus collectives, all-hit | 4 | 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 2212.5 | 2,213 | 1 | `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 754.5 | 1,509 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 957.6 | 1,915 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 3 | 124,296 | 30.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1478.9 | 1,479 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 4 | 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 2164.7 | 2,165 | 1 | `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 518.0 | 2,072 | 4 | `DP4xPP1x(TP8xDPa1|EP1xTPx8)xCP4 b=1` |
| P4 ring1d collectives, all-miss | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 631.9 | 1,264 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 3 | 118,814 | 29.0 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 877.2 | 1,754 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 4 | 135,515 | 33.1 | 4096 | `DP4xPP1x(TP1xDPa8|EP2xTPx4)xCP1 b=1024` | 1119.1 | 2,238 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 0 | 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 594.8 | 1,190 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 1 | 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 788.4 | 1,577 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 3 | 124,296 | 30.3 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 1291.4 | 2,583 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 4 | 143,787 | 35.1 | 4096 | `DP4xPP1x(TP2xDPa4|EP2xTPx4)xCP1 b=1024` | 1685.9 | 3,372 | 2 | `DP2xPP1x(TP16xDPa1|EP16xTPx1)xCP1 b=1` |

### 表 P-1: Pareto 前線（P1 torus collectives, all-miss, k=1）

| agg [t/s] | tsu | users | 構成 | t_slot [ms] | T_tok [ms] | DRAM 車線 [ms] | 依存車線 [ms] | routing |
|---|---|---|---|---|---|---|---|---|
| 71,688 | 17.5 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 78.620 | 78.620 | 78.620 | 38.850 | fitted |
| 71,617 | 35.0 | 2048 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=1024` | 39.349 | 39.349 | 39.349 | 20.813 | fitted |
| 70,200 | 68.6 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 20.072 | 20.072 | 20.072 | 11.977 | fitted |
| 56,137 | 109.6 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.550 | 12.550 | 12.550 | 9.023 | fitted |
| 40,626 | 158.7 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.671 | 8.671 | 8.671 | 8.141 | fitted |
| 25,875 | 202.1 | 128 | `DP1xPP1x(TP16xDPa2|EP32xTPx1)xCP8 b=128` | 6.807 | 6.807 | 6.807 | 6.793 | census |
| 14,747 | 230.4 | 64 | `DP1xPP1x(TP8xDPa4|EP16xTPx2)xCP4 b=64` | 5.972 | 5.972 | 5.972 | 5.813 | census |
| 8,929 | 279.0 | 32 | `DP1xPP1x(TP8xDPa4|EP16xTPx2)xCP1 b=32` | 4.931 | 4.931 | 4.921 | 4.931 | census |
| 5,672 | 354.5 | 16 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP1 b=16` | 3.882 | 3.882 | 3.644 | 3.882 | census |
| 3,901 | 487.6 | 8 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=2` | 2.822 | 2.822 | 2.822 | 2.668 | census |
| 2,523 | 630.7 | 4 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=2` | 2.182 | 2.182 | 1.585 | 2.182 | census |
| 1,592 | 796.1 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` | 1.729 | 1.729 | 1.245 | 1.729 | census |
| 831 | 831.4 | 1 | `DP1xPP1x(TP32xDPa1|EP8xTPx4)xCP1 b=1` | 1.655 | 1.655 | 0.875 | 1.655 | census |

### 表 P-0: Pareto 前線（P1 torus collectives, all-miss, k=0）

| agg [t/s] | tsu | users | 構成 | t_slot [ms] | T_tok [ms] | DRAM 車線 [ms] | 依存車線 [ms] | routing |
|---|---|---|---|---|---|---|---|---|
| 53,433 | 13.0 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 76.656 | 76.656 | 76.656 | 30.921 | fitted |
| 53,206 | 26.0 | 2048 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=1024` | 38.492 | 38.492 | 38.492 | 16.663 | fitted |
| 52,069 | 50.8 | 1024 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=1024` | 19.666 | 19.666 | 19.666 | 9.119 | fitted |
| 42,031 | 82.1 | 512 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=512` | 12.181 | 12.181 | 12.181 | 8.125 | fitted |
| 31,327 | 122.4 | 256 | `DP1xPP1x(TP16xDPa2|EP8xTPx4)xCP8 b=256` | 8.172 | 8.172 | 8.172 | 7.228 | census |
| 20,754 | 162.1 | 128 | `DP1xPP1x(TP16xDPa2|EP16xTPx2)xCP8 b=128` | 6.168 | 6.168 | 6.146 | 6.168 | census |
| 12,538 | 195.9 | 64 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP2 b=64` | 5.104 | 5.104 | 5.096 | 5.104 | census |
| 7,947 | 248.3 | 32 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP1 b=32` | 4.027 | 4.027 | 3.940 | 4.027 | census |
| 5,237 | 327.3 | 16 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP1 b=16` | 3.055 | 3.055 | 3.052 | 3.055 | census |
| 3,440 | 430.0 | 8 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=4` | 2.326 | 2.326 | 1.728 | 2.326 | census |
| 2,218 | 554.4 | 4 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=2` | 1.804 | 1.804 | 1.314 | 1.804 | census |
| 1,325 | 662.6 | 2 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP1 b=1` | 1.509 | 1.509 | 1.085 | 1.509 | census |
| 664 | 664.3 | 1 | `DP1xPP1x(TP32xDPa1|EP4xTPx8)xCP1 b=1` | 1.505 | 1.505 | 0.746 | 1.505 | census |

### 表 P-4: Pareto 前線（P1 torus collectives, all-miss, k=4）

| agg [t/s] | tsu | users | 構成 | t_slot [ms] | T_tok [ms] | DRAM 車線 [ms] | 依存車線 [ms] | routing |
|---|---|---|---|---|---|---|---|---|
| 150,324 | 36.7 | 4096 | `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 82.289 | 82.289 | 82.289 | 56.374 | fitted |
| 150,182 | 73.3 | 2048 | `DP2xPP1x(TP16xDPa1|EP4xTPx4)xCP8 b=1024` | 41.183 | 41.183 | 41.183 | 35.117 | fitted |
| 142,476 | 139.1 | 1024 | `DP1xPP1x(TP8xDPa4|EP8xTPx4)xCP4 b=1024` | 21.705 | 21.705 | 21.705 | 17.298 | fitted |
| 113,352 | 221.4 | 512 | `DP1xPP1x(TP16xDPa2|EP32xTPx1)xCP8 b=512` | 13.641 | 13.641 | 13.008 | 13.641 | fitted |
| 82,002 | 320.3 | 256 | `DP1xPP1x(TP16xDPa2|EP32xTPx1)xCP8 b=256` | 9.428 | 9.428 | 9.018 | 9.428 | fitted |
| 52,395 | 409.3 | 128 | `DP1xPP1x(TP16xDPa2|EP32xTPx1)xCP8 b=128` | 7.378 | 7.378 | 7.023 | 7.378 | fitted |
| 28,714 | 448.7 | 64 | `DP1xPP1x(TP8xDPa4|EP32xTPx1)xCP4 b=64` | 6.731 | 6.731 | 6.731 | 6.647 | fitted |
| 15,498 | 484.3 | 32 | `DP1xPP1x(TP8xDPa4|EP32xTPx1)xCP1 b=32` | 6.236 | 6.236 | 6.171 | 6.236 | fitted |
| 9,111 | 569.4 | 16 | `DP1xPP1x(TP8xDPa4|EP32xTPx1)xCP1 b=16` | 5.304 | 5.304 | 5.209 | 5.304 | fitted |
| 5,931 | 741.3 | 8 | `DP1xPP1x(TP8xDPa4|EP32xTPx1)xCP1 b=8` | 4.074 | 4.074 | 4.068 | 4.074 | fitted |
| 4,050 | 1012.5 | 4 | `DP1xPP1x(TP8xDPa4|EP16xTPx2)xCP4 b=4` | 2.983 | 2.983 | 2.983 | 2.971 | fitted |
| 2,685 | 1342.4 | 2 | `DP2xPP1x(TP16xDPa1|EP8xTPx2)xCP1 b=1` | 2.250 | 2.250 | 1.702 | 2.250 | fitted |
| 1,628 | 1628.1 | 1 | `DP1xPP1x(TP32xDPa1|EP16xTPx2)xCP1 b=1` | 1.855 | 1.855 | 1.074 | 1.855 | fitted |

### 表 I: 今日の実装拘束の中（PP1, TP∈{4,8}, EP=TP, DP_a=1, CP1, lanes×(k+1) ≤ 32; k=0 は 1 stream）での最大

| シナリオ | k | 最大 agg [t/s] | その tsu | 構成 | 最大 tsu | その agg | 構成 |
|---|---|---|---|---|---|---|---|
| P1 torus collectives, all-miss | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P1 torus collectives, all-miss | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P1 torus collectives, all-miss | 3 | 8,225 | 128.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P1 torus collectives, all-miss | 4 | 7,038 | 219.9 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P2 torus collectives, all-hit | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P2 torus collectives, all-hit | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P2 torus collectives, all-hit | 3 | 8,533 | 133.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P2 torus collectives, all-hit | 4 | 7,092 | 221.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 3 | 8,533 | 133.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P3 torus, measured pred1 hit rate (census) | 4 | 7,092 | 221.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 3 | 8,225 | 128.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P4 ring1d collectives, all-miss | 4 | 7,038 | 219.9 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 0 | 2,052 | 256.5 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 430.4 | 1,722 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 1 | 9,630 | 75.2 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 510.1 | 2,041 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 3 | 8,533 | 133.3 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=8` | 670.5 | 2,682 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |
| P5 ring1d collectives, all-hit | 4 | 7,092 | 221.6 | `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=4` | 914.6 | 3,659 | `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=1` |

### 表 R: 参照構成（k=1、torus all-miss と ring1d all-miss）

| 構成 | torus agg | torus tsu | T_tok [ms] | DRAM 車線 | 依存車線 | 通信 link+lat [µs/層] | ring1d agg | ring1d tsu | DRAM/chip [GB] |
|---|---|---|---|---|---|---|---|---|---|
| `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=16` | 9,630 | 75.2 | 18.290 | 18.290 | 17.843 | 5.0+5.5 | 9,630 | 75.2 | 19.2 |
| `DP4xPP1x(TP8xDPa1|EP8xTPx1)xCP1 b=32` | 12,550 | 98.0 | 14.034 | 14.014 | 14.034 | 7.8+8.6 | 12,418 | 97.0 | 9.9 |
| `DP8xPP1x(TP4xDPa1|EP4xTPx1)xCP1 b=1` | 2,433 | 304.2 | 4.524 | 4.524 | 2.876 | 0.3+5.5 | 2,433 | 304.2 | 18.7 |
| `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP1 b=1` | 2,471 | 617.8 | 2.227 | 2.227 | 1.837 | 0.2+8.6 | 2,471 | 617.8 | 9.4 |
| `DP2xPP1x(TP16xDPa1|EP2xTPx8)xCP1 b=1` | 1,552 | 776.1 | 1.773 | 1.222 | 1.773 | 0.2+14.8 | 1,239 | 619.3 | 4.7 |
| `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=1` | 681 | 681.0 | 2.021 | 1.222 | 2.021 | 0.2+17.9 | 391 | 391.3 | 2.3 |
| `DP1xPP1x(TP32xDPa1|EP32xTPx1)xCP1 b=32` | 8,140 | 254.4 | 5.409 | 4.758 | 5.409 | 6.5+17.9 | 6,356 | 198.6 | 2.5 |
| `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=1` | 840 | 26.3 | 52.398 | 1.636 | 0.661 | 0.0+0.0 | 840 | 26.3 | 2.5 |
| `DP1xPP32x(TP1xDPa1|EP1xTPx1)xCP1 b=32` | 10,682 | 10.4 | 131.902 | 4.095 | 3.746 | 0.0+0.0 | 10,682 | 10.4 | 6.2 |
| `DP1xPP4x(TP8xDPa1|EP8xTPx1)xCP1 b=32` | 11,758 | 91.9 | 14.980 | 3.688 | 3.718 | 7.8+8.6 | 11,632 | 90.9 | 2.9 |
| `DP1xPP1x(TP4xDPa8|EP32xTPx1)xCP1 b=256` | 32,326 | 126.3 | 10.897 | 10.897 | 6.659 | 18.4+9.9 | 32,326 | 126.3 | 4.5 |
| `DP4xPP1x(TP8xDPa1|EP2xTPx4)xCP4 b=1024` | 71,688 | 17.5 | 78.620 | 78.620 | 38.850 | 251.1+8.6 | 71,688 | 17.5 | 24.7 |


## 7. 実装拘束と、限界に近づく順序

1. 現行形の効率は 12.3%。失っている場所の分解（§9.4: 非 DRAM 時間、expert ストリーム、collective の実測、admission）は合格判定の側で行い、限界には入れない。
2. 32 行を超える rows（複数 tile）と EP×TP_x の 2D（moe_compute の collective combine）が、agg を実装拘束 9,630 → 71,688 へ開く前提。
3. collective を ring1d から torus 算法へ（B=1 の tsu で P4 632 → P1 831）。
4. 先読み（1 ブロック早い投機 routing、ヒット 0.78〜0.94）は tsu 側にだけ効く。
5. CP は長文脈（≥ 32k）で。

## 8. 未確定（決着方法）

- hop 280 ns（Table 16）は AI clk 1.5 GHz の表で本機 1.35 GHz では最大 311 ns；ケーブル伝搬は仕様値が無く入れていない。§6.2 の感度の範囲内。
- k=3/4 の R_eff（2.46/3.02）は 4×p150 の自然文記録で、本 census の prompt 集合では未測。
- rows > 256 の routing は fitted（census は 256 まで）。b=512/1024 の agg はこの外挿に依存する。
- torus 算法の collective は ttnn に無く、机上値。
- 1 chip あたり DRAM 予約 10% は仮定。
- 演算の単価は 27B README §3 の値（本モデルの行列積の形は同じ 32×32 tile、HiFi2）。expert の行列積は「平均 rows×10/512 行」で数え、行の偏りは最繁忙 group の expert 数にだけ入れている。
