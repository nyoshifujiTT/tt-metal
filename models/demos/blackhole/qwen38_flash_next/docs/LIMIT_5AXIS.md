# Qwen3.8-Flash-Next decode on one Blackhole Galaxy: the five-axis limit (DP, PP, TP, EP, DCP), revision 3

この文書は、32 chip の BH Galaxy で decode を回すときの DP・PP・TP・EP・DCP の全組合せ（系列長 ctx ∈ {1, 128, 8,192, 32,768, 262,144 (native), 1,000,000 (YaRN)} と現行計測条件の 192; 6510 構成 × router の置き方 2 形 × expert の列分割 2 形 × L1 の分け方 4 通り × 周 R ∈ {1,2,4} × 段の切り方・形、batch 1〜2048）について、1 token の時間と機械全体の tokens/s を**物理値と HF 定義だけ**から出し、agg tps と tsu（1 ユーザの tokens/s）それぞれの最大とその構成を示す現在の確定状態である。設計原則（DRAM を止めない等）は課さない: 構成は時間だけで比べ、寝ている chip があっても速ければ採る。実装の実測値は入れない（効率の判定にだけ使う）。経緯・議論は作業記録 `/opt/codex/shared-with-lxd-containers/BH-GLX_Qwen3.8-Flash-Next_SC1_handoff_worklog_20261023.md` §10（追記専用。revision 2 からの修正は §10.H7〜H13）、生成器は `BH-GLX_Qwen3.8-Flash-Next_free5d_v3_20261024.py`（全式と出典を docstring に持つ）、routing の分子は `..._routing_census_20261023.py` → `..._routing_stats_20261023.py`（`..._routing_stats_full_20261023.json`; tile 配置の最繁忙 chip を加えた再集計、既存の値は不変）。この文書は `render_free5d_v3.py` が同じ入力から決定的に生成する。仮定は「この場合には」で断り、代表仮定が置けない所（routed の先読みの当たり方、hop の設計値）は並列列挙する。

## 1. 一般論: 5 つの軸は何を分け、何を払うか

| 軸 | 分けるもの | 重みの読み | 払うもの（1 層あたり） | agg | tsu |
|---|---|---|---|---|---|
| DP | ユーザ集合を複製モデルに | 複製ごとに全部（DRAM 容量を複製分食う。1 chip の dense 分担が DP 倍になり L1 常駐分が減る） | 無し | 複製数に線形 | 不変 |
| PP | 層（＋lm_head）を PP × R 個の枠に切り（バイト均等か鎖均等）、枠 f を chip 群 f mod PP に置く: R = 1 は普通の PP、R > 1 は 27B README §8〜§12 の「周」（1 token が PP × R 段を通り、各群は R 回番が回る。群は空いている間に次の枠を L1 に読む）; b 人を n_mb 個（1〜min(PP, b)、自由）の micro-batch に分けて段を順に通す | 段の dense 部のうち L1 の残り（177 MiB/chip − ストリームバッファ 24 MiB − 活性 − routed 先読みバッファ）に入る分は常駐（pass を跨いで読み直さない）、入らない分は段が空いている間に先読みバッファへ、残りと routed・状態・KV は占有窓の中で流し読み | 段間受け渡し（残差 [rows, 4H]/TP、段の並びで 1〜3 hop） | 段の周期 = max(段の占有 × n_mb, 段の DRAM/pass) | 1 token = 段の周期（Σ占有 ≤ 周期） |
| TP | 1 層の行列を head/列で TP 枚に（同じ token を全員で）; 残差 [rows, 4H] も列で分ける（配備 `read_tp4` の形） | 互いに素な 1/TP | hyper-connection の統計 all-reduce（行が小さいときは部分和の all-gather + 局所 reduce で 1 本）+ ブロック入力の all-gather + ブロック出力の reduce-scatter = 3 collective × 2 ブロック = 6 本/層（router 列分割なら + 候補の all-gather 1 本） | 1 step が短い | 大きく効くが「最後の tile のレイテンシ × 6 本 × 48 層」が床 |
| EP | routed expert を群に丸ごと（群内は列分割 TPx: expert ごとに等分 slice か、20 tile を chip に回し置く tiles の速い方） | expert は 1 回 | 行を持たない chip に expert の分担があるとき（DP_a > 1、EP に依らず）dispatch + combine の all-to-all; 最繁忙 group が時間を決め他は寝る | 偏り分だけ損 | 列分割の tile padding を避けられる |
| DCP | decode context parallel（vLLM `--decode-context-parallel-size` の定義そのまま）: TP 群の中で同じ kv head を複製している rank の内側で、1 ユーザの**系列**（index ブロックと K/V）を 1/DCP ずつ持ち、局所 top-k → 部分 (o, lse) を all-gather して merge。**world size（chip 数）を増やさない**（`vllm/config/parallel.py`: 「DCP does not expand the process world size. Without PCP, DCP reuses TP ranks」、`vllm/config/model.py`: GQA では max DCP = TP / kv head 数） | 無関係 | merge 1 本（QSA 層のみ） | 長文脈で KV 読みが支配する時に効く | 同左 |

**chip 数 = DP × PP × S**（S = 段の chip 数。R は周数で chip を増やさない: `PP4(R2)` は 4 群 × 2 周 = 8 段を 32 chip で）。S の中の分け方が 3 つ同時に乗る: dense は TP_a × DP_a、expert は EP 群 × TP_x、DCP は kv-head 群（同じ kv head を複製する TP_a/2 枚）の内側 — EP・TP_x・DCP は chip を増やさない。構成のラベルは `DP × PP × S chip[TP_a | EP × TPx | DCP/kv-group] b=…` で、例えば `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]` は 1 × 4 × 8 = 32 chip、各段の 8 枚の中で kv head を共有する 4 枚が系列を 2 つに分ける。chip を増やす方の context parallel（Ring Attention / Ulysses / Megatron の `--context-parallel-size`、vLLM の `prefill_context_parallel_size`）は prefill・学習の軸で、decode の本件では列挙していない（この場合には ctx 192 の KV が 1 行 4.9 MB で、系列分割のために chip を足す利点が無い。長文脈では別途）。
TP と DCP は切る方向が直交する: TP は head（GDN の value head 48、QSA の query head 24）を分け、DCP は 1 つの kv head の系列（時間軸）を分ける。本件の QSA は kv head が 2 本なので TP_a > 2 では TP_a/2 chip が同じ kv head を複製しており、DCP はその複製群の中で系列を分ける（vLLM と同じ: KDA/GDN の recurrent state は DCP で分割せず value head で分ける）。GDN の状態は系列長に依らない（O(1)）ので DCP の対象外。

## 2. 専門論: 限界性能の立て方

1. 分母は物理値（DRAM 512 GB/s/chip 読み書き和、chip 間 link 25 GB/s × 2/方向 × 4 方向、1 hop 280 ns + 蓄積転送 2 KiB/(96 B/cycle) = 16 ns、中継 NoC 92 cycle + 32 flit、Tensix 130 core × 1.35 GHz: 行列積 32 cycle/32×32 tile-MAC（HiFi2; 1 行列積の鎖は出力 tile を 130 core に配り、出力 tile が core より少ないときは K を分けて部分和を NoC の木で足す: 1 段 92 cycle + 64 flit + 32 cycle）、同じ入力を読む行列積は並べて走る（27B §9: 和でなく max）、要素演算 16 cycle/bf16 tile・32 cycle/fp32 tile、gated delta rule は 27B README §9 の 1 組 (value head, user) 4,224 cycle を 32 列 4 単位に分け 1 単位 256 + 800 × 行数 cycle（verify 行は再帰で直列）、router の top-10 は logit tile ごとに 10 pass × 32 cycle → NoC 集約 92 cycle → 10 pass、op 固定費は数えない（27B §9 と同じ）、L1 1,536 KiB/core のうち 140 KiB/core 固定専有、ストリームバッファ 1 core 192 KiB（行列積の tile 2 面 + 部分和 80、gated delta rule の状態単位 2 面 + 出力 48、KV/index チャンク 2 面 64）= 24.4 MiB/chip、活性 4 × [rows, H] bf16 + 残差 [rows, 4H]/TP_a、routed の expert 片 2 枚、残りを常駐 dense と routed 先読みバッファに分ける（バッファ 0 / 1 層分 / 2 層分 / 残り全部の 4 通りで速い方、同じ時間なら pass あたりの DRAM バイトが少ない方: dense を pass ごとに読み直す形は、それが厳密に速いときだけ採られる）、DRAM 31.9 GiB/chip の 90%）と HF 定義（H 2560、48 層 = 36 GDN + 12 QSA、512 expert top-10 × 640、GDN 48 value head × 128 の fp32 状態、QSA 24 query head × 256 / 2 kv head、indexer budget 2048 compress 4、vocab 248,320、hc 4 枝 low-rank 320、MTP 1 層；dense bf8、hc mixer/router/norm bf16、expert BF4 2,764,800 B）だけ。
2. 分子はバイト: dense 5.378 GB（32 分割で 1 chip 160 MiB に対し、L1 177 MiB からストリームバッファと b=1 の活性・expert 片を引いた残りは 152 MiB: **dense は 32 分割でも全部は L1 に置けず**、置けるのは残りから routed 先読みバッファを引いた分（tsu 最大点 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` で 0 MB / 段の dense 209 MB、先読みバッファ 159.4 MB）。routed 69.4 GB・状態・KV は DRAM）、GDN 状態 1 ユーザ 1 pass 読み+書き 0.231 GB、QSA KV 1 行 ctx 192 で 4.9 MB。routed は census の実 distinct（B=1 10、B=16×2 行 183、B=256 426）と最繁忙 group（EP 分割時）で、一様仮定は使わない。
3. 層の時間 = 前段（hc 読み、GDN/QSA、shared/router、top-10 とその collective）+ [router 後に読む分があれば発行レイテンシ 0.16 µs] + max(expert 段の演算 + 通信, バッファに先読みできなかった routed + router 後に読む routed のバイト/BW)、かつ ≥ その層の routed 全バイト/BW。先読みできるのは全ヒット: 全部、census: 予測集合（外れ (1−h) は router 後）、全 expert 読み: 全部、全ミス: 無し、いずれも L1 の先読みバッファまで。routed は micro-batch の hidden に依存するので段の空き時間には読めず、最良でも 1 層前の裏で読む。通信は tile ごとに流し、層の link 時間の合計はその層の演算の半分までに隠れる（隠れない分 = link の合計 − 隠れた分 + 最後の tile のレイテンシの合計）。all-reduce は RS+AG（レイテンシ 2 本）と「部分和の all-gather + 局所 reduce」（1 本、link (n−1)×T）の速い方。段 = 48 層 + tail（final mixer, lm_head, MTP 層）を連続分割（DRAM バイト均等と鎖の時間均等の速い方）。枠の占有 occ_f = max(Σ層の占有 + 受け渡し, 窓: (常駐しない dense − 空き時間に先読みした分 + micro-batch の routed + 状態/KV + 残差の DRAM 往復)/BW)。群の DRAM/pass = Σ(群の枠) [常駐しない dense × n_mb + 状態/KV + routed × n_mb + 残差の DRAM 往復 × n_mb]。**T_pass = max(max_g 群の DRAM/pass, max_g Σ(群の枠) occ_f × n_mb, Σ_f occ_f)**、R ∈ {1, 2, 4}（PP > 1）。T_tok = T_pass、agg = DP × b × R_eff / T_pass、tsu = R_eff / T_tok。R_eff = 1（k=0）/ 1.376（k=1、本件プロンプト集合の採択率）。
4. 容量: 各 chip の expert 分 + 段の dense 分 + ユーザの状態/KV ≤ 0.9 × 31.9 GiB（この場合には runtime・trace・CB に 10% 残す）。
5. MoE の router 依存は 4 ケースを並列列挙（§4）。router の重み [512, 2560] bf16 は「全 chip 複製（collective 無し、dense +126 MB/chip）」と「列分割（候補の all-gather 1 本、tile padding）」の両形を評価し速い方を採る（表の router 列）。
6. 実装拘束（TP ∈ {4,8}、EP = TP、PP = DCP = 1、lanes×(k+1) ≤ 32）は別枠。

## 3. 応用: TT デバイス（BH Galaxy）で効いた物理

- **通信の床**: 最後の tile のレイテンシは 4 chip 0.68 µs、8 chip 1.07、16 chip 1.85、32 chip 2.23（submesh の直径: 1×4 で 2、2×4 で 3、4×4 で 5、8×4 で 6 hop）。層 6 本 × 48 層で TP32 は 0.64 ms/token の床（b=1 の TP32×PP1 の隠れない通信 0.643 ms）。link 時間は b=1 の [2, 2560] bf16 の all-gather で 32 chip 0.05 µs（レイテンシの 2%）。
- **L1 の勘定**: 1 chip 177 MiB（130 core × (1,536 − 140) KiB）− ストリームバッファ 24.4 MiB − 活性（b=1 で 40 KiB + 残差）− expert 片 2 枚 = 152 MiB（b=1、TP8）。dense の 32 分割 160 MiB はこれに入らず、PP の段の dense（1/(PP×TP_a) に router/indexer の複製分）も 209〜360 MB で、常駐できるのは残りから routed 先読みバッファ（全ヒットで 1 層分 = 最繁忙 expert × 2.76 MB × pad/TP_x）を引いた分。入らない分は段が空いている間にバッファへ先読み（PP>1）、残りを窓で流し読む。b=1 の TP32×PP1 は dense/chip 360 MB（router 複製）のうち 157 MB 常駐、残りと routed・状態が DRAM/pass 0.643 ms。
- **expert の列分割の 2 形と tile padding（32 列）**: expert の中間 640 列 = 20 tile。slice 形（expert ごとに TPx 等分）は TPx=16 で 40 列 → 64、TPx=32 で 20 列 → 32 に padding = 1.6 / 1.6 倍のバイトで、寝る chip は無い。tiles 形（expert e の 20 tile を群の TPx chip に e mod TPx から回し置く）は padding 無しで、偏りは census の最繁忙 chip（tile 数、B=1×2 行で TPx16 23.5 tile vs 均等 20.6、TPx32 13.4 vs 10.3）。EP は expert を群に丸ごと置くので群内の padding は小さいが、最繁忙群以外の chip が expert 読みを休む（EP2 で B=1×2 行の最繁忙群 9.8 / 16.5 expert）。両形を評価して速い方（表の expert 列）: b=1 の PP2×TP16 では EP1 tiles 1610 vs EP2 1598（slice だけなら EP1 1559: 「寝かす方が速い」に見えたのは slice の padding）。TP32×PP1 で EP8×TPx4 と EP1×TPx32 を b で比べると: b=1 EP1 1514(slice) vs EP8 1494、b=16 EP1 602(tiles) vs EP8 545、b=256 EP1 116(tiles) vs EP8 119、b=1024 EP1 40(tiles) vs EP8 41（EP8 が勝つのは tiles の偏りが群の偏りを超える大 b のみ）。
- **GDN 状態**: fp32 [48,128,128] × 36 層 = 1 ユーザ 0.115 GB 常駐、毎 pass 読み+書き 0.231 GB。B に比例して DRAM の列に乗り（b=128×2 行で 29.6 GB、dense + routed 62.1 GB の 48%）→ 重みを償却しきった agg の漸近は 32 × 512 GB/s ÷ (0.2309 GB 状態 + 0.0049 GB KV) = 69,480 pass/s（k=1 で × 1.376 = 95,605 t/s）。gated delta rule の演算は b=1 でも 1 単位 1,856 cycle（k=1）が core に 1 つずつ乗り、tsu 最大点の演算 0.376 ms のうち 0.049 ms。
- **PP の周（27B の形）と段の空き時間**: 群は micro-batch が他の群にいる間、次の枠の常駐しきれない dense を routed 先読みバッファの空き（バッファ − 1 層分の routed）へ先読みし、残りを窓で流す（§5.2 の「空き時間の先読み」列）。R 周に分けると枠が小さくなり、常駐に頼らず「次の枠を空き時間に読み切る」形（27B の全 chip の DRAM が常に動く形）が成り立つ: b=1 の PP4×TP8 は R1 1,615（常駐 153 MB）→ R2 1,913（常駐 0、先読み 209 MB/pass、受け渡し 8 回）→ R4 1,786。常駐を削ってこの先読みを増やす形も列挙に残し、時間が同じなら DRAM の読みが少ない常駐を採る（結果の「空き時間の先読み」列が 0 でない点がその例）。段の境は「DRAM バイト均等」と「鎖の時間均等」の速い方、PP2 の 16 chip は 4×4 と 8×2（受け渡し 1 hop）の速い方。
- **routed は router の後**: dense は常駐または段が空いている間に L1 へ先読みできるが、層 l の top-10 は router まで分からず、しかも micro-batch の hidden に依存するので段の空き時間には読めない（最良でも 1 層前の演算の裏、先読みバッファの分まで）。先読みの当たり方で 4 ケース。PP の段を分けると、micro-batch が居る段の chip しか routed を読まないので、b ≥ 8 では PP1（全 32 chip の DRAM が同じ layer の routed を分担）が勝つ。
- **DCP が効く条件**: ctx 192 の KV は 1 user 1 pass 4.9 MB（12 QSA 層）で、b=1 では分けても merge（部分 (o, lse) の all-gather、QSA 層ごと 1 本）と相殺する: tsu 最大の形（DCP1 1,913 / DCP2 1,903 / DCP4 1,890）。rows が増えると KV が段の DRAM 窓に比例して乗るので、b ≥ 4 の Pareto 前線は全点 DCP > 1（§5.6: 11 / 13 点）。ctx が長いほど DCP が小さい b にも効く（本版の列挙は ctx 192 のみ）。
- **1 行タイル 32 行**: B × (k+1) ≤ 32 が今日の lanes 形の上限。限界性能は tile を複数並べる形で b を 2048 まで振る。

## 4. 4 ケース（代表仮定が置けないので並列列挙）

| ケース | 意味 | routed の DRAM バイト | router 後に露出する分 |
|---|---|---|---|
| hit | 全ヒット: routed を router の前に読み終える（投機 routing が全部当たる） | distinct × 2.76 MB | 0 |
| census | census ヒット率: 1 ブロック早い投機 routing の実測ヒット率（外れ分は router 後に読み直す）（h: B=1 0.78、B=16×2 0.85、B=256 0.94） | distinct × 2.76 MB × (2−h) | (1−h) 分 + 発行レイテンシ |
| miss | 全ミス: router の後に読む（今の runtime の形） | distinct × 2.76 MB | 全部 + 発行レイテンシ |
| all | 全 expert 読み: 毎 step 512 expert 全部を読み router に依存しない | 512/EP × 2.76 MB（偏り無し） | 0 |

## 5. 結果（分子は census 全数）

### 5.0 系列長の掃引（k=1; ctx 1 = KV を無視した短文、128、8,192、32,768、262,144 = `max_position_embeddings`（native）、1,000,000 = YaRN（factor 4）の最大長）

192（§5.1 以降）は現行配備の計測条件（ISL 128 + decode）で、限界の設計条件ではない。系列長で変わるのは QSA 12 層の KV（1 行 1 層: 選択 K/V min(ctx, 2048) 行 + index block ctx/4 の走査; 1 user 1 pass で ctx 1: 0、128: 3.2 MB、8k: 57、32k: 75、262k: 252、1M: 818）と常駐 KV（1 user で ctx 262k 6.6 GB、1M 25.3 GB = 1 chip の DRAM 30.8 GB の 82% → DCP が容量のためにも要る）。YaRN は RoPE の周波数を変えるだけで decode のバイト・演算は変えない。GDN 状態は系列長に依らない。

| ctx | tsu 最大（全ヒット） | 構成 | T_tok [ms]; DRAM/pass / 演算 / 通信 | tsu 最大（census / 全ミス） | agg 最大（全ヒット） | 構成 | users | T_pass [ms] | agg 最大（census / 全ミス） | DP1 で容量に入る最大 b |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.719; 0.589 / 0.376 / 0.308 | 1,783 / 1,470 | 84,191 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 4096 | 66.9 | 83,576 / 84,191 | 2,048 |
| 128 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | 0.719; 0.591 / 0.376 / 0.308 | 1,779 / 1,470 | 82,215 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 4096 | 68.6 | 81,628 / 82,215 | 2,048 |
| 8,192 | 1,902 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group] b=1` | 0.723; 0.607 / 0.376 / 0.312 | 1,726 / 1,469 | 58,892 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | 2048 | 47.9 | 58,590 / 58,892 | 2,048 |
| 32,768 | 1,889 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | 0.728; 0.601 / 0.376 / 0.317 | 1,708 / 1,468 | 42,281 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | 512 | 16.7 | 41,665 / 42,281 | 512 |
| 262,144 | 1,784 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | 0.771; 0.630 / 0.377 / 0.317 | 1,582 / 1,460 | 18,037 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | 128 | 9.8 | 17,647 / 18,037 | 128 |
| 1,000,000 | 1,579 | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=1` | 0.872; 0.769 / 0.299 / 0.545 | 1,565 / 1,444 | 6,960 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | 32 | 6.3 | 6,657 / 6,960 | 32 |

b ごとの tsu 最良（全ヒット、k=1）:

| ctx \ b | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| 1 | 1,913 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,518 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 645 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 307 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 168 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 72 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 128 | 1,913 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]` | 1,514 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 643 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 305 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 166 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 70 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 8,192 | 1,902 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP2/kv-group]` | 1,458 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 613 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 283 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 138 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 52 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` |
| 32,768 | 1,889 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 1,440 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 603 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 274 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 130 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group]` | 容量超過 |
| 262,144 | 1,784 `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group]` | 1,336 `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 524 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 215 `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group]` | 容量超過 | 容量超過 |
| 1,000,000 | 1,579 `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 1,140 `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group]` | 372 `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group]` | 容量超過 | 容量超過 | 容量超過 |

DCP は ctx ≥ 8k で b=1 から効く（index block の走査が DCP で分かれる）: b=1 の最良は ctx 1 で DCP1 1,913、262k で DCP4 1,784（DCP1 なら 1,502）、1M で DCP8 1,579（`DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=1`; DCP1 の形は 710）。

### 5.1 一行の答え（ctx 192 = 現行配備の計測条件、実測との比較用）

- **agg tps 最大（この場合には全ヒット、k=1）**: **81,253 t/s** = `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048`（router split、expert slice、4096 users、T_pass 69.4 ms = 段の DRAM/pass（演算 19.5 + 通信 16.8 は窓の中）、L1 常駐 68 / 段の dense 392 MB（活性 80 MiB が L1 を食う）、tsu 19.8）。重みを償却しきった漸近 95,605 の 85%。census 80,680、全ミス 81,253、全 expert 読み 81,255（agg はどのケースも DRAM 支配で差が小さい）。k=0 全ヒット 60,392。
- **tsu 最大（全ヒット、k=1）**: **1,913 t/s/u** = `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`（周 R2（R1 1,615 / R2 1,913 / R4 1,786）、router replicated、expert tiles、T_tok 0.719 ms = Σ段の占有: 演算 0.376 + 通信 0.308 + 受け渡し + 窓; 段の DRAM/pass 0.592 ms、L1 は常駐 dense 0 MB + 先読みバッファ 159.4 MB + 空き時間の先読み 209 MB、段の dense 209 MB）。PP × TP の梯子（§5.2、周は自由）で PP1×TP32 1,514(R1) / PP2×TP16 1,610(R1) / PP4×TP8 1,913(R2) / PP8×TP4 1,040(R2): 2 位は 84%、順位は hop の設計値で入れ替わる（×2 で PP4×TP8、×4 で PP8×TP4）。census 1,778（`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1`）、全ミス 1,470（`DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1`）、全 expert 読み 306（`DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1`、1 users）。k=0 全ヒット 1,528。
- **5 軸の答え**: agg 側は **`DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048`**。tsu 側は **`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1`**。中間の users 数ごとの最良は §5.6 の Pareto 前線（PP>1 は 3 点、EP>1 は 5 点、DCP>1 は 11 点、全 13 点）。
- **今日の実装拘束内（TP∈{4,8}, EP=TP, PP=DCP=1, lanes×(k+1) ≤ 32）**: 全ヒット agg 9,824（`DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16`）、tsu 604（`DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1`）；全ミス agg 9,824、tsu 604。現行形 `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` は全ミスで 9,824（T_pass 17.93 ms: 段 DRAM/pass 17.93、演算 2.76 + 通信 0.27 + routed 露出 13.59）、全ヒットで 9,824；**実測 1,186.9 t/s は全ミス机上値の 12.1%**。

### 5.2 PP × TP の梯子（b=1、全ヒット、k=1）と hop 設計値への感度

| PP × TP_a（b=1 の最良、EP/DCP/周 自由） | 構成 | router / expert / 段の切り方 / 段の形 | tsu | T_tok [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | L1: 常駐 dense / 先読みバッファ / 空き時間の先読み [MB]（段の dense） | hop ×1 / ×2 / ×4 |
|---|---|---|---|---|---|---|---|---|---|
| PP1 × TP32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / slice / bytes / 8x4 | 1514 | 0.909 | 0.643 | 0.242 | 0.643 | 157 / 2.3 / 0 (360) | 1514 / 980 / 574 |
| PP2 × TP16 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles / bytes / 8x2 | 1610 | 0.855 | 0.371 | 0.297 | 0.532 | 156 / 3.3 / 0 (261) | 1610 / 1085 / 657 |
| PP4 × TP8 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles / bytes / 2x4 | 1913 | 0.719 | 0.592 | 0.376 | 0.308 | 0 / 159.4 / 209 (209) | 1913 / 1416 / 938 |
| PP8 × TP4 | `DP1 x PP8(R2) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | split / slice / chain / 1x4 | 1040 | 1.323 | 0.696 | 0.528 | 0.233 | 0 / 158.8 / 236 (294) | 1040 / 1039 / 1005 |
| PP16 × TP2 | `DP1 x PP16(R2) x 2chip[TP2 | EP1 x TPx2] b=1` | split / slice / chain / 2x1 | 464 | 2.963 | 1.107 | 0.853 | 0.099 | 0 / 157.5 / 219 (492) | 464 / 464 / 462 |
| PP32 × TP1 | `DP1 x PP32 x 1chip[TP1 | EP1 x TPx1] b=1` | split / slice / chain / 1x1 | 175 | 7.849 | 1.396 | 1.571 | 0.000 | 155 / 0.0 / 0 (792) | 175 / 175 / 175 |

hop 280 ns を 1 / 2 / 4 倍にしたときの最良は PP4×TP8 / PP4×TP8 / PP8×TP4。「`DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` が tsu 最大」は hop が設計値（280 ns、Table 16）に近い場合に限る。

### 5.3 各 TP の tsu 最良（b=1、全ヒット、k=1）

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 383 | 3,066 (8) | `DP1 x PP8(R2) x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=1` | split / slice | 1040 | 1,040 (1) | 1.323; 0.696 / 0.528 / 0.233 |
| 8 | 796 | 3,185 (4) | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles | 1913 | 1,913 (1) | 0.719; 0.592 / 0.376 / 0.308 |
| 16 | 1425 | 2,849 (2) | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` | replicated / tiles | 1610 | 1,610 (1) | 0.855; 0.371 / 0.297 / 0.532 |
| 32 | 1514 | 1,514 (1) | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / slice | 1514 | 1,514 (1) | 0.909; 0.643 / 0.242 / 0.643 |

### 5.4 各 TP の tsu 最良（b=16、全ヒット、k=1; DP が違うと users が違うので agg は同列に比べない）

| TP_a | PP1, EP1×TPx=TP（列等分）, DCP1 tsu | agg (users) | tsu 最良の構成（PP/EP/DCP/n_mb 自由） | router / expert | tsu | agg (users) | T_tok [ms]; 段 DRAM/pass / 演算 / 通信 |
|---|---|---|---|---|---|---|---|
| 4 | 84 | 10,700 (128) | `DP1 x PP8 x 4chip[TP4 | EP1 x TPx4 | DCP2/kv-group] b=16 mb=8` | replicated / slice | 454 | 7,265 (16) | 3.030; 3.030 / 0.679 / 0.200 |
| 8 | 166 | 10,612 (64) | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=16 mb=4` | split / tiles | 508 | 8,121 (16) | 2.711; 2.711 / 0.620 / 0.368 |
| 16 | 328 | 10,485 (32) | `DP1 x PP2(R2) x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=16 mb=2` | split / tiles | 579 | 9,259 (16) | 2.378; 2.378 / 0.557 / 0.722 |
| 32 | 602 | 9,630 (16) | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 642 | 10,270 (16) | 2.144; 2.144 / 0.483 / 1.053 |

### 5.5 ケース × k の最大

| ケース | k | agg 最大 | その tsu | users | 構成 | tsu 最大 | その agg | 構成 |
|---|---|---|---|---|---|---|---|---|
| hit | 0 | 60,392 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1528 | 1,528 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| hit | 1 | 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1913 | 1,913 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 0 | 59,965 | 14.6 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1510 | 1,510 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` |
| census | 1 | 80,680 | 19.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1778 | 1,778 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` |
| miss | 0 | 60,392 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1159 | 1,159 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=1` |
| miss | 1 | 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | 1470 | 1,470 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` |
| all | 0 | 60,393 | 14.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=2048` | 227 | 227 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |
| all | 1 | 81,255 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP1 x TPx16 | DCP8/kv-group] b=2048` | 306 | 306 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=1` |

### 5.6 Pareto 前線（全ヒット、k=1）

前線に乗る構成のうち PP>1 は 3 点、EP>1 は 5 点、DCP>1 は 11 点（全 13 点）。

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.364 | 69.364 | 19.537 | 16.826 | 0.000 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 34.912 | 34.912 | 9.115 | 20.639 | 0.000 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 19.803 | 19.803 | 4.602 | 10.809 | 0.000 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.248 | 12.248 | 2.669 | 5.733 | 0.000 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.352 | 8.352 | 1.698 | 3.198 | 0.000 | 151 / 0.0 (392) | 62.2 | fitted |
| 29,808 | 232.9 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 5.909 | 5.909 | 1.152 | 2.211 | 0.000 | 155 / 0.0 (392) | 13.5 | census |
| 19,468 | 304.2 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | split / tiles | 4.524 | 4.524 | 0.861 | 1.476 | 0.000 | 125 / 32.0 (392) | 11.6 | census |
| 13,772 | 430.4 | 32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | split / tiles | 3.197 | 3.197 | 0.654 | 1.201 | 0.000 | 133 / 24.7 (242) | 8.9 | census |
| 10,270 | 641.9 | 16 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 2.144 | 2.144 | 0.483 | 1.053 | 0.000 | 141 / 17.1 (242) | 6.2 | census |
| 7,936 | 992.0 | 8 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP2/kv-group] b=8` | split / tiles | 1.387 | 1.378 | 0.383 | 0.970 | 0.000 | 148 / 10.5 (242) | 3.8 | census |
| 6,050 | 1512.4 | 4 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=4 mb=4` | split / tiles | 0.910 | 0.910 | 0.391 | 0.368 | 0.000 | 153 / 6.1 (199) | 2.2 | census |
| 3,403 | 1701.5 | 2 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=2 mb=2` | split / tiles | 0.809 | 0.483 | 0.391 | 0.360 | 0.000 | 147 / 12.1 (186) | 2.2 | census |
| 1,913 | 1913.2 | 1 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8] b=1` | replicated / tiles | 0.719 | 0.592 | 0.376 | 0.308 | 0.000 | 0 / 159.4 (209) | 2.2 | census |

### 5.7 Pareto 前線（census ヒット率、k=1）

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80,680 | 19.7 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.857 | 69.857 | 19.537 | 16.826 | 0.501 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,153 | 39.1 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 35.159 | 35.159 | 9.115 | 20.639 | 0.254 | 112 / 0.0 (392) | 64.0 | fitted |
| 70,278 | 68.6 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 20.049 | 20.049 | 4.602 | 10.809 | 0.254 | 135 / 0.0 (392) | 64.0 | fitted |
| 56,386 | 110.1 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.494 | 12.494 | 2.669 | 5.733 | 0.254 | 146 / 0.0 (392) | 64.0 | fitted |
| 40,998 | 160.1 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.592 | 8.592 | 1.698 | 3.198 | 0.247 | 151 / 0.0 (392) | 62.2 | fitted |
| 28,757 | 224.7 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 6.125 | 6.125 | 1.152 | 2.211 | 0.224 | 155 / 0.0 (392) | 13.5 | census |
| 18,357 | 286.8 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | split / tiles | 4.797 | 4.797 | 0.861 | 1.476 | 0.282 | 125 / 32.0 (392) | 11.6 | census |
| 12,634 | 394.8 | 32 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=32` | split / tiles | 3.485 | 3.485 | 0.654 | 1.201 | 0.296 | 133 / 24.7 (242) | 8.9 | census |
| 9,186 | 574.2 | 16 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=16` | split / tiles | 2.397 | 2.397 | 0.483 | 1.053 | 0.261 | 141 / 17.1 (242) | 6.2 | census |
| 7,217 | 902.1 | 8 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32 | DCP16/kv-group] b=8` | split / tiles | 1.525 | 1.525 | 0.383 | 0.988 | 0.188 | 148 / 10.5 (242) | 3.8 | census |
| 5,377 | 1344.2 | 4 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=4 mb=2` | split / tiles | 1.024 | 0.988 | 0.353 | 0.620 | 0.240 | 154 / 6.0 (213) | 2.2 | census |
| 3,154 | 1577.2 | 2 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=2 mb=2` | replicated / tiles | 0.872 | 0.850 | 0.297 | 0.532 | 0.140 | 156 / 3.3 (269) | 1.2 | census |
| 1,778 | 1777.6 | 1 | `DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 | DCP4/kv-group] b=1` | replicated / tiles | 0.774 | 0.624 | 0.376 | 0.317 | 0.128 | 0 / 159.4 (218) | 2.2 | census |

### 5.8 Pareto 前線（全ミス = 今の形、k=1）

| agg [t/s] | tsu | users | 構成 | router / expert | T_pass [ms] | 段 DRAM/pass | 演算 | 隠れない通信 | router 後の routed 読み | L1: 常駐 / バッファ [MB]（段の dense） | 最繁忙 expert/層（tiles は tile/20） | routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 81,253 | 19.8 | 4096 | `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 69.364 | 69.364 | 19.537 | 16.826 | 8.475 | 68 / 0.0 (392) | 128.0 | fitted |
| 80,719 | 39.4 | 2048 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=2048` | split / slice | 34.912 | 34.912 | 9.115 | 20.639 | 4.241 | 112 / 0.0 (392) | 64.0 | fitted |
| 71,153 | 69.5 | 1024 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=1024` | split / slice | 19.803 | 19.803 | 4.602 | 10.809 | 4.241 | 135 / 0.0 (392) | 64.0 | fitted |
| 57,521 | 112.3 | 512 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=512` | split / slice | 12.248 | 12.248 | 2.669 | 5.733 | 4.241 | 146 / 0.0 (392) | 64.0 | fitted |
| 42,174 | 164.7 | 256 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP8 x TPx4 | DCP8/kv-group] b=256` | split / slice | 8.352 | 8.352 | 1.698 | 3.198 | 4.123 | 151 / 0.0 (392) | 62.2 | fitted |
| 29,808 | 232.9 | 128 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=128` | split / tiles | 5.909 | 5.909 | 1.152 | 2.211 | 3.570 | 155 / 0.0 (392) | 13.5 | census |
| 18,771 | 293.3 | 64 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP8/kv-group] b=64` | replicated / tiles | 4.692 | 4.692 | 0.851 | 1.374 | 3.067 | 157 / 0.0 (510) | 11.6 | census |
| 12,385 | 387.0 | 32 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP4/kv-group] b=32` | replicated / tiles | 3.555 | 3.541 | 0.659 | 1.067 | 2.370 | 157 / 0.0 (510) | 8.9 | census |
| 8,422 | 526.4 | 16 | `DP1 x PP1 x 32chip[TP16 x DPa2 | EP1 x TPx32 | DCP2/kv-group] b=16` | replicated / tiles | 2.614 | 2.589 | 0.494 | 0.906 | 1.644 | 158 / 0.0 (510) | 6.2 | census |
| 6,269 | 783.6 | 8 | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=8 mb=4` | replicated / tiles | 1.756 | 1.634 | 0.452 | 0.308 | 4.431 | 159 / 0.0 (234) | 4.2 | census |
| 4,571 | 1142.7 | 4 | `DP1 x PP4 x 8chip[TP8 | EP1 x TPx8] b=4 mb=4` | replicated / tiles | 1.204 | 1.133 | 0.376 | 0.308 | 2.353 | 159 / 0.0 (234) | 2.2 | census |
| 2,672 | 1335.9 | 2 | `DP1 x PP2 x 16chip[TP16 | EP1 x TPx16] b=2 mb=2` | replicated / tiles | 1.030 | 0.777 | 0.297 | 0.532 | 0.639 | 160 / 0.0 (269) | 1.2 | census |
| 1,470 | 1469.6 | 1 | `DP1 x PP1 x 32chip[TP32 | EP1 x TPx32] b=1` | replicated / tiles | 0.936 | 0.598 | 0.244 | 0.643 | 0.185 | 159 / 0.0 (360) | 0.7 | census |

### 5.9 今日の実装拘束内の最大

| ケース | k | agg 最大 | 構成 | tsu 最大 | 構成 |
|---|---|---|---|---|---|
| hit | 0 | 2,263 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 525 (2,101 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| hit | 1 | 9,824 (77 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 604 (2,417 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 0 | 2,117 (265 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 481 (1,925 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| census | 1 | 8,795 (69 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 548 (2,194 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 0 | 2,263 (283 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=1` | 525 (2,101 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| miss | 1 | 9,824 (77 tsu) | `DP8 x PP1 x 4chip[TP4 | EP4 x TPx1] b=16` | 604 (2,417 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 0 | 226 (56 tsu) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` | 56 (226 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |
| all | 1 | 4,615 (72 tsu) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=16` | 76 (304 agg) | `DP4 x PP1 x 8chip[TP8 | EP8 x TPx1] b=1` |

## 6. 未確定（決着方法）

- hop 280 ns は EthernetSsDesignSpec Table 16（AI clk 1.5 GHz）の設計値、本機 1.35 GHz では最大 311 ns、ケーブル伝搬は仕様値無し: §5.2 の感度の範囲。
- rows > 256 の routing は census 外で fitted（distinct の減衰則と偏り則）。b ≥ 256 の agg（agg 最大点 `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` を含む）はこの外挿に依存。
- k=1 の R_eff 1.376 は本件プロンプト集合（ISL128）の採択率。
- 投機 routing のヒット率は CPU oracle（bf16 expert）の greedy 連鎖。device の BF4 連鎖では未測（queue 済）。
- gated delta rule の 1 単位 256 + 800 × 行 cycle は 27B §9 の 4,224 cycle/(head, user) を 4 単位に分けた机上値（単位間の k'S の交換は NoC 1 回で 1 単位の 1/10 未満、数えていない）。
- b=1 の演算は「行列積の鎖」（K 分割の部分和の木）で決まり、27B §9 が「1 µs 未満」として省いた core 間の受け渡しを本件では数えている（tsu 最大点の演算 0.376 ms のうち行列積の鎖が大半）。K 分割の木の段数・NoC の値は設計値。
- L1 のストリームバッファ 1 core 192 KiB（24.4 MiB/chip）と固定専有 140 KiB/core は設計値（27B §3/§7 の枠の入出力・作業域 3〜10 MiB に準ずる、多い側）。バッファを 0 / 2 倍にしたときの tsu 最大点: 1,913 / 1,913（1 倍 1,913）。
- 活性の L1 占有は micro-batch の 4 × [rows, H] bf16 + 残差 [rows, 4H]/TP_a を設計値として置いた（27B §7 の枠の入出力 3〜10 MiB に準ずる）。agg 最大点 `DP2 x PP1 x 16chip[TP16 | EP4 x TPx4 | DCP8/kv-group] b=2048` では 80 MiB で常駐 dense を 68 MB に削る。
- 空いている段の chip が隣の段の routed を自分の DRAM から読んで ETH で渡す形（1 chip の入口 8 link × 25 GB/s = 200 GB/s、DRAM 512 GB/s の 39%）は列挙していない。
- DCP の局所 top-k（各 chip が budget/DCP を自分の範囲から選ぶ）は全体 top-k と選択が異なりうる（数値の差、時間には影響しない）。
- 系列長の掃引（§5.0）は k=1 の 4 ケースのみ（k=0 は ctx 192 だけ）。長文脈の prefill（chunked prefill の時間、KV の書き込み）は decode の限界の外。index block 走査は 1 行 1 QSA 層で 262k 24 µs、1M 93 µs（DCP1、1 chip）の演算で、DCP で分けた分だけ縮む。1M は YaRN（factor 4）前提で、精度は限界の外。
- 枠の chip 群への割付は f mod PP の固定 round-robin（27B は chip ごとの総量を揃えて切る）: 群間の量の差が残る近似。
