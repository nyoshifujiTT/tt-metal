#!/usr/bin/env python
r"""D-1 (plan v2, rule R14): the audit table of every constant of the design and limit generators (tools/design,
tools/limit).  One row per constant: name, value, class, source, status.  render_audit() writes the README table
(section 11.2.3); check() fails when a module-level constant of the generators has no row.

Status:
  spec      the value is printed in a primary source (ISA documentation, design specification, data sheet, HF config)
  derived   computed from spec rows by the formula shown
  hw        read from this machine (tt-smi telemetry decoded per tt-umd, the I-0 link probe): tools/design/hw/
  recount   instruction count or hop count of OUR design; recounted from the README section 12 text at D-2..D-4
  choice    a design / model choice (enumerated or justified in section 11.3 / 12), not a physical value
  stat      a statistic of the model on the prompt set (census, MTP acceptance)
  build     known only from a build product (kernel size: rule R2's exception, pass/fail at K-1)
  load      set by firmware under load (AI clock: rule R2's exception, pass/fail at K-1)
  nosource  searched (record in the worklog) and not found; the design must not depend on it (plan v2 D-1 go-back)
  name      a label or enumeration, not a quantity
"""
TT = "tt-isa-documentation be83310 BlackholeA0/"
HF = "HF config.json text_config"
ROWS = [
    # ---- model (HF) ----
    ("H", 2560, "model", HF + " hidden_size", "spec"),
    ("L, L_GDN, L_QSA, LAYER", "48, 36, 12", "model", HF + " num_hidden_layers, layer_types (full_attention_interval 4)", "spec"),
    ("GDN_QK, GDN_V, GDN_HD", "16, 48, 128", "model", HF + " linear_num_key_heads, linear_num_value_heads, linear_key/value_head_dim", "spec"),
    ("CONV_K", 4, "model", HF + " linear_conv_kernel_dim", "spec"),
    ("QSA_Q, QSA_KV, QSA_HD", "24, 2, 256", "model", HF + " num_attention_heads, num_key_value_heads, head_dim", "spec"),
    ("ROPE", 64, "model", HF + " head_dim 256 x partial_rotary_factor 0.25", "derived"),
    ("IDX_Q, IDX_KV, IDX_HD, IDX_BUDGET, IDX_COMP", "4, 1, 128, 2048, 4", "model", HF + " indexer_n_heads, indexer_kv_heads, indexer_head_dim, indexer_budget, indexer_compress_ratio", "spec"),
    ("E, TOPK (K)", "512, 10", "model", HF + " num_experts, num_experts_per_tok", "spec"),
    ("E_INT, SH_INT", "640, 640", "model", HF + " moe_intermediate_size, shared_expert_intermediate_size", "spec"),
    ("V", 248320, "model", HF + " vocab_size", "spec"),
    ("HC, HC_LOW, HCD", "4, 320, 10240", "model", HF + " hc_count, hc_lowrank; HCD = HC x H", "spec"),
    ("PLE_LAYER, PLE_NGRAM_PARTS, PLE_DIM", "2, 128, 2560", "model", HF + " ple_layer_ids [2], split_ngram_parts, ple_embed_dim", "spec"),
    ("PLE_HEADS, PLE_HEAD_DIM", "16, 160", "model", HF + " (ngram_size 3 - 1) x heads_per_ngram 8 = 16; ple_embed_dim / 16 = 160", "derived"),
    ("PLE_CONV_K, PLE_DIL", "4, 3", "model", HF + " ple_conv_kernel_size; dilation = ngram_size (tt/ple.py:163, the HF n-gram conv)", "spec"),
    ("MTP layers", 1, "model", HF + " mtp_num_hidden_layers", "spec"),
    # ---- formats ----
    ("BF8 (TB bf8)", "1088 B/tile", "format", "tt-metal blockfloat_common.cpp: 1,024 datum bytes + 64 shared-exponent bytes (16 values share one)", "spec"),
    ("BF4 (TB bf4)", "576 B/tile", "format", "same file: 512 datum bytes + 64 exponent bytes", "spec"),
    ("BF16, FP32", "2, 4 B", "format", "IEEE / bfloat16", "spec"),
    ("TILE", 32, "format", "32 x 32 tile of 4 16 x 16 faces (" + TT + "TensixTile/TensixCoprocessor/MVMUL.md)", "spec"),
    ("N_TILES, EXPERT_TILE_COLS", 20, "format", "E_INT 640 / 32", "derived"),
    ("EXPERT_TILE, GU, DN", "138240 B, 2/3, 1/3", "format", "expert (2 x 640 x 2560 + 2560 x 640) x 0.5625 B / 20 columns; gate/up 2 of 3 matrices", "derived"),
    # ---- chip resources ----
    ("CORES", 130, "resource", "this machine: every chip 13 of 14 Tensix columns enabled (tools/design/hw/hw_facts_a03u07.json)", "hw"),
    ("XS, YS", "x 1-7,10-16 minus the chip's harvested column; y 2-11", "resource", TT + "NoC/Coordinates.md + hw_facts (harvested x per chip)", "hw"),
    ("L1 per core", "1536 KiB", "resource", TT + "TensixTile/L1.md; MEM_L1_SIZE", "spec"),
    ("firmware (in L1_FIXED)", "40 KiB", "resource", "MEM_MAP_END = 40,960 B (tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h, compiled /tmp/q38/mm.c)", "spec"),
    ("kernel (in L1_FIXED)", "100 KiB", "resource", "kernel binaries + config: a build product (rule R2 exception); default config 69 KiB (bh_hal_tensix.cpp)", "build"),
    ("DRAM_CHIP", "8 x 4,278,190,080 B x 0.90", "resource", "blackhole_140_arch.yaml dram_bank_size x 8 banks (= 31.875 GiB; the code's 31.9 is rounded, correction 58); 0.90 = reserve for runtime / trace (choice)", "spec+choice"),
    ("BW_DRAM, BW (DRAM)", "512 GB/s", "resource", "8 GDDR6 instances x 2 x16 channels x 16 Gbps (tt-smi ENABLED_GDDR 0xff, DDR_SPEED 16,000, hw_facts); read + write share it (JESD250)", "derived"),
    ("BANK", "47.4 B/cycle", "resource", "512 GB/s / 8 banks / 1.35 GHz", "derived"),
    ("BANK_EPS", "3 DRAM tiles per bank", "resource", "blackhole_140_arch.yaml dram (x 0 and 9)", "spec"),
    ("G", "4096 B", "choice", "DRAM request granule >= 2 KiB contiguous (27B README section 3 GDDR rule (1): one ACT per 2 KiB, tRC/tFAW not binding; K4ZAF325BC Table 98)", "choice"),
    ("CLK", "1.35 GHz", "clock", "tt-smi AICLK fmax 1350 MHz (hw_facts); firmware throttles above TDP 130 W (seen 1306-1312 MHz under other users' load)", "load"),
    # ---- NoC ----
    ("NIU_OUT, NIU_IN, HOP (NoC), FLIT, NOC, NOC_FLIT, LINK", "5, 5, 9 cycles, 64 B", "noc", TT + "NoC/README.md:62-66 (NIU->router ~5, router->router 9, router->NIU ~5, 64 B a cycle)", "spec"),
    ("NX, NY", "17, 12", "noc", "blackhole_140_arch.yaml grid; " + TT + "NoC/Coordinates.md", "spec"),
    ("N_RD", "8 cycles (code still 16)", "noc", "SharePoint SOC 'bh noc deepdive' slide 21: write requests fetch flits from L1, 8 cycles of round trip", "spec"),
    ("ETH (channel coordinates)", "per chip", "noc", "hw_facts: logical ch0-7 -> NoC0 x [1,16,2,15,3,4,13,12] (x 14 and 5 harvested; chip 18: ch6 x 5). gen_core_map.py's list (x 14 as ch5) is wrong for this machine: correction 59", "hw"),
    ("UP_OWN, UP_HOPS, MC_HOPS, DRAM_HOPS", "7/11/14, 14, 17, 14", "noc", "worst over 14 hypothetical harvests (rule R13 violation); recount on this machine's placement at D-3", "recount"),
    ("NOC_ETH_CORE, NOC_HUB_ETH, NOC_ETH_ALL, NOC_ETH_ETH, NEAR_ETH_MIN", "6, 2, 22, 3, 91", "noc", "placement placeholders; recount on this machine's placement and Ethernet tiles at D-3/D-4", "recount"),
    ("NOC_RELAY (limit/mm_cost)", 92, "noc", "revision-5 relay value used by mm_cost.tree_level (step6 TREE); replace by stages.py's stage sums at D-2", "recount"),
    # ---- Tensix ----
    ("MVMUL_LAT", 5, "tensix", TT + "TensixTile/TensixCoprocessor/MatrixUnit.md:21-25; Dst.md:112 (5 cycles before a written 8x16 block is readable)", "spec"),
    ("PHASES", "bf8 2, bf4 1, bf16 4", "tensix", "fidelity phases of the deployed formats (HiFi2, LoFi, HiFi4: MVMUL.md, SrcASrcB.md; ttnn/decode_matmul.py dense plan)", "spec+choice"),
    ("WEIGHT_UNPACK, UNP_BYTES, UNPACK_BYTES, ELT", "8 / 8 / 16 cycles a tile, 128 B/cycle, 16", "tensix", TT + "UNPACR_Regular.md (BH throttle: 8-bit x8 = 128 B, 16-bit x4 '2x' = 128 B, <= 8 rows a cycle)", "spec"),
    ("UNP_ADDR", 2, "tensix", "WormholeB0 UNPACR_Regular.md:735 (2 cycles of address calculation; BH page refers to WH: 'similar, not identical')", "spec"),
    ("SFP_LAT", 2, "tensix", TT + "TensixCoprocessor/VectorUnit.md table; SFPMAD.md:70 (auto stall 1 cycle)", "spec"),
    ("PACK_ROWS", 4, "tensix", TT + "TensixCoprocessor/PACR.md (4 Dst rows of 16 datums a cycle)", "spec"),
    ("SEM, STALL_LAG", "1, 1", "tensix", TT + "TensixCoprocessor/SyncUnit.md, STALLWAIT.md:27", "spec"),
    ("RISC_L1_LOAD, RISC_NOC_REG, RISC_FENCE", "8, 7, 1", "tensix", TT + "TensixTile/BabyRISCV/README.md load latency table (L1 miss >= 8, NoC regs >= 7); MemoryOrdering.md:57-59", "spec"),
    ("ACT silu, sigmoid, exp, rsqrt", "54, 52, 40, 30 cycles a group", "tensix", "repo llk_sfpu compiled with sfpi 7.74.0 -O3 -mcpu=tt-bh-tensix, counted by tools/design/sfpu/sfpu_count.py (worklog 10-28)", "derived"),
    ("ACT softplus", "52 (code 101)", "tensix", "same count, repo's current softplus (correction: 101 was the 27B copy's older version)", "derived"),
    ("ACT recip", "9 + 5 n for n groups (code 6 a group)", "tensix", "SFPLOADMACRO pipeline of _calculate_reciprocal_fast_24b_5c_ (correction 57)", "derived"),
    ("ACT mul, add", "4, 4", "tensix", "SFPLOAD + op + nop (latency 2) + SFPSTORE (VectorUnit.md)", "derived"),
    ("P_PACK", "17 bf16 / 18 fp32 (code 16)", "tensix", "tt-metal #59810 / #58704 (LLK team, BH pack paths read against the RTL): packer drain 16-17 bf16, 18 fp32", "spec"),
    ("P_UNP", "no BH value (code 17 = NEO)", "tensix", "searched: tt-isa, Glean (TA Unpackers Performance = NEO RTL, #59809, #58713, TEN-4865), Glean chat: no BH value. Design must not depend on it (D-2)", "nosource"),
    # ---- RISC sequences (our design) ----
    ("ETH_POLL, NOC_ISSUE, EXPERT_LOOKUP", "22, 10, 10", "risc", "poll = 2 x (fence 1 + L1 load 8 + compare/branch 2); issue = 3 stores + cmd_buf-ready load 7; lookup = L1 load + add", "derived"),
    ("DRISC_PAGE", 110, "risc", "27B core_plan.h kDriscPageCycles (instruction count x ISA latencies); recount from our DRISC sequence at D-2", "recount"),
    ("ETH_SEND", 31, "risc", "27B core_plan.h kEthDm0PacketCycles (4 TXQ registers + reads + cmd_ongoing, EthernetTxRx.md); recount from our ERISC sequence at D-4", "recount"),
    ("GDDR_PIPE", "405 cycles (0.3 us)", "dram", "controller/PHY cycles not in Rambus GDDR6 Controller UG 1.03, BH GDDR Architecture, SOC MAS 4.6 (Glean chat 10-28); only O2O Tdram ~392 (empirical). Keep off the chain (D-2/D-3)", "nosource"),
    # ---- chip to chip ----
    ("BW (link), BW_LINK", "25 GB/s", "link", "BH Galaxy runs 200G (Confluence syseng 'Overview of Blackhole ASIC -> Blackhole Galaxy'); one channel one way", "spec"),
    ("CH, LINKS", 2, "link", "this machine's I-0 link map: 2 channels per neighbour per direction (tools/design/hw/probe_links_a03u07.txt)", "hw"),
    ("HOP (chip)", "306 ns + medium", "link", "EthernetSsDesignSpec 2.13 Table 16 280 ns (400G, L1->remote L1, excludes wire/backplane) + RS-FEC 25.6 ns at 200G (27B README 3); medium per link class at D-4 (AEC DSP latency unpublished)", "spec+recount"),
    ("SAF, SAF_RATE", "96 B/cycle", "link", "EthernetSsDesignSpec 2.8.4 (TX reads L1 by 6 x 128-bit ports); store-and-forward 2.8", "spec"),
    ("HDR", "56 B", "link", "preamble/SFD 8 + Ethernet 14 + TT-link 18 (" + TT + "EthernetTile/EthernetTxRx.md:21) + FCS 4 + IPG 12", "spec"),
    ("PKT", "2048 B", "choice", "packet payload <= 4,096 B (EthernetSsDesignSpec 1.2.10); 2 KiB chosen", "choice"),
    ("TORUS_R, TORUS_C", "8, 4", "link", "this machine's I-0 map: 8 rings of 4 + vertical rings of 8 (hw_facts torus groups)", "hw"),
    # ---- model choices ----
    ("ROWS, ROWS_AR", 2, "choice", "B = 1, MTP k = 1 (token + verify row)", "choice"),
    ("TP, TPX, PP, KVG, GDR_UNITS", "8, 8, 4, 4, 96", "choice", "the chosen form DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8]; KVG = TP8 / 2 kv heads; 6 value heads x 16 units of 8 columns", "choice"),
    ("WAVES", 10, "choice", "partial sums streamed in 10 waves (section 11.3 rule 17)", "choice"),
    ("PIECE", "512 KiB", "choice", "DRAM read piece of step7 (consumption-order issue)", "choice"),
    ("ROUTED_LOOKAHEAD, ROUTED_RESERVE_LAYERS", 2, "choice", "routed reserve of 2 layers in L1 (fastest of 1-5, section 12.6)", "choice"),
    ("L1_MM, L1_GDR, L1_KVC, STREAM_MULT", "80, 48, 64 KiB, x1", "choice", "section 11 stream regions (only when the configuration needs them, correction 33)", "choice"),
    ("PKT_CAP, SUBMESH, CASES_K, FORMS, HC_FORM", "-", "choice", "enumeration / numerical method of the generators", "choice"),
    # ---- statistics ----
    ("T_BUSIEST, X_DISTINCT, h12, h2", "43.9 tiles, 16.5, 0.83, 0.66 (rows 1x2)", "stat", "census of the prompt set (34 items x 256 positions, CPU oracle, routing/stats.json)", "stat"),
    ("R_EFF, TOKENS_PER_PASS", 1.376, "stat", "MTP k=1 tokens per pass on the prompt set (ISL 128)", "stat"),
    # ---- names ----
    ("GB, MiB, KiB, KIND, FMT, VALNAME, CHIPS, TILE_PAIRS, STAGES, FIRST, PIECES, FANIN, KMAX_POOL, FRAMES_SINK, _MESH, _MEET, _MMS",
     "-", "name", "labels, caches and enumerations", "name"),
]


def render_audit():
    out = ["| 定数 | 値 | 種類 | 出典 | 状態 |", "|---|---|---|---|---|"]
    for name, val, cls, src, st in ROWS:
        out.append(f"| {name} | {val} | {cls} | {src} | {st} |")
    n = {}
    for r in ROWS:
        n[r[4]] = n.get(r[4], 0) + 1
    out.append("")
    out.append("状態の内訳: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))
    return "\n".join(out)


if __name__ == "__main__":
    print(render_audit())
