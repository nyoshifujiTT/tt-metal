#!/usr/bin/env python
r"""D-2 (worklog 10-27, integrated plan): the stage-level design of the units of one chip (ctx 1, B=1, k=1, rows 2), from
the specifications only (rule R2).  Every constant below carries its source; constants whose value no document gives are
BUDGETS (rule R2's exception, worklog 10-27 12:3x: pipeline depths of the packer and of the NIU's write-request L1 read),
named P_* / N_*, set as upper bounds and checked pass/fail on the device later (rule R5), never fitted to measurements.

Units of time: AI clock cycles at 1.35 GHz (BabyRISCV/README.md:3).

Sources (tt-isa-documentation unless noted):
  MVMUL: 1/cycle, latency 5 (MatrixUnit.md:21-25); 8 rows of SrcB x 16x16 of SrcA (MVMUL.md); one per fidelity phase.
  Dst: a written aligned 8x16 block is unreadable for 5 cycles (Dst.md:88).
  UNPACR: 2 cycles of address calculation (UNPACR_Regular.md:680), then L1-fetch-bound: BH 8-bit x8 = 128 B/cycle,
    16/32-bit x4 "2x" = 128 B/cycle (UNPACR_Regular.md:349-360), at most 8 rows a cycle into SrcA.
  SFPU: SFPLOAD/SFPSTORE 1/cycle, SFPADD/SFPMAD latency 2 (VectorUnit.md), one 4-row x 8-column group of 32 lanes.
  PACR: up to 4 Dst rows (16 datums each) a cycle (BH PACR.md), L1 write 8 x 16 B a cycle (TA "TDMA non-byte aligned").
  Sync: SEMPOST/SEMGET/STALLWAIT 1 cycle, wait release lag 1 cycle (SyncUnit.md, STALLWAIT.md:27).
  RISC: 1 instr/cycle; load latency: L1 >= 8 (L0 miss), NoC regs >= 7, mailbox/semaphore >= 3, local RAM 2
    (BabyRISCV/README.md:73-83); a poll of L1 written by another agent needs a fence (MemoryOrdering.md:57-59).
  NoC: NIU->router ~5, router->router 9, router->NIU ~5, 64 B (1 flit) a cycle, header 1 flit (NoC/README.md:62-66);
    NoC0 right/down, NoC1 left/up on the 17x12 torus (RoutingPaths.md); hop counts from the Tensix coordinates
    x 1-7,10-16 (one harvested), y 2-11 (Coordinates.md) over all 14 harvest cases (hops()).
  Inter-chip: 306 ns a hop (27B README section 3: EthernetSsDesignSpec Table 16 280 ns at 400G + RS-FEC 25.6 ns at 200G).
"""
import functools
import math

CLK = 1.35e9
# ---- Tensix (specification) ----
MVMUL_LAT = 5
UNP_ADDR = 2
UNP_BYTES = 128            # B/cycle, BH default throttle for 8-bit and 16/32-bit inputs
SFP_LAT = 2
PACK_ROWS = 4              # Dst rows a cycle
SEM = 1
STALL_LAG = 1
RISC_L1_LOAD = 8
RISC_NOC_REG = 7
RISC_FENCE = 1             # the fence itself; its wait (store queue drain) is covered by the load that follows
# ---- NoC (specification) ----
NIU_OUT, NIU_IN, HOP = 5, 5, 9
FLIT = 64
# ---- budgets (no document gives the value; upper bounds, checked pass/fail on the device) ----
P_PACK = 16                # packer pipeline depth from PACR issue to the last L1 write of a short (<= 4 rows) pack
P_UNP = 17                 # unpacker depth from busy to the first Src/Dst write (TA "Unpackers Performance": ~17, a
                           # measurement page; used only as the budget's size, rule R1: never in the limit as a measurement)
N_RD = 16                  # NIU: write request issue -> first flit leaves (L1 read of the payload, one round trip)
# ---- the chip's Tensix grid ----
XS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
YS = list(range(2, 12))


def d0(a, b):
    return (b[0] - a[0]) % 17 + (b[1] - a[1]) % 12


def d1(a, b):
    return (a[0] - b[0]) % 17 + (a[1] - b[1]) % 12


def hops():
    """worst over the 14 harvest cases: (unicast to the hub, multicast hub -> all 130 as two rectangles on NoC0/NoC1),
    for the best hub of each case"""
    worst_up = worst_mc = 0
    for harv in XS:
        cores = [(x, y) for x in XS if x != harv for y in YS]
        xs = sorted({c[0] for c in cores})
        best = None
        for hub in cores:
            up = max(min(d0(c, hub), d1(c, hub)) for c in cores)
            mc = min(max(d0(hub, c) for c in cores), max(d1(hub, c) for c in cores))
            for i in range(1, len(xs)):
                A = [c for c in cores if c[0] in xs[:i]]
                B = [c for c in cores if c[0] in xs[i:]]
                for fa, fb in ((d0, d1), (d1, d0)):
                    mc = min(mc, max(max(fa(hub, c) for c in A), max(fb(hub, c) for c in B)))
            for j in range(1, len(YS)):
                A = [c for c in cores if c[1] in YS[:j]]
                B = [c for c in cores if c[1] in YS[j:]]
                for fa, fb in ((d0, d1), (d1, d0)):
                    mc = min(mc, max(max(fa(hub, c) for c in A), max(fb(hub, c) for c in B)))
            key = (up + mc, up, mc)
            if best is None or key < best:
                best = key
        worst_up = max(worst_up, best[1])
        worst_mc = max(worst_mc, best[2])
    return worst_up, worst_mc


UP_HOPS, MC_HOPS = hops()
# the hub is also the origin of the gathers (AG) and sits within 2 NoC hops of an Ethernet tile: the core map
# (tools/gen_core_map.py, core_map.h static_asserts) keeps the multicast at 17 hops and the up legs at 7 / 11 / 14 with it
# D-2 (ii) candidate 1: owners placed nearest the hub (hubgeo.py, worst over the 14 harvests): up 11 hops for <= 80 owner
# columns, 7 for <= 40; the multicast stays 17
UP_OWN = {40: 7, 80: 11, 130: 14}


def up_hops(n_owner):
    for k in sorted(UP_OWN):
        if n_owner <= k:
            return UP_OWN[k]
    return UP_OWN[130]


def flight(h, nbytes):
    """one packet: NIU out + h router hops + NIU in + its data flits (cut-through, 1 flit a cycle)"""
    return NIU_OUT + HOP * h + NIU_IN + math.ceil(nbytes / FLIT)


def poll():
    """a RISC sees a flag another agent wrote (MemoryOrdering.md: the L0 cache is not coherent, so each poll is a fence
    and an L1 load): one poll period = fence 1 + L1 load 8 + compare and branch 2; worst case the flag lands just after a
    load sampled it: the rest of that period + one more period = 2 periods"""
    return 2 * (RISC_FENCE + RISC_L1_LOAD + 2)


def issue():
    """a RISC issues a preset NoC write: source / destination address + CTRL (3 stores) and the cmd_buf-ready load"""
    return 3 + RISC_NOC_REG


def finalize_partial(rows):
    """fp32 sum of the 2 fidelity phases x 2 K faces into rows 0.. of the 2 N faces on the SFPU (mm_b1_unit.h's
    finalize): per 4-row group x column half x N face: 4 loads + 3 adds (latency 2, one nop each) + 1 store"""
    groups = math.ceil(rows / 4) * 2 * 2
    return groups * (4 + 3 * SFP_LAT + 1) + STALL_LAG


def pack(dst_rows):
    """PACR of `dst_rows` Dst rows (16 datums each) at PACK_ROWS rows a cycle + the packer's pipeline depth (budget)"""
    return math.ceil(dst_rows / PACK_ROWS) + P_PACK


def unpack_to_dst(nbytes):
    return UNP_ADDR + math.ceil(nbytes / UNP_BYTES) + P_UNP


def add_partials(rows, n):
    """n fp32 partials (rows x 32) into the owner's Dst on the SFPU: per partial and 4-row group x half x face: 2 loads,
    1 add, 1 nop, 1 store"""
    return n * math.ceil(rows / 4) * 2 * 2 * (2 + SFP_LAT + 1)


def mac(fmt, rows):
    phases = {"bf8": 2, "bf4": 1, "bf16": 4}[fmt]
    unp = {"bf8": 8, "bf4": 8, "bf16": 16}[fmt]
    return max(math.ceil(rows / 8) * 4 * phases, unp)


def dense_unit(M, N, K, fmt, cores=130, n_part=2):
    """one dependent dense projection on the chip: the input row x_u (M rows, K tiles) is in every core that needs it;
    the output row x_{u+1} must be in every core that needs it.  Stream-K over the cores (work/cores tile-MACs each),
    columns completed by their owner, the hub multicasts the finished columns in groups of 8 tiles."""
    rows = min(M, 32)
    nt, kt = math.ceil(N / 32), math.ceil(K / 32)
    t_mac = math.ceil(nt * kt / cores) * mac(fmt, rows) + MVMUL_LAT
    part = rows * 32 * 4
    col = rows * 32 * 2
    s_send = finalize_partial(rows) + pack(rows * 2) + issue() + N_RD + flight(2, part)
    s_owner = poll() + unpack_to_dst(part) + add_partials(rows, n_part - 1) + finalize_partial(rows) + pack(rows * 2) + issue() + N_RD
    s_up = flight(UP_HOPS, col)
    s_hub = poll() + issue() + N_RD
    s_mc = flight(MC_HOPS, 8 * col)
    s_rx = poll()
    stages = dict(mac=t_mac, send=s_send, owner=s_owner, up=s_up, hub=s_hub, mcast=s_mc, rx=s_rx)
    return sum(stages.values()), stages


if __name__ == "__main__":
    print(f"on-chip hops (worst over 14 harvests, best hub): up {UP_HOPS}, multicast {MC_HOPS}")
    for shape in ((2, 2560, 2560, "bf8"), (2, 2560, 768, "bf8"), (2, 320, 1280, "bf16"), (2, 1280, 320, "bf16")):
        t, st = dense_unit(*shape)
        print(shape, t, st)


def _hashable(x):
    return tuple(_hashable(y) for y in x) if isinstance(x, (list, tuple)) else x


def pool_stage(mms, cores=130):
    return _pool_stage(_hashable(mms), cores)


def pool_fanin(mms, cores=130):
    return _pool_fanin(_hashable(mms), cores)


@functools.lru_cache(maxsize=None)
def _plan(mms, cores=130):
    import pool_plan
    return pool_plan.pool_plan(mms, cores)


def _pool_stage(mms, cores=130):
    """a pool's time (cycles) = the best REALIZABLE core assignment (correction 18, 10-27 17:4x; pool_plan.py): the
    fastest of A (per-column core counts + heap reduce tree), B (linear Stream-K at K-unit boundaries) and C (whole
    columns, no K split), each column's tree timed from its cores' actual finishes, + the hand-over tail (owner -> hub,
    hub multicast, receivers' poll).  The kernels run exactly this assignment (gen_unit_assign.py)."""
    return int(round(_plan(mms, cores)[0]))


# ---- SFPU (fp32) costs per 32-lane group (one 4-row x 8-column load group): instruction counts of the fp32 SFPU
# functions of tt-metal's llk_sfpu counted by the kernel toolchain (27B README section 3, latest 10-26: silu 1,728,
# sigmoid 1,664, exp 1,280, softplus 3,232, rsqrt 960, reciprocal 192 cycles a 32x32 tile = 32 groups) + load/store;
# a simple mul/add: SFPLOAD, op, nop (latency 2), SFPSTORE.  These are instruction counts, not measurements.
ACT = {"silu": 54, "sigmoid": 52, "exp": 40, "softplus": 101, "rsqrt": 30, "recip": 6, "mul": 4, "add": 4}


def col_groups(rows):
    """SFPLOAD groups (4 rows x 8 even or odd columns) of one owner's column of `rows` rows after the finalize: each of
    the 2 N faces holds rows x 16 columns at Dst rows 0.. / 16.. (mm_b1_unit.h) -> ceil(rows/4) x 2 parities x 2 faces
    (D-7 (7): rows <= 4 -> 4 groups)"""
    return math.ceil(rows / 4) * 2 * 2


def owner_act(kind, rows=2):
    return col_groups(rows) * ACT[kind]


@functools.lru_cache(maxsize=None)
def gdr_unit(rows=2, cols=8):
    """gated delta rule, one unit = (value head, `cols` value columns) of S [128, cols] fp32 kept in Dst, issued by the
    math thread at one SFPU instruction a cycle, rows in series (the recurrence).  The issue order interleaves two load
    groups so that no instruction reads a result of the previous cycle (SFPADD.md / SFPMAD.md: latency 2), and two
    accumulators alternate between groups.  Per row:
      pass 1, per 4-row x 8-column load group of S (128/4 x cols/8 groups): SFPLOAD S, SFPMUL g, SFPMAD acc += S' k_rows,
              SFPSTORE S' (4); per 4 rows of S: SFPLOAD k's row pattern, SFPMAD kk += k k (2 x 128/4)
      column sums S'^T k replicated over the 4 row-lane groups (sum_rep): the 2 accumulators added (SFPADD, nop: 2),
              3 SFPMOV copies, SFPTRANSP (SFPTRANSP.md: L_i row r <- L_r row i, so every copy holds one row of the sum
              in all its row-lanes), 3 SFPADDs as a tree (5): 11; the k.k sum the same way (11)
      delta = (v - S'^T k / |k|) beta: SFPLOAD v, rsqrt of k.k (ACT), SFPMAD, nop, SFPMUL beta, nop (rsqrt + 5)
      pass 2, per group: SFPLOAD S', SFPMAD S'' = S' + k_rows (delta / |k|), SFPMAD acc += S'' q_rows, SFPSTORE S'' (4);
              per 4 rows: SFPLOAD k's and q's row patterns, SFPMAD qq += q q (3 x 128/4)
      output o = S''^T q / |q| x scale (row 0 of the lanes is enough, no replication): the 2 accumulators added (2),
              2 steps of (SFPMOV save, SFPSHFT2 SUBVEC_CHAINED_COPY4 (8-lane shift), SFPADD, nop) (8); q.q the same (10);
              rsqrt (ACT) + 2
    then the unit's output: sigmoid(z) and the gated norm's scale (2 mul), the pack of rows x cols fp32 and the hand-over
    write (issue + N_RD)"""
    groups = (128 // 4) * (cols // 8)
    pat = 128 // 4
    sum_rep = 2 + 3 + 1 + 5
    p1 = groups * 4 + 2 * pat + 2 * sum_rep
    mid = 1 + ACT["rsqrt"] + 4
    p2 = groups * 4 + 3 * pat
    tail = (2 + 8) + (2 + 8) + ACT["rsqrt"] + 2
    per_row = p1 + mid + p2 + tail
    out = 2 * ACT["mul"] + 10 + math.ceil(rows * cols * 4 / (PACK_ROWS * 64)) + P_PACK + issue() + N_RD
    return rows * per_row + out


def hub_tail(rows=2):
    col = rows * 32 * 2
    return flight(UP_HOPS, col) + poll() + issue() + N_RD + flight(MC_HOPS, 8 * col) + poll()


@functools.lru_cache(maxsize=None)
def topk_local(rows=2, ntile=16):
    """router top-10 up to the selector holding the row's 10 winners (D-2 (ii) candidate 3): the `ntile` logit column
    tiles are `ntile` router owners' outputs; each owner selects its own top-10 of 32 x rows (10 rounds of SFPSWAP max +
    a 5-step cross-lane max + mask), sends its 10 (score, index) pairs (7 hops worst, owners near the selector) to the
    selector, which merges the ntile x 10 candidates (ceil(ntile x 10 / 32) groups: 10 rounds)"""
    # D-7 (7): one round = SFPSWAP (argmax form) over the groups + 5 cross-lane steps (3 rotations within 8 lanes,
    # SUBVEC_SHFLROR1, + 2 row shifts, SUBVEC_CHAINED_COPY4) x (move, SFPSWAP) + the winner's mask (compare, SETCC,
    # write, ENCC); rev8 read-through (10-27 16:2x): the merge's group count follows the candidates (was fixed at 5)
    own = 10 * (2 * rows // 2 + 5 * 2 + 4)
    return own + issue() + N_RD + flight(7, 160) + poll() + merge10(ntile * 10)


def merge10(n_cand):
    """the selector's merge of n_cand (score, index) candidates into the top-10: 10 rounds over ceil(n_cand / 32) groups"""
    return 10 * (math.ceil(n_cand / 32) + 5 * 2 + 4)


def topk_post():
    """after the merge: softmax of the 10 (exp, sum, reciprocal) and the multicast of the 10 weights + ids to the cores"""
    return ACT["exp"] + 10 + ACT["recip"] + issue() + N_RD + flight(MC_HOPS, 128) + poll()


@functools.lru_cache(maxsize=None)
def topk10(rows=2, ntile=16):
    return topk_local(rows, ntile) + topk_post()


# DRAM tiles (x 0 and 9, y 0-11) -> any Tensix core: worst unicast hops over the 14 harvest cases (NoC0/NoC1 the shorter);
# a broadcast cannot target DRAM tiles (NoC/README.md:32: "receivers can only be Tensix tiles")
DRAM_HOPS = 14
DRISC_PAGE = 110           # one DRISC page: reservation, DMA issue, NoC write, oldest-trid check, arrival word (27B
                           # README section 16 item 10, core_plan.h kDriscPageCycles: instruction counts x specified latencies)
GDDR_PIPE = 405            # budget (rule R2's exception): GDDR read pipeline, NoC round trip + tRCD + RL <= 0.3 us (27B
                           # README section 16 item 7)
EXPERT_LOOKUP = RISC_L1_LOAD + 2   # the DRISC turns an expert id into its bank address: one L1 table load + add


def routed_issue():
    """D-3 (rev8 read-through 10-27 16:2x): cycles from the end of the router unit (topk10, whose post-merge stage is
    topk_post) to the first routed page landing in the cores.  The selector packs the 10 ids right after the merge, in
    parallel with the softmax, and writes them to the 8 bank DRISCs as 8 unicasts (no broadcast to DRAM tiles); the last
    DRISC sees them, looks the first expert up and issues its first page; the GDDR pipeline (budget); the page (2 KiB)
    flies to the core, which sees it."""
    ids_to_drisc = pack(4) + 8 * issue() + N_RD + flight(DRAM_HOPS, 64) + poll()
    first_page = EXPERT_LOOKUP + DRISC_PAGE + GDDR_PIPE + flight(DRAM_HOPS, 2048) + poll()
    return max(0, ids_to_drisc + first_page - topk_post())


# ---- per-unit stage formulas shared by step4_item_time.py (section 12) and free5d_v8.py (section 11), so the two
# models cannot diverge (worklog 10-27 15:0x: free5d had kept older formulas for the SDPA hand-over and the gated norm)
def hc_read_elt(rows=2):
    """one hyper-connection read's elementwise work fused into the owners: 1/rms_g (rsqrt + 2 mul), silu / 4 in the down
    owners (silu + mul), sigmoid in the up owners, gate x normalized and the mean of the 4 branches (8 mul)"""
    return math.ceil(rows / 32) * (ACT["rsqrt"] + 2 * ACT["mul"] + owner_act("silu") + ACT["mul"] + owner_act("sigmoid") + 8 * ACT["mul"])


def hc_write_elt(rows=2):
    """residual += inject[b] x block output on the residual columns' owners (4 branches x (mul, add))"""
    return math.ceil(rows / 32) * 8 * ACT["mul"]


def conv1d_elt():
    """depthwise conv1d, 4 taps from the L1-resident tap state, + silu, in the in_proj owners' finalize"""
    return 2 * 4 * 3 + owner_act("silu")


def norm_gated():
    """gated RMS norm per value head: the head's 16 units' sums of squares gathered to its first unit (7 hops), rsqrt,
    the scale back to the 16 units (7 hops), applied with sigmoid(z) before the hand-over to out_proj"""
    return (flight(7, 64) + poll() + ACT["rsqrt"] + issue() + N_RD + flight(7, 64) + poll() + 2 * ACT["mul"]
            + owner_act("sigmoid"))


def sdpa_ctx1(rows=2):
    """SDPA at ctx 1 on the kv-head cores: 3 q heads x rows, q.k (16 x 4), softmax (max 20, exp, sum 10, reciprocal), P.v
    (16 x 4), sigmoid(gate); the attention output multicast straight to the o_proj cores (no owner -> hub leg)"""
    return (rows * (16 * 4 + 20 + ACT["exp"] + 10 + ACT["recip"] + 16 * 4) + owner_act("sigmoid") + issue() + N_RD
            + flight(MC_HOPS, 8 * rows * 64) + poll())


def rope_elt():
    """partial RoPE (64 of 256 columns) on q and k in their owners' finalize: cos / sin pairs"""
    return 2 * 2 * 3 * 2


def index_ctx1_elt():
    """index scoring at ctx 1: one block"""
    return 2 * 4 * ACT["mul"]


def moe_act_elt(rows=2, score=True):
    """correction 20/21: in the owner of a gate/up pair column: silu(gate) x up (x the routing score for a routed expert:
    the score folded into the down's input so the down only accumulates) on the SFPU"""
    return col_groups(rows) * (ACT["silu"] + (2 if score else 1) * ACT["mul"])


def moe_combine_elt(rows=2):
    """the down owner's epilogue (correction 19): the routed accumulator (scores folded into the down's input) + the
    shared accumulator x sigmoid(shared_expert_gate . x) (the gate scalar arrives with the router pool's multicast; the
    owner takes its sigmoid once) -> block output"""
    return ACT["sigmoid"] + col_groups(rows) * 2 * ACT["mul"]


def gdr_layer(rows, units, cores=130):
    """the GDN layer's gated delta rule: ceil(units / cores) rounds of the unit, then the hand-over to out_proj"""
    return math.ceil(units / cores) * gdr_unit(rows, 8) + hub_tail(min(rows, 32))


# ---- QSA beyond ctx 1 (rev8 read-through 10-27 16:4x): stages for the section 11 enumeration at ctx > 1 ----
@functools.lru_cache(maxsize=None)
def attn_merge(R):
    """SFPU merge of one partial (o [R, 256] fp32, m, l) into the running one: per 4-row group the new max and the two
    rescale factors (2 exp), then o = o a + p b over the 8 column tiles (2 loads, mul, mad, store with latency nops)"""
    return math.ceil(R / 4) * (2 * ACT["exp"] + 6) + col_groups(R) * 8 * 7


@functools.lru_cache(maxsize=None)
def sdpa_stage(R, sel, cores=130):
    """SDPA of R q rows (rows x the q heads of one kv head; with DCP the gathered heads of the group) over `sel` K/V
    positions (sel > 1) on this chip, Stream-K over the positions (rev8 read-through 10-27 16:5x).  The position tiles
    are split over n cores (n chosen to minimize the time):
      A  each core: S = q.K^T over its nt position tiles (8 tile-MACs of bf16 HiFi4 each), its row max (SFPSWAP over
         the groups + a 5-step cross-lane max);
      M  the row maxes to the hub (unicast), the hub's max over the n (one SFPSWAP a core), multicast back: one global
         max, so the partial sums below are plain sums (no rescale);
      B  exp(S - m) and the row sum l on the SFPU, P packed and unpacked into SrcB, P.V over the nt tiles into the 8
         output column tiles (+ l as one more column);
      R  Stream-K: the 9 column owners add the n partials [R, 32] fp32 over the best tree (pool_stage's level);
      F  the owner divides by l, applies the gate (sigmoid) and multicasts the output to the o_proj cores.
    Returns cycles."""
    pt = math.ceil(sel / 32)
    g = col_groups(R)
    xl = math.ceil(R / 4) * 10
    part = R * 32 * 4
    best = None
    for n in range(1, min(cores, pt) + 1):
        nt = math.ceil(pt / n)
        a = nt * 8 * mac("bf16", R) + MVMUL_LAT + nt * g * 2 + xl
        m = 0 if n == 1 else (issue() + N_RD + flight(UP_HOPS, 64) + poll() + n * math.ceil(R / 4) * 2 + 10
                              + issue() + N_RD + flight(MC_HOPS, 64) + poll())
        b = (nt * g * (1 + ACT["add"] + ACT["exp"] + 1) + nt * g * ACT["add"] + xl + pack(2 * R * nt) + P_UNP
             + nt * 9 * mac("bf16", R) + MVMUL_LAT)
        red = 0
        if n > 1:
            send = pack(2 * R * 9) + 9 * issue() + N_RD + flight(2, 9 * part)   # correction 13: the 9 partials' flits in series
            lv = None
            for L in (1, 2, 3):
                mm = max(1, math.ceil(n ** (1.0 / L) - 1e-9))
                while mm ** L < n:
                    mm += 1
                t = L * (poll() + mm * (UNP_ADDR + math.ceil(part / UNP_BYTES)) + P_UNP + add_partials(R, mm)) + (L - 1) * (
                    pack(2 * R) + issue() + N_RD + flight(2, part)) + send
                lv = t if lv is None else min(lv, t)
            red = lv
        t = a + m + b + red
        best = t if best is None else min(best, t)
    fin = math.ceil(R / 4) * (ACT["recip"] + 4) + g * ACT["mul"] + owner_act("sigmoid")
    return best + fin + issue() + N_RD + flight(MC_HOPS, R * 256 * 2) + poll()


def index_select(rows, nblk, cores=130, round_extra=None):
    """round_extra: None (DCP 1) or the cycles of the cross-chip count all-reduce for 1, 3, 7, 15 counts (a tuple)"""
    ex = {1: 0, 3: 0, 7: 0, 15: 0} if round_extra is None else dict(zip((1, 3, 7, 15), round_extra))
    return _index_select(rows, nblk, cores, tuple(ex[k] for k in (1, 3, 7, 15)))


@functools.lru_cache(maxsize=None)
def _index_select(rows, nblk, cores, ex4):
    """ctx > 2,048 (HF definition, reference.py: the complete index blocks' pooled keys scored by the 4 index heads,
    relu(q.k) summed over the heads / sqrt(128), the top 512 = budget 2,048 / compress 4 selected).  Scoring: the
    nblk blocks of this chip split over the cores, 4 tile-MACs (128 deep, bf16 HiFi4) per 32-block tile + relu and the
    head sum on the SFPU.  Selection: a radix threshold search on the fp32 scores (>= 0: 31 bits) with b bits a round
    (b in 1..4, the fastest): every core counts its scores above the 2^b - 1 candidate thresholds (per 32-score group:
    load, (compare, conditional add) x (2^b - 1), end), sums the counters across lanes, sends them to the selector
    (14 hops), which adds the 130 cores' counters, picks the bucket and multicasts the next prefix.  Then every core
    writes its selected block ids.  Returns (cycles, rounds): the caller adds a cross-chip count all-reduce per round
    under DCP."""
    R = rows * 4
    bt = math.ceil(nblk / 32)
    per = math.ceil(bt / cores)
    score = per * (4 * mac("bf16", R) + col_groups(R) * 4 + 16) + MVMUL_LAT
    gpc = math.ceil(nblk / cores / 32) * rows
    best = None
    for b in (1, 2, 3, 4):
        k = 2 ** b - 1
        rounds = math.ceil(31 / b)
        local = gpc * (2 + 2 * k) + k * rows * 10
        sel_core = (issue() + N_RD + flight(UP_HOPS, 64) + poll() + k * rows * (5 * 2 + 10) + 10
                    + issue() + N_RD + flight(MC_HOPS, 64) + poll())
        t = rounds * (local + sel_core + ex4[b - 1])   # correction 17: the 2^b - 1 counts of the round
        if best is None or t < best[0]:
            best = (t, rounds)
    # correction 14: exactly 512 like the HF topk: the ties at the threshold are numbered by a prefix sum over the cores
    # (each core's tie count to the selector, the selector's running sum back), then every core writes its blocks
    ties = (gpc * 3 + issue() + N_RD + flight(UP_HOPS, 64) + poll() + 130 // 8 + issue() + N_RD + flight(MC_HOPS, 64 * 4)
            + poll() + ex4[0])
    compact = gpc * 3 + issue() + N_RD
    return score + best[0] + ties + compact, best[1]


def kv_gather_issue():
    """the selected blocks' K/V read issued by the SDPA cores (NoC read requests to the DRAM tiles): issue, the request
    flight, the GDDR pipeline (budget), the first 4 KiB block (K + V of 4 rows of one kv head) back, the poll"""
    return issue() + flight(DRAM_HOPS, 0) + GDDR_PIPE + flight(DRAM_HOPS, 4096) + poll()


# ---- per-core landing areas (rev8, D-3 correction 12, 10-27 17:0x): the same address on every core, out of the frame
# capacity; chip_l1.h (tools/design/gen_chip_l1.py) lays them out with these sizes ----
def _pool_fanin(mms, cores=130):
    """the largest number of fp32 partials one tree node of the pool's chosen assignment receives (0: no K split)"""
    return _plan(mms, cores)[3]


def l1_landing(rows, meet_max, hand_max, fanin):
    """bytes per core: flags 64 x 16 B; a hub's multicast group (8 tiles of a rows-row bf16 column) double buffered;
    the Stream-K partials one tree node receives (fanin fp32 partials of rows x 32: the largest pool_fanin of the
    design's pools, correction 15); the meeting landing (2 x the largest meeting payload); the hand-over landing (the
    largest hand-over); each rounded to 16 B"""
    def al(x):
        return (int(math.ceil(x)) + 15) // 16 * 16
    return (al(64 * 16) + al(2 * 8 * rows * 32 * 2) + al(fanin * rows * 32 * 4) + al(2 * meet_max)
            + al(hand_max))



def landing_core(rows, kmax, fanin, meet_max, hand_max):
    """correction 33-34 (10-27 19:1x): the per-core landing areas of section 12: flags 64 x 16 B; the x receive area =
    the whole next input of a pool (2 faces: the next unit's x lands while the current one is still read) = 2 x rows x
    kmax x 2 B (bf16); the Stream-K partials of one tree node (fanin x rows x 32 x 4 B); the meeting landing (2 x the
    largest meeting payload); the hand-over landing (the largest hand-over); each rounded to 16 B"""
    def al(x):
        return (int(math.ceil(x)) + 15) // 16 * 16
    return (al(64 * 16) + al(2 * rows * kmax * 2) + al(fanin * rows * 32 * 4) + al(2 * meet_max) + al(hand_max))
