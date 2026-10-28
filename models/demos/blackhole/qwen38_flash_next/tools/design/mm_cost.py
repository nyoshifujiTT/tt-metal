#!/usr/bin/env python
r"""The matmul unit costs of the limit (worklog 14, revision 6), from the primary sources only:
  MVMUL (tt-isa-documentation MVMUL.md / MatrixUnit.md): one instruction a cycle multiplies 8 rows of SrcB by a 16x16 of
    SrcA, one instruction per fidelity phase -> a 32x32 tile-MAC with r valid rows is ceil(r/8) x 2 x 2 x phases cycles.
  the weight tile into SrcA (UNPACR_Regular.md, Blackhole: BurstRows = throughput / row bytes, at most 8 rows a cycle;
    8-bit (BFP8) x8 = 128 B, BFP4 x4 = 64 B -> 8 rows, 16-bit x4 "2x" = 128 B -> 4 rows; a tile is 64 rows of 16) ->
    bf8 8, BF4 8, bf16 16 cycles; SrcA has two banks, so the next tile's unpack hides under the MVMULs; the activation
    (unpacker 1, r rows x 32 bf16 a K tile) fits the 160 B/cycle of the two unpackers together (LLK Team, 2025-05-13).
  the served fidelity per weight format (ttnn/decode_matmul.py:102 and the bf16 mixers / router at HiFi4): bf8 HiFi2,
    BF4 LoFi, bf16 HiFi4.
  tile-MAC = max(MVMUL, weight unpack).
  K-split partial tree, per level: NoC relay 92 cycles + the partial's flits (r x 32 x fp32 / 64 B) + its unpack into
    SrcA (/128 B) and ELWADD (ceil(r/8) x 2 instructions) -> 98 cycles at r = 2 (revision 5: 188, a full 4 KiB tile).
  input distribution (revision 7), per matmul unit: its input row (the previous unit's output, 2 rows) has to reach
    the cores holding its K slices; the bytes a core needs (its K range, <= 80 x 128 B) arrive under its own MACs when
    sent in K order, so the exposed part is one NoC hop of the first tile = one tree level (98 cycles at r = 2;
    revisions 5-6 started a unit at the previous unit's end, the hand-over uncounted).
Shapes are (M, N, K, fmt); M is the matmul's row count (rows of one row tile at most 32 here)."""
import math

CORES = 130
TILE = 32
NOC_RELAY = 92
NOC_FLIT = 64
UNPACK_BYTES = 128
PHASES = {"bf8": 2, "bf4": 1, "bf16": 4}
WEIGHT_UNPACK = {"bf8": 8, "bf4": 8, "bf16": 16}


def rows_in_tile(M):
    return min(M, TILE)


def mac(fmt, rows):
    mvmul = math.ceil(rows / 8) * 2 * 2 * PHASES[fmt]
    return max(mvmul, WEIGHT_UNPACK[fmt])


def tree_level(rows):
    part = rows * TILE * 4
    return NOC_RELAY + math.ceil(part / NOC_FLIT) + max(math.ceil(part / UNPACK_BYTES), math.ceil(rows / 8) * 2)


import stages


def pool(*mms):
    return stages.pool_stage(mms)


def _pool_model(*mms):
    """cycles of matmuls that read the same input, their output tiles pooled over the 130 cores"""
    out = 0
    work = 0.0
    kt_max = 0
    nosplit = 0.0
    rows = 0
    for M, N, K, fmt in mms:
        rt = math.ceil(M / TILE)
        nt = math.ceil(N / TILE)
        kt = math.ceil(K / TILE)
        r = rows_in_tile(M)
        c = mac(fmt, r)
        out += rt * nt
        work += rt * nt * kt * c
        kt_max = max(kt_max, kt)
        nosplit = max(nosplit, kt * c)
        rows = max(rows, r)
    dist = tree_level(rows)
    if out >= CORES:
        return dist + work / CORES
    cpo = min(math.ceil(CORES / max(out, 1)), kt_max)
    levels = math.ceil(math.log2(cpo)) if cpo > 1 else 0
    return dist + min(nosplit, work / CORES + levels * tree_level(rows))


def one(M, N, K, fmt):
    return pool((M, N, K, fmt))
