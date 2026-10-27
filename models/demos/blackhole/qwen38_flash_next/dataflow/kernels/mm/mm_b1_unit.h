// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-7 (worklog 14): the B=1 matmul unit at the served fidelity, built from the primary sources.
//   out[rows <= 8, 32 ct] (fp32 DEST) += in0[rows, 32 kt] (bf16, SrcB) x in1[32 kt, 32 ct] (Bfp8_b / Bfp4_b, SrcA)
// Why not matmul_block: one tile-MAC at <= 8 rows is 4 MVMULs a fidelity phase (MVMUL.md: SrcB 8 rows x SrcA 16x16 a
// cycle), but Dst.md:88 bars reading an aligned 8x16 Dst block for 5 cycles after a write, so an accumulating loop
// needs >= 6 distinct blocks in flight; a 2-row column accumulated in place touches 2 blocks a phase (stall-bound,
// measured 25-33 cycles/tile in I-1c-4).
// Here every tile-MAC writes 4 x phases distinct blocks: phase p's (K face kf, N face nf) partial goes to Dst rows
// 32 p + 16 nf + 8 kf (8 blocks at HiFi2: 0, 16, 8, 24, 32, 48, 40, 56), the fidelity loop outermost inside a tile
// (Dst.md: "the fidelity loop should be the outer loop"); finalize sums the 2 phases x 2 K faces in fp32 on the SFPU
// into rows 0 (N face 0) and 16 (N face 1) of the output tile slot (no FPU move: MOVD2B would round to bf16/tf32).
// The unpack is DeepSeek Blitz custom_mm's (api/compute/experimental/custom_mm.h, llk_unpack_AB_custom_mm.h, ct 1):
// SrcB (the activation's K tile) and SrcA (the weight tile) unpacked per K tile, the weight's L1 address advanced by
// CFGSHIFTMASK from a scratch stride (3 instructions a tile; 8.85 cycles a Bfp8_b tile in tt-metal #58712), one MOP
// for the K loop (kt even); the math here consumes each source bank pair for all the fidelity phases.
// Operand layout: in1 tiles of one output column are consecutive in L1 (k-major per column); ct columns follow.
#pragma once
#include <cstdint>
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#endif

namespace qwen38_b1 {

#ifdef TRISC_MATH
using namespace ckernel;
// Dst addressing of one tile-MAC (rows relative to the tile slot):
//   M0: kf0 nf0 -> +16 (to nf1 of kf0)    M1: kf0 nf1 -> back to nf0 of kf1 (-8: rows 8)   (srca +16 each)
//   M2: kf1 nf0 -> +16                     M3: kf1 nf1 -> next phase (+32 - 24 = +8 from 24) / end of tile
template <uint32_t phases>
inline void b1_mm_math_init() {
    // M0 / M2: next N face of SrcA (srca +16), same SrcB rows, Dst +16
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 0}, .dest = {.incr = 16}}.set(ADDR_MOD_0);
    // M1: next K face: srca +16 (A face 2), srcb +16 (B face 1), Dst -8 (row 24 -> 8 ... from 16: to 8)
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 16}, .dest = {.incr = static_cast<int16_t>(-8)}}.set(ADDR_MOD_1);
    // M3 inside the tile: next phase: srca / srcb back to 0, Dst from 24 to 32 p, fidelity +1
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = 8},
        .fidelity = {.incr = 1, .clr = 0}}
        .set(ADDR_MOD_2);
    // M3 at the end of the tile: srca / srcb to 0, Dst back to the tile slot base (c-to-cr: Dst_Cr), fidelity cleared
    addr_mod_t{.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 0, .cr = 1}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
    // SFPLOAD / SFPSTORE of the finalize: no counter change
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    // one tile-MAC = phases x {M0 M1 M0 M3}; M3 of the last phase releases both source banks (SrcA: the weight tile,
    // SrcB: this K tile of the activation) and rewinds
    load_replay_buf(ckernel::math::replay_buf_offset, 4, [] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    });
    const uint32_t body3 = lltt::replay_insn(ckernel::math::replay_buf_offset, 3);
    const uint32_t m3_phase = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    const uint32_t m3_tile = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
    // MOP template 1: outer = tiles (set per run), inner = phases of {body3, m3}; the last inner op of every outer
    // iteration is the end-of-tile M3
    ckernel_template tmp(1, phases, body3, m3_phase);
    tmp.set_last_inner_loop_instr(m3_tile);
    tmp.set_last_outer_loop_instr(m3_tile);
    tmp.program();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    ckernel::sfpu::_init_sfpu_config_reg();
}

// ct form (custom_mm's unpack order: for each K tile, SrcB once, then the ct weight tiles of that K row into SrcA):
// the math walks c inside k, tile slot c at Dst rows 64 c (dense: 64 rows a slot), SrcB released only after the last
// column of a K tile.  Per tile: phases x {M0 M1 M0 M3}; M3 of the last phase: SrcA released and Dst to the next slot
// (or back to slot 0 with SrcB released after column ct - 1).
template <uint32_t phases>
inline void b1_mm_math_init_ct(uint32_t ct) {
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 0}, .dest = {.incr = 16}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 16}, .dest = {.incr = static_cast<int16_t>(-8)}}.set(ADDR_MOD_1);
    addr_mod_t{.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 8}, .fidelity = {.incr = 1, .clr = 0}}
        .set(ADDR_MOD_2);
    // end of a tile, more columns follow: srca / srcb to 0, Dst to the next slot (from row 32 (phases - 1) + 24 to
    // the slot base + 64: cr walk), fidelity cleared
    addr_mod_t{.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 64, .clr = 0, .cr = 1}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
    // end of the K tile's last column: srca / srcb to 0, Dst back to slot 0 (clear), fidelity cleared
    addr_mod_t{.srca = {.incr = 0, .clr = 1}, .srcb = {.incr = 0, .clr = 1}, .dest = {.incr = 0, .clr = 1, .cr = 0}, .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_4);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    load_replay_buf(ckernel::math::replay_buf_offset, 4, [] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    });
    const uint32_t body3 = lltt::replay_insn(ckernel::math::replay_buf_offset, 3);
    const uint32_t m3_phase = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    const uint32_t m3_col = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_3, 0);
    const uint32_t m3_krow = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_4, 0);
    // template 1: outer = ct columns, inner = phases x {body3, m3}; last inner op of a column: m3_col, of the last
    // column: m3_krow
    ckernel_template tmp(ct, phases, body3, m3_phase);
    tmp.set_last_inner_loop_instr(m3_col);
    tmp.set_last_outer_loop_instr(m3_krow);
    tmp.program();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    ckernel::sfpu::_init_sfpu_config_reg();
}

// kt K rows of ct columns into the tile slots 0 .. ct - 1 (Dst rows 64 c ..); no finalize
inline void b1_mm_math_run_ct(uint32_t kt) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    for (uint32_t k = 0; k < kt; ++k) {
        TTI_MOP(1, 0, 0);
    }
}

// kt tile-MACs into the tile slot `dst_index` (Dst rows dst_index x 64 .. + 32 phases); no finalize
inline void b1_mm_math_run(uint32_t dst_index, uint32_t kt) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    for (uint32_t k = 0; k < kt; ++k) {
        TTI_MOP(1, 0, 0);
    }
}

// fp32 finalize on the SFPU of the tile slots 0 .. n - 1 (after their MVMULs): rows 0..1 += rows 8, 32, 40 (N face 0)
// and rows 16..17 += 24, 48, 56 (N face 1); SFPLOAD reads one aligned 4-row group, even or odd columns (SFPLOAD.md:
// Imm10 bits [9:2] the group, bit 1 the column parity), so a face's 2 valid rows are one group (rows 2..3 carry the
// zeros of SrcB's padding rows); 2 column halves x 2 N faces a slot; the slots are addressed through Imm10 (64 rows a
// slot) with one Dst offset, so no SETC16 between SFPU instructions
constexpr uint32_t kSfpFp32 = 3;  // SFPLOAD / SFPSTORE MOD0_FMT_FP32 (SFPLOAD.md)
template <uint32_t phases>
inline void b1_mm_math_finalize(uint32_t n) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    for (uint32_t c = 0; c < n; ++c) {
#pragma GCC unroll 4
        for (uint32_t q = 0; q < 4; ++q) {
            const uint32_t base = 64 * c + 16 * (q >> 1) + (q & 1) * 2;
            TT_SFPLOAD(p_sfpu::LREG0, kSfpFp32, ADDR_MOD_7, base + 0);
            TT_SFPLOAD(p_sfpu::LREG1, kSfpFp32, ADDR_MOD_7, base + 8);
            if constexpr (phases > 1) {
                TT_SFPLOAD(p_sfpu::LREG2, kSfpFp32, ADDR_MOD_7, base + 32);
                TT_SFPLOAD(p_sfpu::LREG3, kSfpFp32, ADDR_MOD_7, base + 40);
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
                TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG1, 0);
                TTI_SFPNOP;
            }
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPNOP;
            TT_SFPSTORE(p_sfpu::LREG0, kSfpFp32, ADDR_MOD_7, base + 0);
        }
    }
    math::clear_dst_reg_addr();
}
#endif

}  // namespace qwen38_b1
