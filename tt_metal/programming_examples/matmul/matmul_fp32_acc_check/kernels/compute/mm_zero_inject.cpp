// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// C: zero injection, so that at most one useful value lands in each 8-lane SOP group.
//
// Background. One MVMUL reduces 16 K-elements as two independent 8-lane sum-of-products groups.
// Inside a group every product is right-shifted to the group's maximum exponent, and a product
// carries only ~12 bits with no guard or sticky bit, so a term more than ~11 binades below its
// group maximum is dropped outright. That happens before the FP32 accumulator, so
// fp32_dest_acc_en cannot recover it. Putting only one useful value in each group removes the
// intra-group alignment entirely: with a single non-zero term the group maximum is that term's
// own exponent and nothing is shifted away.
//
// Only SrcA is zeroed. mvmul() tests the operands with an OR:
//     zero_a = (a & 0x3FFFF) < 0x400;  zero_b = ...;  if (zero_a || zero_b) { man = 0; exp = ... }
// so a zero in SrcA alone makes the lane contribute nothing, and the lane's exponent is forced to
// a neutral value rather than anything derived from the data, so it cannot raise the group
// maximum. Whatever SrcB holds in those lanes is never read, which is why SrcB needs no zero
// padding and no reordering.
//
// Loop order, l, k, i, j, f. B1/B2/B3 consume a whole 16-wide K window per MVMUL, so K=32 is
// exhausted by k alone. Here 2 of the 16 lanes are used, so the window is subdivided further by
// l (0..7), selecting the K pair (l, l+8) - 8 apart so the two land in different SOP groups.
// K = k * l * 2 = 32.
//
// l is outermost and f innermost, and in both cases the reason is the number of SrcA rewrites.
// SrcB carries the useful values of all four faces after one UNPACR, so i (M half) and k (K half)
// are reached by moving its row counter and cost no refetch. SrcA can hold only 2 useful rows per
// face, so advancing l means rewriting it; with l outermost that is 8 times per tile, and each
// rewrite fills all four faces at once because SrcA's face index is 2k+j. A fidelity phase only
// selects which mantissa bits are read out of Src, not which rows or their contents, so the four
// phases belong at one (l, k, i, j) position; hoisting f above l would take the rewrites to 32.
// k, i and j are free, as B2 and B3 established, and are ordered as in B3.
//
// Cost. 512 MVMULs against B1's 64, the 8x the analysis predicts, plus 8 SrcA rewrites and one
// bank clear per tile-row instead of a single whole-tile unpack. The bytes written into SrcA are
// the same, only split.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

// A 32x32 tile holds 32 K-elements. One l covers a pair (l, l+8) of each 16-wide K window.
constexpr uint32_t kLPerTile = 8;

#ifdef TRISC_MATH
namespace {

// Address modes for the k, i, j, f walk performed at one l.
//
// SrcA face = 2*k + j, SrcB face = 2*(M half) + k, Dst face = 2*(M half) + j, each face being 16
// rows. i is the 8-row half within the M dimension, so i = 2*(M half) + (M eighth).
//
// Positions visited, in order (srcA, srcB, dst), i as 0..3:
//   k=0: (0,0,0) (0,8,8) (0,32,32) (0,40,40)      j=0
//        (16,0,16) (16,8,24) (16,32,48) (16,40,56) j=1
//   k=1: (32,16,0) (32,24,8) (32,48,32) (32,56,40) j=0
//        (48,16,16) (48,24,24) (48,48,48) (48,56,56) j=1
inline void zi_configure_addrmod() {
    constexpr uint32_t fidelity_increment = (MATH_FIDELITY != ckernel::MathFidelity::LoFi) ? 1 : 0;

    // f step: nothing moves, only the fidelity phase.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = fidelity_increment, .clr = 0},
    }
        .set(ckernel::ADDR_MOD_0);

    // i step inside an M half: SrcB and Dst on by 8 rows, fidelity back to 0.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ckernel::ADDR_MOD_1);

    // i step crossing to the M16-31 faces: SrcB and Dst on by 24 rows (8 -> 32).
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 24, .clr = 0, .cr = 0},
        .dest = {.incr = 24, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ckernel::ADDR_MOD_2);

    // j step: SrcA on by one face, SrcB back to its marker, Dst to its marker + 16.
    // Dst increments are not modular (10-bit counter, wraps at 1024), so the backward move is
    // done with the cr marker rather than by adding the complement.
    ckernel::addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 1},
        .dest = {.incr = 16, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ckernel::ADDR_MOD_3);

    // k step: SrcA on to the K16-31 faces, SrcB marker to row 16, Dst marker back to 0.
    ckernel::addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ckernel::ADDR_MOD_4);

    // End of an l: everything back to 0, ready for the next SrcA contents.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ckernel::ADDR_MOD_5);
}

// The 16 (k, i, j) positions at one l, each repeated over the fidelity phases.
//
// SrcA holds useful values only in rows l and l+8 of each face, so each MVMUL consumes exactly 2
// useful K-elements, one per SOP group. SrcB holds the whole in0 tile and is not moved between
// l values.
inline void zi_matmul_one_l() {
    constexpr bool high_fidelity = (MATH_FIDELITY != ckernel::MathFidelity::LoFi);
    constexpr uint32_t phases = high_fidelity ? static_cast<uint32_t>(MATH_FIDELITY) : 1;

    // One (k, i, j) position: the fidelity phases, then the address mode that moves on.
    auto position = [](uint32_t step_mode) {
        for (uint32_t phase = 1; phase < phases; ++phase) {
            TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);
        }
        TT_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, step_mode, 0);
    };

    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t j = 0; j < 2; ++j) {
            position(ckernel::ADDR_MOD_1);  // i = 0 -> 1
            position(ckernel::ADDR_MOD_2);  // i = 1 -> 2, crossing to the M16-31 faces
            position(ckernel::ADDR_MOD_1);  // i = 2 -> 3
            // i = 3: j step, except on the last j of the last k, which ends the l.
            const bool last = (k == 1) && (j == 1);
            position(last ? ckernel::ADDR_MOD_5 : (j == 1 ? ckernel::ADDR_MOD_4 : ckernel::ADDR_MOD_3));
        }
    }

    // Release SrcA so the next l can be unpacked into it. SrcB is released by the caller, once
    // per tile.
    TTI_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
}

// SrcB is fetched once per tile, so it is released once per tile.
inline void zi_release_srcb() { TTI_SETRWC(ckernel::p_setrwc::CLR_B, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F); }

}  // namespace
#endif

#ifdef TRISC_UNPACK
namespace {

inline void zi_unpack_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    // SrcB takes a whole 32x32 tile (4 faces), as in B1. SrcA takes one 16-datum row per UNPACR.
    TT_SETADCXX(p_setadc::UNP_B, 4 * 16 * 16 - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_A, 1 * 16 - 1, 0x0);
}

// Write one 16-datum L1 row into a chosen SrcA row.
//   l1_addr_16b: 16B-granular L1 address of the source row
//   srca_row:    destination row within SrcA's 64 rows
//
// The cfg registers are programmed through SETDMAREG + WRCFG rather than by storing to the cfg
// pointer. A store is executed by ThCon as the TRISC issues it, while the UNPACR it configures
// sits in the instruction FIFO waiting for the matrix unit to release the SrcA bank, so the
// TRISC runs ahead and the UNPACR sees a later row's addresses. Measured that way, only l=0
// landed on rows 0,8,...,56; from l=1 on, the destination row and the source address came from
// different iterations. WRCFG goes through the same FIFO as UNPACR, so the order is kept.
inline void zi_unpack_srca_row(uint32_t l1_addr_16b, uint32_t srca_row, uint32_t set_dvalid) {
    // Both config-context slots are written, because the kernel alternates contexts and the
    // unpacker reads the slot belonging to the current one. Writing only the context-0 slot left
    // every context-1 step reading a stale address, i.e. exactly half the output tile wrong.
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_addr_16b), 0, LO_16(ckernel::p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(l1_addr_16b), 0, HI_16(ckernel::p_gpr_unpack::TMP0));
    TTI_WRCFG(ckernel::p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
    TTI_WRCFG(ckernel::p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);

    // Destination row, in datums. The SrcA path subtracts 64 datums (4 rows) from this address,
    // so add that back to land on the requested row. Both context slots live in the low and high
    // halves of this one 32-bit register.
    const uint32_t dest_datums = (srca_row + 4) * 16;
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_datums), 0, LO_16(ckernel::p_gpr_unpack::TMP1));
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_datums), 0, HI_16(ckernel::p_gpr_unpack::TMP1));
    TTI_WRCFG(ckernel::p_gpr_unpack::TMP1, 0, THCON_SEC0_REG5_Dest_cntx0_address_ADDR32);

    TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::TRISC_CFG);
    TT_UNPACR(SrcA, 0, 0, 0, 0, 1 /* OvrdThreadId */, set_dvalid, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
}

// Fetch the whole in0 tile into SrcB. Called once per tile: SrcB carries the useful values of all
// four faces, so both k and both M halves are reached by moving the row counter, and nothing here
// depends on l. Programmed through WRCFG for the ordering reason described above.
inline void zi_unpack_srcb_tile(uint32_t cb_id_in0) {
    const uint32_t base_in0 = get_local_cb_interface(cb_id_in0).fifo_rd_ptr - 1;

    TT_SETDMAREG(0, LOWER_HALFWORD(base_in0), 0, LO_16(ckernel::p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(base_in0), 0, HI_16(ckernel::p_gpr_unpack::TMP0));
    TTI_WRCFG(ckernel::p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);
    TTI_WRCFG(ckernel::p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
    TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::TRISC_CFG);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
}

// Write the SrcA rows for one l: rows l and l+8 of each of the four faces.
//
// in1 is the K-by-N operand, so K is the row index of the tile, and in TILED_NFACES layout a
// 32x32 tile is four 16x16 faces (TL, TR, BL, BR), each row-major. Tile row kk lives in faces
// (kk/16)*2 and (kk/16)*2+1 at row kk%16. SrcA's face f = 2*k + j holds K window k and output
// column half j, which is exactly the same mapping, so face f of SrcA takes face f of the tile.
//
// One bf16 row of 16 datums is 32 B, and the unpacker address is in 16 B units, so a row is 2
// units and a face is 32 units.
inline void zi_unpack_srca_rows(uint32_t cb_id_in1, uint32_t l) {
    const uint32_t base_in1 = get_local_cb_interface(cb_id_in1).fifo_rd_ptr - 1;
    constexpr uint32_t kRowUnits = 2;
    constexpr uint32_t kFaceUnits = 32;

    ckernel::unpacker::wait_for_next_context(2);
    ckernel::semaphore_post(ckernel::semaphore::UNPACK_SYNC);

    // Zero the bank about to be written, and wait until the matrix unit has released it.
    // ZEROSRC clears whole banks so it cannot maintain the invariant incrementally, and clearing
    // individual rows from L1's zero region does not work either: those clearing UNPACRs advance
    // the config context themselves, so a clear cannot be aimed at the bank holding the rows it
    // was meant to undo. Measured, the window filled up as 0,8 then 0,2,8,10 then
    // 0,2,4,6,8,10,12,14. This makes the invariant local instead: on entry to the writes below,
    // every row of this bank is zero.
    TTI_UNPACR_NOP(SrcA, 0, 0, 0 /* no dvalid */, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    // Eight rows: (l, l+8) of each face. dvalid on the last one only, so the math thread sees a
    // complete SrcA.
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_base = base_in1 + face * kFaceUnits;
        const uint32_t srca_face = face * 16;
        zi_unpack_srca_row(face_base + l * kRowUnits, srca_face + l, 0);
        zi_unpack_srca_row(face_base + (l + 8) * kRowUnits, srca_face + l + 8, face == 3 ? 1 : 0);
    }

    ckernel::t6_semaphore_get(ckernel::semaphore::UNPACK_SYNC);
    ckernel::unpacker::switch_config_context(unp_cfg_context);
}

}  // namespace
#endif

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    state_configure(cb_in1, cb_in0, __builtin_LINE());

    MATH((zi_configure_addrmod()));
    MATH((ckernel::math::reset_counters(ckernel::p_setrwc::SET_ABD_F)));
    UNPACK((zi_unpack_init()));

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                UNPACK((zi_unpack_srcb_tile(cb_in0)));

                for (uint32_t l = 0; l < kLPerTile; ++l) {
                    UNPACK((zi_unpack_srca_rows(cb_in1, l)));
                    MATH((ckernel::math::set_dst_write_addr<ckernel::DstTileShape::Tile32x32,
                                                           ckernel::UnpackDestination::SrcRegs>(0)));
                    MATH((zi_matmul_one_l()));
                }

                // SrcB was fetched once for the tile, so it is released once here. Releasing only
                // one of the two source registers leaves the other permanently valid and the
                // unpacker blocks on the next tile.
                MATH((zi_release_srcb()));

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
