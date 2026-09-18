// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// B3: the same matmul again, walked as k -> i -> j with N innermost.
//
// B2 showed the LLK's M split is not required by keeping M contiguous and innermost (k, j, i).
// B3 is the remaining permutation: M outer and N inner (k, i, j). Together they show the order is
// free, not that one particular replacement happens to work.
//
// Same 16 products and same Dst as B1/B2, different issue order, so again not bit-identical -
// judged against the same Higham bound.
//
// Dst needs two different backward moves here (-8 between i steps, -56 on the k step), which one
// marker cannot park directly. Expressed as forward motion instead: the marker walks the pair
// bases 0 -> 8 -> 32 -> 40 (advance +8 or +24) and the j step moves +16 forward from it.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

#ifdef TRISC_MATH
// Mirror of matmul_configure_addrmod() for the full-tile, non-transposed case.
// MVMUL computes D = B*A over 8 SrcB rows at a time: D[8,16] += B[8,16] * A[16,16].
namespace {

inline void custom_matmul_configure_addrmod() {
    constexpr uint32_t fidelity_increment = (MATH_FIDELITY != ckernel::MathFidelity::LoFi) ? 1 : 0;

    // j step (N, innermost): SrcA to the other N face, Dst +16 forward. SrcB held.
    ckernel::addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ckernel::ADDR_MOD_0);

    // i step within a face pair: SrcA back to its marker, SrcB +8, Dst marker +8.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 1},
    }
        .set(ckernel::ADDR_MOD_1);

    // i step crossing to the M16-31 faces: SrcB +24, Dst marker +24.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 24, .clr = 0, .cr = 0},
        .dest = {.incr = 24, .clr = 0, .cr = 1},
    }
        .set(ckernel::ADDR_MOD_2);

    // k step: SrcA marker to the K16-31 faces, SrcB +40, Dst marker cleared to 0.
    ckernel::addr_mod_t{
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 40, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ckernel::ADDR_MOD_3);

    // End of a fidelity phase: reset SrcA/SrcB/Dst, step the fidelity counter.
    ckernel::addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = fidelity_increment, .clr = 0},
    }
        .set(ckernel::ADDR_MOD_5);
}

// The full-tile MVMUL sequence, walked as k -> i -> j with N innermost.
inline void custom_matmul_one_tile() {
    constexpr bool high_fidelity = (MATH_FIDELITY != ckernel::MathFidelity::LoFi);
    constexpr uint32_t phases = high_fidelity ? static_cast<uint32_t>(MATH_FIDELITY) : 1;

    for (uint32_t phase = 0; phase < phases; ++phase) {
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k0 i0 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_1, 0);  // k0 i0 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k0 i1 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_2, 0);  // k0 i1 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k0 i2 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_1, 0);  // k0 i2 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k0 i3 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_3, 0);  // k0 i3 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k1 i0 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_1, 0);  // k1 i0 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k1 i1 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_2, 0);  // k1 i1 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k1 i2 j0
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_1, 0);  // k1 i2 j1
        TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_0, 0);  // k1 i3 j0

        if constexpr (high_fidelity) {
            TTI_MVMUL(ckernel::p_setrwc::CLR_NONE, 0, ckernel::ADDR_MOD_5, 0);
        } else {
            TTI_MVMUL(ckernel::p_setrwc::CLR_A, 0, ckernel::ADDR_MOD_5, 0);
        }
    }

    if constexpr (high_fidelity) {
        TTI_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
    }

    TTI_SETRWC(ckernel::p_setrwc::CLR_B, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
}

}  // namespace
#endif

#ifdef TRISC_UNPACK
namespace {

// In place of llk_unpack_AB_matmul_init for the full-tile, non-transposed case.
inline void custom_unpack_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    // Full 32x32 tiles: 4 faces of 16x16 datums on both operands.
    TT_SETADCXX(p_setadc::UNP_A, 4 * 16 * 16 - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, 4 * 16 * 16 - 1, 0x0);
}

// Mirror of the full-tile path of _llk_unpack_AB_matmul_: one tile of in0 into SrcB and one tile
// of in1 into SrcA. Note the operand-to-section mapping, which is reversed for matmul: in1's
// address goes to SEC0 (feeding SrcA) and in0's to SEC1 (feeding SrcB).
inline void custom_unpack_one_tile(uint32_t cb_id_in0, uint32_t cb_id_in1) {
    volatile uint32_t* cfg = ckernel::get_cfg_pointer();

    // Same address derivation as llk_unpack_AB_matmul: the unpacker takes a 16B-granular address,
    // which is fifo_rd_ptr - 1 in the CB bookkeeping.
    const uint32_t address_in0 = get_local_cb_interface(cb_id_in0).fifo_rd_ptr - 1;
    const uint32_t address_in1 = get_local_cb_interface(cb_id_in1).fifo_rd_ptr - 1;

    ckernel::unpacker::wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_in1, address_in0, cfg);
    ckernel::semaphore_post(ckernel::semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::TRISC_CFG);

    // ct_dim == rt_dim == 1: exactly one SrcB fetch and one SrcA fetch per call. LLK issues the
    // SrcA fetch through a MOP because it has to repeat it ct_dim times and post-increment the L1
    // address between repeats; with a single repeat and the base address rewritten above on every
    // call, the fetch can be issued directly.
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

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

    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse. Same as mm.cpp.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);

    // In place of matmul_init(). state_configure() only sets up the format and tile-size tracking
    // shared by all compute ops; the matmul-specific programming below is ours.
    state_configure(cb_in1, cb_in0, __builtin_LINE());

    MATH((custom_matmul_configure_addrmod()));
    MATH((ckernel::math::reset_counters(ckernel::p_setrwc::SET_ABD_F)));

    UNPACK((custom_unpack_init()));

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                UNPACK((custom_unpack_one_tile(cb_in0, cb_in1)));

                MATH((ckernel::math::set_dst_write_addr<ckernel::DstTileShape::Tile32x32,
                                                       ckernel::UnpackDestination::SrcRegs>(0)));
                MATH((custom_matmul_one_tile()));

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
