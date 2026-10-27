// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-7 (worklog 14): single-core bench of the B=1 matmul unit (kernels/mm/mm_b1_unit.h) at the served fidelity:
// in0 = one tiny tile row (rows <= 8) x kt K tiles (bf16), in1 = ct columns x kt K tiles (column-major, Bfp8_b HiFi2 /
// Bfp4_b LoFi), fp32 DEST; reps repetitions on the resident inputs, the pack thread stamps the wall clock before the
// first acquire and after the last pack.  Compile args: kt, ct, reps, phases.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "mm_b1_unit.h"

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t phases = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_out = 16, cb_stamp = 24;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    UNPACK((llk_unpack_AB_custom_mm_init<false>(cb_act, cb_w, 1)));
    MATH((qwen38_b1::b1_mm_math_init<phases>()));
    cb_wait_front(cb_act, kt);
    cb_wait_front(cb_w, kt * ct);
    uint32_t t0 = 0;
    PACK({ t0 = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L); });
    for (uint32_t r = 0; r < reps; ++r) {
        tile_regs_acquire();
        for (uint32_t c = 0; c < ct; ++c) {
            custom_mm_block_unpack<false, true>(cb_act, cb_w, 0, c * kt, kt, 1);
            MATH((qwen38_b1::b1_mm_math_run(c, kt)));
        }
        MATH((qwen38_b1::b1_mm_math_finalize<phases>(ct)));
        tile_regs_commit();
        cb_reserve_back(cb_out, ct);
        tile_regs_wait();
        for (uint32_t c = 0; c < ct; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, ct);
    }
    cb_reserve_back(cb_stamp, 1);
    PACK({
        const uint32_t t1 = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_stamp).fifo_wr_ptr << cb_addr_shift);
        s[0] = t0;
        s[1] = t1;
        s[2] = 0x600d0000u;
    });
    cb_push_back(cb_stamp, 1);
}
