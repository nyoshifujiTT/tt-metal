// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c (worklog 14): the single-core matmul unit-cost bench: an activation row (1 tile row: M = 32, the decode rows
// in it) times a weight block of kt x ct tiles, accumulated in the fp32 DEST over the kt K tiles with one
// matmul_block call per K tile (ct output columns a call, rt 1), packed, repeated reps times on the resident inputs.
// The pack thread stamps the wall clock before the first repetition's acquire and after the last pack (into CB 24, which
// the reader copies to DRAM); the cycles per tile-MAC are (end - start) / (reps x kt x ct).
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t ct = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_out = 16, cb_stamp = 24;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    matmul_block_init(cb_act, cb_w, 0, ct, 1, kt);
    cb_wait_front(cb_act, kt);
    cb_wait_front(cb_w, kt * ct);
    uint32_t t0 = 0;
    PACK({ t0 = *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L); });
    for (uint32_t r = 0; r < reps; ++r) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_act, cb_w, k, k * ct, 0, 0, ct, 1, kt);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, ct);
        tile_regs_wait();
        for (uint32_t c = 0; c < ct; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, ct);
    }
    // the stamps go to a one-page CB the reader forwards to DRAM (a CB's write pointer is valid on the pack thread)
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
