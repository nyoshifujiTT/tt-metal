// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3a (worklog 14): the DRAM lane running beside the matmul chain on the same core -- the NCRISC (NOC1) reads this
// core's weight segments (the balanced NOC1-only assignment: up to 2 segments of (bank, offset, bytes, endpoint tile))
// into a scratch region of its L1, `reps` times back to back with one barrier each, with nothing else on NOC1.  It
// stamps the start and the end (wall clock) into its result page.  It starts on the chain's first go (all cores at
// once) so it overlaps the chain's rounds; the host compares its rate with the lane alone (I-1b', 96.4% on NOC1).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t a = 0;
    const uint32_t buf_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dst = get_arg_val<uint32_t>(a++);
    const uint32_t reps = get_arg_val<uint32_t>(a++);
    const uint32_t res_addr = get_arg_val<uint32_t>(a++);
    const uint32_t res_page = get_arg_val<uint32_t>(a++);
    const uint32_t scratch = get_arg_val<uint32_t>(a++);
    const uint32_t nseg = get_arg_val<uint32_t>(a++);
    uint32_t sb[2], so[2], sl[2], sx[2], sy[2];
    for (uint32_t i = 0; i < 2; ++i) {
        sb[i] = get_arg_val<uint32_t>(a++);
        so[i] = get_arg_val<uint32_t>(a++);
        sl[i] = get_arg_val<uint32_t>(a++);
        sx[i] = get_arg_val<uint32_t>(a++);
        sy[i] = get_arg_val<uint32_t>(a++);
    }
    // the start: when the chain's go semaphore reaches 1 (its first multicast go, every core at once); 0 = start at once
    const uint32_t go_sem_id = get_arg_val<uint32_t>(a++);
    const uint32_t use_go = get_arg_val<uint32_t>(a++);
    constexpr auto res_args = TensorAccessorArgs<0>();
    const auto res = TensorAccessor(res_args, res_addr);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t timeout = 0;
    if (use_go) {
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id));
        uint32_t i = 0;
        for (; i < (1u << 26); ++i) {
            invalidate_l1_cache();
            if (*go >= 1) {
                break;
            }
        }
        timeout = i == (1u << 26) ? 1 : 0;
    }
    const uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t bytes = 0;
    for (uint32_t r = 0; r < reps; ++r) {
        uint32_t d = dst;
        for (uint32_t i = 0; i < nseg; ++i) {
            uint64_t a64 = get_noc_addr_from_bank_id<true>(sb[i], buf_addr + so[i]);
            if (sx[i] != 0) {
                a64 = get_noc_addr(sx[i], sy[i], 0) | (a64 & ((1ull << NOC_ADDR_LOCAL_BITS) - 1));
            }
            noc_async_read(a64, d, sl[i]);
            d += sl[i];
            bytes += sl[i];
        }
        noc_async_read_barrier();
    }
    const uint32_t t1 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    out[0] = t0;
    out[1] = t1;
    out[2] = bytes;
    out[15] = timeout ? 0xdead0000u : 0x600d0000u;
    noc_async_write_page(res_page, res, scratch);
    noc_async_write_barrier();
}
