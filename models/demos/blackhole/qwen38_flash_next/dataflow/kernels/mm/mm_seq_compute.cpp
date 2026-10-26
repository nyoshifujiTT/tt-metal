// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3b (worklog 14): a chain of dependent projections x_{s+1} = x_s W on the 130 cores of one chip (K = N = Kt tiles,
// bf8 HiFi2, fp32 DEST, 2-row tiny tiles), Stream-K as I-1c-5 v9.  Step s: the activation x_s arrives tile by tile in
// cb_act (the reader pushes each K tile when it has landed); the core runs its segments (column, K range) over the
// landed tiles; an owner waits for its column's partials (cb_in_part, 32-row fp32 slots) and adds them with the matmul
// against the 2-row selector (cb_sel), then packs the finished column (2-row bf16) to cb_out for its reader to
// multicast as x_{s+1}'s tile; a sender packs each segment's fp32 partial (2-row, cb_part) for its reader to send.
// The weight segments stay resident (the same W every step); the matmul init runs once.
// Compile args: role (0 owner, 1 sender), Kt, steps.  Runtime args: n_seg, per segment (k0, kt), n_in.
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t steps = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_sel = 5, cb_out = 16, cb_part = 17;
    constexpr uint32_t cb_dst = role == 1 ? cb_part : cb_out;
    constexpr uint32_t in_slots = 3;
    const uint32_t n_seg = get_arg_val<uint32_t>(0);
    uint32_t seg_k0[3], seg_kt[3];
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_k0[s] = get_arg_val<uint32_t>(1 + 2 * s);
        seg_kt[s] = get_arg_val<uint32_t>(2 + 2 * s);
    }
    const uint32_t n_in = get_arg_val<uint32_t>(1 + 2 * n_seg);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_dst);
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += seg_kt[s];
    }
    cb_wait_front(cb_w, w_tiles);
    if constexpr (role == 0) {
        cb_wait_front(cb_sel, 1);
    }
    matmul_block_init(cb_act, cb_w, 0, 1, 1, Kt);
    for (uint32_t st = 0; st < steps; ++st) {
        // x_s: Kt tiles, each pushed by the reader as it lands; the core's segments read the K tiles they need, so
        // wait up to the last K tile any of its segments uses, in segment order (the tiles land in K order)
        uint32_t w0 = 0;
        uint32_t have = 0;
        for (uint32_t s = 0; s < n_seg; ++s) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < seg_kt[s]; ++k) {
                const uint32_t kk = seg_k0[s] + k;
                if (kk + 1 > have) {
                    cb_wait_front(cb_act, kk + 1);
                    have = kk + 1;
                }
                matmul_block(cb_act, cb_w, kk, w0 + k, 0, 0, 1, 1, Kt);
            }
            w0 += seg_kt[s];
            if constexpr (role == 0) {
                cb_wait_front(cb_in_part, in_slots);
                reconfig_data_format_srca(cb_in_part);
                for (uint32_t j = 0; j < n_in; ++j) {
                    matmul_block(cb_sel, cb_in_part, 0, j, 0, 0, 1, 1, Kt);
                }
                reconfig_data_format_srca(cb_w);
                cb_pop_front(cb_in_part, in_slots);
            }
            tile_regs_commit();
            cb_reserve_back(cb_dst, 1);
            tile_regs_wait();
            pack_tile(0, cb_dst);
            tile_regs_release();
            cb_push_back(cb_dst, 1);
        }
        // the whole x_s has to be consumed before the next step's tiles reuse the CB: pop every tile of the step
        cb_wait_front(cb_act, Kt + 1);
        cb_pop_front(cb_act, Kt + 1);
    }
}
