// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-6 (worklog 14): the K-split projection with column pairs.  The output columns go in pairs (ct = 2: one unpack of
// the activation tile serves both columns' MVMULs); an owner core holds a pair over K [0, k_own) and adds the pair's
// partials; a sender core holds 1..3 segments (pair, K range) of the pair-major stream and packs each segment's two
// fp32 partials (2-row tiles, CB 17) for its reader to send.  The owner adds a partial P (32-row fp32 slot, rows 0..1
// valid) by the matmul with the 2-row selector (sel x P, I-1c-5): P's pair is column c0 and c1 in DEST tiles 0 / 1.
// Activation, output and selector are 2-row tiny tiles (shaped on the host).  The matmul init runs once.
// Compile args: role (0 owner, 1 sender), Kt, rounds.  Runtime args: n_seg, per segment (k0, kt), n_in (owner: the
// partials per column, <= 3; cb_in_part holds 2 x 4 slots: column 0's four then column 1's four).
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
    constexpr uint32_t rounds = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_go = 3, cb_ready = 4, cb_sel = 5, cb_out = 16, cb_part = 17;
    constexpr uint32_t cb_dst = role == 1 ? cb_part : cb_out;
    constexpr uint32_t kPerCol = 4, in_slots = 2 * kPerCol;
    const uint32_t n_seg = get_arg_val<uint32_t>(0);
    uint32_t seg_k0[3], seg_kt[3];
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_k0[s] = get_arg_val<uint32_t>(1 + 2 * s);
        seg_kt[s] = get_arg_val<uint32_t>(2 + 2 * s);
    }
    const uint32_t n_in = get_arg_val<uint32_t>(1 + 2 * n_seg);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_dst);
    cb_wait_front(cb_act, Kt);
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += 2 * seg_kt[s];
    }
    cb_wait_front(cb_w, w_tiles);
    if constexpr (role == 0) {
        cb_wait_front(cb_sel, 1);
    }
    matmul_block_init(cb_act, cb_w, 0, 2, 1, Kt);
    for (uint32_t r = 0; r < rounds; ++r) {
        cb_reserve_back(cb_ready, 1);
        cb_push_back(cb_ready, 1);
        cb_wait_front(cb_go, 1);
        cb_pop_front(cb_go, 1);
        uint32_t w0 = 0;
        for (uint32_t s = 0; s < n_seg; ++s) {
            tile_regs_acquire();
            // the weight block of a segment: kt rows of the pair (column c0, c1 interleaved per K row: index 2k, 2k+1)
            for (uint32_t k = 0; k < seg_kt[s]; ++k) {
                matmul_block(cb_act, cb_w, seg_k0[s] + k, w0 + 2 * k, 0, 0, 2, 1, Kt);
            }
            w0 += 2 * seg_kt[s];
            if constexpr (role == 0) {
                cb_wait_front(cb_in_part, in_slots);
                reconfig_data_format_srca(cb_in_part);
                for (uint32_t j = 0; j < n_in; ++j) {
                    matmul_block(cb_sel, cb_in_part, 0, j, 0, 0, 1, 1, Kt);
                    matmul_block(cb_sel, cb_in_part, 0, kPerCol + j, 1, 0, 1, 1, Kt);
                }
                reconfig_data_format_srca(cb_w);
                cb_pop_front(cb_in_part, in_slots);
            }
            tile_regs_commit();
            cb_reserve_back(cb_dst, 2);
            tile_regs_wait();
            pack_tile(0, cb_dst);
            pack_tile(1, cb_dst);
            tile_regs_release();
            cb_push_back(cb_dst, 2);
        }
    }
}
