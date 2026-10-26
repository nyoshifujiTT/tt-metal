// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-2 (worklog 14): one core's share of a K-split projection on the 130 cores of a chip (README 12.4 step 4: when
// the output tiles are fewer than the cores, K is split so that every core carries work / 130 tile-MACs and the fp32
// partials are added over the NoC), in the Stream-K assignment: the owner of a column computes its K range [0, k_own)
// and adds the partials of the 1..n_in senders of the column; a sender computes 1..3 segments (column, K range) and
// packs each finished segment as one fp32 tile for its reader to send to the segment's owner.
// x (bf16, one tile row, Kt tiles) and the weight blocks (bf8, the segments' K ranges one after the other, K-major per
// segment) are resident; the init of the round runs before the round's go (cb_go, pushed by the reader), as in the
// pass, where the projection's init precedes the arrival of x: the compute kernel reports ready (cb_ready) after the
// init and the reader reports the core ready for the go only then.
// Compile args: role (0 owner, 1 sender), Kt, rounds.  Runtime args: n_seg, then per segment (k0, kt), then n_in (owner:
// the number of partials of its column, <= 3; cb_in_part has 3 slots and moves by 3 pages a round).
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t rounds = get_compile_time_arg_val(2);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_go = 3, cb_ready = 4, cb_out = 16, cb_part = 17;
    constexpr uint32_t cb_dst = role == 1 ? cb_part : cb_out;
    constexpr uint32_t in_slots = 3;  // cb_in_part always moves by its 3 pages a round, so its slots stay at fixed addresses
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
        w_tiles += seg_kt[s];
    }
    cb_wait_front(cb_w, w_tiles);
    for (uint32_t r = 0; r < rounds; ++r) {
        if constexpr (role == 0) {
            reconfig_data_format<SrcOrder::Reverse>(cb_act, cb_w);
        }
        matmul_block_init(cb_act, cb_w, 0, 1, 1, Kt);
        cb_reserve_back(cb_ready, 1);
        cb_push_back(cb_ready, 1);
        cb_wait_front(cb_go, 1);
        cb_pop_front(cb_go, 1);
        uint32_t w0 = 0;
        for (uint32_t s = 0; s < n_seg; ++s) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < seg_kt[s]; ++k) {
                matmul_block(cb_act, cb_w, seg_k0[s] + k, w0 + k, 0, 0, 1, 1, Kt);
            }
            w0 += seg_kt[s];
            if constexpr (role == 0) {
                cb_wait_front(cb_in_part, in_slots);
                reconfig_data_format_srca(cb_in_part);
                reconfig_data_format_srcb(cb_in_part);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in_part);
                for (uint32_t j = 0; j < n_in; ++j) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in_part, j, 0);
                }
                cb_pop_front(cb_in_part, in_slots);
            }
            tile_regs_commit();
            cb_reserve_back(cb_dst, 1);
            tile_regs_wait();
            pack_tile(0, cb_dst);
            tile_regs_release();
            cb_push_back(cb_dst, 1);
        }
    }
}
