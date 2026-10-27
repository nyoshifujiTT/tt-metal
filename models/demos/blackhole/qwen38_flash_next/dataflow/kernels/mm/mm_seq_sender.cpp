// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3b v6 (worklog 14): the NOC1 sends of one core of the dependent-projection chain, on NCRISC: a sender's packed
// 2-row fp32 partials (cb_part) to their owners' slots, an owner's column (cb_out) to the hub's next half of x.  The
// BRISC reader (mm_seq_reader.cpp) keeps NOC0 for the hub's persistent linked multicast and the polls.  Moving these
// unicasts off NOC0 tests whether the linked multicast's held VCs slow the NOC0 unicasts (v3a / v4: the partial's
// flight grew from ~120 to ~210 cycles after the linked multicast came in).
// Runtime args: role, n_seg, per segment (x, y, slot), hub_x, hub_y, col, is_hub, Kt, steps (the CB L1 addresses are
// the same on every core: read from the local interfaces of cb_in_part / cb_act, which NCRISC has set up too).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_part = 17, cb_out = 16;
    constexpr uint32_t kChunk = 128, kFace1 = 1024, kSlot = 4096 + 32, kTile = 128, kSlotsHalf = 4;
    uint32_t a = 0;
    const uint32_t role = get_arg_val<uint32_t>(a++);
    const uint32_t n_seg = get_arg_val<uint32_t>(a++);
    uint32_t seg_x[3], seg_y[3], seg_slot[3];
    for (uint32_t g = 0; g < n_seg; ++g) {
        seg_x[g] = get_arg_val<uint32_t>(a++);
        seg_y[g] = get_arg_val<uint32_t>(a++);
        seg_slot[g] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hub_x = get_arg_val<uint32_t>(a++);
    const uint32_t hub_y = get_arg_val<uint32_t>(a++);
    const uint32_t col = get_arg_val<uint32_t>(a++);
    const uint32_t is_hub = get_arg_val<uint32_t>(a++);
    constexpr uint32_t cb_act = 0, cb_in_part = 2;
    const uint32_t in_base = get_write_ptr(cb_in_part);
    const uint32_t act_base = get_write_ptr(cb_act);
    const uint32_t Kt = get_arg_val<uint32_t>(a++);
    const uint32_t steps = get_arg_val<uint32_t>(a++);
    const uint32_t Kp = Kt + 1;
    for (uint32_t s = 0; s < steps; ++s) {
        if (role == 1) {
            for (uint32_t g = 0; g < n_seg; ++g) {
                cb_wait_front(cb_part, 1);
                const uint32_t src = get_read_ptr(cb_part);
                const uint32_t slot = in_base + ((s & 1) * kSlotsHalf + seg_slot[g]) * kSlot;
                noc_async_write(src, get_noc_addr(seg_x[g], seg_y[g], slot), kChunk);
                noc_async_write(src + kChunk, get_noc_addr(seg_x[g], seg_y[g], slot + kFace1), kChunk);
                noc_async_writes_flushed();
                cb_pop_front(cb_part, 1);
            }
        } else if (!is_hub && s + 1 < steps) {
            cb_wait_front(cb_out, 1);
            noc_async_write(
                get_read_ptr(cb_out), get_noc_addr(hub_x, hub_y, act_base + (((s + 1) & 1) * Kp + col) * kTile), kTile);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
