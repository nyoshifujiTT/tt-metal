// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c (worklog 14): the reader of the single-core matmul unit-cost bench: copies the activation row (kt tiles) and
// the weight block (kt x ct tiles) from DRAM into their circular buffers once and pushes them (they stay resident:
// the compute kernel indexes them for every repetition and never pops them), then drains the output CB.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t kt = get_arg_val<uint32_t>(3);
    const uint32_t ct = get_arg_val<uint32_t>(4);
    const uint32_t reps = get_arg_val<uint32_t>(5);
    const uint32_t stamp_addr = get_arg_val<uint32_t>(6);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_out = 16, cb_stamp = 24;
    const uint32_t act_tile = get_tile_size(cb_act);
    const uint32_t w_tile = get_tile_size(cb_w);
    const uint32_t out_tile = get_tile_size(cb_out);
    constexpr auto act_args = TensorAccessorArgs<0>();
    const auto act = TensorAccessor(act_args, act_addr);
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    cb_reserve_back(cb_act, kt);
    uint32_t dst = get_write_ptr(cb_act);
    for (uint32_t i = 0; i < kt; ++i) {
        noc_async_read_page(i, act, dst + i * act_tile);
    }
    cb_reserve_back(cb_w, kt * ct);
    dst = get_write_ptr(cb_w);
    for (uint32_t i = 0; i < kt * ct; ++i) {
        noc_async_read_page(i, w, dst + i * w_tile);
    }
    noc_async_read_barrier();
    cb_push_back(cb_act, kt);
    cb_push_back(cb_w, kt * ct);
    for (uint32_t r = 0; r < reps; ++r) {
        cb_wait_front(cb_out, ct);
        if (r + 1 == reps) {
            const uint32_t src = get_read_ptr(cb_out);
            for (uint32_t c = 0; c < ct; ++c) {
                noc_async_write_page(c, out, src + c * out_tile);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, ct);
    }
    cb_wait_front(cb_stamp, 1);
    constexpr auto stamp_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto stamp = TensorAccessor(stamp_args, stamp_addr);
    noc_async_write_page(0, stamp, get_read_ptr(cb_stamp));
    noc_async_write_barrier();
    cb_pop_front(cb_stamp, 1);
}
