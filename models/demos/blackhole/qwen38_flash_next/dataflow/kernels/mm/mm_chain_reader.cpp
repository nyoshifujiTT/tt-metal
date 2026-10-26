// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-2 (worklog 14): the data movement of one core of the K-split projection (kernels/mm/mm_chain_compute.cpp).
// Setup: x (Kt bf16 tiles) and the core's weight segments (bf8, each a contiguous tile range of the column-major weight)
// are copied from DRAM into their CBs once; an owner zeroes the trailers of its partial slots.
// Round r: once the compute kernel has run the round's init (cb_ready), the core increments the done count on the go core, which multicasts go = r + 1 to the 130 cores (with
// itself) once all have; the core stamps the go, pushes cb_go to the compute kernel, and then
//   sender: for every segment, waits for the packed fp32 partial (cb_part, a page of 4,096 B + a 32 B trailer), writes
//           the round's tag into the trailer's two 16 B lines, reads the store back (so the NIU reads the stored tag,
//           the rule of the Ethernet kernels' trailers, worklog 10-26 I-1e) and sends the page in one NoC packet to its
//           slot in the owner's cb_in_part.  The two trailer lines fall in the two write streams of the receiving NIU
//           (16 B mod 32) and within a stream the writes of one packet land in ascending address order
//           (tt-isa-documentation NoC/Ordering.md), so both tags visible = the whole tile landed.  No ack wait: the
//           page is released once the write has left L1 (noc_async_writes_flushed);
//   owner:  polls the trailers of its n_in slots for the round's tags, pushes cb_in_part (always its 3 pages, so that
//           the slots keep the addresses every core computes from its own cb_in_part base), waits for the finished
//           column (cb_out) and stamps it; on the last round it writes the column to the output buffer.
// Stamps of the first 40 rounds (one 512 B page per core): go seen; sender: its last partial packed, owner: all its
// partials landed; sender: its last partial sent, owner: its column packed.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kSpin = 1u << 24;  // a bounded wait gives up after 2^24 polls (~0.1 s) and flags the round
bool wait_min(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
    }
    return false;
}
bool wait_tag(volatile tt_l1_ptr uint32_t* t, uint32_t tag) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (t[0] == tag && t[4] == tag) {
            return true;
        }
    }
    return false;
}
}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t act_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stamp_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stamp_page = get_arg_val<uint32_t>(a++);
    const uint32_t role = get_arg_val<uint32_t>(a++);
    const uint32_t rounds = get_arg_val<uint32_t>(a++);
    const uint32_t sem_go = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t sem_done = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t go_core = get_arg_val<uint32_t>(a++);
    const uint32_t n_cores = get_arg_val<uint32_t>(a++);
    const uint32_t mc_x0 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_y0 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_x1 = get_arg_val<uint32_t>(a++);
    const uint32_t mc_y1 = get_arg_val<uint32_t>(a++);
    const uint32_t go_x = get_arg_val<uint32_t>(a++);
    const uint32_t go_y = get_arg_val<uint32_t>(a++);
    const uint32_t n_seg = get_arg_val<uint32_t>(a++);
    uint32_t seg_w0[3], seg_kt[3], seg_x[3], seg_y[3], seg_slot[3];
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_w0[s] = get_arg_val<uint32_t>(a++);
        seg_kt[s] = get_arg_val<uint32_t>(a++);
        seg_x[s] = get_arg_val<uint32_t>(a++);
        seg_y[s] = get_arg_val<uint32_t>(a++);
        seg_slot[s] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_in = get_arg_val<uint32_t>(a++);
    const uint32_t out_page = get_arg_val<uint32_t>(a++);
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t part_page = get_compile_time_arg_val(1);
    constexpr uint32_t part_bytes = 4096;
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_go = 3, cb_ready = 4, cb_out = 16, cb_part = 17, cb_stamp = 24;
    constexpr auto act_args = TensorAccessorArgs<2>();
    const auto act = TensorAccessor(act_args, act_addr);
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto st = TensorAccessor(st_args, stamp_addr);
    const uint32_t act_tile = get_tile_size(cb_act);
    const uint32_t w_tile = get_tile_size(cb_w);
    cb_reserve_back(cb_act, Kt);
    uint32_t dst = get_write_ptr(cb_act);
    for (uint32_t i = 0; i < Kt; ++i) {
        noc_async_read_page(i, act, dst + i * act_tile);
    }
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += seg_kt[s];
    }
    cb_reserve_back(cb_w, w_tiles);
    dst = get_write_ptr(cb_w);
    for (uint32_t s = 0; s < n_seg; ++s) {
        for (uint32_t i = 0; i < seg_kt[s]; ++i) {
            noc_async_read_page(seg_w0[s] + i, w, dst);
            dst += w_tile;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_act, Kt);
    cb_push_back(cb_w, w_tiles);
    const uint32_t in_base = get_write_ptr(cb_in_part);
    if (role == 0) {
        for (uint32_t j = 0; j < 3; ++j) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + j * part_page + part_bytes);
            t[0] = 0;
            t[4] = 0;
        }
    }
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done);
    const uint64_t go_done = get_noc_addr(go_x, go_y, sem_done);
    const uint64_t mc_go = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_go);
    cb_reserve_back(cb_stamp, 1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stamp));
    uint32_t errors = 0;
    for (uint32_t r = 0; r < rounds; ++r) {
        cb_wait_front(cb_ready, 1);
        cb_pop_front(cb_ready, 1);
        noc_semaphore_inc(go_done, 1);
        if (go_core) {
            errors += wait_min(done, (r + 1) * n_cores) ? 0 : 1;
            *go = r + 1;
            noc_semaphore_set_multicast_loopback_src(sem_go, mc_go, n_cores);
        }
        errors += wait_min(go, r + 1) ? 0 : 0x100;
        const uint32_t t_go = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        cb_reserve_back(cb_go, 1);
        cb_push_back(cb_go, 1);
        uint32_t t_mid = 0, t_end = 0;
        const uint32_t tag = ((r + 1) << 4) | 0x5u;
        if (role == 1) {
            for (uint32_t s = 0; s < n_seg; ++s) {
                cb_wait_front(cb_part, 1);
                t_mid = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                const uint32_t src = get_read_ptr(cb_part);
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + part_bytes);
                t[0] = tag;
                t[4] = tag;
                const uint32_t back = t[4];
                asm volatile("addi x0, %0, 0" : : "r"(back));
                noc_async_write(src, get_noc_addr(seg_x[s], seg_y[s], in_base + seg_slot[s] * part_page), part_bytes + 32);
                noc_async_writes_flushed();
                cb_pop_front(cb_part, 1);
            }
            t_end = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        } else {
            for (uint32_t j = 0; j < n_in; ++j) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + j * part_page + part_bytes);
                errors += wait_tag(t, tag) ? 0 : 0x10000;
            }
            t_mid = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            cb_reserve_back(cb_in_part, 3);
            cb_push_back(cb_in_part, 3);
            cb_wait_front(cb_out, 1);
            t_end = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            if (r + 1 == rounds) {
                noc_async_write_page(out_page, out, get_read_ptr(cb_out));
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, 1);
        }
        if (r < 40) {
            stamps[3 * r] = t_go;
            stamps[3 * r + 1] = t_mid;
            stamps[3 * r + 2] = t_end;
        }
    }
    stamps[127] = 0x600d0000u | (errors & 0xFFFFu) | (errors >> 16 ? 0x8000u : 0u);
    noc_async_write_page(stamp_page, st, get_write_ptr(cb_stamp));
    noc_async_write_barrier();
}
