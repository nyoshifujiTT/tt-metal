// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-6 (worklog 14): the data movement of one core of the column-pair K-split projection
// (kernels/mm/mm_chain6_compute.cpp).  Setup: x (Kt 2-row bf16 tiles), the weight segments (bf8; per segment the pair's
// kt K rows, columns interleaved: tile (k, c0), (k, c1), ...) and, on an owner, the 2-row selector, from DRAM into the
// CBs once; an owner zeroes its 8 partial slots (32-row fp32 tiles, 2 columns x 4) and arms their sentinels.
// Round r: after the compute kernel's init (cb_ready) the core increments the go core's done count; the go core
// multicasts go = r + 1 once all have; the core stamps the go and pushes cb_go, then
//   sender: per segment, waits for the two packed 2-row fp32 partials (CB 17, 256 B each: face 0 rows 0..1 at +0, face 1
//           rows 0..1 at +128) and sends each partial's two 128 B chunks to rows 0..1 of faces 0 / 1 of its slot in the
//           owner's cb_in_part (column j's slot: (j x 4 + seg_slot) x 4,096 B);
//   owner:  waits for the sentinels of its 2 x n_in slots to be overwritten (the last 16 B line of each write stream of
//           each chunk: within a stream one packet's writes land in ascending address order, 16 B atomic --
//           tt-isa-documentation NoC/Ordering.md), pushes cb_in_part (8 pages), waits for the two packed columns, stamps,
//           re-arms the sentinels; on the last round writes the columns to the output buffer.
// Stamps of rounds 0..39: go seen, mid (sender: last partials packed, owner: partials landed), end.  All waits bounded.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kSpin = 1u << 24;
constexpr uint32_t kSentinel = 0x7FC0DEADu;
constexpr uint32_t kChunk = 128, kFace1 = 1024, kSlot = 4096, kPerCol = 4;
bool wait_min(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
    }
    return false;
}
bool wait_landed(uint32_t slot) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (w[96 / 4] != kSentinel && w[112 / 4] != kSentinel && w[(kFace1 + 96) / 4] != kSentinel &&
            w[(kFace1 + 112) / 4] != kSentinel) {
            return true;
        }
    }
    return false;
}
void arm(uint32_t slot) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    w[96 / 4] = kSentinel;
    w[112 / 4] = kSentinel;
    w[(kFace1 + 96) / 4] = kSentinel;
    w[(kFace1 + 112) / 4] = kSentinel;
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
    const uint32_t sel_addr = get_arg_val<uint32_t>(a++);
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_go = 3, cb_ready = 4, cb_sel = 5, cb_out = 16, cb_part = 17,
                       cb_stamp = 24;
    constexpr auto act_args = TensorAccessorArgs<1>();
    const auto act = TensorAccessor(act_args, act_addr);
    constexpr auto w_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto st = TensorAccessor(st_args, stamp_addr);
    const uint32_t act_tile = get_tile_size(cb_act);
    const uint32_t w_tile = get_tile_size(cb_w);
    const uint32_t out_tile = get_tile_size(cb_out);
    cb_reserve_back(cb_act, Kt);
    uint32_t dst = get_write_ptr(cb_act);
    for (uint32_t i = 0; i < Kt; ++i) {
        noc_async_read_page(i, act, dst + i * act_tile);
    }
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += 2 * seg_kt[s];
    }
    cb_reserve_back(cb_w, w_tiles);
    dst = get_write_ptr(cb_w);
    for (uint32_t s = 0; s < n_seg; ++s) {
        for (uint32_t i = 0; i < 2 * seg_kt[s]; ++i) {
            noc_async_read_page(seg_w0[s] + i, w, dst);
            dst += w_tile;
        }
    }
    if (role == 0) {
        cb_reserve_back(cb_sel, 1);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, sel_addr), get_write_ptr(cb_sel), get_tile_size(cb_sel));
    }
    noc_async_read_barrier();
    cb_push_back(cb_act, Kt);
    cb_push_back(cb_w, w_tiles);
    if (role == 0) {
        cb_push_back(cb_sel, 1);
    }
    const uint32_t in_base = get_write_ptr(cb_in_part);
    if (role == 0) {
        for (uint32_t j = 0; j < 2 * kPerCol; ++j) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + j * kSlot);
            for (uint32_t q = 0; q < kSlot / 4; ++q) {
                p[q] = 0;
            }
            arm(in_base + j * kSlot);
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
        if (role == 1) {
            for (uint32_t s = 0; s < n_seg; ++s) {
                cb_wait_front(cb_part, 2);
                t_mid = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                const uint32_t src = get_read_ptr(cb_part);
                const uint32_t part = get_tile_size(cb_part);
                for (uint32_t j = 0; j < 2; ++j) {
                    const uint32_t slot = in_base + (j * kPerCol + seg_slot[s]) * kSlot;
                    const uint32_t p = src + j * part;
                    noc_async_write(p, get_noc_addr(seg_x[s], seg_y[s], slot), kChunk);
                    noc_async_write(p + kChunk, get_noc_addr(seg_x[s], seg_y[s], slot + kFace1), kChunk);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_part, 2);
            }
            t_end = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        } else {
            for (uint32_t j = 0; j < 2; ++j) {
                for (uint32_t i = 0; i < n_in; ++i) {
                    errors += wait_landed(in_base + (j * kPerCol + i) * kSlot) ? 0 : 0x10000;
                }
            }
            t_mid = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            cb_reserve_back(cb_in_part, 2 * kPerCol);
            cb_push_back(cb_in_part, 2 * kPerCol);
            cb_wait_front(cb_out, 2);
            t_end = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            for (uint32_t j = 0; j < 2; ++j) {
                for (uint32_t i = 0; i < n_in; ++i) {
                    arm(in_base + (j * kPerCol + i) * kSlot);
                }
            }
            if (r + 1 == rounds) {
                noc_async_write_page(2 * out_page, out, get_read_ptr(cb_out));
                noc_async_write_page(2 * out_page + 1, out, get_read_ptr(cb_out) + out_tile);
                noc_async_write_barrier();
            }
            cb_pop_front(cb_out, 2);
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
