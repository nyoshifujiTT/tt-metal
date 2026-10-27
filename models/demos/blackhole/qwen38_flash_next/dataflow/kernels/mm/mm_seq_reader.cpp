// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3b (worklog 14): the data movement of one core of the dependent-projection chain (kernels/mm/mm_seq_compute.cpp).
// Layout: cb_act holds 2 halves of Kt + 1 pages of 128 B (Kt 2-row bf16 x tiles and a trailer page), x_s in the half
// s & 1; cb_in_part (owner) 2 x 3 slots of 4,128 B (32-row fp32 partials, rows 0..1 valid), step s in the half s & 1;
// cb_out (owner) pages of 128 B.
// Setup: the weight segments from DRAM (resident for every step), the owner's selector; x_0 from DRAM into half 0 with
// tag 1 in its trailer; the owner zeroes its partial slots and arms their sentinels.  One go for the whole chain.
// Step s: the reader waits until both 16 B lines at the end of the trailer of half s & 1 show tag s + 1 (the hub sends
// the half as one packet; the trailer's lines are the last line of each NIU write stream, tt-isa-documentation
// NoC/Ordering.md, so the whole half has landed) and pushes the half;
//   sender: per segment, sends the packed 2-row fp32 partial (cb_part) to rows 0..1 of faces 0 / 1 of its slot in the
//           owner's cb_in_part;
//   owner:  waits for the sentinels of its slots, pushes cb_in_part, waits for its packed column (cb_out), re-arms the
//           sentinels, writes the column (non-posted) to slot col of the hub's half (s + 1) & 1;
//   hub (one owner): copies its own column, waits until its NIU has completed (79 + 2 n_in) (s + 1) incoming
//           non-posted writes since the go (the other owners' columns and its partials, NIU_SLV_WR_ACK_SENT), writes
//           tag s + 2 in its trailer and sends the half on its persistent linked multicast
//           (Kt + 1 pages, one packet, one path reservation) to the other 129 cores.
// (Worklog I-3b v2: 80 owners each multicasting their column, two path-reserving multicasts per owner per step,
// measured 141 us/step: the concurrent path reservations back off exponentially.  27B rule 105 / Blitz mcast.)
// All waits are bounded (errors counted in the status word).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kSpin = 1u << 24;
constexpr uint32_t kSentinel = 0x7FC0DEADu;
constexpr uint32_t kChunk = 128, kFace1 = 1024, kSlot = 4096 + 32, kTile = 128;
bool wait_min(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
    }
    return false;
}
bool wait_tag(uint32_t trailer, uint32_t tag) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(trailer + 96);
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (t[0] == tag && t[4] == tag) {
            return true;
        }
    }
    return false;
}
bool wait_acks(uint32_t target) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        if (static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, NIU_SLV_WR_ACK_SENT) - target) >= 0) {
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
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t steps = get_compile_time_arg_val(1);
    uint32_t a = 0;
    const uint32_t x0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stamp_addr = get_arg_val<uint32_t>(a++);
    const uint32_t stamp_page = get_arg_val<uint32_t>(a++);
    const uint32_t role = get_arg_val<uint32_t>(a++);
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
    const uint32_t col = get_arg_val<uint32_t>(a++);
    const uint32_t n_in = get_arg_val<uint32_t>(a++);
    const uint32_t sel_addr = get_arg_val<uint32_t>(a++);
    const uint32_t is_hub = get_arg_val<uint32_t>(a++);
    const uint32_t hub_x = get_arg_val<uint32_t>(a++);
    const uint32_t hub_y = get_arg_val<uint32_t>(a++);
    a++;  // the counter semaphore (v2 atomics; unused since the NIU counter)
    const uint32_t n_owners = get_arg_val<uint32_t>(a++);
    constexpr uint32_t Kp = Kt + 1;  // pages per half of cb_act: the Kt x tiles and the trailer page
    const uint32_t n_seg = get_arg_val<uint32_t>(a++);
    uint32_t seg_w0[3], seg_kt[3], seg_x[3], seg_y[3], seg_slot[3];
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_w0[s] = get_arg_val<uint32_t>(a++);
        seg_kt[s] = get_arg_val<uint32_t>(a++);
        seg_x[s] = get_arg_val<uint32_t>(a++);
        seg_y[s] = get_arg_val<uint32_t>(a++);
        seg_slot[s] = get_arg_val<uint32_t>(a++);
    }
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_sel = 5, cb_out = 16, cb_part = 17, cb_stamp = 24,
                       cb_scratch = 25;
    constexpr auto x0_args = TensorAccessorArgs<2>();
    const auto x0 = TensorAccessor(x0_args, x0_addr);
    constexpr auto w_args = TensorAccessorArgs<x0_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    constexpr auto st_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto st = TensorAccessor(st_args, stamp_addr);
    const uint32_t w_tile = get_tile_size(cb_w);
    // weights, selector, x_0
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += seg_kt[s];
    }
    cb_reserve_back(cb_w, w_tiles);
    uint32_t dst = get_write_ptr(cb_w);
    for (uint32_t s = 0; s < n_seg; ++s) {
        for (uint32_t i = 0; i < seg_kt[s]; ++i) {
            noc_async_read_page(seg_w0[s] + i, w, dst);
            dst += w_tile;
        }
    }
    if (role == 0) {
        cb_reserve_back(cb_sel, 1);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, sel_addr), get_write_ptr(cb_sel), get_tile_size(cb_sel));
    }
    const uint32_t act_base = get_write_ptr(cb_act);
    for (uint32_t k = 0; k < Kt; ++k) {
        noc_async_read_page(k, x0, act_base + k * kTile);
    }
    noc_async_read_barrier();
    for (uint32_t h = 0; h < 2; ++h) {
        volatile tt_l1_ptr uint32_t* t =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_base + (h * Kp + Kt) * kTile + 96);
        t[0] = h == 0 ? 1 : 0;
        t[4] = h == 0 ? 1 : 0;
    }
    cb_push_back(cb_w, w_tiles);
    if (role == 0) {
        cb_push_back(cb_sel, 1);
    }
    const uint32_t in_base = get_write_ptr(cb_in_part);
    if (role == 0) {
        for (uint32_t j = 0; j < 6; ++j) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + j * kSlot);
            for (uint32_t q = 0; q < kSlot / 4; ++q) {
                p[q] = 0;
            }
            arm(in_base + j * kSlot);
        }
    }
    // one go for the chain
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done);
    uint32_t errors = 0;
    noc_semaphore_inc(get_noc_addr(go_x, go_y, sem_done), 1);
    const uint64_t mc_go = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_go);
    if (go_core) {
        errors += wait_min(done, n_cores) ? 0 : 1;
        *go = 1;
        noc_semaphore_set_multicast_loopback_src(sem_go, mc_go, n_cores);
    }
    errors += wait_min(go, 1) ? 0 : 0x100;
    // the hub counts the owners' columns by its NIU's completed incoming non-posted writes (NIU_SLV_WR_ACK_SENT,
    // tt-isa-documentation NoC/Counters.md: incremented after the L1 write); the only such writes to the hub on NOC0
    // during the chain are the owners' columns
    // (the go multicast is a non-posted write too: let its acknowledgement settle first; no owner column can arrive
    // within this wait, an owner needs its first step's matmul before its first column)
    const uint32_t settle = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while (reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - settle < 200) {
    }
    const uint32_t ack_base = NOC_STATUS_READ_REG(noc_index, NIU_SLV_WR_ACK_SENT);
    // the hub (an owner: no other NOC0 request of this core during the chain) opens a persistent linked multicast to the
    // grid: the path is reserved once by this 32 B scratch write (NOC_CMD_PATH_RESERVE on the first request of the
    // transaction), every step's half rides the same transaction (linked, no reservation; tt-isa-documentation
    // NoC/MemoryMap.md NOC_CMD_VC_LINKED), an unlinked scratch write closes it before the hub's DRAM writes
    // (DeepSeek Blitz unified_kernels/mcast.hpp, tt-blaze mcast_common.hpp; non-posted: errata BH-85)
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint64_t mc_scr = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, scratch);
    const uint32_t per_step = (n_owners - 1) + 2 * n_in;  // the hub's incoming non-posted writes a step
    cb_reserve_back(cb_stamp, 1);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stamp));
    stamps[0] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint64_t mc_act = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, 0);
    if (is_hub) {
        noc_async_write_multicast(scratch, mc_scr, 32, n_cores - 1, true);
    }
    // profile of one step (kProf): words 64.. of the stamp page, each core's own clock
    constexpr uint32_t kProf = steps > 21 ? 20 : 0xFFFFu;
    auto prof = [&](uint32_t s, uint32_t w) {
        if (s == kProf || (s == kProf + 1 && w == 64)) {
            stamps[(s == kProf + 1) ? 76 : w] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        }
    };
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t h = s & 1, nh = (s + 1) & 1;
        // x_s: the hub's one multicast packet (Kt tiles + trailer) landed when both trailer lines show tag s + 1
        errors += wait_tag(act_base + (h * Kp + Kt) * kTile, s + 1) ? 0 : 0x10000;
        prof(s, 64);
        cb_reserve_back(cb_act, Kp);
        cb_push_back(cb_act, Kp);
        if (role == 1) {
            for (uint32_t g = 0; g < n_seg; ++g) {
                cb_wait_front(cb_part, 1);
                prof(s, 65 + 2 * g);
                const uint32_t src = get_read_ptr(cb_part);
                const uint32_t slot = in_base + ((s & 1) * 3 + seg_slot[g]) * kSlot;
                noc_async_write(src, get_noc_addr(seg_x[g], seg_y[g], slot), kChunk);
                noc_async_write(src + kChunk, get_noc_addr(seg_x[g], seg_y[g], slot + kFace1), kChunk);
                noc_async_writes_flushed();
                prof(s, 66 + 2 * g);
                cb_pop_front(cb_part, 1);
            }
        } else {
            for (uint32_t j = 0; j < n_in; ++j) {
                errors += wait_landed(in_base + ((s & 1) * 3 + j) * kSlot) ? 0 : 0x10000;
            }
            prof(s, 71);
            cb_reserve_back(cb_in_part, 3);
            cb_push_back(cb_in_part, 3);
            cb_wait_front(cb_out, 1);
            prof(s, 72);
            for (uint32_t j = 0; j < n_in; ++j) {
                arm(in_base + ((s & 1) * 3 + j) * kSlot);
            }
            const uint32_t src = get_read_ptr(cb_out);
            if (s + 1 < steps) {
                if (is_hub) {
                    // the hub's own column: a local copy into slot col of its half nh (no NoC request while linked)
                    volatile tt_l1_ptr uint32_t* d =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_base + (nh * Kp + col) * kTile);
                    volatile tt_l1_ptr uint32_t* q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                    for (uint32_t w = 0; w < kTile / 4; ++w) {
                        d[w] = q[w];
                    }
                } else {
                    // the column (non-posted) to slot col of the hub's half nh; the hub counts its NIU's completed
                    // incoming non-posted writes
                    noc_async_write(src, get_noc_addr(hub_x, hub_y, act_base + (nh * Kp + col) * kTile), kTile);
                    noc_async_writes_flushed();
                }
            prof(s, 73);
            } else {
                if (is_hub) {
                    noc_async_write_multicast(scratch, mc_scr, 32, n_cores - 1, false);
                    noc_async_write_barrier();
                }
                noc_async_write_page(col, out, src);
                noc_async_write_barrier();
            }
            if (s + 1 < 120) {
                stamps[1 + s] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            }
            cb_pop_front(cb_out, 1);
        }
        if (is_hub && s + 1 < steps) {
            // all owners' columns of x_{s+1} are in this core's half nh: tag the trailer, one multicast to the others
            errors += wait_acks(ack_base + per_step * (s + 1)) ? 0 : 0x10000;
            prof(s, 74);
            volatile tt_l1_ptr uint32_t* t =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_base + (nh * Kp + Kt) * kTile + 96);
            t[0] = s + 2;
            t[4] = s + 2;
            const uint32_t back = t[4];
            asm volatile("addi x0, %0, 0" : : "r"(back));
            const uint32_t half = act_base + nh * Kp * kTile;
            noc_async_write_multicast(half, mc_act | half, Kp * kTile, n_cores - 1, true);
            noc_async_writes_flushed();
            prof(s, 75);
        }
    }
    stamps[127] = 0x600d0000u | (errors & 0xFFFFu) | (errors >> 16 ? 0x8000u : 0u);
    noc_async_write_page(stamp_page, st, get_write_ptr(cb_stamp));
    noc_async_write_barrier();
}
