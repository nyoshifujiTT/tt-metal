// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1b' (worklog 14): the DRAM lane of one chip as the design uses it -- every one of the 130 cores reads its own weight
// slice (s bytes, contiguous in one DRAM bank) straight into its L1, all cores at once.  Round r: the go core takes its
// clock, multicasts go = r + 1 to the 130 cores (itself included), every core reads its slice (noc_async_read of the
// whole slice: the NIU cuts it into maximum-burst packets and keeps them in flight; one barrier at the end -- mode 0 --
// or 27B section 601's pipeline of stateful maximum-burst reads with three transaction ids in rotation -- mode 1) and
// increments the done count on the go core; the go core takes its clock when all 130 have.  Both stamps on one clock
// (the go core's), so the round time carries no clock offset between cores.  All waits are bounded.  After the last
// round every core sums its slice (32-bit words) for the host's check.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kSpin = 1u << 26;
bool wait_min(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < kSpin; ++i) {
        invalidate_l1_cache();
        if (*p >= v) {
            return true;
        }
    }
    return false;
}
}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t buf_addr = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    const uint32_t bank_off = get_arg_val<uint32_t>(a++);
    const uint32_t slice = get_arg_val<uint32_t>(a++);
    const uint32_t dst = get_arg_val<uint32_t>(a++);
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
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_page = get_arg_val<uint32_t>(a++);
    const uint32_t scratch = get_arg_val<uint32_t>(a++);
    const uint32_t mode = get_arg_val<uint32_t>(a++);   // 0: one noc_async_read of the slice, 1: 27B's TRID pipeline
    const uint32_t page = get_arg_val<uint32_t>(a++);   // mode 1: bytes per read packet (slice is a multiple)
    const uint32_t group = get_arg_val<uint32_t>(a++);  // mode 1: packets per transaction id
    const uint32_t vc = get_arg_val<uint32_t>(a++);     // mode 1: the request VC
    // mode 2: the slice as up to 2 segments (bank, byte offset in the buffer's bank region, bytes), each read with one
    // noc_async_read (worklog 14 I-1b': the link-load LP's assignment of slices to banks)
    const uint32_t nseg = get_arg_val<uint32_t>(a++);
    uint32_t sb[2], so[2], sl[2];
    for (uint32_t i = 0; i < 2; ++i) {
        sb[i] = get_arg_val<uint32_t>(a++);
        so[i] = get_arg_val<uint32_t>(a++);
        sl[i] = get_arg_val<uint32_t>(a++);
    }
    // mode 2 with an explicit endpoint tile (translated coordinates; 0 = the bank's default NOC0 endpoint): the
    // row-local assignment reads every bank through the tile in the reader's own NOC0 row
    const uint32_t ep_x = get_arg_val<uint32_t>(a++);
    const uint32_t ep_y = get_arg_val<uint32_t>(a++);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done);
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank, buf_addr + bank_off);
    const uint64_t src_lo = get_noc_addr_from_bank_id<true>(bank, buf_addr);
    if (mode == 1 && slice) {
        noc_async_read_one_packet_set_state<true>(src_lo, page, vc);
    }
    const uint64_t go_done = get_noc_addr(go_x, go_y, sem_done);
    const uint64_t mc_go = get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_go);
    uint32_t errors = 0;
    for (uint32_t i = 0; i < 64; ++i) {
        res[i] = 0;
    }
    for (uint32_t r = 0; r < rounds; ++r) {
        if (go_core) {
            if (r > 0) {
                errors += wait_min(done, r * n_cores) ? 0 : 1;
                res[2 * (r - 1) + 1] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            }
            res[2 * r] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            *go = r + 1;
            noc_semaphore_set_multicast_loopback_src(sem_go, mc_go, n_cores);
        }
        errors += wait_min(go, r + 1) ? 0 : 0x100;
        if (slice) {
            if (mode == 0) {
                noc_async_read(src, dst, slice);
                noc_async_read_barrier();
            } else if (mode == 2) {
                uint32_t d = dst;
                for (uint32_t i = 0; i < nseg; ++i) {
                    uint64_t a64 = get_noc_addr_from_bank_id<true>(sb[i], buf_addr + so[i]);
                    if (ep_x != 0) {
                        a64 = get_noc_addr(ep_x, ep_y, 0) | (a64 & ((1ull << NOC_ADDR_LOCAL_BITS) - 1));
                    }
                    noc_async_read(a64, d, sl[i]);
                    d += sl[i];
                }
                noc_async_read_barrier();
            } else {
                // 27B section 601's pipeline: maximum-burst pages issued with the stateful read on a fixed VC, a group
                // of pages per transaction id, three ids in rotation, only the oldest id drained before reusing it
                uint32_t off = 0, d = dst, trid = 1, oldest = 1, open = 0;
                while (off < slice) {
                    noc_async_read_set_trid(trid);
                    for (uint32_t h = 0; h < group && off < slice; ++h) {
                        noc_async_read_one_packet_with_state_with_trid(src_lo, bank_off + off, d, trid);
                        off += page;
                        d += page;
                    }
                    if (open == 2) {
                        noc_async_read_barrier_with_trid(oldest);
                        oldest = oldest == 3 ? 1 : oldest + 1;
                    } else {
                        ++open;
                    }
                    trid = trid == 3 ? 1 : trid + 1;
                }
                for (uint32_t t = 1; t <= 3; ++t) {
                    noc_async_read_barrier_with_trid(t);
                }
            }
        }
        noc_semaphore_inc(go_done, 1);
    }
    if (go_core) {
        errors += wait_min(done, rounds * n_cores) ? 0 : 1;
        res[2 * (rounds - 1) + 1] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    uint32_t sum = 0;
    uint32_t got = slice;
    if (mode == 2) {
        got = 0;
        for (uint32_t i = 0; i < nseg; ++i) {
            got += sl[i];
        }
    }
    for (uint32_t i = 0; i < got / 4; ++i) {
        sum += w[i];
    }
    res[60] = sum;
    res[61] = errors;
    res[63] = 0x600d0000u;
    noc_async_write_page(out_page, out, scratch);
    noc_async_write_barrier();
}
