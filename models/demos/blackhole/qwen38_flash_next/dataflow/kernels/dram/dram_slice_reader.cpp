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
    // each segment can name its endpoint tile (translated coordinates; 0 = the bank's default NOC0 endpoint) and its
    // NoC (the response path: NOC0 right then down, NOC1 up then left; translated coordinates address the same tile
    // from both NoCs on Blackhole)
    const uint32_t nseg = get_arg_val<uint32_t>(a++);
    constexpr uint32_t role = get_compile_time_arg_val(0);  // 0: BRISC (NOC0, the round sync), 1: NCRISC (NOC1 segments)
    uint32_t sb[2], so[2], sl[2], sx[2], sy[2], sn[2];
    for (uint32_t i = 0; i < 2; ++i) {
        sb[i] = get_arg_val<uint32_t>(a++);
        so[i] = get_arg_val<uint32_t>(a++);
        sl[i] = get_arg_val<uint32_t>(a++);
        sx[i] = get_arg_val<uint32_t>(a++);
        sy[i] = get_arg_val<uint32_t>(a++);
        sn[i] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t sem_local = get_semaphore(get_arg_val<uint32_t>(a++));
    // reads of the slice per round, back to back without any sync between them (a round of `chain` reads after one go
    // measures the stream without the per-round go / done cost)
    const uint32_t chain = get_arg_val<uint32_t>(a++);
    // pacing (I-1b' stream): with period > 0 the q-th read of a round is issued no earlier than go + q x period cycles
    // (wall clock, read by each core from the go time it saw), so every core starts its reads in lockstep
    const uint32_t period = get_arg_val<uint32_t>(a++);
    constexpr auto out_args = TensorAccessorArgs<1>();
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
    // the NCRISC's per-round count of its NOC1 segments, a semaphore (host-initialised to 0) the BRISC polls locally
    volatile tt_l1_ptr uint32_t* local_sync = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_local);
    if constexpr (role == 1) {
        uint32_t has = 0;
        for (uint32_t i = 0; i < nseg; ++i) {
            has |= (sn[i] == 1) ? 1u : 0u;
        }
        for (uint32_t r = 0; r < rounds; ++r) {
            wait_min(go, r + 1);
            if (has && mode == 2) {
                const uint32_t t_go_seen = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                for (uint32_t q = 0; q < chain; ++q) {
                    if (period) {
                        while (reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t_go_seen < q * period) {
                        }
                        if (q > 0) {
                            noc_async_read_barrier();
                        }
                    }
                    uint32_t d = dst;
                    for (uint32_t i = 0; i < nseg; ++i) {
                        if (sn[i] == 1) {
                            uint64_t a64 = get_noc_addr_from_bank_id<true>(sb[i], buf_addr + so[i]);
                            if (sx[i] != 0) {
                                a64 = get_noc_addr(sx[i], sy[i], 0) | (a64 & ((1ull << NOC_ADDR_LOCAL_BITS) - 1));
                            }
                            noc_async_read(a64, d, sl[i]);
                        }
                        d += sl[i];
                    }
                }
                noc_async_read_barrier();
            }
            *local_sync = r + 1;
        }
        return;
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
                // this RISC reads its own NoC's segments (a RISC issues only on its dedicated NoC: the firmware sets up
                // the counters of that NoC alone, brisc.cc noc_local_state_init(noc_index)); the other RISC reads the rest
                const uint32_t t_go_seen = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
                for (uint32_t q = 0; q < chain; ++q) {
                    if (period) {
                        while (reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t_go_seen < q * period) {
                        }
                        if (q > 0) {
                            noc_async_read_barrier();
                        }
                    }
                    d = dst;
                    for (uint32_t i = 0; i < nseg; ++i) {
                        if (sn[i] == role) {
                            uint64_t a64 = get_noc_addr_from_bank_id<true>(sb[i], buf_addr + so[i]);
                            if (sx[i] != 0) {
                                a64 = get_noc_addr(sx[i], sy[i], 0) | (a64 & ((1ull << NOC_ADDR_LOCAL_BITS) - 1));
                            }
                            noc_async_read(a64, d, sl[i]);
                        }
                        d += sl[i];
                    }
                }
                noc_async_read_barrier();
                if (role == 0) {
                    // wait for the NCRISC's NOC1 segments of this round (its count in the local sync word)
                    errors += wait_min(local_sync, r + 1) ? 0 : 0x1000000;
                }
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
