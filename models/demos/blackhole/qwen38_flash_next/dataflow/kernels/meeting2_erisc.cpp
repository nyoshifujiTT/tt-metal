// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The group meeting, version 2 (README 12.5, worklog 14 I-1b'): the all-gather of one chunk per chip along the BFS
// trees of the 2 x 4 block, sent directly over the links in two planes.  A direction toward a neighbour is two channels
// (the two Ethernet cores of the link, I-1d: 24.31 GB/s each, 48.62 together); the channels of a link come in pairs
// (2k, 2k+1) and the peer keeps the parity, so plane p = the cores of channel parity p is a full copy of the torus with
// one channel per link-direction.  Chunk half p travels in plane p only (the packet model's even cut, comm_model.py
// revision 5), and a relay stays between cores of the same plane (the same side of the chip).
// Each Ethernet core runs two kernels:
//   tx (ERISC0, NoC 0, TXQ 0)  sends on its channel: relayed frames from its staging rings (one ring per feeding rx
//                              core), the own half once the chip holds every other root's half of the previous
//                              iteration in this plane, and the link flow control (forwards the rx role's ring
//                              releases to the peer's stream register 0; turns link credits into staging releases).
//   rx (ERISC1, NoC 1)         receives on its channel into one FIFO ring: per frame it polls the ring head's trailer,
//                              relays the frame to the staging rings of the children tx cores (one NoC write each),
//                              counts a completed half at the plane's own-half tx cores, and releases the ring slot
//                              once the relay writes have left the NIU.
// A frame is [payload | trailer 32 B] in a slot of `stride` bytes; the last packet of a half may be shorter and is then
// right-aligned against the trailer (the frame starts at payload - length), so the trailer sits at the same offset in
// every slot.  The trailer's two 16 B words (one in each NIU write stream)
// carry (sequence, meta); the sequence of a ring frame is the link's cumulative packet count (the receiver polls for
// the next value); a frame the rx role relays gets its two sequence words set to (top bit | ring count) before the NoC
// writes, and a tx core takes a staging slot when its two words are equal with the top bit set (it overwrites them with
// its link sequence when it sends, which clears the bit; the slot is rewritten only after the link credit for that send
// came back); meta = root << 24 | packet << 16 | iteration.  Counters are cumulative
// and the host zeroes the words before the launch.  Every loop ends in finite time (a stall watchdog writes a state).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

namespace {

constexpr uint32_t kHello = 0;
constexpr uint32_t kFlag = 32;
constexpr uint32_t kArrive = 64;
constexpr uint32_t kReleased = 96;
constexpr uint32_t kRxDone = 128;
constexpr uint32_t kStageRel = 160;
constexpr uint32_t kState = 256;
constexpr uint32_t kBad = 512;
constexpr uint32_t kStamps = 1024;
constexpr uint32_t kProbe = 4096;
constexpr uint32_t kStallLimit = 50000000;
constexpr uint32_t kMaxSrc = 2;
constexpr uint32_t kMaxChildren = 2;
constexpr uint32_t kMaxOwnCores = 3;
constexpr uint32_t kMaxRing = 64;
constexpr uint32_t kMaxPackets = 8;
constexpr uint32_t kStageFlag = 0x80000000u;

inline uint32_t load(uint32_t addr) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }

inline void store(uint32_t addr, uint32_t value) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value; }

inline uint32_t clock_low() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// write the two trailer sequence words and make sure the writes are processed by L1 before the caller's next request
// (a NoC command or an Ethernet TXQ command, both MMIO): store, load the same address back, consume the value
// (tt-isa-documentation BabyRISCV/MemoryOrdering.md "Enforcing stronger ordering"; lighter than a fence, which also
// drops the L0 data cache and only guarantees that the writes were emitted)
inline void store_trailer(uint32_t at, uint32_t value) {
    store(at, value);
    store(at + 16, value);
    const uint32_t back = load(at + 16);
    asm volatile("" : : "r"(back) : "memory");
    asm volatile("addi x0, %0, 0" : : "r"(back));
}

inline void store_clock64(uint32_t addr) {
    const uint32_t low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    store(addr, low);
    store(addr + 4, high);
}

inline void prepare_write_buf(uint32_t cmd_buf) {
    while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_RET_ADDR_MID, 0);
}

inline void issue_write(uint32_t cmd_buf, uint32_t src, uint32_t coord, uint32_t dst, uint32_t len) {
    while (!noc_cmd_buf_ready(noc_index, cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_TARG_ADDR_LO, src);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_RET_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_AT_LEN_BE, len);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc_index] += 1;
    noc_nonposted_writes_acked[noc_index] += 1;
}

inline uint32_t coord_of(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

inline uint32_t pattern(uint32_t root, uint32_t plane, uint32_t word) {
    return (root << 24) ^ (plane << 20) ^ word ^ 0x5a5a0000u;
}

}  // namespace

constexpr uint32_t kRole = get_compile_time_arg_val(0);  // 0: tx (ERISC0), 1: rx (ERISC1)

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t base = get_arg_val<uint32_t>(arg++);
    const uint32_t iters = get_arg_val<uint32_t>(arg++);
    const uint32_t packets = get_arg_val<uint32_t>(arg++);   // packets per half
    const uint32_t payload = get_arg_val<uint32_t>(arg++);   // bytes per packet (multiple of 16)
    const uint32_t last_payload = get_arg_val<uint32_t>(arg++);  // the last packet of a half (multiple of 16, <= payload)
    const uint32_t stride = get_arg_val<uint32_t>(arg++);    // slot stride (multiple of 32, >= payload + 32)
    const uint32_t ring = get_arg_val<uint32_t>(arg++);      // receive ring address (same on every core)
    const uint32_t ring_slots = get_arg_val<uint32_t>(arg++);  // a power of two (slot = count & mask: no division)
    const uint32_t stage = get_arg_val<uint32_t>(arg++);     // staging rings address (kMaxSrc rings)
    const uint32_t stage_slots = get_arg_val<uint32_t>(arg++);  // a power of two
    const uint32_t own_buf = get_arg_val<uint32_t>(arg++);   // own half frames: [parity][packet]
    const uint32_t plane = get_arg_val<uint32_t>(arg++);
    const uint32_t me = get_arg_val<uint32_t>(arg++);        // this chip's position in the block (root id)
    const uint32_t run_id = get_arg_val<uint32_t>(arg++);
    const uint32_t probe_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t stamp = get_arg_val<uint32_t>(arg++);
    const uint32_t startup_spin = get_arg_val<uint32_t>(arg++);
    const uint32_t full_check = get_arg_val<uint32_t>(arg++);  // 1: the rx role checks every word of every packet
    // tx role
    const uint32_t own = get_arg_val<uint32_t>(arg++);
    const uint32_t arrivals_per_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t n_src = get_arg_val<uint32_t>(arg++);
    uint32_t src_expected[kMaxSrc];
    uint64_t src_release[kMaxSrc];
    for (uint32_t s = 0; s < kMaxSrc; ++s) {
        const uint32_t x = get_arg_val<uint32_t>(arg++);
        const uint32_t y = get_arg_val<uint32_t>(arg++);
        const uint32_t child_index = get_arg_val<uint32_t>(arg++);
        src_expected[s] = get_arg_val<uint32_t>(arg++);
        src_release[s] = s < n_src ? get_noc_addr(x, y, base + kStageRel + 16 * child_index) : 0;
    }
    // rx role
    const uint32_t rx_total = get_arg_val<uint32_t>(arg++);
    const uint32_t child_mask = get_arg_val<uint32_t>(arg++);  // 2 bits per root: which children relay it
    const uint32_t n_children = get_arg_val<uint32_t>(arg++);
    uint32_t child_coord[kMaxChildren];
    uint32_t child_stage[kMaxChildren];
    for (uint32_t c = 0; c < kMaxChildren; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(arg++);
        const uint32_t y = get_arg_val<uint32_t>(arg++);
        const uint32_t src_index = get_arg_val<uint32_t>(arg++);
        child_coord[c] = c < n_children ? coord_of(x, y) : 0;
        child_stage[c] = stage + src_index * stage_slots * stride;
    }
    const uint32_t n_own_cores = get_arg_val<uint32_t>(arg++);
    uint64_t arrive_addr[kMaxOwnCores];
    for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(arg++);
        const uint32_t y = get_arg_val<uint32_t>(arg++);
        arrive_addr[c] = c < n_own_cores ? get_noc_addr(x, y, base + kArrive) : 0;
    }

    const uint32_t ring_mask = ring_slots - 1;
    const uint32_t stage_mask = stage_slots - 1;
    const uint32_t trailer = payload;
    const uint32_t words = payload / 4;
    auto length_of = [&](uint32_t pk) { return pk == packets - 1 ? last_payload : payload; };

    for (uint32_t i = 0; i < startup_spin; ++i) {
        asm volatile("nop");
    }
    if constexpr (kRole == 0) {
        init_ptr_val(0, 0);
        store(base + kFlag, run_id);
        while (internal_::eth_txq_is_busy(0)) {
        }
        internal_::eth_send_packet_bytes_unsafe(0, base + kFlag, base + kHello, 16);
        uint32_t wait = 0;
        while (++wait < kStallLimit) {
            invalidate_l1_cache();
            if (load(base + kHello) == run_id) {
                break;
            }
        }
        if (wait >= kStallLimit) {
            store(base + kState, 0xdead0001u);
            return;
        }
    }

    if constexpr (kRole == 1) {
        // ---------------- rx ----------------
        prepare_write_buf(write_cmd_buf);
        prepare_write_buf(write_reg_cmd_buf);
        uint32_t received = 0;
        uint32_t released = 0;
        uint32_t head = ring;
        uint32_t head_index = 0;
        uint32_t staged[kMaxChildren] = {0, 0};
        uint32_t stage_released[kMaxChildren] = {0, 0};
        uint32_t need[kMaxRing];
        uint32_t has_need[kMaxRing];
        for (uint32_t i = 0; i < ring_slots; ++i) {
            has_need[i] = 0;
        }
        uint32_t bad = 0;
        uint32_t next_meta[8];
        for (uint32_t r = 0; r < 8; ++r) {
            next_meta[r] = 0;
        }
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while (received < rx_total || released < received) {
            ++passes;
            invalidate_l1_cache();
            bool moved = false;
            if (received < rx_total) {
                const uint32_t seq = received + 1;
                const uint32_t w0 = load(head + trailer);
                const uint32_t w1 = load(head + trailer + 16);
                if ((w0 == seq) & (w1 == seq)) {
                    const uint32_t seen = clock_low();
                    const uint32_t meta = load(head + trailer + 4);
                    const uint32_t root = meta >> 24;
                    const uint32_t pk = (meta >> 16) & 0xff;
                    const uint32_t it = meta & 0xffff;
                    const uint32_t len = length_of(pk);
                    const uint32_t off = payload - len;
                    bool go = true;
                    const uint32_t mask = (child_mask >> (2 * root)) & 3;
                    if (mask) {
                        for (uint32_t c = 0; c < n_children; ++c) {
                            if (((mask >> c) & 1) && staged[c] - stage_released[c] >= stage_slots) {
                                stage_released[c] = load(base + kStageRel + 16 * c);
                                if (staged[c] - stage_released[c] >= stage_slots) {
                                    go = false;
                                }
                            }
                        }
                        if (go) {
                            store_trailer(head + trailer, kStageFlag | seq);
                            for (uint32_t c = 0; c < n_children; ++c) {
                                if ((mask >> c) & 1) {
                                    issue_write(
                                        (c & 1) ? write_reg_cmd_buf : write_cmd_buf,
                                        head + off,
                                        child_coord[c],
                                        child_stage[c] + (staged[c] & stage_mask) * stride + off,
                                        len + 32);
                                    ++staged[c];
                                }
                            }
                            need[head_index] = noc_nonposted_writes_num_issued[noc_index];
                            has_need[head_index] = 1;
                            if (it == probe_iter && pk == 0) {
                                store(base + kProbe + 192 + 4 * root, seen);
                                store(base + kProbe + 224 + 4 * root, clock_low());
                            }
                        }
                    }
                    if (go) {
                        if (pk == packets - 1) {
                            if (it == probe_iter) {
                                store_clock64(base + kProbe + 128 + 8 * root);
                            }
                            for (uint32_t c = 0; c < n_own_cores; ++c) {
                                noc_semaphore_inc(arrive_addr[c], 1);
                            }
                        }
                        if (root >= 8 || ((pk << 16) | it) != next_meta[root]) {
                            ++bad;
                        } else {
                            next_meta[root] = pk + 1 == packets ? it + 1 : (((pk + 1) << 16) | it);
                        }
                        const uint32_t first = pk * words;
                        const uint32_t n_words = len / 4;
                        if (full_check) {
                            for (uint32_t w = 0; w < n_words; ++w) {
                                if (load(head + off + 4 * w) != pattern(root, plane, first + w)) {
                                    ++bad;
                                }
                            }
                        } else if (
                            (load(head + off) != pattern(root, plane, first)) |
                            (load(head + payload - 4) != pattern(root, plane, first + n_words - 1))) {
                            ++bad;
                        }
                        ++received;
                        moved = true;
                        if (++head_index == ring_slots) {
                            head_index = 0;
                            head = ring;
                        } else {
                            head += stride;
                        }
                    }
                }
            }
            if (released < received) {
                const uint32_t sent_count = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
                uint32_t r = released;
                while (r < received) {
                    const uint32_t slot = r & ring_mask;
                    if (has_need[slot] && static_cast<int32_t>(sent_count - need[slot]) < 0) {
                        break;
                    }
                    has_need[slot] = 0;
                    ++r;
                }
                if (r != released) {
                    released = r;
                    store(base + kReleased, released);
                    moved = true;
                }
            }
            if (moved) {
                idle = 0;
            } else if (++idle == kStallLimit) {
                break;
            }
        }
        store(base + kProbe + 264, passes);
        store(base + kProbe + 268, clock_low() - t0);
        store(base + kBad, bad);
        store(base + kState + 32, idle >= kStallLimit ? 0xdead0000u : 0x600d0000u);
        store(base + kState + 36, received);
        store(base + kState + 40, released);
        store(base + kState + 44, staged[0] | (staged[1] << 16));
        store(base + kState + 48, load(head + trailer));
        store(base + kRxDone, 1);
        noc_async_atomic_barrier();
        noc_async_write_barrier();
    } else {
        // ---------------- tx ----------------
        uint32_t sent = 0;
        uint32_t credits = 0;
        uint32_t fwd = 0;
        uint32_t own_sent = 0;
        uint32_t own_it = 0;
        uint32_t own_pk = 0;
        const uint32_t own_total = own ? iters * packets : 0;
        uint32_t src_count[kMaxSrc] = {0, 0};
        uint32_t src_freed[kMaxSrc] = {0, 0};
        uint32_t src_head[kMaxSrc];
        uint8_t rec[kMaxRing];
        for (uint32_t s = 0; s < kMaxSrc; ++s) {
            src_head[s] = stage + s * stage_slots * stride;
        }
        uint32_t own_last[2 * kMaxPackets];
        for (uint32_t i = 0; i < 2 * kMaxPackets; ++i) {
            own_last[i] = 0;
        }
        uint32_t stamped = 0;
        bool own_seen = false;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        auto refresh_credits = [&]() {
            const uint32_t c = static_cast<uint32_t>(get_ptr_val(0));
            if (c != credits) {
                uint32_t freed[kMaxSrc] = {0, 0};
                for (uint32_t q = credits; q != c; ++q) {
                    const uint32_t s = rec[q & ring_mask];
                    if (s < kMaxSrc) {
                        ++freed[s];
                    }
                }
                credits = c;
                for (uint32_t s = 0; s < n_src; ++s) {
                    if (freed[s]) {
                        noc_semaphore_inc(src_release[s], freed[s]);
                        src_freed[s] += freed[s];
                    }
                }
                return true;
            }
            return false;
        };
        auto send_frame = [&](uint32_t src, uint32_t len, uint32_t tag_src) {
            const uint32_t off = payload - len;
            store_trailer(src + trailer, sent + 1);
            while (internal_::eth_txq_is_busy(0)) {
            }
            internal_::eth_send_packet_bytes_unsafe(0, src + off, ring + (sent & ring_mask) * stride + off, len + 32);
            rec[sent & ring_mask] = static_cast<uint8_t>(tag_src);
            ++sent;
        };
        bool done = false;
        while (!done) {
            ++passes;
            invalidate_l1_cache();
            bool moved = false;
            for (uint32_t s = 0; s < n_src; ++s) {
                if (src_count[s] == src_expected[s]) {
                    continue;
                }
                const uint32_t at = src_head[s];
                const uint32_t w0 = load(at + trailer);
                const uint32_t w1 = load(at + trailer + 16);
                if ((w0 != w1) | ((w0 & kStageFlag) == 0)) {
                    continue;
                }
                if (sent - credits >= ring_slots && !refresh_credits()) {
                    continue;
                }
                if (sent - credits >= ring_slots) {
                    continue;
                }
                const uint32_t seen = clock_low();
                const uint32_t meta = load(at + trailer + 4);
                send_frame(at, length_of((meta >> 16) & 0xff), s);
                if ((meta & 0xffffff) == probe_iter) {
                    const uint32_t root = meta >> 24;
                    store(base + kProbe + 64 + 4 * root, seen);
                    store(base + kProbe + 96 + 4 * root, clock_low());
                }
                ++src_count[s];
                src_head[s] = stage + s * stage_slots * stride + (src_count[s] & stage_mask) * stride;
                moved = true;
            }
            if (own_sent < own_total) {
                const bool ready = own_pk != 0 || load(base + kArrive) >= arrivals_per_iter * own_it;
                if (ready && !own_seen && own_pk == 0) {
                    own_seen = true;
                    if (own_it == probe_iter || own_it == probe_iter + 1) {
                        store_clock64(base + kProbe + 8 * (own_it - probe_iter));
                    }
                    if (stamp && stamped == own_it) {
                        store_clock64(base + kStamps + 8 * own_it);
                        ++stamped;
                    }
                }
                const uint32_t own_slot = (own_it & 1) * packets + own_pk;
                if (ready && (sent - credits >= ring_slots || credits < own_last[own_slot])) {
                    refresh_credits();
                }
                if (ready && credits >= own_last[own_slot]) {
                    if (sent - credits < ring_slots) {
                        const uint32_t src = own_buf + own_slot * stride;
                        store(src + trailer + 4, (me << 24) | (own_pk << 16) | own_it);
                        store(src + trailer + 20, (me << 24) | (own_pk << 16) | own_it);
                        send_frame(src, length_of(own_pk), 0xff);
                        own_last[own_slot] = sent;
                        ++own_sent;
                        if (++own_pk == packets) {
                            own_pk = 0;
                            ++own_it;
                            own_seen = false;
                        }
                        moved = true;
                    }
                }
            }
            const uint32_t rel = load(base + kReleased);
            if (rel != fwd) {
                while (internal_::eth_txq_is_busy(0)) {
                }
                remote_update_ptr_val<0>(0, static_cast<int32_t>(rel - fwd));
                fwd = rel;
                moved = true;
            }
            bool staging_pending = false;
            for (uint32_t s = 0; s < n_src; ++s) {
                if (src_count[s] - src_freed[s] >= stage_slots / 2 ||
                    (src_count[s] == src_expected[s] && src_freed[s] != src_count[s])) {
                    staging_pending = true;
                }
            }
            if (staging_pending || (!moved && sent != credits)) {
                if (refresh_credits()) {
                    moved = true;
                }
            }
            bool all = own_sent == own_total && credits == sent;
            for (uint32_t s = 0; s < n_src; ++s) {
                all = all && src_count[s] == src_expected[s] && src_freed[s] == src_count[s];
            }
            if (all) {
                invalidate_l1_cache();
                const bool rx_done = load(base + kRxDone) == 1;
                invalidate_l1_cache();
                done = rx_done && fwd == load(base + kReleased);
            }
            if (moved) {
                idle = 0;
            } else if (++idle == kStallLimit) {
                break;
            }
        }
        store(base + kProbe + 256, passes);
        store(base + kProbe + 260, clock_low() - t0);
        store(base + kState, idle >= kStallLimit ? 0xdead0000u : 0x600d0000u);
        store(base + kState + 4, sent);
        store(base + kState + 8, credits);
        store(base + kState + 12, own_sent);
        store(base + kState + 16, src_count[0]);
        store(base + kState + 20, src_count[1]);
        store(base + kState + 24, load(base + kArrive));
        store(base + kState + 28, fwd);
        while (internal_::eth_txq_is_busy(0)) {
        }
        noc_async_atomic_barrier();
    }
}
