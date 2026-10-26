// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The group meeting, version 3 (README 12.5, worklog 14 I-1b''): the all-gather of one chunk per chip along the BFS
// trees of the 2 x 4 block, in two planes (channel parity; chunk half p in plane p), as version 2, with frames that are
// never rewritten after the root built them.
//   A frame is [payload | trailer 32 B] in a slot of `stride` bytes (the last, shorter packet of a half is right-aligned
//   against the trailer).  The trailer's two 16 B words (one in each NIU write stream, each written last in its
//   stream: tt-isa-documentation NoC/Ordering.md) both start with the frame's tag = 1 << 31 | root << 24 |
//   packet << 16 | iteration: unique per frame, so a reader that sees both words equal to a tag different from the
//   one it last saw in that slot holds the whole new frame.  The root writes the tag once (store, load back, use:
//   BabyRISCV/MemoryOrdering.md) before its own send; relays copy frames byte for byte (NoC into a staging ring,
//   Ethernet into the next receive ring) without touching them.
//   rx (ERISC1, NoC 1): polls the head slot of its receive ring; on a new frame it writes the frame to the staging
//                       rings of the children tx cores of the same plane (one NoC write each, fixed-state command
//                       buffers); on the last packet of a half it copies the frame's trailer line (16 B: the tag) into
//                       the arrival table [root] of the plane's own-half tx cores (a NoC write from the untouched
//                       frame, same source / destination / VC as every earlier notice, so the table entry of a root
//                       only grows); it releases ring slots once its writes have left the NIU (the tx core of the same
//                       Ethernet core forwards releases to the sender's stream register in batches).
//   tx (ERISC0, NoC 0, TXQ 0): polls the head slot of each staging ring and sends new frames as they come; sends the
//                       own half when the arrival table shows every other root's half of the previous iteration (a
//                       cursor over the roots: one load per pass); turns returned link credits into staging releases.
// Every count is cumulative from zero (the host zeroes the words); every loop ends in finite time (stall watchdog).
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
constexpr uint32_t kArriveTab = 3072;  // [root] 16 B: the trailer line of the last packet of the root's half (its tag);
                                       // after the stamps (kStamps + 8 x 256 iterations = 3072)
constexpr uint32_t kState = 256;
constexpr uint32_t kBad = 512;
constexpr uint32_t kStamps = 1024;
constexpr uint32_t kProbe = 4096;
constexpr uint32_t kStallLimit = 50000000;
constexpr uint32_t kSpinLimit = 10000000;  // an inner hardware wait (command buffer ready, TXQ busy) that exceeds this
                                           // marks the core broken; every loop then ends (finite time)
constexpr uint32_t kMaxSrc = 2;
constexpr uint32_t kMaxChildren = 2;
constexpr uint32_t kMaxOwnCores = 3;
constexpr uint32_t kMaxRing = 64;
constexpr uint32_t kMaxPackets = 8;
constexpr uint32_t kTagBit = 0x80000000u;
// the rx role (ERISC1, NoC 1) issues the relay writes on two command buffers of its own NIU: the write and the
// write-register buffers of this processor (dataflow_cmd_bufs.h), set up once with a fixed NOC_CTRL
constexpr uint32_t kRelayBuf0 = write_cmd_buf;
constexpr uint32_t kRelayBuf1 = write_reg_cmd_buf;
// the arrival notices (the rx role writes the 16 B trailer line of a half's last packet into the own-half tx cores'
// arrival table) go on the atomic command buffer, which the rx role does not use for atomics
constexpr uint32_t kNoticeBuf = write_at_cmd_buf;

bool g_broken = false;

inline uint32_t load(uint32_t addr) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }

inline void store(uint32_t addr, uint32_t value) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value; }

inline uint32_t clock_low() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

inline void store_clock64(uint32_t addr) {
    const uint32_t low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    store(addr, low);
    store(addr + 4, high);
}

inline void prepare_write_buf(uint32_t cmd_buf) {
    for (uint32_t spin = 0; !noc_cmd_buf_ready(noc_index, cmd_buf); ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return;
        }
    }
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_RET_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_PACKET_TAG, 0);  // no transaction id (an atomic buffer may hold one)
    // the source side of a write is this core's L1: noc_init sets TARG_ADDR_MID / TARG_ADDR_COORDINATE only on the
    // write and write-register command buffers (noc_nonblocking_api.h noc_init); set them here for any buffer used
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        static_cast<uint32_t>(get_noc_addr(my_x[noc_index], my_y[noc_index], 0) >> NOC_ADDR_COORD_SHIFT) &
            NOC_COORDINATE_MASK);
}

// a NoC write on a command buffer whose NOC_CTRL / RET_ADDR_MID were set once (prepare_write_buf): the register block
// address is a compile-time constant, the five stores go straight to it; the caller counts the issued writes
template <uint32_t kCmdBuf>
inline __attribute__((always_inline)) void issue_write(uint32_t src, uint32_t coord, uint32_t dst, uint32_t len) {
    constexpr uint32_t kBase = (kCmdBuf << NOC_CMD_BUF_OFFSET_BIT) + (static_cast<uint32_t>(noc_index) << NOC_INSTANCE_OFFSET_BIT);
    auto reg = [](uint32_t addr) { return reinterpret_cast<volatile uint32_t*>(kBase + addr); };
    for (uint32_t spin = 0; *reg(NOC_CMD_CTRL) != NOC_CTRL_STATUS_READY; ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return;
        }
    }
    *reg(NOC_TARG_ADDR_LO) = src;
    *reg(NOC_RET_ADDR_LO) = dst;
    *reg(NOC_RET_ADDR_COORDINATE) = coord;
    *reg(NOC_AT_LEN_BE) = len;
    *reg(NOC_CMD_CTRL) = NOC_CTRL_SEND_REQ;
}

inline uint32_t coord_of(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

inline uint32_t pattern(uint32_t root, uint32_t plane, uint32_t word) {
    return (root << 24) ^ (plane << 20) ^ word ^ 0x5a5a0000u;
}

inline uint32_t tag_of(uint32_t root, uint32_t pk, uint32_t it) { return kTagBit | (root << 24) | (pk << 16) | it; }

// the end-of-kernel waits for this RISC's NoC requests (atomics acknowledged, writes acknowledged), bounded
inline void bounded_noc_drain() {
    for (uint32_t spin = 0; !ncrisc_noc_nonposted_atomics_flushed(noc_index); ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return;
        }
    }
    for (uint32_t spin = 0; !ncrisc_noc_nonposted_writes_flushed(noc_index); ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return;
        }
    }
}

inline bool txq_wait() {
    for (uint32_t spin = 0; internal_::eth_txq_is_busy(0); ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return false;
        }
    }
    return true;
}

inline void eth_send(uint32_t src, uint32_t dst, uint32_t len) {
    if (txq_wait()) {
        internal_::eth_send_packet_bytes_unsafe(0, src, dst, len);
    }
}

}  // namespace

constexpr uint32_t kRole = get_compile_time_arg_val(0);  // 0: tx (ERISC0), 1: rx (ERISC1)

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t base = get_arg_val<uint32_t>(arg++);
    const uint32_t iters = get_arg_val<uint32_t>(arg++);
    const uint32_t packets = get_arg_val<uint32_t>(arg++);
    const uint32_t payload = get_arg_val<uint32_t>(arg++);
    const uint32_t last_payload = get_arg_val<uint32_t>(arg++);
    const uint32_t stride = get_arg_val<uint32_t>(arg++);
    const uint32_t ring = get_arg_val<uint32_t>(arg++);
    const uint32_t ring_slots = get_arg_val<uint32_t>(arg++);  // a power of two
    const uint32_t stage = get_arg_val<uint32_t>(arg++);
    const uint32_t stage_slots = get_arg_val<uint32_t>(arg++);  // a power of two
    const uint32_t own_buf = get_arg_val<uint32_t>(arg++);
    const uint32_t plane = get_arg_val<uint32_t>(arg++);
    const uint32_t me = get_arg_val<uint32_t>(arg++);
    const uint32_t run_id = get_arg_val<uint32_t>(arg++);
    const uint32_t probe_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t stamp = get_arg_val<uint32_t>(arg++);
    const uint32_t startup_spin = get_arg_val<uint32_t>(arg++);
    const uint32_t full_check = get_arg_val<uint32_t>(arg++);
    const uint32_t own = get_arg_val<uint32_t>(arg++);
    get_arg_val<uint32_t>(arg++);  // arrivals per iteration (7: implied by the arrival table)
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
    const uint32_t rx_total = get_arg_val<uint32_t>(arg++);
    const uint32_t child_mask = get_arg_val<uint32_t>(arg++);
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
    const uint32_t last_off = payload - last_payload;

    for (uint32_t i = 0; i < startup_spin; ++i) {
        asm volatile("nop");
    }
    if constexpr (kRole == 0) {
        init_ptr_val(0, 0);
        store(base + kFlag, run_id);
        eth_send(base + kFlag, base + kHello, 16);
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
        prepare_write_buf(kRelayBuf0);
        prepare_write_buf(kRelayBuf1);
        prepare_write_buf(kNoticeBuf);
        uint32_t own_coord[kMaxOwnCores];
        for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
            own_coord[c] = static_cast<uint32_t>(arrive_addr[c] >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
        }
        uint32_t child_slot[kMaxChildren] = {child_stage[0], child_stage[1]};
        uint32_t seen_tag[kMaxRing];
        uint32_t need[kMaxRing];
        uint8_t has_need[kMaxRing];
        for (uint32_t i = 0; i < ring_slots; ++i) {
            seen_tag[i] = 0;
            has_need[i] = 0;
        }
        uint32_t next_meta[8];
        for (uint32_t r = 0; r < 8; ++r) {
            next_meta[r] = 0;
        }
        uint32_t received = 0;
        uint32_t released = 0;
        uint32_t head = ring;
        uint32_t head_index = 0;
        uint32_t staged[kMaxChildren] = {0, 0};
        uint32_t stage_released[kMaxChildren] = {0, 0};
        uint32_t bad = 0;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while ((received < rx_total || released < received) && !g_broken) {
            ++passes;
            invalidate_l1_cache();
            bool moved = false;
            if (received < rx_total) {
                const uint32_t w0 = load(head + trailer);
                const uint32_t w1 = load(head + trailer + 16);
                if ((w0 == w1) & (w0 != seen_tag[head_index]) & ((w0 & kTagBit) != 0)) {
                    const uint32_t it = w0 & 0xffff;
                    const uint32_t seen = it == probe_iter ? clock_low() : 0;
                    const uint32_t root = (w0 >> 24) & 0x7f;
                    const uint32_t pk = (w0 >> 16) & 0xff;
                    const bool last_pk = pk == packets - 1;
                    const uint32_t off = last_pk ? last_off : 0;
                    const uint32_t len = payload - off;
                    const uint32_t mask = (child_mask >> (2 * root)) & 3;
                    bool go = true;
                    for (uint32_t c = 0; c < n_children; ++c) {
                        if (((mask >> c) & 1) && staged[c] - stage_released[c] >= stage_slots) {
                            stage_released[c] = load(base + kStageRel + 16 * c);
                            if (staged[c] - stage_released[c] >= stage_slots) {
                                go = false;
                            }
                        }
                    }
                    if (go) {
                        const bool probe = it == probe_iter && pk == 0 && mask;
                        const uint32_t t_checked = probe ? clock_low() : 0;
                        uint32_t t_issue[kMaxChildren] = {0, 0};
                        uint32_t issued = 0;
                        if (mask) {
                            if (mask & 1) {
                                issue_write<kRelayBuf0>(head + off, child_coord[0], child_slot[0] + off, len + 32);
                                t_issue[0] = probe ? clock_low() : 1;
                                ++staged[0];
                                child_slot[0] = (staged[0] & stage_mask) == 0 ? child_stage[0] : child_slot[0] + stride;
                                ++issued;
                            }
                            if (mask & 2) {
                                issue_write<kRelayBuf1>(head + off, child_coord[1], child_slot[1] + off, len + 32);
                                t_issue[1] = probe ? clock_low() : 1;
                                ++staged[1];
                                child_slot[1] = (staged[1] & stage_mask) == 0 ? child_stage[1] : child_slot[1] + stride;
                                ++issued;
                            }
                        }
                        if (last_pk) {
                            for (uint32_t c = 0; c < n_own_cores; ++c) {
                                issue_write<kNoticeBuf>(head + trailer, own_coord[c], base + kArriveTab + 16 * root, 16);
                            }
                            issued += n_own_cores;
                        }
                        if (issued) {
                            noc_nonposted_writes_num_issued[noc_index] += issued;
                            noc_nonposted_writes_acked[noc_index] += issued;
                            need[head_index] = noc_nonposted_writes_num_issued[noc_index];
                            has_need[head_index] = 1;
                        }
                        if (probe) {
                            store(base + kProbe + 288 + 4 * root, t_checked - seen);
                            store(base + kProbe + 320 + 4 * root, t_issue[0] ? t_issue[0] - t_checked : 0);
                            store(base + kProbe + 352 + 4 * root, t_issue[1] ? t_issue[1] - t_checked : 0);
                        }
                        if (it == probe_iter) {
                            if (last_pk) {
                                store(base + kProbe + 128 + 8 * root, seen);
                                store(base + kProbe + 128 + 8 * root + 4, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H));
                            }
                            if (pk == 0 && mask) {
                                store(base + kProbe + 192 + 4 * root, seen);
                                store(base + kProbe + 224 + 4 * root, clock_low());
                            }
                        }
                        if (root >= 8 || ((pk << 16) | it) != next_meta[root]) {
                            ++bad;
                        } else {
                            next_meta[root] = last_pk ? it + 1 : (((pk + 1) << 16) | it);
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
                        seen_tag[head_index] = w0;
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
        store(base + kState + 32, g_broken ? 0xdead0002u : (idle >= kStallLimit ? 0xdead0000u : 0x600d0000u));
        store(base + kState + 36, received);
        store(base + kState + 40, released);
        store(base + kState + 44, staged[0] | (staged[1] << 16));
        store(base + kState + 48, load(head + trailer));
        store(base + kRxDone, 1);
        bounded_noc_drain();
        store(base + kState + 52, g_broken ? 0xdead0003u : 0x600d0000u);
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
        uint32_t src_seen[kMaxSrc][kMaxRing];
        uint8_t rec[kMaxRing];
        for (uint32_t s = 0; s < kMaxSrc; ++s) {
            src_head[s] = stage + s * stage_slots * stride;
            for (uint32_t i = 0; i < stage_slots; ++i) {
                src_seen[s][i] = 0;
            }
        }
        uint32_t own_last[2 * kMaxPackets];
        for (uint32_t i = 0; i < 2 * kMaxPackets; ++i) {
            own_last[i] = 0;
        }
        uint32_t stamped = 0;
        bool own_seen = false;
        uint32_t idle = 0;
        uint32_t passes = 0;
        uint32_t wait_root = me == 0 ? 1 : 0;  // the next other root whose half of the previous iteration is awaited
        const uint32_t t0 = clock_low();
        auto refresh_credits = [&]() {
            const uint32_t c = static_cast<uint32_t>(get_ptr_val(0));
            if (c == credits) {
                return false;
            }
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
        };
        bool done = false;
        while (!done && !g_broken) {
            ++passes;
            invalidate_l1_cache();
            bool moved = false;
            for (uint32_t s = 0; s < n_src; ++s) {
                if (src_count[s] == src_expected[s]) {
                    continue;
                }
                const uint32_t slot = src_count[s] & stage_mask;
                const uint32_t at = src_head[s];
                const uint32_t w0 = load(at + trailer);
                const uint32_t w1 = load(at + trailer + 16);
                if ((w0 != w1) | (w0 == src_seen[s][slot]) | ((w0 & kTagBit) == 0)) {
                    continue;
                }
                if (sent - credits >= ring_slots) {
                    refresh_credits();
                    if (sent - credits >= ring_slots) {
                        continue;
                    }
                }
                const uint32_t seen = clock_low();
                const uint32_t off = ((w0 >> 16) & 0xff) == packets - 1 ? last_off : 0;
                eth_send(at + off, ring + (sent & ring_mask) * stride + off, payload - off + 32);
                rec[sent & ring_mask] = static_cast<uint8_t>(s);
                ++sent;
                src_seen[s][slot] = w0;
                if ((w0 & 0xffffff) == probe_iter) {
                    const uint32_t root = (w0 >> 24) & 0x7f;
                    store(base + kProbe + 64 + 4 * root, seen);
                    store(base + kProbe + 96 + 4 * root, clock_low());
                }
                ++src_count[s];
                src_head[s] = (src_count[s] & stage_mask) == 0 ? stage + s * stage_slots * stride : at + stride;
                moved = true;
            }
            if (own_sent < own_total) {
                if (own_pk == 0 && own_it > 0 && wait_root < 8) {
                    const uint32_t e = load(base + kArriveTab + 16 * wait_root);
                    if ((e & kTagBit) && (e & 0xffff) + 1 >= own_it) {
                        wait_root = wait_root + 1 == me ? me + 1 : wait_root + 1;
                    }
                }
                const bool ready = own_pk != 0 || own_it == 0 || wait_root >= 8;
                if (ready) {
                    const uint32_t own_slot = (own_it & 1) * packets + own_pk;
                    if (sent - credits >= ring_slots || credits < own_last[own_slot]) {
                        refresh_credits();
                    }
                    if (credits >= own_last[own_slot] && sent - credits < ring_slots) {
                        if (!own_seen) {
                            own_seen = true;
                            if (own_it == probe_iter || own_it == probe_iter + 1) {
                                store_clock64(base + kProbe + 8 * (own_it - probe_iter));
                            }
                            if (stamp && stamped == own_it) {
                                store_clock64(base + kStamps + 8 * own_it);
                                ++stamped;
                            }
                        }
                        const uint32_t src = own_buf + own_slot * stride;
                        const uint32_t tag = tag_of(me, own_pk, own_it);
                        store(src + trailer, tag);
                        store(src + trailer + 16, tag);
                        const uint32_t back = load(src + trailer + 16);
                        asm volatile("addi x0, %0, 0" : : "r"(back));
                        const uint32_t off = own_pk == packets - 1 ? last_off : 0;
                        eth_send(src + off, ring + (sent & ring_mask) * stride + off, payload - off + 32);
                        rec[sent & ring_mask] = 0xff;
                        ++sent;
                        own_last[own_slot] = sent;
                        ++own_sent;
                        if (++own_pk == packets) {
                            own_pk = 0;
                            ++own_it;
                            own_seen = false;
                            wait_root = me == 0 ? 1 : 0;
                        }
                        moved = true;
                    }
                }
            }
            const uint32_t rel = load(base + kReleased);
            if (rel - fwd >= 8 || (rel != fwd && !moved)) {
                txq_wait();
                remote_update_ptr_val<0>(0, static_cast<int32_t>(rel - fwd));
                fwd = rel;
                moved = true;
            }
            if (!moved && sent != credits) {
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
        store(base + kState, g_broken ? 0xdead0002u : (idle >= kStallLimit ? 0xdead0000u : 0x600d0000u));
        store(base + kState + 4, sent);
        store(base + kState + 8, credits);
        store(base + kState + 12, own_sent);
        store(base + kState + 16, src_count[0]);
        store(base + kState + 20, src_count[1]);
        store(base + kState + 24, wait_root | (own_it << 8));
        store(base + kState + 28, fwd);
        txq_wait();
        bounded_noc_drain();
        store(base + kState + 60, g_broken ? 0xdead0003u : 0x600d0000u);
    }
}
