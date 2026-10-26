// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The group meeting, version 5 (README 12.5, worklog 14 I-1b, v5): one kernel for every collective of a group,
// driven by host tables.  The transport is version 4's: two planes (channel parity), a receive FIFO ring per Ethernet
// core, staging rings toward the tx cores of the same plane, frames never rewritten after the source built them, and no
// flow control (rings sized by the causality bound: a link carries frames of at most three iterations).
// A flow is (source chip, destination set) with a route tree; its frames carry the tag
//     1 << 31 | flow << 24 | packet << 16 | iteration          (flow < 128)
// so a rx core looks up, per flow, which children tx cores to relay to (2 bits) and whether this chip is a final
// destination of the flow (then the frame's 16 B trailer line is copied into the arrival table [flow] of the chip's
// "waiting" tx cores: the cores that send the next iteration of a flow of this chip).
//   AG (all-gather): flow = root, destinations = every other chip, BFS tree; the next iteration of a chip's own flow
//       waits for every other flow (its "inputs").
//   RS (reduce-scatter, direct): flow = (source, owner) for the 7 owners; a chip's next iteration waits for the 7
//       flows that end at it (its slices from every other chip), modelling the owner's reduce dependency.
// The tx core sends its own flows' frames (own slots: [flow index][parity][packet]) when all its input flows have
// arrived for the previous iteration, and relays staging frames as they come.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"

namespace {

constexpr uint32_t kHello = 0;
constexpr uint32_t kFlag = 32;
constexpr uint32_t kRxDone = 128;
constexpr uint32_t kState = 256;
constexpr uint32_t kBad = 512;
constexpr uint32_t kStamps = 1024;
constexpr uint32_t kArriveTab = 3072;  // [flow] 16 B, flows < 64
constexpr uint32_t kProbe = 4096;
constexpr uint32_t kWindow = 64;
constexpr uint32_t kArrArr = 8192;   // [arrival slot][window] u32 (rx core): last packet of an arriving flow
constexpr uint32_t kOwnArr = 16384;  // [window] u32 (own tx core): the first own frame of the iteration
constexpr uint32_t kRegion = 16640;
constexpr uint32_t kStallLimit = 50000000;
constexpr uint32_t kSpinLimit = 10000000;
constexpr uint32_t kMaxFlows = 64;
constexpr uint32_t kMaxOwnFlows = 7;
constexpr uint32_t kMaxInputs = 8;
constexpr uint32_t kMaxChildren = 2;
constexpr uint32_t kMaxOwnCores = 3;
constexpr uint32_t kMaxRing = 64;
constexpr uint32_t kTagBit = 0x80000000u;
constexpr uint32_t kRelayBuf0 = write_cmd_buf;
constexpr uint32_t kRelayBuf1 = write_reg_cmd_buf;
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
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_PACKET_TAG, 0);
    NOC_CMD_BUF_WRITE_REG(noc_index, cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        cmd_buf,
        NOC_TARG_ADDR_COORDINATE,
        static_cast<uint32_t>(get_noc_addr(my_x[noc_index], my_y[noc_index], 0) >> NOC_ADDR_COORD_SHIFT) &
            NOC_COORDINATE_MASK);
}

template <uint32_t kCmdBuf>
inline __attribute__((always_inline)) void issue_write(uint32_t src, uint32_t coord, uint32_t dst, uint32_t len) {
    constexpr uint32_t kBase =
        (kCmdBuf << NOC_CMD_BUF_OFFSET_BIT) + (static_cast<uint32_t>(noc_index) << NOC_INSTANCE_OFFSET_BIT);
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

inline __attribute__((always_inline)) bool txq_wait() {
    for (uint32_t spin = 0; internal_::eth_txq_is_busy(0); ++spin) {
        if (spin == kSpinLimit) {
            g_broken = true;
            return false;
        }
    }
    return true;
}

inline __attribute__((always_inline)) void eth_send(uint32_t src, uint32_t dst, uint32_t len) {
    if (txq_wait()) {
        volatile uint32_t* q = reinterpret_cast<volatile uint32_t*>(ETH_TXQ0_REGS_START);
        q[ETH_TXQ_TRANSFER_START_ADDR / 4] = src;
        q[ETH_TXQ_DEST_ADDR / 4] = dst;
        q[ETH_TXQ_TRANSFER_SIZE_BYTES / 4] = len;
        q[ETH_TXQ_CMD / 4] = ETH_TXQ_CMD_START_DATA;
    }
}

inline bool l0_cache_off() {
    uint32_t cfg0;
    asm volatile("csrr %0, 0x7c0" : "=r"(cfg0));
    return (cfg0 & 0x8) != 0;
}

inline uint32_t coord_of(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

inline uint32_t pattern(uint32_t flow, uint32_t plane, uint32_t word) {
    return (flow << 24) ^ (plane << 20) ^ word ^ 0x5a5a0000u;
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
    const uint32_t ring_slots = get_arg_val<uint32_t>(arg++);
    const uint32_t stage = get_arg_val<uint32_t>(arg++);
    const uint32_t stage_slots = get_arg_val<uint32_t>(arg++);
    const uint32_t own_buf = get_arg_val<uint32_t>(arg++);
    const uint32_t plane = get_arg_val<uint32_t>(arg++);
    const uint32_t run_id = get_arg_val<uint32_t>(arg++);
    const uint32_t probe_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t stamp = get_arg_val<uint32_t>(arg++);
    const uint32_t startup_spin = get_arg_val<uint32_t>(arg++);
    const uint32_t full_check = get_arg_val<uint32_t>(arg++);
    // tx: own flows sent on this link and the input flows they wait for
    const uint32_t n_own = get_arg_val<uint32_t>(arg++);
    uint32_t own_flow[kMaxOwnFlows];
    for (uint32_t i = 0; i < kMaxOwnFlows; ++i) {
        own_flow[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t n_inputs = get_arg_val<uint32_t>(arg++);
    uint32_t input_flow[kMaxInputs];
    for (uint32_t i = 0; i < kMaxInputs; ++i) {
        input_flow[i] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t src_expected[2];
    src_expected[0] = get_arg_val<uint32_t>(arg++);
    src_expected[1] = get_arg_val<uint32_t>(arg++);
    // rx: frames to receive, per flow the relay mask (2 bits) and the destination bit, children, waiting cores
    const uint32_t rx_total = get_arg_val<uint32_t>(arg++);
    uint32_t flow_action[kMaxFlows / 8];  // 4 bits per flow: bit 0, 1 relay to child 0 / 1, bit 2 destination here
    for (uint32_t i = 0; i < kMaxFlows / 8; ++i) {
        flow_action[i] = get_arg_val<uint32_t>(arg++);
    }
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
    const uint32_t n_wait_cores = get_arg_val<uint32_t>(arg++);
    uint32_t wait_coord[kMaxOwnCores];
    for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(arg++);
        const uint32_t y = get_arg_val<uint32_t>(arg++);
        wait_coord[c] = c < n_wait_cores ? coord_of(x, y) : 0;
    }
    const uint32_t n_arrival_slots = get_arg_val<uint32_t>(arg++);
    uint32_t arrival_slot_of_flow[kMaxFlows / 4];  // 8 bits per flow: the stamp slot (0xff: none)
    for (uint32_t i = 0; i < kMaxFlows / 4; ++i) {
        arrival_slot_of_flow[i] = get_arg_val<uint32_t>(arg++);
    }

    const bool need_fence = !l0_cache_off();
    const uint32_t ring_mask = ring_slots - 1;
    const uint32_t stage_mask = stage_slots - 1;
    const uint32_t trailer = payload;
    const uint32_t words = payload / 4;
    const uint32_t last_off = payload - last_payload;
    (void)n_arrival_slots;

    for (uint32_t i = 0; i < startup_spin; ++i) {
        asm volatile("nop");
    }
    if constexpr (kRole == 0) {
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
        uint32_t child_slot[kMaxChildren] = {child_stage[0], child_stage[1]};
        uint32_t staged[kMaxChildren] = {0, 0};
        uint32_t seen_tag[kMaxRing];
        for (uint32_t i = 0; i < ring_slots; ++i) {
            seen_tag[i] = 0;
        }
        uint32_t next_meta[kMaxFlows];
        for (uint32_t f = 0; f < kMaxFlows; ++f) {
            next_meta[f] = 0;
        }
        uint32_t received = 0;
        uint32_t head = ring;
        uint32_t head_index = 0;
        uint32_t head_prev = 0;
        uint32_t issued_total = 0;
        uint32_t bad = 0;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while (received < rx_total && !g_broken) {
            ++passes;
            // wait for the head slot's next frame: a tight poll (two loads, the compare), bounded
            uint32_t w0 = 0;
            const uint32_t at = head + trailer;
            const uint32_t prev = head_prev;
            uint32_t spin = 0;
            for (;;) {
                if (need_fence) {
                    invalidate_l1_cache();
                }
                w0 = load(at);
                const uint32_t w1 = load(at + 16);
                if ((w0 == w1) & (w0 != prev) & ((w0 & kTagBit) != 0)) {
                    break;
                }
                if (++spin == kStallLimit) {
                    break;
                }
            }
            if (spin == kStallLimit) {
                idle = kStallLimit;
                break;
            }
            passes += spin;
            const uint32_t it = w0 & 0xffff;
            const uint32_t flow = (w0 >> 24) & 0x7f;
            const uint32_t pk = (w0 >> 16) & 0xff;
            const bool last_pk = pk == packets - 1;
            const bool window = it - probe_iter < kWindow;
            const uint32_t seen = window && last_pk ? clock_low() : 0;
            const uint32_t off = last_pk ? last_off : 0;
            const uint32_t len = payload - off;
            const uint32_t action = (flow_action[(flow >> 3) & 7] >> (4 * (flow & 7))) & 0xf;
            uint32_t issued = 0;
            if (action & 1) {
                issue_write<kRelayBuf0>(head + off, child_coord[0], child_slot[0] + off, len + 32);
                ++staged[0];
                child_slot[0] = (staged[0] & stage_mask) == 0 ? child_stage[0] : child_slot[0] + stride;
                ++issued;
            }
            if (action & 2) {
                issue_write<kRelayBuf1>(head + off, child_coord[1], child_slot[1] + off, len + 32);
                ++staged[1];
                child_slot[1] = (staged[1] & stage_mask) == 0 ? child_stage[1] : child_slot[1] + stride;
                ++issued;
            }
            if ((action & 4) && last_pk) {
                for (uint32_t c = 0; c < n_wait_cores; ++c) {
                    issue_write<kNoticeBuf>(head + trailer, wait_coord[c], base + kArriveTab + 16 * flow, 16);
                }
                issued += n_wait_cores;
                if (window) {
                    const uint32_t slot = (arrival_slot_of_flow[(flow >> 2) & 15] >> (8 * (flow & 3))) & 0xff;
                    if (slot != 0xff) {
                        store(base + kArrArr + 4 * (slot * kWindow + (it - probe_iter)), seen);
                    }
                }
            }
            issued_total += issued;
            if (flow >= kMaxFlows || ((pk << 16) | it) != next_meta[flow]) {
                ++bad;
            } else {
                next_meta[flow] = last_pk ? it + 1 : (((pk + 1) << 16) | it);
            }
            if (action & 4) {
                const uint32_t first = pk * words;
                const uint32_t n_words = len / 4;
                if (full_check) {
                    for (uint32_t w = 0; w < n_words; ++w) {
                        if (load(head + off + 4 * w) != pattern(flow, plane, first + w)) {
                            ++bad;
                        }
                    }
                } else if (
                    (load(head + off) != pattern(flow, plane, first)) |
                    (load(head + payload - 4) != pattern(flow, plane, first + n_words - 1))) {
                    ++bad;
                }
            }
            seen_tag[head_index] = w0;
            ++received;
            if (++head_index == ring_slots) {
                head_index = 0;
                head = ring;
            } else {
                head += stride;
            }
            head_prev = seen_tag[head_index];
        }
        noc_nonposted_writes_num_issued[noc_index] += issued_total;
        noc_nonposted_writes_acked[noc_index] += issued_total;
        store(base + kProbe + 264, passes);
        store(base + kProbe + 268, clock_low() - t0);
        store(base + kBad, bad);
        store(base + kState + 32, g_broken ? 0xdead0002u : (idle >= kStallLimit ? 0xdead0000u : 0x600d0000u));
        store(base + kState + 36, received);
        store(base + kState + 44, staged[0] | (staged[1] << 16));
        store(base + kState + 48, load(head + trailer));
        store(base + kRxDone, 1);
        bounded_noc_drain();
        store(base + kState + 52, g_broken ? 0xdead0003u : 0x600d0000u);
    } else {
        // ---------------- tx ----------------
        uint32_t sent = 0;
        uint32_t dst = ring;
        const uint32_t own_total = n_own * iters * packets;
        uint32_t own_sent = 0;
        uint32_t own_it = 0;
        uint32_t own_i = 0;   // which own flow
        uint32_t own_pk = 0;  // which packet of it
        uint32_t wait_i = 0;  // the next input flow awaited for own_it
        uint32_t src_count[2] = {0, 0};
        uint32_t head0 = stage;
        uint32_t head1 = stage + stage_slots * stride;
        uint32_t prev0 = 0;
        uint32_t prev1 = 0;
        uint32_t src_seen[2][kMaxRing];
        for (uint32_t s = 0; s < 2; ++s) {
            for (uint32_t i = 0; i < stage_slots; ++i) {
                src_seen[s][i] = 0;
            }
        }
        uint32_t remaining = own_total + src_expected[0] + src_expected[1];
        uint32_t stamped = 0;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while (remaining && !g_broken) {
            ++passes;
            // while the own half is mid-flight (packets left in this iteration) the loop must keep sending; otherwise
            // wait in a tight poll for a new staging frame or the next input-flow arrival (bounded)
            if (own_sent == own_total || (own_i == 0 && own_pk == 0 && own_it > 0)) {
                const bool s0 = src_count[0] != src_expected[0];
                const bool s1 = src_count[1] != src_expected[1];
                const bool own_wait = own_sent != own_total && wait_i < n_inputs;
                const uint32_t in_addr = own_wait ? base + kArriveTab + 16 * input_flow[wait_i] : base;
                const uint32_t need_it = own_it;
                uint32_t spin = 0;
                for (;; ++spin) {
                    if (need_fence) {
                        invalidate_l1_cache();
                    }
                    if (s0) {
                        const uint32_t a = load(head0 + trailer);
                        const uint32_t b = load(head0 + trailer + 16);
                        if ((a == b) & (a != prev0) & ((a & kTagBit) != 0)) {
                            break;
                        }
                    }
                    if (s1) {
                        const uint32_t a = load(head1 + trailer);
                        const uint32_t b = load(head1 + trailer + 16);
                        if ((a == b) & (a != prev1) & ((a & kTagBit) != 0)) {
                            break;
                        }
                    }
                    if (own_wait) {
                        const uint32_t e = load(in_addr);
                        if ((e & kTagBit) && (e & 0xffff) + 1 >= need_it) {
                            break;
                        }
                    } else if (own_sent != own_total) {
                        break;
                    }
                    if (spin == kStallLimit) {
                        break;
                    }
                }
                passes += spin;
                if (spin == kStallLimit) {
                    idle = kStallLimit;
                    break;
                }
            }
            bool moved = false;
            if (src_count[0] != src_expected[0]) {
                const uint32_t w0 = load(head0 + trailer);
                const uint32_t w1 = load(head0 + trailer + 16);
                if ((w0 == w1) & (w0 != prev0) & ((w0 & kTagBit) != 0)) {
                    const uint32_t off = ((w0 >> 16) & 0xff) == packets - 1 ? last_off : 0;
                    eth_send(head0 + off, dst + off, payload - off + 32);
                    ++sent;
                    dst = (sent & ring_mask) == 0 ? ring : dst + stride;
                    const uint32_t slot = src_count[0] & stage_mask;
                    src_seen[0][slot] = w0;
                    ++src_count[0];
                    head0 = (src_count[0] & stage_mask) == 0 ? stage : head0 + stride;
                    prev0 = src_seen[0][(slot + 1) & stage_mask];
                    --remaining;
                    moved = true;
                }
            }
            if (src_count[1] != src_expected[1]) {
                const uint32_t w0 = load(head1 + trailer);
                const uint32_t w1 = load(head1 + trailer + 16);
                if ((w0 == w1) & (w0 != prev1) & ((w0 & kTagBit) != 0)) {
                    const uint32_t off = ((w0 >> 16) & 0xff) == packets - 1 ? last_off : 0;
                    eth_send(head1 + off, dst + off, payload - off + 32);
                    ++sent;
                    dst = (sent & ring_mask) == 0 ? ring : dst + stride;
                    const uint32_t slot = src_count[1] & stage_mask;
                    src_seen[1][slot] = w0;
                    ++src_count[1];
                    head1 = (src_count[1] & stage_mask) == 0 ? stage + stage_slots * stride : head1 + stride;
                    prev1 = src_seen[1][(slot + 1) & stage_mask];
                    --remaining;
                    moved = true;
                }
            }
            if (own_sent < own_total) {
                const bool starting = own_i == 0 && own_pk == 0;
                while (starting && own_it > 0 && wait_i < n_inputs) {
                    const uint32_t e = load(base + kArriveTab + 16 * input_flow[wait_i]);
                    if (!((e & kTagBit) && (e & 0xffff) + 1 >= own_it)) {
                        break;
                    }
                    ++wait_i;
                }
                if (!starting || own_it == 0 || wait_i >= n_inputs) {
                    if (starting) {
                        if (own_it - probe_iter < kWindow) {
                            store(base + kOwnArr + 4 * (own_it - probe_iter), clock_low());
                        }
                        if (stamp && stamped == own_it) {
                            store_clock64(base + kStamps + 8 * own_it);
                            ++stamped;
                        }
                    }
                    const uint32_t flow = own_flow[own_i];
                    const uint32_t src = own_buf + ((own_i * 2 + (own_it & 1)) * packets + own_pk) * stride;
                    const uint32_t tag = kTagBit | (flow << 24) | (own_pk << 16) | own_it;
                    store(src + trailer, tag);
                    store(src + trailer + 16, tag);
                    const uint32_t back = load(src + trailer + 16);
                    asm volatile("addi x0, %0, 0" : : "r"(back));
                    const uint32_t off = own_pk == packets - 1 ? last_off : 0;
                    eth_send(src + off, dst + off, payload - off + 32);
                    ++sent;
                    dst = (sent & ring_mask) == 0 ? ring : dst + stride;
                    ++own_sent;
                    --remaining;
                    moved = true;
                    if (++own_pk == packets) {
                        own_pk = 0;
                        if (++own_i == n_own) {
                            own_i = 0;
                            ++own_it;
                            wait_i = 0;
                        }
                    }
                }
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
        store(base + kState + 12, own_sent);
        store(base + kState + 16, src_count[0]);
        store(base + kState + 20, src_count[1]);
        store(base + kState + 24, wait_i | (own_it << 8));
        txq_wait();
        bounded_noc_drain();
        store(base + kState + 60, g_broken ? 0xdead0003u : 0x600d0000u);
    }
}
