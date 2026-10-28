// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The group meeting, version 4 (README 12.5, worklog 14 I-1b'''): the all-gather of one chunk per chip along the BFS
// trees of the 2 x 4 block in two planes (channel parity; chunk half p in plane p), frames never rewritten after the
// root built them (version 3), and no flow control at all: no link credits, no staging releases.
// Why no flow control is needed (worklog 14, v4 design): a chip starts iteration k of its own half in a plane only once
// it holds every other root's half of iteration k-1 in that plane.  If m is the newest iteration started anywhere,
// every root has started m-1 and every chip holds all of m-2, so every frame of an iteration <= m-2 has reached every
// destination (relay NoC writes and TT-link re-sends included).  A receive ring (one link's FIFO) whose slot of frame
// f is rewritten by frame f+R while f is not yet everywhere: f has an iteration >= m-1, and every frame sent on a
// link after a frame of iteration j has an iteration >= j-1 (a frame of iteration j leaves a chip only after its root
// started j, i.e. after every chip held all of j-2), so frames f..f+R all lie in iterations m-2..m, at most 3F frames
// for F frames per iteration on the link: a ring of R >= 3F slots is never rewritten under a live frame (likewise a
// staging ring: R >= 3 x its frames per iteration; the own frames: two parities, iteration k+2 is built after every
// chip holds k).  A TT-link re-send of an overwritten slot carries a sequence number the receiver already accepted
// and is dropped (EthernetTxRx.md).  The host checks the ring sizes.
//   A frame is [payload | trailer 32 B] in a slot of `stride` bytes (the last, shorter packet of a half is right-aligned
//   against the trailer).  The trailer's two 16 B words (one per NIU write stream, each written last in its stream:
//   tt-isa-documentation NoC/Ordering.md) both start with the frame's tag = 1 << 31 | root << 24 | packet << 16 |
//   iteration; a reader that sees both words equal to a tag different from the last one in that slot holds the frame.
//   rx (ERISC1, NoC 1): polls its receive ring's head; on a new frame it writes it to the staging rings of the children
//                       tx cores of the same plane (one NoC write each); on the last packet of a half it copies the
//                       frame's 16 B trailer line into the arrival table [root] of the plane's own-half tx cores.
//   tx (ERISC0, NoC 0, TXQ 0): polls the heads of its staging rings and sends new frames; sends the own half when the
//                       arrival table shows every other root's half of the previous iteration.
// Every wait is bounded (a broken core ends all loops; the end-of-kernel NoC drain is bounded too).
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
constexpr uint32_t kArriveTab = 3072;
constexpr uint32_t kProbe = 4096;
constexpr uint32_t kWindow = 64;
constexpr uint32_t kArrArr = 8192;
constexpr uint32_t kOwnArr = 10240;
constexpr uint32_t kStallLimit = 50000000;
constexpr uint32_t kSpinLimit = 10000000;
constexpr uint32_t kMaxSrc = 2;
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

// every register a write uses is set here or per write: CTRL (unicast copy write, static VC, acknowledged), return
// address high bits, no transaction id, and the source = this core's L1 (noc_init sets the source fields only on the
// write and write-register buffers; the atomic buffer used for the notices keeps its own values otherwise)
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

// one TT-link L1 write packet on TXQ 0: the three TXQ registers and the command, straight to the queue's register
// block (tunneling.h eth_send_packet_bytes_unsafe without the debug hooks; same registers, same order)
inline __attribute__((always_inline)) void eth_send(uint32_t src, uint32_t dst, uint32_t len) {
    if (txq_wait()) {
        volatile uint32_t* q = reinterpret_cast<volatile uint32_t*>(ETH_TXQ0_REGS_START);
        q[ETH_TXQ_TRANSFER_START_ADDR / 4] = src;
        q[ETH_TXQ_DEST_ADDR / 4] = dst;
        q[ETH_TXQ_TRANSFER_SIZE_BYTES / 4] = len;
        q[ETH_TXQ_CMD / 4] = ETH_TXQ_CMD_START_DATA;
    }
}

// the L0 data cache of this RISC: the firmware disables it unless ENABLE_L1_DATA_CACHE (firmware_common.h
// configure_l1_data_cache: cfg0 bit 3 DisLowCash set); with it off every L1 load reads L1 and the polls need no fence
inline bool l0_cache_off() {
    uint32_t cfg0;
    asm volatile("csrr %0, 0x7c0" : "=r"(cfg0));
    return (cfg0 & 0x8) != 0;
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
    const uint32_t me = get_arg_val<uint32_t>(arg++);
    const uint32_t run_id = get_arg_val<uint32_t>(arg++);
    const uint32_t probe_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t stamp = get_arg_val<uint32_t>(arg++);
    const uint32_t startup_spin = get_arg_val<uint32_t>(arg++);
    const uint32_t full_check = get_arg_val<uint32_t>(arg++);
    const uint32_t own = get_arg_val<uint32_t>(arg++);
    get_arg_val<uint32_t>(arg++);
    const uint32_t n_src = get_arg_val<uint32_t>(arg++);
    uint32_t src_expected[kMaxSrc];
    for (uint32_t s = 0; s < kMaxSrc; ++s) {
        get_arg_val<uint32_t>(arg++);
        get_arg_val<uint32_t>(arg++);
        get_arg_val<uint32_t>(arg++);
        src_expected[s] = get_arg_val<uint32_t>(arg++);
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
    uint32_t own_coord[kMaxOwnCores];
    for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(arg++);
        const uint32_t y = get_arg_val<uint32_t>(arg++);
        own_coord[c] = c < n_own_cores ? coord_of(x, y) : 0;
    }

    const bool need_fence = !l0_cache_off();
    const uint32_t ring_mask = ring_slots - 1;
    const uint32_t stage_mask = stage_slots - 1;
    const uint32_t trailer = payload;
    const uint32_t words = payload / 4;
    const uint32_t last_off = payload - last_payload;

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
        uint32_t next_meta[8];
        for (uint32_t r = 0; r < 8; ++r) {
            next_meta[r] = 0;
        }
        uint32_t received = 0;
        uint32_t head = ring;
        uint32_t head_index = 0;
        uint32_t head_prev = 0;  // the tag last seen in the head slot (seen_tag[head_index]), kept in a register
        uint32_t issued_total = 0;
        uint32_t bad = 0;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while (received < rx_total && !g_broken) {
            ++passes;
            if (need_fence) {
                invalidate_l1_cache();
            }
            const uint32_t w0 = load(head + trailer);
            const uint32_t w1 = load(head + trailer + 16);
            if ((w0 != w1) | (w0 == head_prev) | ((w0 & kTagBit) == 0)) {
                if (++idle == kStallLimit) {
                    break;
                }
                continue;
            }
            idle = 0;
            const uint32_t it = w0 & 0xffff;
            const uint32_t root = (w0 >> 24) & 0x7f;
            const uint32_t pk = (w0 >> 16) & 0xff;
            const bool last_pk = pk == packets - 1;
            const bool window = it - probe_iter < kWindow;
            const uint32_t seen = (window && last_pk) || (it == probe_iter && pk == 0) ? clock_low() : 0;
            const uint32_t off = last_pk ? last_off : 0;
            const uint32_t len = payload - off;
            const uint32_t mask = (child_mask >> (2 * root)) & 3;
            uint32_t issued = 0;
            if (mask & 1) {
                issue_write<kRelayBuf0>(head + off, child_coord[0], child_slot[0] + off, len + 32);
                ++staged[0];
                child_slot[0] = (staged[0] & stage_mask) == 0 ? child_stage[0] : child_slot[0] + stride;
                ++issued;
            }
            if (mask & 2) {
                issue_write<kRelayBuf1>(head + off, child_coord[1], child_slot[1] + off, len + 32);
                ++staged[1];
                child_slot[1] = (staged[1] & stage_mask) == 0 ? child_stage[1] : child_slot[1] + stride;
                ++issued;
            }
            const uint32_t t_issued = it == probe_iter && pk == 0 ? clock_low() : 0;
            if (last_pk) {
                for (uint32_t c = 0; c < n_own_cores; ++c) {
                    issue_write<kNoticeBuf>(head + trailer, own_coord[c], base + kArriveTab + 16 * root, 16);
                }
                issued += n_own_cores;
                if (window) {
                    store(base + kArrArr + 4 * (root * kWindow + (it - probe_iter)), seen);
                    if (it == probe_iter) {
                        store(base + kProbe + 128 + 8 * root, seen);
                        store(base + kProbe + 128 + 8 * root + 4, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H));
                    }
                }
            }
            issued_total += issued;
            if (it == probe_iter && pk == 0 && mask) {
                store(base + kProbe + 192 + 4 * root, seen);
                store(base + kProbe + 224 + 4 * root, t_issued);
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
        uint32_t dst = ring;  // the peer's receive ring slot of the next frame on this link
        uint32_t own_sent = 0;
        uint32_t own_it = 0;
        uint32_t own_pk = 0;
        const uint32_t own_total = own ? iters * packets : 0;
        uint32_t src_count[kMaxSrc] = {0, 0};
        uint32_t head0 = stage;
        uint32_t head1 = stage + stage_slots * stride;
        uint32_t prev0 = 0;
        uint32_t prev1 = 0;
        uint32_t src_seen[kMaxSrc][kMaxRing];
        for (uint32_t s = 0; s < kMaxSrc; ++s) {
            for (uint32_t i = 0; i < stage_slots; ++i) {
                src_seen[s][i] = 0;
            }
        }
        if (n_src < 2) {
            src_expected[1] = 0;
        }
        if (n_src < 1) {
            src_expected[0] = 0;
        }
        uint32_t remaining = own_total + src_expected[0] + src_expected[1];
        uint32_t stamped = 0;
        uint32_t wait_root = me == 0 ? 1 : 0;
        uint32_t idle = 0;
        uint32_t passes = 0;
        const uint32_t t0 = clock_low();
        while (remaining && !g_broken) {
            ++passes;
            if (need_fence) {
                invalidate_l1_cache();
            }
            bool moved = false;
            // the staging rings first: per ring the head address and the tag last seen at the head slot are in
            // registers (a slot's previous tag is fetched once, when the head moves onto it)
            if (src_count[0] != src_expected[0]) {
                const uint32_t w0 = load(head0 + trailer);
                const uint32_t w1 = load(head0 + trailer + 16);
                if ((w0 == w1) & (w0 != prev0) & ((w0 & kTagBit) != 0)) {
                    const bool probe = (w0 & 0xffffff) == probe_iter;
                    const uint32_t seen = probe ? clock_low() : 0;
                    const uint32_t off = ((w0 >> 16) & 0xff) == packets - 1 ? last_off : 0;
                    eth_send(head0 + off, dst + off, payload - off + 32);
                    ++sent;
                    dst = (sent & ring_mask) == 0 ? ring : dst + stride;
                    const uint32_t slot = src_count[0] & stage_mask;
                    src_seen[0][slot] = w0;
                    ++src_count[0];
                    head0 = (src_count[0] & stage_mask) == 0 ? stage : head0 + stride;
                    prev0 = src_seen[0][(slot + 1) & stage_mask];
                    if (probe) {
                        const uint32_t root = (w0 >> 24) & 0x7f;
                        store(base + kProbe + 64 + 4 * root, seen);
                        store(base + kProbe + 96 + 4 * root, clock_low());
                    }
                    --remaining;
                    moved = true;
                }
            }
            if (src_count[1] != src_expected[1]) {
                const uint32_t w0 = load(head1 + trailer);
                const uint32_t w1 = load(head1 + trailer + 16);
                if ((w0 == w1) & (w0 != prev1) & ((w0 & kTagBit) != 0)) {
                    const bool probe = (w0 & 0xffffff) == probe_iter;
                    const uint32_t seen = probe ? clock_low() : 0;
                    const uint32_t off = ((w0 >> 16) & 0xff) == packets - 1 ? last_off : 0;
                    eth_send(head1 + off, dst + off, payload - off + 32);
                    ++sent;
                    dst = (sent & ring_mask) == 0 ? ring : dst + stride;
                    const uint32_t slot = src_count[1] & stage_mask;
                    src_seen[1][slot] = w0;
                    ++src_count[1];
                    head1 = (src_count[1] & stage_mask) == 0 ? stage + stage_slots * stride : head1 + stride;
                    prev1 = src_seen[1][(slot + 1) & stage_mask];
                    if (probe) {
                        const uint32_t root = (w0 >> 24) & 0x7f;
                        store(base + kProbe + 64 + 4 * root, seen);
                        store(base + kProbe + 96 + 4 * root, clock_low());
                    }
                    --remaining;
                    moved = true;
                }
            }
            if (own_sent < own_total) {
                while (own_pk == 0 && own_it > 0 && wait_root < 8) {
                    const uint32_t e = load(base + kArriveTab + 16 * wait_root);
                    if (!((e & kTagBit) && (e & 0xffff) + 1 >= own_it)) {
                        break;
                    }
                    wait_root = wait_root + 1 == me ? me + 1 : wait_root + 1;
                }
                if (own_pk != 0 || own_it == 0 || wait_root >= 8) {
                    if (own_pk == 0) {
                        if (own_it - probe_iter < kWindow) {
                            store(base + kOwnArr + 4 * (own_it - probe_iter), clock_low());
                        }
                        if (own_it == probe_iter || own_it == probe_iter + 1) {
                            store_clock64(base + kProbe + 8 * (own_it - probe_iter));
                        }
                        if (stamp && stamped == own_it) {
                            store_clock64(base + kStamps + 8 * own_it);
                            ++stamped;
                        }
                    }
                    const uint32_t src = own_buf + ((own_it & 1) * packets + own_pk) * stride;
                    const uint32_t tag = kTagBit | (me << 24) | (own_pk << 16) | own_it;
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
                        ++own_it;
                        wait_root = me == 0 ? 1 : 0;
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
        store(base + kState + 24, wait_root | (own_it << 8));
        txq_wait();
        bounded_noc_drain();
        store(base + kState + 60, g_broken ? 0xdead0003u : 0x600d0000u);
    }
}
