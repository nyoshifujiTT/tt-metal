// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One Ethernet core of a group's meeting (README 12.5): the all-gather of one chunk per chip along the BFS trees of the
// 2 x 4 block, sent directly over the link (no fabric router).  The core drives one link (its peer core on the
// neighbouring chip) and runs two job lists given by the host:
//   tx jobs  a root's chunk sent to the peer frame by frame: the own chunk (root = this chip) as soon as the chip has
//            every chunk of the previous iteration, a relayed chunk as soon as the packet's frame has landed in this
//            core's slot (its trailer words show the packet's count).  One Ethernet packet per frame: the receiver
//            polls the trailer in its own slot (upstream's merged message, ethernet_bidirectional_ubench.cpp MERGE_MSG:
//            the last word of a packet lands last).
//   rx jobs  a root's chunk arriving from the peer: every packet's frame is copied with one NoC write (no flush, no
//            atomic) into the slots of the cores of this chip that forward it; when the whole chunk is in, every core
//            of this chip counts one arrival.
// A frame is [data rounded up to 32 B | trailer 32 B]: the trailer's two 16 B words carry the packet's cumulative
// count, one in each of the receiving NIU's two write streams (0 mod 32, 16 mod 32), each written last in its stream
// (tt-isa-documentation NoC/Ordering.md: within a stream the writes of one packet happen in ascending address order),
// so a forwarder that sees both words has the whole frame.  The root writes the trailer of its own frames; relays pass
// it through unchanged (no RISC store between a NoC landing and a NoC/Ethernet read).
// The loop is self-timed: iteration k of the own chunk starts when the chip holds every chunk of iteration k-1, so the
// period of the loop is the meeting's time (the farthest chunk's arrival) once the chips run in step.  The stamp core
// writes the wall clock at the start of every iteration.  Slots alternate by iteration parity; flags and counters are
// cumulative counts the host zeroes before the launch (no word is reset on the device).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

namespace {

constexpr uint32_t kHello = 0;
constexpr uint32_t kArrivals = 272;
constexpr uint32_t kFlagRing = 288;
constexpr uint32_t kStamps = 1024;
constexpr uint32_t kProbe = 4096;
constexpr uint32_t kState = 4608;
constexpr uint32_t kCredits = 5120;
constexpr uint32_t kSeen = 6144;
constexpr uint32_t kRelayProbe = 14336;
constexpr uint32_t kReleased = 14592;
constexpr uint32_t kReleaseSum = 14720;
constexpr uint32_t kStallLimit = 50000000;
constexpr uint32_t kData = 16384;
constexpr uint32_t kRoots = 8;
constexpr uint32_t kMaxJobs = 8;
constexpr uint32_t kMaxTargets = 4;
constexpr uint32_t kMaxChipCores = 6;

struct TxJob {
    uint32_t root;
    uint32_t relayed;
    uint32_t sent;
    uint32_t up_x;
    uint32_t up_y;
    uint32_t up_index;
    uint32_t iter;
    uint32_t index;
    uint32_t addr;
    uint32_t span;
    uint32_t root_base[2];
    uint64_t credit_dst;
    uint32_t link_ok_until;
};

struct RxJob {
    uint32_t root;
    uint32_t done;
    uint32_t n_targets;
    uint32_t target_x[kMaxTargets];
    uint32_t target_y[kMaxTargets];
    uint32_t iter;
    uint32_t index;
    uint32_t addr;
    uint32_t span;
    uint32_t root_base[2];
    uint32_t target_coord[kMaxTargets];
    uint32_t credit_addr[kMaxTargets];
    uint32_t room_limit;
};

inline uint32_t load(uint32_t addr) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }

inline void store(uint32_t addr, uint32_t value) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value; }

inline uint64_t wall_clock() {
    const uint32_t low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (static_cast<uint64_t>(high) << 32) | low;
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

}  // namespace

constexpr uint32_t kRole =
    get_compile_time_arg_val(0);  // 0: the tx jobs (ERISC0, TXQ 0), 1: the rx jobs (ERISC1, TXQ 1)

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t base = get_arg_val<uint32_t>(arg++);
    const uint32_t iters = get_arg_val<uint32_t>(arg++);
    const uint32_t payload = get_arg_val<uint32_t>(arg++);
    const uint32_t packet = get_arg_val<uint32_t>(arg++);
    const uint32_t packets = get_arg_val<uint32_t>(arg++);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(arg++);
    const uint32_t arrivals_per_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t run_id = get_arg_val<uint32_t>(arg++);
    const uint32_t stamp = get_arg_val<uint32_t>(arg++);
    const uint32_t startup_spin = get_arg_val<uint32_t>(arg++);
    const uint32_t probe_iter = get_arg_val<uint32_t>(arg++);
    const uint32_t n_chip_cores = get_arg_val<uint32_t>(arg++);
    uint32_t chip_core_x[kMaxChipCores];
    uint32_t chip_core_y[kMaxChipCores];
    for (uint32_t i = 0; i < n_chip_cores; ++i) {
        chip_core_x[i] = get_arg_val<uint32_t>(arg++);
        chip_core_y[i] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t n_tx = get_arg_val<uint32_t>(arg++);
    TxJob tx[kMaxJobs];
    for (uint32_t i = 0; i < n_tx; ++i) {
        tx[i].root = get_arg_val<uint32_t>(arg++);
        tx[i].relayed = get_arg_val<uint32_t>(arg++);
        tx[i].up_x = get_arg_val<uint32_t>(arg++);
        tx[i].up_y = get_arg_val<uint32_t>(arg++);
        tx[i].up_index = get_arg_val<uint32_t>(arg++);
        tx[i].sent = 0;
    }
    uint32_t n_rx = get_arg_val<uint32_t>(arg++);
    RxJob rx[kMaxJobs];
    for (uint32_t i = 0; i < n_rx; ++i) {
        rx[i].root = get_arg_val<uint32_t>(arg++);
        rx[i].done = 0;
        rx[i].n_targets = get_arg_val<uint32_t>(arg++);
        for (uint32_t t = 0; t < rx[i].n_targets; ++t) {
            rx[i].target_x[t] = get_arg_val<uint32_t>(arg++);
            rx[i].target_y[t] = get_arg_val<uint32_t>(arg++);
        }
    }

    uint32_t credit_roots[kMaxJobs];
    uint32_t credits_sent[kMaxJobs];
    const uint32_t n_credit = n_rx;
    for (uint32_t i = 0; i < n_rx; ++i) {
        credit_roots[i] = rx[i].root;
        credits_sent[i] = 0;
    }
    if constexpr (kRole == 0) {
        n_rx = 0;
    } else {
        n_tx = 0;
    }
    const uint32_t total = iters * packets;
    const uint32_t last_len = payload - (packets - 1) * packet;
    const uint32_t frame = ((packet + 31) & ~31u) + 32;
    auto slot_of = [&](uint32_t count, uint32_t root) {
        const uint32_t iteration = count / packets;
        const uint32_t index = count % packets;
        return base + kData + ((iteration & 1) * kRoots + root) * slot_bytes + index * frame;
    };
    auto span_of = [&](uint32_t count) {
        return ((((count % packets) == packets - 1 ? last_len : packet) + 31) & ~31u) + 32;
    };
    auto trailer_seen = [&](uint32_t addr, uint32_t value) {
        const uint32_t first = load(addr);
        const uint32_t second = load(addr + 16);
        return (first == value) & (second == value);
    };
    const uint32_t last_span = ((last_len + 31) & ~31u) + 32;
    auto step = [&](uint32_t& index, uint32_t& iter, uint32_t& addr, uint32_t& span, const uint32_t* root_base) {
        if (++index == packets) {
            index = 0;
            ++iter;
            addr = root_base[iter & 1];
        } else {
            addr += frame;
        }
        span = index == packets - 1 ? last_span : frame;
    };
    for (uint32_t i = 0; i < n_rx; ++i) {
        RxJob& job = rx[i];
        for (uint32_t p = 0; p < 2; ++p) {
            job.root_base[p] = base + kData + (p * kRoots + job.root) * slot_bytes;
        }
        job.iter = 0;
        job.index = 0;
        job.addr = job.root_base[0];
        job.span = packets == 1 ? last_span : frame;
        for (uint32_t t = 0; t < job.n_targets; ++t) {
            job.target_coord[t] =
                static_cast<uint32_t>(get_noc_addr(job.target_x[t], job.target_y[t], 0) >> NOC_ADDR_COORD_SHIFT) &
                NOC_COORDINATE_MASK;
            job.credit_addr[t] = base + kCredits + 16 * (job.root * kMaxTargets + t);
        }
        job.room_limit = job.n_targets == 0 ? 0xffffffffu : 2 * packets - 1;
    }
    uint64_t arrival_dst[kMaxChipCores];
    for (uint32_t c = 0; c < n_chip_cores; ++c) {
        arrival_dst[c] = get_noc_addr(chip_core_x[c], chip_core_y[c], base + kArrivals);
    }
    uint32_t release_sum = 0;
    uint32_t release_seen = 0;
    if (n_rx > 0) {
        prepare_write_buf(write_cmd_buf);
        prepare_write_buf(write_reg_cmd_buf);
    }
    for (uint32_t i = 0; i < n_tx; ++i) {
        TxJob& job = tx[i];
        for (uint32_t p = 0; p < 2; ++p) {
            job.root_base[p] = base + kData + (p * kRoots + job.root) * slot_bytes;
        }
        job.iter = 0;
        job.index = 0;
        job.addr = job.root_base[0];
        job.span = packets == 1 ? last_span : frame;
        job.credit_dst =
            get_noc_addr(job.up_x, job.up_y, base + kCredits + 16 * (job.root * kMaxTargets + job.up_index));
        job.link_ok_until = 1;
    }

    for (uint32_t i = 0; i < startup_spin; ++i) {
        asm volatile("nop");
    }
    if constexpr (kRole == 0) {
        for (uint32_t r = 0; r < kRoots; ++r) {
            init_ptr_val(r, 0);
        }
        store(base + kFlagRing, run_id);
        while (internal_::eth_txq_is_busy(0)) {
        }
        internal_::eth_send_packet_bytes_unsafe(0, base + kFlagRing, base + kHello, 16);
    }
    uint32_t hello_wait = 0;
    while (++hello_wait < kStallLimit) {
        invalidate_l1_cache();
        if (load(base + kHello) == run_id) {
            break;
        }
    }
    if (hello_wait >= kStallLimit) {
        store(base + kState + 4 * kRole, 0xdead0001u);
        return;
    }

    uint32_t stamped = 0;
    uint32_t first_own = n_tx;
    for (uint32_t i = 0; i < n_tx; ++i) {
        if (!tx[i].relayed && first_own == n_tx) {
            first_own = i;
        }
    }
    const uint32_t loop_t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t passes = 0;
    bool busy = true;
    uint32_t idle = 0;
    const uint32_t chunks = iters;
    uint32_t progress_mark = 0;
    uint32_t progress = 0;
    while (busy) {
        busy = false;
        ++passes;
        if (progress == progress_mark) {
            if (++idle == kStallLimit) {
                break;
            }
        } else {
            idle = 0;
            progress_mark = progress;
        }
        invalidate_l1_cache();
        if constexpr (kRole == 0) {
            for (uint32_t i = 0; i < n_credit; ++i) {
                if (credits_sent[i] < chunks) {
                    busy = true;
                }
            }
        }
        for (uint32_t i = 0; i < n_rx; ++i) {
            RxJob& job = rx[i];
            if (job.done == total) {
                continue;
            }
            busy = true;
            if (!trailer_seen(job.addr + job.span - 32, job.done + 1)) {
                continue;
            }
            const bool probe = job.done == probe_iter * packets;
            if (probe) {
                store(base + kRelayProbe + 4 * job.root, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L));
            }
            if (job.done > job.room_limit) {
                uint32_t least = load(job.credit_addr[0]);
                for (uint32_t t = 1; t < job.n_targets; ++t) {
                    const uint32_t credit = load(job.credit_addr[t]);
                    least = credit < least ? credit : least;
                }
                job.room_limit = least + 2 * packets - 1;
                if (job.done > job.room_limit) {
                    continue;
                }
            }
            const uint32_t src = job.addr;
            const uint32_t span = job.span;
            for (uint32_t t = 0; t < job.n_targets; ++t) {
                issue_write((t & 1) ? write_reg_cmd_buf : write_cmd_buf, src, job.target_coord[t], src, span);
            }
            if (probe) {
                store(base + kRelayProbe + 96 + 4 * job.root, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L));
            }
            if (job.index == 0) {
                store(base + kSeen + 4 * (job.root * 256 + (job.iter & 255)), load(src));
            }
            ++job.done;
            ++progress;
            step(job.index, job.iter, job.addr, job.span, job.root_base);
            if (job.done == (probe_iter + 1) * packets) {
                const uint64_t now = wall_clock();
                store(base + kProbe + 8 * job.root, static_cast<uint32_t>(now));
                store(base + kProbe + 8 * job.root + 4, static_cast<uint32_t>(now >> 32));
            }
            if (job.index == 0) {
                for (uint32_t c = 0; c < n_chip_cores; ++c) {
                    noc_semaphore_inc(arrival_dst[c], 1);
                }
                noc_async_writes_flushed();
                store(base + kReleased + 16 * job.root, job.iter);
                store(base + kReleaseSum, ++release_sum);
            }
        }
        if constexpr (kRole == 0) {
            const uint32_t sum = load(base + kReleaseSum);
            if (sum != release_seen) {
                release_seen = sum;
                for (uint32_t i = 0; i < n_credit; ++i) {
                    const uint32_t released = load(base + kReleased + 16 * credit_roots[i]);
                    if (released > credits_sent[i]) {
                        while (internal_::eth_txq_is_busy(0)) {
                        }
                        remote_update_ptr_val<0>(credit_roots[i], static_cast<int32_t>(released - credits_sent[i]));
                        credits_sent[i] = released;
                        ++progress;
                    }
                }
            }
        }
        for (uint32_t i = 0; i < n_tx; ++i) {
            TxJob& job = tx[i];
            if (job.sent == total) {
                continue;
            }
            busy = true;
            const uint32_t iteration = job.iter;
            const uint32_t src = job.addr;
            const uint32_t span = job.span;
            const bool ready = job.relayed ? trailer_seen(src + span - 32, job.sent + 1)
                                           : load(base + kArrivals) >= arrivals_per_iter * iteration;
            if (ready && job.relayed && job.sent == probe_iter * packets &&
                load(base + kRelayProbe + 128 + 4 * job.root) == 0) {
                store(base + kRelayProbe + 128 + 4 * job.root, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L));
            }
            if (!ready) {
                continue;
            }
            if (iteration > job.link_ok_until) {
                job.link_ok_until = static_cast<uint32_t>(get_ptr_val(job.root)) + 1;
                if (iteration > job.link_ok_until) {
                    continue;
                }
            }
            if (stamp && i == first_own && job.index == 0 && stamped == iteration) {
                const uint64_t now = wall_clock();
                store(base + kStamps + 8 * iteration, static_cast<uint32_t>(now));
                store(base + kStamps + 8 * iteration + 4, static_cast<uint32_t>(now >> 32));
                ++stamped;
            }
            if (!job.relayed && job.index == 0 && (iteration == probe_iter || iteration == probe_iter + 1)) {
                const uint64_t now = wall_clock();
                const uint32_t at = base + kProbe + 8 * (kRoots + iteration - probe_iter);
                store(at, static_cast<uint32_t>(now));
                store(at + 4, static_cast<uint32_t>(now >> 32));
            }
            if (!job.relayed) {
                store(src + span - 32, job.sent + 1);
                store(src + span - 16, job.sent + 1);
            }
            if (job.sent == probe_iter * packets) {
                store(base + kRelayProbe + 32 + 4 * job.root, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L));
            }
            while (internal_::eth_txq_is_busy(0)) {
            }
            internal_::eth_send_packet_bytes_unsafe(0, src, src, span);
            ++job.sent;
            ++progress;
            step(job.index, job.iter, job.addr, job.span, job.root_base);
            if (job.relayed) {
                noc_inline_dw_write<InlineWriteDst::L1, true>(job.credit_dst, job.sent);
            }
        }
    }
    store(base + kRelayProbe + 64 + 8 * kRole, passes);
    store(base + kRelayProbe + 68 + 8 * kRole, reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - loop_t0);
    store(base + kState + 4 * kRole, idle >= kStallLimit ? 0xdead0000u : 0x600d0000u);
    store(base + kState + 8, load(base + kArrivals));
    for (uint32_t i = 0; i < n_rx; ++i) {
        store(base + kState + 16 + 16 * i, rx[i].root);
        store(base + kState + 16 + 16 * i + 4, rx[i].done);
        store(
            base + kState + 16 + 16 * i + 8,
            rx[i].done < total ? load(slot_of(rx[i].done, rx[i].root) + span_of(rx[i].done) - 32) : 0);
    }
    for (uint32_t i = 0; i < n_tx; ++i) {
        store(base + kState + 160 + 16 * i, tx[i].root | (tx[i].relayed << 16));
        store(base + kState + 160 + 16 * i + 4, tx[i].sent);
        if (tx[i].sent < total) {
            const uint32_t src = slot_of(tx[i].sent, tx[i].root);
            const uint32_t span = span_of(tx[i].sent);
            store(base + kState + 160 + 16 * i + 8, load(src + span - 32));
            store(base + kState + 160 + 16 * i + 12, load(src + span - 16));
        }
    }
    while (internal_::eth_txq_is_busy(0)) {
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
