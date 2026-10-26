// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1d (worklog 14): the sustained rate of one Ethernet channel (one Ethernet core of a link) sending a stream of
// packets to its peer, to decide whether a direction toward a neighbour is one channel (25 GB/s) or two channels
// (2 x 25 GB/s) -- the host runs one or both cores of the same link at once.  The sender (role 0, ERISC0, TXQ 0)
// sends n packets of `bytes` from a ring of `slots` slots into the receiver's ring, each packet's last 16 B carrying
// its sequence number (the merged message of ethernet_bidirectional_ubench.cpp); the receiver (role 1) polls the
// sequence word of the slot it expects and returns one credit per packet to the sender's stream register 0 over its
// own TXQ 0 (remote_update_ptr_val).  Both cores run on ERISC0 (TXQ 0 only; the TXQ / MAC configuration is never
// written).  The sender stamps the wall clock at the first and the last send; the receiver at the first and the last
// arrival.  A stall watchdog ends every loop in finite time.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

namespace {
constexpr uint32_t kHello = 0;
constexpr uint32_t kFlag = 32;
constexpr uint32_t kResult = 64;
constexpr uint32_t kData = 4096;
constexpr uint32_t kStallLimit = 50000000;
inline uint32_t load(uint32_t addr) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
inline void store(uint32_t addr, uint32_t value) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value; }
}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t bytes = get_arg_val<uint32_t>(1);
    const uint32_t slots = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    const uint32_t run_id = get_arg_val<uint32_t>(4);
    const uint32_t mode = get_arg_val<uint32_t>(5);  // 1: sender, 2: receiver
    const uint32_t kRole = mode - 1;
    if (mode == 0) {
        return;
    }
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
        store(base + kResult + 16 * kRole, 0xdead0001u);
        return;
    }
    uint32_t idle = 0;
    if (kRole == 0) {
        uint32_t t_first = 0;
        uint32_t slot = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t src = base + kData + slot * bytes;
            store(src + bytes - 16, i + 1);
            if (i >= slots) {
                while (get_ptr_val(0) < static_cast<int32_t>(i - slots + 1)) {
                    if (++idle == kStallLimit) {
                        store(base + kResult, 0xdead0002u);
                        store(base + kResult + 4, i);
                        return;
                    }
                }
            }
            while (internal_::eth_txq_is_busy(0)) {
            }
            if (i == 0) {
                t_first = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            }
            internal_::eth_send_packet_bytes_unsafe(0, src, src, bytes);
            slot = slot + 1 == slots ? 0 : slot + 1;
        }
        const uint32_t t_last = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        while (internal_::eth_txq_is_busy(0)) {
        }
        store(base + kResult, 0x600d0000u);
        store(base + kResult + 4, t_first);
        store(base + kResult + 8, t_last);
    } else {
        uint32_t t_first = 0;
        uint32_t slot = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t at = base + kData + slot * bytes + bytes - 16;
            while (true) {
                invalidate_l1_cache();
                if (load(at) == i + 1) {
                    break;
                }
                if (++idle == kStallLimit) {
                    store(base + kResult + 16, 0xdead0003u);
                    store(base + kResult + 20, i);
                    return;
                }
            }
            if (i == 0) {
                t_first = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            }
            while (internal_::eth_txq_is_busy(0)) {
            }
            remote_update_ptr_val<0>(0, 1);
            slot = slot + 1 == slots ? 0 : slot + 1;
        }
        const uint32_t t_last = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        while (internal_::eth_txq_is_busy(0)) {
        }
        store(base + kResult + 16, 0x600d0000u);
        store(base + kResult + 20, t_first);
        store(base + kResult + 24, t_last);
    }
}
