// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1e (worklog 14): the cycle cost of the steps on the relay path of an Ethernet core's RISC (ERISC1, NoC 1), each
// repeated n times between two wall-clock reads: an empty loop, the wall-clock read itself, an L1 load (dependent),
// an L1 store + load back + use, a NoC command-buffer ready poll, a 5-register NoC write issue to this core's own L1
// (with the ready poll), a noc_semaphore_inc to this core, the NIU "write requests sent" status read, and a 672 B NoC
// write to the peer Ethernet core of the same chip side waited until sent.  Results in cycles per repetition.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t load(uint32_t addr) { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
inline void store(uint32_t addr, uint32_t value) { *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value; }
inline uint32_t now() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t peer_x = get_arg_val<uint32_t>(2);
    const uint32_t peer_y = get_arg_val<uint32_t>(3);
    const uint32_t out = base;
    const uint32_t scratch = base + 1024;
    const uint32_t self_coord =
        static_cast<uint32_t>(get_noc_addr(my_x[noc_index], my_y[noc_index], 0) >> NOC_ADDR_COORD_SHIFT) &
        NOC_COORDINATE_MASK;
    const uint32_t peer_coord =
        static_cast<uint32_t>(get_noc_addr(peer_x, peer_y, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    uint32_t slot = 0;
    auto mark = [&](uint32_t t0) { store(out + 4 * slot++, now() - t0); };
    uint32_t t0;
    uint32_t sink = 0;
    // 0: empty loop
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        asm volatile("" ::: "memory");
    }
    mark(t0);
    // 1: wall clock read
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        sink += now();
    }
    mark(t0);
    // 2: dependent L1 load
    uint32_t addr = scratch;
    store(scratch, scratch);
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        addr = load(addr);
    }
    mark(t0);
    sink += addr;
    // 3: store + load back + use
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        store(scratch + 16, i);
        const uint32_t back = load(scratch + 16);
        asm volatile("addi x0, %0, 0" : : "r"(back));
    }
    mark(t0);
    // 4: store + fence
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        store(scratch + 32, i);
        asm volatile("fence" ::: "memory");
    }
    mark(t0);
    // 5: command buffer ready poll (idle buffer)
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
    }
    mark(t0);
    // 6: a 64 B NoC write to this core's own L1, issued with 5 registers after the ready poll (fixed CTRL)
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(
        noc_index,
        write_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID, 0);
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, scratch);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, scratch + 4096);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, self_coord);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, 64);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;
    }
    mark(t0);
    noc_async_write_barrier();
    // 7: NIU status read (write requests sent)
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        sink += NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    }
    mark(t0);
    // 8: semaphore increment to this core (atomic, non-posted)
    const uint64_t sem = get_noc_addr(my_x[noc_index], my_y[noc_index], scratch + 64);
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        noc_semaphore_inc(sem, 1);
    }
    mark(t0);
    noc_async_atomic_barrier();
    // 9: one 704 B write to the peer core, issue -> all write requests sent (NIU status), one at a time
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, scratch);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, scratch + 8192);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, peer_coord);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, 704);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;
        while (NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT) != noc_nonposted_writes_num_issued[noc_index]) {
        }
    }
    mark(t0);
    // 10: the same, waited until acknowledged (round trip to the peer core)
    t0 = now();
    for (uint32_t i = 0; i < n; ++i) {
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_TARG_ADDR_LO, scratch);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_LO, scratch + 8192);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_COORDINATE, peer_coord);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_AT_LEN_BE, 704);
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
        noc_nonposted_writes_num_issued[noc_index] += 1;
        noc_nonposted_writes_acked[noc_index] += 1;
        noc_async_write_barrier();
    }
    mark(t0);
    store(out + 4 * 31, sink);
    store(out + 4 * 30, 0x600d0000u);
}
