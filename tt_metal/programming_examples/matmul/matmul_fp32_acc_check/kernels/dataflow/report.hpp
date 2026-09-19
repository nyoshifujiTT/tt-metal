// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Getting a line of text out of a kernel, on silicon and under ttsim.
//
// On silicon that is DPRINT. ttsim does not implement the device print buffer, so a kernel that
// uses DPRINT there waits forever for a host flush that never comes; instead it offers RISC-V
// semihosting, which is just an `ecall` the simulator intercepts. Setting REPORT_VIA_ECALL picks
// that path, and then the kernels report the same lines under both.
//
// Two constraints on the ttsim path, both from the simulator's side:
//   - semihosting has to be enabled with TTSIM_SEMIHOSTING=1, or libttsim_syscall raises
//     ConfigurationError
//   - only BRISC may issue it (riscv_id 0); other cores raise UnsupportedFunctionality. The
//     reporting kernels here are all DataMovementProcessor::RISCV_0, which is BRISC
//
// The buffer has to live in L1, because that is all the simulator's syscall_mem_rd reaches.

#pragma once

#include <cstdint>

// Size of the L1 line buffer the reporting kernels assemble text in. Must match the circular
// buffer the host places for it.
constexpr uint32_t kReportLineBytes = 256;

#ifdef REPORT_VIA_ECALL

namespace report {

// sys_write(1, buf, count), newlib's syscall numbering, which is what ttsim implements.
inline void write_stdout(const char* buf, uint32_t count) {
    register uint32_t a0 asm("a0") = 1;  // stdout
    register uint32_t a1 asm("a1") = reinterpret_cast<uint32_t>(buf);
    register uint32_t a2 asm("a2") = count;
    register uint32_t a7 asm("a7") = 64;  // SYS_write
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
}

// Just enough formatting to render the reports, since there is no libc here.
class Line {
public:
    explicit Line(char* buf, uint32_t capacity) : buf_(buf), capacity_(capacity) {}

    Line& str(const char* s) {
        while (*s != '\0' && len_ + 1 < capacity_) {
            buf_[len_++] = *s++;
        }
        return *this;
    }

    void flush() {
        if (len_ + 1 < capacity_) {
            buf_[len_++] = '\n';
        }
        write_stdout(buf_, len_);
        len_ = 0;
    }

private:
    char* buf_;
    uint32_t capacity_;
    uint32_t len_ = 0;
};

}  // namespace report

#endif  // REPORT_VIA_ECALL
