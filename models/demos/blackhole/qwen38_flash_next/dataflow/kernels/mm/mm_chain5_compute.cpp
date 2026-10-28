// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-5 (worklog 14): the fixed costs cut -- partials as 2-row fp32 tiny tiles (256 B, CB 2 / 17 shaped by the host),
// the matmul init once before the rounds (the owner re-inits only around its add), no reconfig on the senders.
// Otherwise as I-1c-2: one core's share of a K-split projection on the 130 cores of a chip (README 12.4 step 4: when
// the output tiles are fewer than the cores, K is split so that every core carries work / 130 tile-MACs and the fp32
// partials are added over the NoC), in the Stream-K assignment: the owner of a column computes its K range [0, k_own)
// and adds the partials of the 1..n_in senders of the column; a sender computes 1..3 segments (column, K range) and
// packs each finished segment as one fp32 tile for its reader to send to the segment's owner.
// x (bf16, one tile row, Kt tiles) and the weight blocks (bf8, the segments' K ranges one after the other, K-major per
// segment) are resident; the init of the round runs before the round's go (cb_go, pushed by the reader), as in the
// pass, where the projection's init precedes the arrival of x: the compute kernel reports ready (cb_ready) after the
// init and the reader reports the core ready for the go only then.
// Compile args: role (0 owner, 1 sender), Kt, rounds.  Runtime args: n_seg, then per segment (k0, kt), then n_in (owner:
// the number of partials of its column, <= 3; cb_in_part has 3 slots and moves by 3 pages a round).
#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

namespace {
// the per-thread stamps of rounds kStamp0 .. kStamp0 + 7 (worklog 14 I-1c-3), in the one-page CB 25 that the pack thread
// pushes at the end for the reader to copy: unpack words [8 r + 0] go seen, [+1] partials seen (owner), [+2 + s] the
// segment's unpacks issued; pack words [64 + 8 r + s] the segment's math done (tile_regs_wait returned), [64 + 8 r + 4 + s]
// the segment's tile packed and pushed
constexpr uint32_t kStamp0 = 8, kStampRounds = 8, cb_tstamp = 25;
inline uint32_t clock_low() { return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t rounds = get_compile_time_arg_val(2);
    constexpr bool kDrain = get_compile_time_arg_val(3) != 0;  // measurement build: drain before every stamp
    // the owner adds a partial P (fp32 32-row tile, rows 0..1 valid) with the matmul itself: sel x P, sel the constant
    // 2-row selector (row r = e_r, bf16, CB 5) -> rows r of P into the DEST accumulator.  P sits in SrcA as tf32 and
    // HiFi2's phases 0 and 1 consume its implicit bit + 9 of its 10 mantissa bits (tt-isa-documentation SrcASrcB.md,
    // "Fidelity phases": the last SrcA bit is consumed by no phase), one bit less than the ELWADD path; sel in SrcB is
    // exact (1.0).  The matmul MOP stays programmed, so neither the add's init nor the matmul's re-init is needed
    constexpr bool kSelAdd = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t cb_sel = 5;
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_go = 3, cb_ready = 4, cb_out = 16, cb_part = 17;
    constexpr uint32_t cb_dst = role == 1 ? cb_part : cb_out;
    constexpr uint32_t in_slots = 3;  // cb_in_part always moves by its 3 pages a round, so its slots stay at fixed addresses
    const uint32_t n_seg = get_arg_val<uint32_t>(0);
    uint32_t seg_k0[3], seg_kt[3];
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_k0[s] = get_arg_val<uint32_t>(1 + 2 * s);
        seg_kt[s] = get_arg_val<uint32_t>(2 + 2 * s);
    }
    const uint32_t n_in = get_arg_val<uint32_t>(1 + 2 * n_seg);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_dst);
    cb_wait_front(cb_act, Kt);
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += seg_kt[s];
    }
    cb_wait_front(cb_w, w_tiles);
    if constexpr (role == 0 && kSelAdd) {
        cb_wait_front(cb_sel, 1);
    }
    volatile tt_l1_ptr uint32_t* ts = nullptr;
    // the unpack thread has only the read side of its CB interfaces set up (trisc.cc: cb_init_write on the pack thread
    // alone), so it takes the page from the read pointer, which equals the write pointer of the unpushed CB
    UNPACK({ ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_tstamp).fifo_rd_ptr << cb_addr_shift); });
    PACK({ ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_tstamp).fifo_wr_ptr << cb_addr_shift); });
    matmul_block_init(cb_act, cb_w, 0, 1, 1, Kt);
    for (uint32_t r = 0; r < rounds; ++r) {
        const bool st = r >= kStamp0 && r < kStamp0 + kStampRounds;
        const uint32_t sr = (r - kStamp0) * 8;
        if constexpr (role == 0 && !kSelAdd) {
            if (r > 0) {
                reconfig_data_format<SrcOrder::Reverse>(cb_act, cb_w);
                matmul_block_init(cb_act, cb_w, 0, 1, 1, Kt);
            }
        }
        cb_reserve_back(cb_ready, 1);
        cb_push_back(cb_ready, 1);
        cb_wait_front(cb_go, 1);
        cb_pop_front(cb_go, 1);
        UNPACK({
            if (st) {
                ts[sr] = clock_low();
            }
        });
        if constexpr (kDrain) {
            // measurement only: wait for the coprocessor to finish what this thread pushed, so the stamps below are
            // execution times (the TRISC otherwise runs ahead of the Tensix pipeline, worklog 14 I-1c-3)
            UNPACK(tensix_sync());
            MATH(tensix_sync());
            PACK(tensix_sync());
        }
        uint32_t w0 = 0;
        for (uint32_t s = 0; s < n_seg; ++s) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < seg_kt[s]; ++k) {
                matmul_block(cb_act, cb_w, seg_k0[s] + k, w0 + k, 0, 0, 1, 1, Kt);
            }
            if constexpr (kDrain) {
                UNPACK(tensix_sync());
            }
            UNPACK({
                if (st) {
                    ts[sr + 2 + s] = clock_low();
                }
            });
            w0 += seg_kt[s];
            if constexpr (role == 0) {
                cb_wait_front(cb_in_part, in_slots);
                UNPACK({
                    if (st) {
                        ts[sr + 1] = clock_low();
                    }
                });
                if constexpr (kSelAdd) {
                    reconfig_data_format_srca(cb_in_part);
                    for (uint32_t j = 0; j < n_in; ++j) {
                        matmul_block(cb_sel, cb_in_part, 0, j, 0, 0, 1, 1, Kt);
                    }
                    reconfig_data_format_srca(cb_w);
                } else {
                reconfig_data_format_srca(cb_in_part);
                reconfig_data_format_srcb(cb_in_part);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in_part);
                for (uint32_t j = 0; j < n_in; ++j) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_in_part, j, 0);
                }
                }
                cb_pop_front(cb_in_part, in_slots);
            }
            tile_regs_commit();
            cb_reserve_back(cb_dst, 1);
            tile_regs_wait();
            if constexpr (kDrain) {
                PACK(tensix_sync());
            }
            PACK({
                if (st) {
                    ts[64 + sr + s] = clock_low();
                }
            });
            pack_tile(0, cb_dst);
            if constexpr (kDrain) {
                PACK(tensix_sync());
            }
            tile_regs_release();
            cb_push_back(cb_dst, 1);
            PACK({
                if (st) {
                    ts[64 + sr + 4 + s] = clock_low();
                }
            });
        }
    }
    cb_reserve_back(cb_tstamp, 1);
    cb_push_back(cb_tstamp, 1);
}
