// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3b v7 (worklog 14): column pairs (CT output columns a core, SrcB reused across them: the B=1 unit's ct form).
// I-3b v4 (worklog 14): the compute of one core of the dependent-projection chain with the B=1 matmul unit
// (kernels/mm/mm_b1_unit.h: bf8 HiFi2, 4 x 2 distinct Dst blocks a tile, fp32 SFPU finalize) and custom_mm's unpack.
// Step s: x_s (cb_act, one half of Kt + 1 pages pushed at once by the reader) times the core's resident weight
// segments (cb_w, one column a segment, k-major);
//   sender: per segment, kt tile-MACs into Dst slot 0, finalize, pack the 2-row fp32 partial to cb_part;
//   owner:  its own segment's tile-MACs, then its partials (cb_in_part: 4 fp32 32-row slots a half, n_in of them
//           written by senders, the others zero) through the same unit with SrcA = the partial slot (fp32 -> tf32;
//           HiFi2's two phases take all of tf32's mantissa) and SrcB = the 2-row selector (rows 0 / 1 identity, 4
//           copies), accumulating into the same Dst blocks; one finalize, pack the 2-row bf16 column to cb_out.
// custom_mm's unpack MOP takes 2 K tiles an iteration: every segment and the padded partial count are even.
// The hub owner packs its column straight into slot col of its half (s + 1) & 1 of cb_act (an explicit pack address;
// v3a copied it there with the RISC, ~400 cycles) and signals its reader through cb_out (a 1-page token).
// Compile args: role (0 owner, 1 sender), Kt, steps.  Runtime args: n_seg, per segment (k0, kt), n_in, hub, col.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "mm_b1_unit.h"

namespace {
inline uint32_t clock_low() { return *reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
constexpr uint32_t cb_tstamp = 26, kProfStep = 20;
// the unpacker base address of a CB position (16 B units, minus one: llk_unpack_AB_custom_mm_api.h)
#ifdef TRISC_UNPACK
template <uint32_t ct>
inline void unpack_seg(uint32_t a16, uint32_t b16, uint32_t a_tile16, uint32_t b_tile16, uint32_t kt) {
    ::_llk_unpack_AB_custom_mm_<false, true>(a16, b16, 0, 0, a_tile16, b_tile16, kt, ct);
}
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t steps = get_compile_time_arg_val(2);
    constexpr uint32_t CT = get_compile_time_arg_val(3);  // output columns per core (weights k-major: [k][c])
    constexpr uint32_t Kp = Kt + 1;
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_in_part = 2, cb_sel = 5, cb_out = 16, cb_part = 17;
    constexpr uint32_t cb_dst = role == 1 ? cb_part : cb_out;
    constexpr uint32_t kSlotsHalf = 4;
    const uint32_t n_seg = get_arg_val<uint32_t>(0);
    uint32_t seg_k0[3] = {0, 0, 0};
    uint32_t seg_kt[3] = {0, 0, 0};
    for (uint32_t s = 0; s < n_seg; ++s) {
        seg_k0[s] = get_arg_val<uint32_t>(1 + 2 * s);
        seg_kt[s] = get_arg_val<uint32_t>(2 + 2 * s);
    }
    const uint32_t n_in = get_arg_val<uint32_t>(1 + 2 * n_seg);
    const uint32_t is_hub = get_arg_val<uint32_t>(2 + 2 * n_seg);
    const uint32_t col = get_arg_val<uint32_t>(3 + 2 * n_seg);
    const uint32_t n_pad = n_in + (n_in & 1);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_dst);
    UNPACK((llk_unpack_AB_custom_mm_init<false>(cb_act, cb_w, CT)));
    MATH((qwen38_b1::b1_mm_math_init_ct<2>(CT)));
    uint32_t w_tiles = 0;
    for (uint32_t s = 0; s < n_seg; ++s) {
        w_tiles += seg_kt[s] * CT;
    }
    cb_wait_front(cb_w, w_tiles);
    if constexpr (role == 0) {
        cb_wait_front(cb_sel, kSlotsHalf);
    }
    // per-thread stamps of step kProfStep (measurement build: drained, tensix_sync, so a stamp is the moment the
    // coprocessor finished what this thread issued): words 0..15 unpack, 16..31 math, 32..47 pack of a 64-word page
    volatile tt_l1_ptr uint32_t* ts = nullptr;
    UNPACK({ ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_tstamp).fifo_rd_ptr << cb_addr_shift); });
    PACK({ ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_local_cb_interface(cb_tstamp).fifo_wr_ptr << cb_addr_shift); });
    for (uint32_t st = 0; st < steps; ++st) {
        const bool pf = st == kProfStep;
        cb_wait_front(cb_act, Kp);
        if (pf) {
            UNPACK({ ts[0] = clock_low(); });
            PACK({ ts[32] = clock_low(); });
        }
        uint32_t w0 = 0;
        for (uint32_t s = 0; s < n_seg; ++s) {
            tile_regs_acquire();
            UNPACK({
                const auto& a = get_local_cb_interface(cb_w);
                const auto& b = get_local_cb_interface(cb_act);
                unpack_seg<CT>(
                    a.fifo_rd_ptr - 1 + w0 * a.fifo_page_size,
                    b.fifo_rd_ptr - 1 + seg_k0[s] * b.fifo_page_size,
                    a.fifo_page_size,
                    b.fifo_page_size,
                    seg_kt[s]);
            });
            MATH((qwen38_b1::b1_mm_math_run_ct(seg_kt[s])));
            if (pf) {
                UNPACK({ tensix_sync(); ts[1 + 4 * s] = clock_low(); });
            }
            w0 += seg_kt[s] * CT;
            if (role == 0 && n_in > 0) {
                cb_wait_front(cb_in_part, kSlotsHalf * CT);
                if (pf) {
                    UNPACK({ ts[2] = clock_low(); });
                }
                reconfig_data_format_srca(cb_w, cb_in_part);
                UNPACK({
                    const auto& a = get_local_cb_interface(cb_in_part);
                    const auto& b = get_local_cb_interface(cb_sel);
                    unpack_seg<CT>(a.fifo_rd_ptr - 1, b.fifo_rd_ptr - 1, a.fifo_page_size, b.fifo_page_size, n_pad);
                });
                MATH((qwen38_b1::b1_mm_math_run_ct(n_pad)));
                if (pf) {
                    UNPACK({ tensix_sync(); ts[3] = clock_low(); });
                }
                reconfig_data_format_srca(cb_in_part, cb_w);
                cb_pop_front(cb_in_part, kSlotsHalf * CT);
            }
            MATH((qwen38_b1::b1_mm_math_finalize<2>(CT)));
            if (pf) {
                MATH({ tensix_sync(); });
            }
            tile_regs_commit();
            cb_reserve_back(cb_dst, CT);
            tile_regs_wait();
            if (pf) {
                PACK({ ts[33 + 4 * s] = clock_low(); });
            }
            if (role == 0 && is_hub && st + 1 < steps) {
                // the hub's CT columns straight into slots col .. col + CT - 1 of the next half of x (this core's
                // cb_act; that half is free: the reader re-pushes it only after the hub has multicast it)
                PACK({
                    const auto& a = get_local_cb_interface(cb_act);
                    const uint32_t slot = ((st + 1) & 1) * Kp + col;
                    for (uint32_t c = 0; c < CT; ++c) {
                        ::_llk_pack_<DST_SYNC_MODE, DST_ACCUM_MODE, PackMode::Default>(
                            c, a.fifo_wr_ptr + (slot + c) * a.fifo_page_size - 1);
                    }
                });
            } else {
                for (uint32_t c = 0; c < CT; ++c) {
                    pack_tile(c, cb_dst);
                }
            }
            if (pf) {
                PACK({ tensix_sync(); ts[34 + 4 * s] = clock_low(); });
            }
            tile_regs_release();
            cb_push_back(cb_dst, CT);
        }
        cb_pop_front(cb_act, Kp);
    }
    cb_reserve_back(cb_tstamp, 1);
    cb_push_back(cb_tstamp, 1);
}
