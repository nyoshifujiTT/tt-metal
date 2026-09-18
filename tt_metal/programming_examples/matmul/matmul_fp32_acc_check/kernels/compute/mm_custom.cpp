// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// B1: the same single-core matmul as matmul_single_core/kernels/compute/mm.cpp, but with the
// compute API calls replaced by the LLK calls they expand to.
//
// matmul_init() and matmul_tiles() are thin wrappers (see api/compute/matmul.h):
//   matmul_init  -> state_configure
//                   MATH  (llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose))
//                   UNPACK(llk_unpack_AB_matmul_init(in0, in1, transpose))
//   matmul_tiles -> UNPACK(llk_unpack_AB_matmul(in0, in1, in0_tile, in1_tile))
//                   MATH  (llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst))
//
// Spelling them out here is what makes the later zero-injection work possible: that change needs
// to sit between the unpack and the math call, which is not reachable through matmul_tiles().
// This kernel must stay bit-exact with mm.cpp; the host checks that.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse. Same as mm.cpp.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);

    // Expansion of matmul_init(cb_in0, cb_in1).
    state_configure(cb_in1, cb_in0, __builtin_LINE());
    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(cb_in0, cb_in1, 0 /* transpose */)));
    UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0 /* transpose */)));

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Expansion of matmul_tiles(cb_in0, cb_in1, 0, 0, 0). The unpack call moves one
                // tile of each operand into SrcB/SrcA; the math call issues the MVMUL sequence
                // that accumulates into DST.
                UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, 0 /* in0_tile */, 0 /* in1_tile */)));
                MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(0 /* idst */)));

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
