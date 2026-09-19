// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader that materialises the operands directly into L1.
//
// The operands are functions of compile-time constants (see problem.hpp), so there is nothing to
// fetch: no DRAM buffers, no TensorAccessor, and no host-side upload. The reader evaluates the
// same constexpr definitions the reference uses and writes the tile-layout datums into the
// circular buffers.
//
// This also takes DRAM out of the measurement. What the compute kernel waits on is a local L1
// write rather than a NoC read from DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The kernel's own directory is on the include path (Kernel::process_include_paths), so the
// header is reached relative to this file.
#include "../../problem.hpp"

using namespace mm_fp32_acc_check;

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr Layout kLayout = static_cast<Layout>(LAYOUT_ID);
    constexpr uint32_t kK = k_dim(kLayout);
    constexpr uint32_t kKt = kK / kTileWidth;

    // One tile of in0 (A, M-by-K) and one of in1 (B, K-by-N) per step, in the order the compute
    // kernels consume them: for each output tile, the Kt pairs along the reduction dimension.
    // Mt and Nt are 1 here, so this is just the k loop.
    for (uint32_t kt = 0; kt < kKt; ++kt) {
        {
            cb_reserve_back(cb_id_in0, 1);
            volatile tt_l1_ptr uint16_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_in0));
            for (uint32_t r = 0; r < kTileHeight; ++r) {
                for (uint32_t c = 0; c < kTileWidth; ++c) {
                    // Tile-local index: the tile grid stride is handled by kt, not by the map.
                    dst[tiled_index(r, c, kTileWidth)] = to_bf16_bits(a_at(kLayout, r, kt * kTileWidth + c));
                }
            }
            cb_push_back(cb_id_in0, 1);
        }

        {
            cb_reserve_back(cb_id_in1, 1);
            volatile tt_l1_ptr uint16_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_in1));
            for (uint32_t r = 0; r < kTileHeight; ++r) {
                for (uint32_t c = 0; c < kTileWidth; ++c) {
                    // B does not depend on k, but the tile still has to be pushed once per kt
                    // because the compute kernel consumes one of each per step.
                    dst[tiled_index(r, c, kTileWidth)] = to_bf16_bits(b_at(c));
                }
            }
            cb_push_back(cb_id_in1, 1);
        }
    }
}
