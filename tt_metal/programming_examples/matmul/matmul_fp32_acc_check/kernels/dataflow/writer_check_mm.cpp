// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer that checks the result on the device, and forwards it to an aggregator core.
//
// The reference and the error bound are constant expressions (see problem.hpp), so the verdict
// can be reached without the host: the writer drains the output into a scratch area of L1, then
// walks it and reports.
//
// The two phases are deliberate. Draining and checking are separated so that no DPRINT is issued
// while the compute kernel is still running: a full print buffer stalls the device until the host
// drains it, which would back-pressure through the output CB into the compute kernel and show up
// in any timing measurement. By the time anything is printed, the matmul has finished.
//
// The output still goes to DRAM as well. The device-side verdict covers the error bound, but the
// example also compares whole output tiles between kernels for bit-exactness, which needs the
// values themselves on the host.
//
// With AGGREGATOR_SLOT defined, the drained tile is also written into the aggregator core's
// scratch, at the slot this run owns, and a semaphore there is incremented. The aggregator waits
// for all its slots before comparing them, so each sender writes to a distinct region and no
// handshake is needed before the write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// The kernel's own directory is on the include path, so the header is reached relative to this
// file.
#include "../../problem.hpp"

using namespace mm_fp32_acc_check;

namespace {

constexpr Layout kLayout = static_cast<Layout>(LAYOUT_ID);
constexpr uint32_t kOutDatums = kM * kN;

constexpr double const_abs(double x) { return x < 0.0 ? -x : x; }

}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
#ifdef AGGREGATOR_SLOT
    const uint32_t agg_x = get_arg_val<uint32_t>(1);
    const uint32_t agg_y = get_arg_val<uint32_t>(2);
    const uint32_t agg_semaphore = get_semaphore(get_arg_val<uint32_t>(3));
#endif

    constexpr uint32_t cb_id_out0 = 16;
    // Scratch to drain the output into. It is a circular buffer only so that the framework places
    // the L1 region; the writer addresses it directly rather than using the CB protocol.
    constexpr uint32_t cb_id_scratch = 24;
    constexpr uint32_t kTileDatums = kTileHeight * kTileWidth;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr);

    volatile tt_l1_ptr float* scratch = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_id_scratch));

    // Phase 1: drain. Copy out and write to DRAM, but print nothing.
    cb_wait_front(cb_id_out0, 1);
    {
        const uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write_page(0, s, l1_read_addr);
        noc_async_write_barrier();

        volatile tt_l1_ptr float* src = reinterpret_cast<volatile tt_l1_ptr float*>(l1_read_addr);
        for (uint32_t i = 0; i < kTileDatums; ++i) {
            scratch[i] = src[i];
        }

#ifdef AGGREGATOR_SLOT
        // Send this run's tile to the aggregator, then tell it one more slot is filled.
        //
        // The aggregator's scratch is at the same circular buffer index on every core, and
        // circular buffers are placed at the same L1 address across a program, so this core's own
        // scratch address is also the aggregator's. Only the slot within it differs.
        const uint32_t bytes = kTileDatums * sizeof(float);
        const uint32_t agg_scratch_addr = get_write_ptr(cb_id_scratch);
        const uint64_t slot_addr = get_noc_addr(agg_x, agg_y, agg_scratch_addr + AGGREGATOR_SLOT * bytes);
        noc_async_write(l1_read_addr, slot_addr, bytes);
        noc_async_write_barrier();

        const uint64_t sem_addr = get_noc_addr(agg_x, agg_y, agg_semaphore);
        noc_semaphore_inc(sem_addr, 1);
        noc_async_atomic_barrier();
#endif
    }
    cb_pop_front(cb_id_out0, 1);

    // Phase 2: check. The matmul is over, so the stalls a full print buffer causes are harmless.
    //
    // Higham's bound is what a correct FP32-accumulating dot product must respect; the ratio of
    // the observed error to it is the figure of merit. Seed with element 0 so the element the
    // verdict names is always a real one even when nothing exceeds the seed.
    double worst_ratio = 0.0;
    double worst_expected = expected_at(kLayout, 0, 0);
    double worst_actual = static_cast<double>(scratch[tiled_index(0, 0, kN)]);
    double worst_bound = bound_at(kLayout, 0, 0);
    uint32_t worst_index = 0;
    uint32_t within_bound = 0;

    for (uint32_t m = 0; m < kM; ++m) {
        for (uint32_t n = 0; n < kN; ++n) {
            const double got = static_cast<double>(scratch[tiled_index(m, n, kN)]);
            const double want = expected_at(kLayout, m, n);
            const double bound = bound_at(kLayout, m, n);
            const double err = const_abs(got - want);

            // A zero bound means every term was zero, so only an exact result is acceptable.
            const bool ok = (bound == 0.0) ? (err == 0.0) : (err <= bound);
            if (ok) {
                ++within_bound;
            }
            if (bound != 0.0) {
                const double ratio = err / bound;
                if (ratio > worst_ratio) {
                    worst_ratio = ratio;
                    worst_expected = want;
                    worst_actual = got;
                    worst_bound = bound;
                    worst_index = m * kN + n;
                }
            }
        }
    }

    DPRINT(
        "device_check layout={} worst_index={} expected={:.17g} actual={:.17g} bound={:.17g} "
        "err_over_bound={:.17g} within={}/{}\n",
        static_cast<uint32_t>(kLayout),
        worst_index,
        worst_expected,
        worst_actual,
        worst_bound,
        worst_ratio,
        within_bound,
        kOutDatums);
}
