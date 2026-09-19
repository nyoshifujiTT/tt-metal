// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Aggregator: collects the output tiles of every variant and reports on all of them together.
//
// The nine variants - USEFUL_PER_SOP 1 through 8 and the LLK matmul - run at the same time on
// nine cores, each writing its tile into this core's scratch at its own slot and then bumping the
// semaphore. This core waits for all of them, so the table it prints comes from one run under one
// set of conditions rather than from nine separate launches.
//
// Two things are checked here. Each slot is judged against the Higham bound from problem.hpp, as
// the writers do individually. Then the S=8 slot is compared against the LLK slot: they issue the
// same 64 MVMULs over the same fully occupied SrcA, so their results should differ only by the
// order in which the partial sums reach Dst. That difference is bounded by twice the Higham
// bound, and checking it is what distinguishes "these are the same computation in a different
// order" from "one of them is wrong".

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#include "../../problem.hpp"

using namespace mm_fp32_acc_check;

namespace {

constexpr Layout kLayout = static_cast<Layout>(LAYOUT_ID);
constexpr uint32_t kSlots = NUM_SLOTS;
// Slots 0..7 are USEFUL_PER_SOP 1..8; the last slot is the LLK matmul.
constexpr uint32_t kLlkSlot = kSlots - 1;
constexpr uint32_t kOutDatums = kM * kN;

constexpr double const_abs(double x) { return x < 0.0 ? -x : x; }

}  // namespace

void kernel_main() {
    const uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(0));

    constexpr uint32_t cb_id_scratch = 24;
    volatile tt_l1_ptr float* scratch = reinterpret_cast<volatile tt_l1_ptr float*>(get_write_ptr(cb_id_scratch));

    // Wait for every variant to have delivered its tile.
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
    noc_semaphore_wait_min(sem_ptr, kSlots);

    // Per-slot verdict against the bound, keeping each ratio so the sweep can be judged as a
    // whole afterwards.
    double slot_ratio[kSlots];
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
        volatile tt_l1_ptr float* tile = scratch + slot * kOutDatums;

        double worst_ratio = 0.0;
        uint32_t worst_index = 0;
        double worst_expected = expected_at(kLayout, 0, 0);
        double worst_actual = static_cast<double>(tile[tiled_index(0, 0, kN)]);
        uint32_t within_bound = 0;

        for (uint32_t m = 0; m < kM; ++m) {
            for (uint32_t n = 0; n < kN; ++n) {
                const double got = static_cast<double>(tile[tiled_index(m, n, kN)]);
                const double want = expected_at(kLayout, m, n);
                const double bound = bound_at(kLayout, m, n);
                const double err = const_abs(got - want);

                // A zero bound means every term was zero, so only an exact result will do.
                if ((bound == 0.0) ? (err == 0.0) : (err <= bound)) {
                    ++within_bound;
                }
                if (bound != 0.0) {
                    const double ratio = err / bound;
                    if (ratio > worst_ratio) {
                        worst_ratio = ratio;
                        worst_index = m * kN + n;
                        worst_expected = want;
                        worst_actual = got;
                    }
                }
            }
        }

        slot_ratio[slot] = worst_ratio;

        if (slot == kLlkSlot) {
            DPRINT(
                "variant=llk worst_index={} expected={:.17g} actual={:.17g} err_over_bound={:.17g} "
                "within={}/{}\n",
                worst_index,
                worst_expected,
                worst_actual,
                worst_ratio,
                within_bound,
                kOutDatums);
        } else {
            const uint32_t s = slot + 1;
            const uint32_t passes = (8 + s - 1) / s;
            DPRINT(
                "variant=useful_per_sop={} passes={} mvmuls_per_tile={} worst_index={} "
                "expected={:.17g} actual={:.17g} err_over_bound={:.17g} within={}/{}\n",
                s,
                passes,
                64 * passes,
                worst_index,
                worst_expected,
                worst_actual,
                worst_ratio,
                within_bound,
                kOutDatums);
        }
    }

    // The accuracy has to degrade monotonically as more values share a group: that is the whole
    // claim the knob makes. Judged here rather than on the host because it spans all nine runs,
    // which only this core sees together.
    {
        bool monotone = true;
        uint32_t first_bad = 0;
        for (uint32_t s = 1; s < 8; ++s) {
            if (slot_ratio[s] < slot_ratio[s - 1]) {
                monotone = false;
                if (first_bad == 0) {
                    first_bad = s + 1;
                }
            }
        }
        DPRINT(
            "check=accuracy_monotone_in_useful_per_sop expected=1 actual={} first_regression_at={}\n",
            monotone ? 1u : 0u,
            first_bad);

        // S=1 must satisfy the bound and S=8 must not: the knob really does span from FP32
        // accuracy back to what the LLK matmul achieves.
        DPRINT(
            "check=s1_within_bound expected=1 actual={} err_over_bound={:.17g}\n",
            slot_ratio[0] <= 1.0 ? 1u : 0u,
            slot_ratio[0]);
        DPRINT(
            "check=s8_over_bound expected=1 actual={} err_over_bound={:.17g}\n",
            slot_ratio[7] > 1.0 ? 1u : 0u,
            slot_ratio[7]);
    }

    // S=8 against the LLK matmul. Both reduce the same 32 products with the same intra-group
    // truncation; only the order in which the partial sums reach Dst differs. Summing n terms in
    // two different orders can separate the results by at most 2 * gamma_n * sum|a_k b_k|, so
    // that is the bound the difference has to respect. Exceeding it would mean the two are not
    // the same computation.
    {
        volatile tt_l1_ptr float* s8 = scratch + 7 * kOutDatums;
        volatile tt_l1_ptr float* llk = scratch + kLlkSlot * kOutDatums;

        double worst_ratio = 0.0;
        uint32_t worst_index = 0;
        uint32_t differing = 0;

        for (uint32_t m = 0; m < kM; ++m) {
            for (uint32_t n = 0; n < kN; ++n) {
                const uint32_t idx = tiled_index(m, n, kN);
                const double a = static_cast<double>(s8[idx]);
                const double b = static_cast<double>(llk[idx]);
                if (a != b) {
                    ++differing;
                }
                const double bound = 2.0 * bound_at(kLayout, m, n);
                if (bound != 0.0) {
                    const double ratio = const_abs(a - b) / bound;
                    if (ratio > worst_ratio) {
                        worst_ratio = ratio;
                        worst_index = m * kN + n;
                    }
                }
            }
        }

        DPRINT(
            "check=s8_vs_llk_order_only expected=1 actual={:.17g} differing={}/{} worst_index={} "
            "within_order_bound={}\n",
            worst_ratio,
            differing,
            kOutDatums,
            worst_index,
            // DPRINT has no string or char type, so the verdict is a number: 1 means the two
            // differ by no more than reordering an FP32 sum permits.
            worst_ratio <= 1.0 ? 1u : 0u);
    }
}
