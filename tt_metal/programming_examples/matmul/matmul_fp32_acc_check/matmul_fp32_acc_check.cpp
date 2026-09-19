// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// What the SOP alignment window costs, measured across three K-layouts.
//
// The same 32 products are fed through layouts that differ only in how they sit inside the 8-lane
// SOP groups: packed together, spread one per group, or grouped with a shared exponent. The
// difference in accuracy is therefore attributable to the intra-group alignment and nothing else.
//
// This is also the regression test of the pair. It judges on the host, so it runs under ttsim,
// which has no device print buffer and cannot run the device-side reporting that
// matmul_fp32_accurate does.

#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>

#include "runner.hpp"

using namespace mm_host;








int main() {
    constexpr int device_id = 0;
    // Each MVMUL reduces 16 K-elements as two independent 8-lane SOP groups. Products inside a
    // group are aligned to the group's max exponent with only ~12 bits of headroom, so a term more
    // than ~11 binades below its group maximum is dropped before the FP32 accumulator ever sees
    // it. The three K-layouts that probe this are defined in problem.hpp.
    constexpr uint32_t M = problem::kM;
    constexpr uint32_t N = problem::kN;

    bool pass = true;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // One run: build the operands, quantize the reference from the same BF16 datums the device
        // actually receives, run the device matmul, and compare every output element against
        // Higham's bound. Returns the worst observed ratio of |error| to the bound.
        struct RunResult {
            double worst_ratio;      // max over outputs of |err| / bound
            double worst_expected;   // expected value at the worst element
            double worst_actual;     // device value at the worst element
            uint32_t worst_index;
            double bound_at_worst;
            uint32_t within_bound;   // how many of the M*N outputs satisfy the bound
        };

        auto run = [&](problem::Layout layout,
                       bool fp32_dest_acc_en,
                       KernelVariant variant = KernelVariant::LlkInaccurate,
                       uint32_t useful_per_sop = 1) {
            const uint32_t k = problem::k_dim(layout);

            // Reference and bound come from problem.hpp, which the reader uses too, so both sides
            // are built from the same definitions.
            std::vector<double> expected(M * N, 0.0);
            std::vector<double> abs_sum(M * N, 0.0);
            for (uint32_t mm = 0; mm < M; ++mm) {
                for (uint32_t nn = 0; nn < N; ++nn) {
                    expected[mm * N + nn] = problem::expected_at(layout, mm, nn);
                    abs_sum[mm * N + nn] = problem::abs_sum_at(layout, mm, nn);
                }
            }

            std::vector<float> out_tiled(M * N, 0.0f);
            run_single_core_matmul(
                out_tiled, M, N, k, fp32_dest_acc_en, mesh_device, variant, useful_per_sop, layout);
            auto out = untilize_nfaces(out_tiled, M, N);

            // The bound scales with the number of accumulated products, which is K.
            const double g = problem::gamma_n(k);
            // Seed with element 0 so the reported "worst element" is always a real element, even
            // when the layout is exact everywhere and no ratio ever exceeds the seed.
            RunResult r{0.0, expected[0], static_cast<double>(out[0]), 0, g * abs_sum[0], 0};
            for (uint32_t i = 0; i < M * N; ++i) {
                const double bound = g * abs_sum[i];
                const double err = std::fabs(static_cast<double>(out[i]) - expected[i]);
                const double ratio = (bound == 0.0) ? ((err == 0.0) ? 0.0 : INFINITY) : err / bound;
                if (ratio <= 1.0) {
                    ++r.within_bound;
                }
                if (ratio > r.worst_ratio) {
                    r.worst_ratio = ratio;
                    r.worst_expected = expected[i];
                    r.worst_actual = static_cast<double>(out[i]);
                    r.worst_index = i;
                    r.bound_at_worst = bound;
                }
            }
            return r;
        };

        const RunResult packed_r = run(problem::Layout::Packed, true);
        const RunResult spread_r = run(problem::Layout::Spread, true);
        const RunResult uniform_r = run(problem::Layout::Uniform, true);

        print_terms(problem::Layout::Packed);

        fmt::print(
            "note gamma_32={} gamma_256={} checked_elements={}\n", problem::gamma_n(32), problem::gamma_n(256), M * N);

        // spread and uniform must satisfy the FP32 bound; packed must violate it, which is the
        // whole point: the violation is caused by the intra-SOP alignment, not by fp32_dest_acc_en.
        const bool spread_within_bound = spread_r.worst_ratio <= 1.0;
        const bool uniform_within_bound = uniform_r.worst_ratio <= 1.0;
        const bool packed_exceeds_bound = packed_r.worst_ratio > 1.0;

        print_check(
            "spread_layout_worst_element", spread_r.worst_expected, spread_r.worst_actual, spread_within_bound);
        print_check("spread_layout_err_over_fp32_bound", 1.0, spread_r.worst_ratio, spread_within_bound);
        print_check(
            "uniform_layout_worst_element", uniform_r.worst_expected, uniform_r.worst_actual, uniform_within_bound);
        print_check("uniform_layout_err_over_fp32_bound", 1.0, uniform_r.worst_ratio, uniform_within_bound);
        print_check(
            "packed_layout_worst_element", packed_r.worst_expected, packed_r.worst_actual, packed_exceeds_bound);
        print_check("packed_layout_err_over_fp32_bound", 1.0, packed_r.worst_ratio, packed_exceeds_bound);

        // Per-element counts, so a verdict never rests on one element.
        fmt::print(
            "detail elements_within_fp32_bound spread={}/{} uniform={}/{} packed={}/{}\n",
            spread_r.within_bound,
            M * N,
            uniform_r.within_bound,
            M * N,
            packed_r.within_bound,
            M * N);

        fmt::print(
            "detail packed worst_index={} expected={} actual={} bound={} ratio={}\n",
            packed_r.worst_index,
            packed_r.worst_expected,
            packed_r.worst_actual,
            packed_r.bound_at_worst,
            packed_r.worst_ratio);
        fmt::print(
            "note packed_K={} spread_K={} mvmul_instruction_ratio={}x\n",
            problem::k_dim(problem::Layout::Packed),
            problem::k_dim(problem::Layout::Spread),
            problem::k_dim(problem::Layout::Spread) / problem::k_dim(problem::Layout::Packed));

        // C: the zero-injecting kernel runs the same packed layout, at K=32, and must satisfy the
        // FP32 bound that the baseline violates by a factor of 620. Same input, same
        // fp32_dest_acc_en, same fidelity: the only difference is that each SOP group now holds
        // one useful value instead of eight, so there is no intra-group alignment to lose bits to.
        const RunResult acc_r = run(problem::Layout::Packed, true, KernelVariant::Fp32Accurate);
        const bool accurate_within_bound = acc_r.worst_ratio <= 1.0;
        print_check("fp32_accurate_worst_element", acc_r.worst_expected, acc_r.worst_actual, accurate_within_bound);
        print_check("fp32_accurate_err_over_fp32_bound", 1.0, acc_r.worst_ratio, accurate_within_bound);
        fmt::print(
            "detail fp32_accurate elements_within_fp32_bound={}/{} worst_index={} bound={}\n",
            acc_r.within_bound,
            M * N,
            acc_r.worst_index,
            acc_r.bound_at_worst);
        fmt::print(
            "detail fp32_accurate_vs_llk packed_ratio={} fp32_accurate_ratio={}\n",
            packed_r.worst_ratio,
            acc_r.worst_ratio);

        pass = spread_within_bound && uniform_within_bound && packed_exceeds_bound && accurate_within_bound;
        if (!mesh_device->close()) {
            pass = false;
        }

    } catch (const std::exception& e) {
        fmt::print(stderr, "check=exception expected=no_exception actual={} result=NG\n", e.what());
        pass = false;
    }

    if (pass) {
        fmt::print("check=overall expected=SOP_LIMIT_CONFIRMED actual=SOP_LIMIT_CONFIRMED result=OK\n");
        return 0;
    }

    fmt::print("check=overall expected=SOP_LIMIT_CONFIRMED actual=NOT_CONFIRMED result=NG\n");
    return 1;
}
