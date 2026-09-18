// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

// A[m][kk] and B[kk][nn] both vary along every axis, so a transposed or face-swapped operand
// cannot go unnoticed: the K-dependent factor carries the exponent spread under test, and the
// m/n-dependent factors make each output element a distinct value.
//
// A[m][kk] = (1 + m/M) * terms[kk], B[kk][nn] = (1 + nn/N)
// so exact C[m][nn] = (1 + m/M) * (1 + nn/N) * sum_kk terms[kk].
// Every output element is a different multiple of the same sum, which is what makes the
// K-layout comparison meaningful across the whole tile rather than at element 0 only.
std::vector<bfloat16> build_input_a(uint32_t m, uint32_t k, const std::vector<float>& terms) {
    std::vector<bfloat16> a(m * k, bfloat16(0.0f));
    for (uint32_t mm = 0; mm < m; ++mm) {
        const float row_scale = 1.0f + static_cast<float>(mm) / static_cast<float>(m);
        for (uint32_t kk = 0; kk < k; ++kk) {
            a[mm * k + kk] = bfloat16(row_scale * terms[kk]);
        }
    }
    return a;
}

std::vector<bfloat16> build_input_b(uint32_t k, uint32_t n) {
    std::vector<bfloat16> b(k * n, bfloat16(0.0f));
    for (uint32_t kk = 0; kk < k; ++kk) {
        for (uint32_t nn = 0; nn < n; ++nn) {
            b[kk * n + nn] = bfloat16(1.0f + static_cast<float>(nn) / static_cast<float>(n));
        }
    }
    return b;
}


void run_single_core_matmul(
    const std::vector<bfloat16>& a_tiled,
    const std::vector<bfloat16>& b_tiled,
    std::vector<float>& output_tiled,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool fp32_dest_acc_en,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program{};
    CoreCoord core({0, 0});

    const uint32_t mt = m / TILE_HEIGHT;
    const uint32_t kt = k / TILE_WIDTH;
    const uint32_t nt = n / TILE_WIDTH;

    const uint32_t input_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    const uint32_t output_tile_size = sizeof(float) * TILE_HEIGHT * TILE_WIDTH;

    distributed::DeviceLocalBufferConfig dram_input_config{
        .page_size = input_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    distributed::DeviceLocalBufferConfig dram_output_config{
        .page_size = output_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config_a{.size = static_cast<uint32_t>(sizeof(bfloat16) * a_tiled.size())};
    distributed::ReplicatedBufferConfig buffer_config_b{.size = static_cast<uint32_t>(sizeof(bfloat16) * b_tiled.size())};
    distributed::ReplicatedBufferConfig buffer_config_c{.size = static_cast<uint32_t>(sizeof(float) * output_tiled.size())};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_a, dram_input_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_b, dram_input_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_c, dram_output_config, mesh_device.get());

    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat cb_output_format = tt::DataFormat::Float32;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * input_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, input_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * input_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, input_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * output_tile_size, {{output_cb_index, cb_output_format}})
            .set_page_size(output_cb_index, output_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

    const auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args,
        });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    const auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
        });

    std::vector<uint32_t> compute_compile_time_args = {mt, kt, nt};
    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
        });

    tt_metal::SetRuntimeArgs(
        program, reader_id, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), mt, kt, nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_dram_buffer->address(), mt, nt});

    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, a_tiled, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, b_tiled, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output_tiled, dst_dram_buffer, true);
}

const char* okng(bool ok) { return ok ? "OK" : "NG"; }

void print_check(const char* name, double expected, double actual, bool ok) {
    fmt::print("check={} expected={} actual={} result={}\n", name, expected, actual, okng(ok));
}

// Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.1:
// a length-n inner product accumulated in a format with unit roundoff u satisfies
//   |computed - exact| <= gamma_n * sum_k |a_k * b_k|,   gamma_n = n*u / (1 - n*u).
// With an FP32 accumulator u = 2^-24. This is the bound a correct FP32-accumulating dot product
// must respect; it replaces the hand-picked "effective bits" thresholds this example used before,
// which had no derivation behind them.
double gamma_n(uint32_t n) {
    constexpr double u = 1.0 / 16777216.0;  // 2^-24
    const double nu = static_cast<double>(n) * u;
    return nu / (1.0 - nu);
}

void print_terms(const std::vector<float>& terms) {
    fmt::print("input_terms=[");
    for (size_t i = 0; i < terms.size(); ++i) {
        fmt::print("{}{}", terms[i], (i + 1 == terms.size()) ? "" : ",");
    }
    fmt::print("]\n");
}

}  // namespace







int main() {
    constexpr int device_id = 0;
    constexpr uint32_t M = TILE_HEIGHT;
    constexpr uint32_t N = TILE_WIDTH;
    constexpr uint32_t NUM_VALUES = 32;
    // Each MVMUL reduces 16 K-elements as two independent 8-lane SOP groups. Products inside a
    // group are aligned to the group's max exponent with only ~12 bits of headroom, so a term more
    // than ~11 binades below its group maximum is dropped before the FP32 accumulator ever sees it.
    constexpr uint32_t SOP_GROUP_LANES = 8;

    bool pass = true;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // 32 values whose exponents span ~21 bits. Packed together they exceed the intra-group
        // alignment window; spread one-per-group they do not.
        std::vector<float> values(NUM_VALUES);
        for (uint32_t j = 0; j < NUM_VALUES; ++j) {
            values[j] = std::ldexp(1.0f + 0.125f * (j % 8), -3 * static_cast<int>(j % 8));
        }

        std::vector<float> packed(values);
        std::vector<float> spread(NUM_VALUES * SOP_GROUP_LANES, 0.0f);
        for (uint32_t j = 0; j < NUM_VALUES; ++j) {
            spread[SOP_GROUP_LANES * j] = values[j];
        }

        // Third layout: every lane of every SOP group carries a useful value, but all values within
        // a group share one exponent, so no intra-group right-shift happens. This isolates the
        // alignment window from the group size: a full 8-wide group is not itself a problem.
        std::vector<float> uniform(NUM_VALUES, 0.0f);
        for (uint32_t j = 0; j < NUM_VALUES; ++j) {
            const uint32_t group = j / SOP_GROUP_LANES;
            uniform[j] = std::ldexp(1.0f, -3 * static_cast<int>(group));
        }

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

        auto run = [&](const std::vector<float>& terms, bool fp32_dest_acc_en) {
            const uint32_t k = static_cast<uint32_t>(terms.size());
            auto a = build_input_a(M, k, terms);
            auto b = build_input_b(k, N);

            // Reference from the BF16 datums, not from the pre-rounding floats: the device never
            // sees the float values, so comparing against them would fold BF16 input quantization
            // into what is meant to be a measurement of accumulator behaviour.
            std::vector<double> expected(M * N, 0.0);
            std::vector<double> abs_sum(M * N, 0.0);
            for (uint32_t mm = 0; mm < M; ++mm) {
                for (uint32_t nn = 0; nn < N; ++nn) {
                    double acc = 0.0;
                    double abs_acc = 0.0;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        const double term = static_cast<double>(static_cast<float>(a[mm * k + kk])) *
                                            static_cast<double>(static_cast<float>(b[kk * N + nn]));
                        acc += term;
                        abs_acc += std::fabs(term);
                    }
                    expected[mm * N + nn] = acc;
                    abs_sum[mm * N + nn] = abs_acc;
                }
            }

            auto a_tiled = tilize_nfaces(a, M, k);
            auto b_tiled = tilize_nfaces(b, k, N);
            std::vector<float> out_tiled(M * N, 0.0f);
            run_single_core_matmul(a_tiled, b_tiled, out_tiled, M, N, k, fp32_dest_acc_en, mesh_device);
            auto out = untilize_nfaces(out_tiled, M, N);

            // The bound scales with the number of accumulated products, which is K.
            const double g = gamma_n(k);
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

        const RunResult packed_r = run(packed, true);
        const RunResult spread_r = run(spread, true);
        const RunResult uniform_r = run(uniform, true);

        print_terms(values);
        fmt::print(
            "note gamma_32={} gamma_256={} checked_elements={}\n", gamma_n(32), gamma_n(256), M * N);

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
            packed.size(),
            spread.size(),
            spread.size() / packed.size());

        pass = spread_within_bound && uniform_within_bound && packed_exceeds_bound;
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
