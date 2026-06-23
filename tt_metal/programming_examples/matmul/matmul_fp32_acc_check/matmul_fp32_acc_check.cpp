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

std::vector<bfloat16> build_input_a(uint32_t m, uint32_t k) {
    std::vector<bfloat16> a(m * k, bfloat16(1.0f));
    return a;
}

std::vector<bfloat16> build_input_b_from_terms(uint32_t k, uint32_t n, const std::vector<float>& terms) {
    std::vector<bfloat16> b(k * n, bfloat16(0.0f));
    for (uint32_t kk = 0; kk < k; ++kk) {
        for (uint32_t nn = 0; nn < n; ++nn) {
            b[kk * n + nn] = bfloat16(terms[kk]);
        }
    }
    return b;
}

float run_reference_fp32_acc(const std::vector<float>& terms) {
    float acc = 0.0f;
    for (const float term : terms) {
        acc += term;
    }
    return acc;
}

float run_reference_bf16_sequential_acc(const std::vector<float>& terms) {
    bfloat16 acc(0.0f);
    for (const float term : terms) {
        acc = bfloat16(static_cast<float>(acc) + term);
    }
    return static_cast<float>(acc);
}

void run_single_core_matmul(
    const std::vector<bfloat16>& a_tiled,
    const std::vector<bfloat16>& b_tiled,
    std::vector<bfloat16>& output_tiled,
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
    const uint32_t output_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

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
    distributed::ReplicatedBufferConfig buffer_config_c{.size = static_cast<uint32_t>(sizeof(bfloat16) * output_tiled.size())};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_a, dram_input_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_b, dram_input_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_c, dram_output_config, mesh_device.get());

    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    constexpr tt::DataFormat cb_output_format = tt::DataFormat::Float16_b;
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

void print_check(const char* name, float expected, float actual, bool ok) {
    fmt::print("check={} expected={} actual={} result={}\n", name, expected, actual, okng(ok));
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
    constexpr uint32_t K = 32;  // single matmul_tiles call => single K tile (Kt=1)
    constexpr float TOL = 1e-7f;
    constexpr float EXPECTED_NON_FP32_THEORY = -0.44921875f;  // 8-step rounded accumulation model for this fixed test vector

    bool pass = true;

    try {
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        // Fixed K=32 test vector chosen to make fp32_dest_acc_en=true/false diverge in HiFi4.
        const std::vector<float> terms = {
            1.0f,         0.25f,        -0.00390625f, -0.03125f,    -0.00390625f, 1.0f,         -1.0f,       -0.5f,
            -0.00390625f, -1.0f,        0.5f,         1.0f,         -0.5f,        -1.0f,        -0.5f,       -0.5f,
            0.03125f,     -0.5f,        -0.00390625f, -0.25f,       0.00390625f,  0.25f,        -0.5f,       0.03125f,
            -0.25f,       0.5f,         0.25f,        0.00390625f,  0.5f,         1.0f,         -0.25f,      0.03125f,
        };

        auto a = build_input_a(M, K);
        auto b = build_input_b_from_terms(K, N, terms);
        auto a_tiled = tilize_nfaces(a, M, K);
        auto b_tiled = tilize_nfaces(b, K, N);

        std::vector<bfloat16> out_fp32_tiled(M * N, bfloat16(0.0f));
        std::vector<bfloat16> out_non_fp32_tiled(M * N, bfloat16(0.0f));

        run_single_core_matmul(a_tiled, b_tiled, out_non_fp32_tiled, M, N, K, false, mesh_device);
        run_single_core_matmul(a_tiled, b_tiled, out_fp32_tiled, M, N, K, true, mesh_device);

        auto out_fp32 = untilize_nfaces(out_fp32_tiled, M, N);
        auto out_non_fp32 = untilize_nfaces(out_non_fp32_tiled, M, N);

        const float expected_fp32 = run_reference_fp32_acc(terms);
        const float expected_fp32_rounded_to_bf16 = static_cast<float>(bfloat16(expected_fp32));
        const float expected_bf16_sequential = run_reference_bf16_sequential_acc(terms);
        const float expected_non_fp32_theory = EXPECTED_NON_FP32_THEORY;
        const float actual_fp32 = static_cast<float>(out_fp32[0]);
        const float actual_non_fp32 = static_cast<float>(out_non_fp32[0]);

        const float fp32_err_vs_fp32_ref = std::fabs(actual_fp32 - expected_fp32);
        const float non_fp32_err_vs_fp32_ref = std::fabs(actual_non_fp32 - expected_fp32);

        const bool fp32_matches_fp32_rounded = std::fabs(actual_fp32 - expected_fp32_rounded_to_bf16) <= TOL;
        const bool non_fp32_matches_theory = std::fabs(actual_non_fp32 - expected_non_fp32_theory) <= TOL;
        const bool true_false_different = std::fabs(actual_fp32 - actual_non_fp32) > TOL;
        const bool fp32_is_closer_to_fp32_ref = fp32_err_vs_fp32_ref + TOL < non_fp32_err_vs_fp32_ref;
        const bool non_fp32_differs_from_fp32_rounded = std::fabs(actual_non_fp32 - expected_fp32_rounded_to_bf16) > TOL;

        print_terms(terms);
        print_check("reference_fp32_value", expected_fp32, expected_fp32, true);
        print_check("reference_fp32_rounded_to_bf16", expected_fp32_rounded_to_bf16, expected_fp32_rounded_to_bf16, true);
        print_check("reference_bf16_sequential_acc", expected_bf16_sequential, expected_bf16_sequential, true);
        print_check("reference_non_fp32_theory", expected_non_fp32_theory, expected_non_fp32_theory, true);
        print_check("actual_fp32_mode", expected_fp32_rounded_to_bf16, actual_fp32, fp32_matches_fp32_rounded);
        print_check("actual_non_fp32_mode_matches_theory", expected_non_fp32_theory, actual_non_fp32, non_fp32_matches_theory);
        print_check("actual_non_fp32_mode", expected_fp32_rounded_to_bf16, actual_non_fp32, non_fp32_differs_from_fp32_rounded);
        print_check("actual_fp32_mode_error_vs_reference_fp32", 0.0f, fp32_err_vs_fp32_ref, true);
        print_check("actual_non_fp32_mode_error_vs_reference_fp32", 0.0f, non_fp32_err_vs_fp32_ref, true);
        print_check("true_false_modes_produce_different_results", 1.0f, true_false_different ? 1.0f : 0.0f, true_false_different);
        print_check(
            "actual_fp32_mode_is_closer_to_reference_fp32_than_non_fp32_mode",
            1.0f,
            fp32_is_closer_to_fp32_ref ? 1.0f : 0.0f,
            fp32_is_closer_to_fp32_ref);

        pass =
            fp32_matches_fp32_rounded && non_fp32_matches_theory && true_false_different && fp32_is_closer_to_fp32_ref &&
            non_fp32_differs_from_fp32_rounded;
        if (!mesh_device->close()) {
            pass = false;
        }

    } catch (const std::exception& e) {
        fmt::print(stderr, "check=exception expected=no_exception actual={} result=NG\n", e.what());
        pass = false;
    }

    if (pass) {
        fmt::print("check=overall expected=FP32_ACC_CONFIRMED actual=FP32_ACC_CONFIRMED result=OK\n");
        return 0;
    }

    fmt::print("check=overall expected=FP32_ACC_CONFIRMED actual=NOT_CONFIRMED result=NG\n");
    return 1;
}
