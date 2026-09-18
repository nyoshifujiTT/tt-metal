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
    constexpr uint32_t NUM_VALUES = 32;
    // Each MVMUL reduces 16 K-elements as two independent 8-lane SOP groups. Products inside a
    // group are aligned to the group's max exponent with only ~12 bits of headroom, so a term more
    // than ~11 binades below its group maximum is dropped before the FP32 accumulator ever sees it.
    constexpr uint32_t SOP_GROUP_LANES = 8;
    constexpr double MIN_BITS_SOP_LIMITED = 16.0;
    constexpr double MIN_BITS_FP32 = 23.0;

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

        auto reference_sum = [](const std::vector<float>& terms) {
            double acc = 0.0;
            for (const float term : terms) {
                acc += static_cast<double>(term);
            }
            return acc;
        };

        auto run = [&](const std::vector<float>& terms) {
            const uint32_t k = static_cast<uint32_t>(terms.size());
            auto a = build_input_a(M, k);
            auto b = build_input_b_from_terms(k, N, terms);
            auto a_tiled = tilize_nfaces(a, M, k);
            auto b_tiled = tilize_nfaces(b, k, N);
            std::vector<float> out_tiled(M * N, 0.0f);
            run_single_core_matmul(a_tiled, b_tiled, out_tiled, M, N, k, true, mesh_device);
            auto out = untilize_nfaces(out_tiled, M, N);
            return static_cast<double>(out[0]);
        };

        const double exact = reference_sum(values);
        const double packed_value = run(packed);
        const double spread_value = run(spread);

        const double uniform_exact = reference_sum(uniform);
        const double uniform_value = run(uniform);

        auto effective_bits_vs = [](double value, double reference) {
            const double rel = std::fabs(value - reference) / std::fabs(reference);
            return (rel == 0.0) ? 24.0 : -std::log2(rel);
        };
        auto effective_bits = [&](double value) { return effective_bits_vs(value, exact); };

        const double packed_bits = effective_bits(packed_value);
        const double spread_bits = effective_bits(spread_value);
        const double uniform_bits = effective_bits_vs(uniform_value, uniform_exact);

        // Same 32 products, same fp32_dest_acc_en, same MathFidelity: only the K-layout differs.
        const bool packed_is_sop_limited = packed_bits < MIN_BITS_SOP_LIMITED;
        const bool spread_reaches_fp32 = spread_bits >= MIN_BITS_FP32;
        const bool layout_changes_accuracy = spread_bits > packed_bits + 1.0;
        // A fully populated 8-wide group keeps FP32-class accuracy as long as its exponents agree,
        // which shows the loss comes from the alignment shift and not from the group width.
        const bool uniform_reaches_fp32 = uniform_bits >= MIN_BITS_FP32;

        print_terms(values);
        print_check("reference_fp64_sum", exact, exact, true);
        print_check("packed_layout_value", exact, packed_value, packed_is_sop_limited);
        print_check("spread_layout_value", exact, spread_value, spread_reaches_fp32);
        print_check("packed_layout_effective_bits", MIN_BITS_SOP_LIMITED, packed_bits, packed_is_sop_limited);
        print_check("spread_layout_effective_bits", MIN_BITS_FP32, spread_bits, spread_reaches_fp32);
        print_check("uniform_exponent_layout_value", uniform_exact, uniform_value, uniform_reaches_fp32);
        print_check("uniform_exponent_layout_effective_bits", MIN_BITS_FP32, uniform_bits, uniform_reaches_fp32);
        print_check(
            "layout_alone_changes_accuracy",
            1.0,
            layout_changes_accuracy ? 1.0 : 0.0,
            layout_changes_accuracy);
        fmt::print(
            "note packed_K={} spread_K={} mvmul_instruction_ratio={}x\n",
            packed.size(),
            spread.size(),
            spread.size() / packed.size());

        pass = packed_is_sop_limited && spread_reaches_fp32 && layout_changes_accuracy && uniform_reaches_fp32;
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
