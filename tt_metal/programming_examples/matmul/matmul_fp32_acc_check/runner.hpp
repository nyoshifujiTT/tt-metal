// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-side scaffolding shared by the two programs in this directory: placing a variant on a
// core, running one to completion, and running all nine at once behind an aggregator.
//
// The problem itself - operands, layout, reference, bound - is in problem.hpp and is shared with
// the kernels.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "tt-metalium/core_coord.hpp"

#include <fmt/base.h>

#include "problem.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace mm_host {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
// tt_metal has its own Layout, so the problem definitions are reached through an alias rather
// than a using-directive.
namespace problem = mm_fp32_acc_check;
// Text buffer the reporting kernels assemble a line in, under ttsim.
constexpr uint32_t kReportLineBytes = 256;

// ttsim has no device print buffer, so the reporting kernels go through RISC-V semihosting there
// instead of DPRINT, and report only their verdict. Detected from TT_METAL_SIMULATOR, which is
// how the simulator is selected in the first place.
void add_report_define(std::map<std::string, std::string>& defines) {
    if (std::getenv("TT_METAL_SIMULATOR") != nullptr) {
        defines["REPORT_VIA_ECALL"] = "1";
    }
}

// Which compute kernel to run.
//
// LlkInaccurate is the LLK matmul: the ordinary path, which cannot reach FP32 accuracy on data
// whose exponents spread within an SOP group. Fp32Accurate is this example's answer to that.
enum class KernelVariant { LlkInaccurate, Fp32Accurate };

// Where a variant should forward its output tile, when several run at once.
struct AggregatorTarget {
    CoreCoord core;
    uint32_t slot = 0;
    uint32_t semaphore_id = 0;
};

// Places one variant on one core: the reader that materialises the operands, the compute kernel
// under test, and the writer that drains and checks the result.
void place_variant(
    Program& program,
    CoreCoord core,
    const std::shared_ptr<distributed::MeshBuffer>& dst_dram_buffer,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool fp32_dest_acc_en,
    KernelVariant variant,
    uint32_t useful_per_sop,
    problem::Layout layout,
    const AggregatorTarget* aggregator = nullptr) {

    const uint32_t mt = m / TILE_HEIGHT;
    const uint32_t kt = k / TILE_WIDTH;
    const uint32_t nt = n / TILE_WIDTH;

    // The operands never leave the device: the reader materialises them straight into L1 from
    // constant expressions, so only the output needs a DRAM buffer, and the caller owns it.
    const uint32_t input_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    const uint32_t output_tile_size = sizeof(float) * TILE_HEIGHT * TILE_WIDTH;

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

    // Scratch for the writer to drain the output into before checking it. A circular buffer is
    // just a convenient way to have the framework place an L1 region; the writer addresses it
    // directly rather than going through the CB protocol. When several variants share an
    // aggregator the caller places this buffer across all the cores at once, so skip it here.
    if (aggregator == nullptr) {
        constexpr uint32_t scratch_cb_index = CBIndex::c_24;
        CircularBufferConfig cb_scratch_config =
            CircularBufferConfig(output_tile_size, {{scratch_cb_index, cb_output_format}})
                .set_page_size(scratch_cb_index, output_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_scratch_config);

        // Line buffer for the ttsim reporting path, which assembles its text in L1 because that
        // is all the simulator's semihosting reads.
        constexpr uint32_t report_cb_index = CBIndex::c_25;
        CircularBufferConfig cb_report_config =
            CircularBufferConfig(kReportLineBytes, {{report_cb_index, cb_output_format}})
                .set_page_size(report_cb_index, kReportLineBytes);
        tt_metal::CreateCircularBuffer(program, core, cb_report_config);
    }

    tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/dataflow/reader_constexpr_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            // The operands are compile-time constants; the layout under test is all the reader
            // needs to know.
            .defines = {{"LAYOUT_ID", std::to_string(static_cast<uint32_t>(layout))}},
        });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> writer_defines{
        // The writer reaches its own verdict from the same constant expressions the reader uses,
        // so it needs to know which layout is being run.
        {"LAYOUT_ID", std::to_string(static_cast<uint32_t>(layout))}};
    add_report_define(writer_defines);
    if (aggregator != nullptr) {
        writer_defines["AGGREGATOR_SLOT"] = std::to_string(aggregator->slot);
    }

    const auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/dataflow/writer_check_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines,
        });

    std::vector<uint32_t> compute_compile_time_args = {mt, kt, nt};
    const char* compute_kernel = nullptr;
    switch (variant) {
        case KernelVariant::LlkInaccurate:
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp";
            break;
        case KernelVariant::Fp32Accurate:
            // C: one useful value per SOP group, walked as l -> k -> i -> j -> f.
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/compute/mm_fp32_accurate.cpp";
            break;
    }

    std::map<std::string, std::string> compute_defines;
    // Opt-in device-profiler zone around one K iteration, for the timing comparison in the
    // README. Off by default so the accuracy runs are unaffected.
    if (std::getenv("MM_ZONE_PER_K_TILE") != nullptr) {
        compute_defines["ZONE_PER_K_TILE"] = "1";
    }
    if (variant == KernelVariant::Fp32Accurate) {
        compute_defines["USEFUL_PER_SOP"] = std::to_string(useful_per_sop);
    }

    tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines,
        });

    // The reader takes no runtime args: its operands are compile-time constants.
    // Mt and Nt are 1 here, and the writer takes its shape from problem.hpp, so the output
    // address is the only runtime argument it needs.
    std::vector<uint32_t> writer_args{dst_dram_buffer->address()};
    if (aggregator != nullptr) {
        writer_args.push_back(aggregator->core.x);
        writer_args.push_back(aggregator->core.y);
        writer_args.push_back(aggregator->semaphore_id);
    }
    tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
}

// One variant on one core, run to completion, with the output tile read back.
void run_single_core_matmul(
    std::vector<float>& output_tiled,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool fp32_dest_acc_en,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    KernelVariant variant = KernelVariant::LlkInaccurate,
    uint32_t useful_per_sop = 1,
    problem::Layout layout = problem::Layout::Packed) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program{};
    CoreCoord core({0, 0});

    const uint32_t output_tile_size = sizeof(float) * TILE_HEIGHT * TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_output_config{
        .page_size = output_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config_c{
        .size = static_cast<uint32_t>(sizeof(float) * output_tiled.size())};
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_c, dram_output_config, mesh_device.get());

    place_variant(program, core, dst_dram_buffer, m, n, k, fp32_dest_acc_en, variant, useful_per_sop, layout);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output_tiled, dst_dram_buffer, true);
}

// All nine variants at once: USEFUL_PER_SOP 1 through 8, plus the LLK matmul, each on its own
// core, all forwarding their output tile to an aggregator core that reports on the lot.
//
// Running them together is the point. A table assembled from nine separate launches would leave
// the reader to take on trust that the conditions were the same each time; here they are the same
// by construction, and the aggregator can also compare two variants against each other, which no
// single launch can do.
void run_all_variants(
    uint32_t m,
    uint32_t n,
    uint32_t k,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    problem::Layout layout) {
    constexpr uint32_t kVariants = 9;  // USEFUL_PER_SOP 1..8, then the LLK matmul

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program{};

    const CoreCoord aggregator_core({0, 1});
    const CoreRange variant_cores(CoreCoord{0, 0}, CoreCoord{kVariants - 1, 0});
    const CoreRangeSet all_cores(std::vector<CoreRange>{variant_cores, CoreRange(aggregator_core)});

    const uint32_t output_tile_size = sizeof(float) * TILE_HEIGHT * TILE_WIDTH;
    distributed::DeviceLocalBufferConfig dram_output_config{
        .page_size = output_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config_c{.size = output_tile_size};
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_c, dram_output_config, mesh_device.get());

    // Scratch for the output tiles, one slot per variant. Declared identically on every core, and
    // on all of them at once, so the framework places it at the same L1 address everywhere: that
    // is what lets a writer compute the aggregator's slot address from its own scratch pointer.
    // The variant cores use only their own slot; the aggregator reads all of them.
    constexpr uint32_t scratch_cb_index = CBIndex::c_24;
    CircularBufferConfig cb_scratch_config =
        CircularBufferConfig(kVariants * output_tile_size, {{scratch_cb_index, tt::DataFormat::Float32}})
            .set_page_size(scratch_cb_index, output_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_scratch_config);

    const uint32_t agg_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);

    AggregatorTarget target;
    // The writers address the aggregator over the NoC, which uses physical coordinates rather
    // than the logical ones the program is written in.
    target.core = mesh_device->worker_core_from_logical_core(aggregator_core);
    target.semaphore_id = agg_semaphore;

    for (uint32_t i = 0; i < kVariants; ++i) {
        const bool is_llk = (i == kVariants - 1);
        target.slot = i;
        place_variant(
            program,
            CoreCoord{i, 0},
            dst_dram_buffer,
            m,
            n,
            k,
            true,
            is_llk ? KernelVariant::LlkInaccurate : KernelVariant::Fp32Accurate,
            is_llk ? 1 : (i + 1),
            layout,
            &target);
    }

    const auto aggregator_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/dataflow/aggregator.cpp",
        aggregator_core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .defines =
                {{"LAYOUT_ID", std::to_string(static_cast<uint32_t>(layout))},
                 {"NUM_SLOTS", std::to_string(kVariants)}},
        });
    tt_metal::SetRuntimeArgs(program, aggregator_id, aggregator_core, {agg_semaphore});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
}

const char* okng(bool ok) { return ok ? "OK" : "NG"; }

void print_check(const char* name, double expected, double actual, bool ok) {
    fmt::print("check={} expected={} actual={} result={}\n", name, expected, actual, okng(ok));
}

void print_terms(problem::Layout layout) {
    const uint32_t k = problem::k_dim(layout);
    fmt::print("input_terms=[");
    for (uint32_t i = 0; i < k; ++i) {
        fmt::print("{}{}", problem::term_at(layout, i), (i + 1 == k) ? "" : ",");
    }
    fmt::print("]\n");
}

}  // namespace mm_host
