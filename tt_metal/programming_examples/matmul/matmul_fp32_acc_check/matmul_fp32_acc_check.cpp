// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "tt-metalium/core_coord.hpp"

#include "problem.hpp"

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
// This file already pulls in three namespaces wholesale, and tt_metal has its own Layout, so the
// problem definitions are reached through an alias instead of a fourth using-directive.
namespace problem = mm_fp32_acc_check;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

// Which compute kernel to run: the stock compute-API matmul, the B1 rewrite of it, or the C
// zero-injecting variant.
enum class KernelVariant { ComputeApi, Custom, CustomB2, CustomB3, ZeroInject };

void run_single_core_matmul(
    std::vector<float>& output_tiled,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    bool fp32_dest_acc_en,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    KernelVariant variant = KernelVariant::ComputeApi,
    // Useful values per 8-lane SOP group for the ZeroInject kernel, 1 to 8. Ignored otherwise.
    uint32_t useful_per_sop = 1,
    // Which K-layout the reader should materialise.
    problem::Layout layout = problem::Layout::Packed) {
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

    distributed::DeviceLocalBufferConfig dram_output_config{
        .page_size = output_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config_c{.size = static_cast<uint32_t>(sizeof(float) * output_tiled.size())};

    // The operands never leave the device: the reader materialises them straight into L1 from
    // constant expressions, so only the output needs a DRAM buffer.
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
    const char* compute_kernel = nullptr;
    switch (variant) {
        case KernelVariant::ComputeApi:
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp";
            break;
        case KernelVariant::Custom:
            // B1: same matmul, written out as direct UNPACR and MVMUL sequences.
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/compute/mm_custom.cpp";
            break;
        case KernelVariant::CustomB2:
            // B2: same matmul as B1 with the M dimension contiguous (no i split).
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/compute/mm_custom_b2.cpp";
            break;
        case KernelVariant::CustomB3:
            // B3: same matmul walked as k -> i -> j, with N innermost.
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/compute/mm_custom_b3.cpp";
            break;
        case KernelVariant::ZeroInject:
            // C: one useful value per SOP group, walked as l -> k -> i -> j -> f.
            compute_kernel = OVERRIDE_KERNEL_PREFIX "matmul/matmul_fp32_acc_check/kernels/compute/mm_zero_inject.cpp";
            break;
    }

    std::map<std::string, std::string> compute_defines;
    // Opt-in device-profiler zone around one K iteration, for the timing comparison in the
    // README. Off by default so the accuracy runs are unaffected.
    if (std::getenv("MM_ZONE_PER_K_TILE") != nullptr) {
        compute_defines["ZONE_PER_K_TILE"] = "1";
    }
    if (variant == KernelVariant::ZeroInject) {
        compute_defines["ZI_USEFUL_PER_SOP"] = std::to_string(useful_per_sop);
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
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_dram_buffer->address(), mt, nt});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output_tiled, dst_dram_buffer, true);
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

}  // namespace







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

        // Raw device output for one layout, so the two compute kernels can be compared directly.
        auto run_raw = [&](problem::Layout layout,
                           bool fp32_dest_acc_en,
                           KernelVariant variant,
                           uint32_t useful_per_sop = 1) {
            const uint32_t k = problem::k_dim(layout);
            std::vector<float> out_tiled(M * N, 0.0f);
            run_single_core_matmul(
                out_tiled, M, N, k, fp32_dest_acc_en, mesh_device, variant, useful_per_sop, layout);
            return untilize_nfaces(out_tiled, M, N);
        };

        auto run = [&](problem::Layout layout,
                       bool fp32_dest_acc_en,
                       KernelVariant variant = KernelVariant::ComputeApi,
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

        // B1: the hand-written LLK kernel must reproduce the compute-API kernel bit for bit. Run
        // it on all three layouts so the comparison covers both the exact and the lossy regimes.
        bool custom_matches = true;
        for (const auto& [name, layout] : std::vector<std::pair<const char*, problem::Layout>>{
                 {"packed", problem::Layout::Packed},
                 {"spread", problem::Layout::Spread},
                 {"uniform", problem::Layout::Uniform}}) {
            const auto ref = run_raw(layout, true, KernelVariant::ComputeApi);
            const auto cus = run_raw(layout, true, KernelVariant::Custom);
            uint32_t mismatches = 0;
            for (uint32_t i = 0; i < M * N; ++i) {
                // Bit-exact, not approximate: the two kernels issue the same instructions.
                if (ref[i] != cus[i]) {
                    ++mismatches;
                }
            }
            const bool ok = mismatches == 0;
            custom_matches = custom_matches && ok;
            fmt::print(
                "check=custom_kernel_bitexact_{} expected=0 actual={} result={}\n", name, mismatches, okng(ok));
        }

        // B2 and B3: the same 16 products as B1 without the LLK's M split, issued as k -> j -> i
        // (M innermost) and k -> i -> j (N innermost). The issue order differs from B1, so the
        // accumulation order differs and the results are not bit-identical to B1. What must hold
        // is that they still behave like correct FP32-accumulating matmuls, so they are judged
        // against the same Higham bound: spread and uniform within bound, packed over it.
        bool reorder_ok = true;
        for (const auto& [tag, variant] : std::vector<std::pair<const char*, KernelVariant>>{
                 {"b2_no_i_split", KernelVariant::CustomB2}, {"b3_n_innermost", KernelVariant::CustomB3}}) {
            for (const auto& [name, layout, expect_within] :
                 std::vector<std::tuple<const char*, problem::Layout, bool>>{
                     {"spread", problem::Layout::Spread, true}, {"uniform", problem::Layout::Uniform, true},
                     {"packed", problem::Layout::Packed, false}}) {
                const RunResult r = run(layout, true, variant);
                const bool within = r.worst_ratio <= 1.0;
                const bool ok = (within == expect_within);
                reorder_ok = reorder_ok && ok;
                fmt::print(
                    "check={}_{} expected={} actual={} result={}\n",
                    tag,
                    name,
                    expect_within ? "within_bound" : "over_bound",
                    within ? "within_bound" : "over_bound",
                    okng(ok));
                fmt::print(
                    "detail {} {} err_over_bound={} elements_within={}/{}\n",
                    tag,
                    name,
                    r.worst_ratio,
                    r.within_bound,
                    M * N);
            }
        }

        // C: the zero-injecting kernel runs the same packed layout, at K=32, and must satisfy the
        // FP32 bound that the baseline violates by a factor of 620. Same input, same
        // fp32_dest_acc_en, same fidelity: the only difference is that each SOP group now holds
        // one useful value instead of eight, so there is no intra-group alignment to lose bits to.
        const RunResult zi_r = run(problem::Layout::Packed, true, KernelVariant::ZeroInject);
        const bool zi_within_bound = zi_r.worst_ratio <= 1.0;
        print_check("zero_inject_worst_element", zi_r.worst_expected, zi_r.worst_actual, zi_within_bound);
        print_check("zero_inject_err_over_fp32_bound", 1.0, zi_r.worst_ratio, zi_within_bound);
        fmt::print(
            "detail zero_inject elements_within_fp32_bound={}/{} worst_index={} bound={}\n",
            zi_r.within_bound,
            M * N,
            zi_r.worst_index,
            zi_r.bound_at_worst);
        fmt::print(
            "detail zero_inject_vs_baseline packed_ratio={} zero_inject_ratio={}\n",
            packed_r.worst_ratio,
            zi_r.worst_ratio);

        // The same kernel with 1, 2, 4 and 8 useful values per SOP group. This is the accuracy
        // knob: with S per group, one MVMUL takes 2*S useful K-elements, so the tile needs 8/S
        // passes and 64*8/S MVMULs. S=1 leaves no intra-group alignment at all; raising S widens
        // the alignment window again and the error grows back towards the baseline.
        //
        // S=8 fills every lane, so SrcA holds what B1/B2/B3 unpack in one go and the same 16
        // MVMULs per fidelity phase are issued over it. B3 is the one to compare against: it
        // walks k, i, j in the same order, whereas B1 splits M around N. The remaining difference
        // is where the fidelity phase sits - B3 has it outermost, this kernel innermost - so the
        // four phases and the two k values reach each Dst element in a different order, which FP
        // addition does not have to be indifferent to. Whether that shows up is measured here
        // rather than assumed.
        bool sweep_ok = true;
        const auto b3_ref = run_raw(problem::Layout::Packed, true, KernelVariant::CustomB3);
        double prev_ratio = -1.0;
        for (uint32_t s = 1; s <= 8; ++s) {
            const RunResult r = run(problem::Layout::Packed, true, KernelVariant::ZeroInject, s);
            // S need not divide 8: the passes that absorb the remainder are narrower, and the
            // knob's guarantee is an upper bound on how many values share a group. The cost only
            // changes when ceil(8/S) does, so S=5,6,7 cost the same as S=4 and only lose
            // accuracy - included here to show that, not because they are useful settings.
            const uint32_t passes = (8 + s - 1) / s;
            const uint32_t mvmuls = 64 * passes;
            fmt::print(
                "detail sweep useful_per_sop={} passes={} mvmuls_per_tile={} err_over_bound={} "
                "elements_within={}/{}\n",
                s,
                passes,
                mvmuls,
                r.worst_ratio,
                r.within_bound,
                M * N);

            // Accuracy must degrade monotonically as more values share a group.
            const bool monotone = r.worst_ratio >= prev_ratio;
            if (!monotone) {
                fmt::print(
                    "detail sweep_non_monotone at useful_per_sop={} previous={} current={}\n",
                    s,
                    prev_ratio,
                    r.worst_ratio);
            }
            sweep_ok = sweep_ok && monotone;
            prev_ratio = r.worst_ratio;

            if (s == 1) {
                const bool ok = r.worst_ratio <= 1.0;
                sweep_ok = sweep_ok && ok;
                fmt::print(
                    "check=sweep_s1_within_fp32_bound expected=1 actual={} result={}\n",
                    r.worst_ratio,
                    okng(ok));
            }
            if (s == 8) {
                const auto full = run_raw(problem::Layout::Packed, true, KernelVariant::ZeroInject, 8);
                uint32_t mismatches = 0;
                for (uint32_t i = 0; i < M * N; ++i) {
                    if (full[i] != b3_ref[i]) {
                        ++mismatches;
                    }
                }
                fmt::print(
                    "detail sweep_s8_vs_b3 differing_elements={}/{}\n", mismatches, M * N);
                // S=8 must at least land in the same regime as the ordinary matmul: every lane
                // useful again, so the intra-group alignment is back and the packed layout must
                // exceed the FP32 bound just as B1/B2/B3 do.
                const bool ok = r.worst_ratio > 1.0;
                sweep_ok = sweep_ok && ok;
                fmt::print(
                    "check=sweep_s8_back_to_baseline_regime expected=over_bound actual={} result={}\n",
                    r.worst_ratio > 1.0 ? "over_bound" : "within_bound",
                    okng(ok));
            }
        }
        fmt::print(
            "check=sweep_accuracy_monotone_in_useful_per_sop expected=monotone actual={} result={}\n",
            sweep_ok ? "monotone" : "not_monotone",
            okng(sweep_ok));

        pass = spread_within_bound && uniform_within_bound && packed_exceeds_bound && custom_matches &&
               reorder_ok && zi_within_bound && sweep_ok;
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
