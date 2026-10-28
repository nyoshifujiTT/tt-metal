// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c (worklog 14): the Tensix matmul unit cost that README 12.4 (step 4) uses -- HiFi2, bf8 weights, bf16
// activation row (one tile row: the decode rows in it), fp32 DEST -- measured on one core with the inputs resident in
// L1 (kernels/mm/mm_bench_*.cpp): cycles per tile-MAC for ct output columns a call (1, 2, 4, 8) and kt K tiles
// (8, 40, 80).  Budget: 32 cycles per tile-MAC (step 4).  Accuracy: PCC of the output row against a host fp64
// reference on the bf8-quantized weights and the bf16 activation.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <cstring>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/context/metal_context.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

struct BenchResult {
    double cycles_per_mac = 0;
    double pcc = 0;
    bool ok = false;
};

enum class Fmt { Bf8, Bf4, Bf16 };

// one core: in0 = `rows` x K (a full 32-row tile, or a tiny tile of 16 / 8 / 4 / 2 / 1 rows with partial faces), in1 =
// kt x ct weight tiles in `fmt` at its served fidelity (bf8 HiFi2, BF4 LoFi, bf16 HiFi4), fp32 DEST
BenchResult bench(
    const std::shared_ptr<distributed::MeshDevice>& mesh, uint32_t kt, uint32_t ct, uint32_t reps, uint32_t rows = 32,
    Fmt fmt = Fmt::Bf8) {
    const uint32_t K = kt * 32, N = ct * 32, M = rows;
    std::mt19937 rng(kt * 131 + ct);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(M * K, 0.0f), w(K * N);
    const uint32_t valid = std::min<uint32_t>(rows, 2);  // the decode rows (2) that carry data
    for (uint32_t r = 0; r < valid; ++r) {
        for (uint32_t k = 0; k < K; ++k) {
            a[r * K + k] = dist(rng);
        }
    }
    for (auto& v : w) {
        v = dist(rng) * 0.05f;
    }
    // device layouts: activation [M, K] bf16 tilized; weight [K, N] bf8 tiles in (k, c) order, k major
    std::vector<bfloat16> a_bf(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a_bf[i] = bfloat16(a[i]);
    }
    const tt::tt_metal::Tile in0_tile({rows, 32});
    const auto a_tiled = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(a_bf),
        {M, K},
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        in0_tile.get_tile_shape(),
        in0_tile.get_face_shape());
    std::vector<float> w_tiles_rm;
    w_tiles_rm.reserve(K * N);
    for (uint32_t k = 0; k < kt; ++k) {
        for (uint32_t c = 0; c < ct; ++c) {
            for (uint32_t i = 0; i < 32; ++i) {
                for (uint32_t j = 0; j < 32; ++j) {
                    w_tiles_rm.push_back(w[(k * 32 + i) * N + c * 32 + j]);
                }
            }
        }
    }
    // pack_as_bfp8_tiles with row_major_input = true takes a row-major tile per 1024 values
    std::vector<uint32_t> w_bf8;
    std::vector<float> w_q;
    if (fmt == Fmt::Bf8) {
        w_bf8 = pack_as_bfp8_tiles(ttsl::Span<const float>(w_tiles_rm), true, false);
        w_q = unpack_bfp8_tiles_into_float_vec(ttsl::Span<const uint32_t>(w_bf8), true, false);
    } else if (fmt == Fmt::Bf4) {
        w_bf8 = pack_as_bfp4_tiles(ttsl::Span<const float>(w_tiles_rm), true, false);
        w_q = unpack_bfp4_tiles_into_float_vec(ttsl::Span<const uint32_t>(w_bf8), true, false);
    } else {
        // bf16 tiles in the nfaces layout, packed two to a word; the reference uses the bf16-rounded values
        std::vector<bfloat16> w16;
        w16.reserve(w_tiles_rm.size());
        for (float v : w_tiles_rm) {
            w16.push_back(bfloat16(v));
        }
        std::vector<bfloat16> w16_nf;
        w16_nf.reserve(w16.size());
        for (size_t t = 0; t < w16.size() / 1024; ++t) {
            std::vector<bfloat16> one(w16.begin() + t * 1024, w16.begin() + (t + 1) * 1024);
            const auto nf = tilize_nfaces(one, 32, 32);
            w16_nf.insert(w16_nf.end(), nf.begin(), nf.end());
        }
        w_bf8.resize(w16_nf.size() / 2);
        std::memcpy(w_bf8.data(), w16_nf.data(), w16_nf.size() * sizeof(bfloat16));
        w_q.resize(w16.size());
        for (size_t i = 0; i < w16.size(); ++i) {
            w_q[i] = static_cast<float>(w16[i]);
        }
    }
    // the reference on the quantized operands
    std::vector<double> ref(valid * N, 0.0);
    for (uint32_t r = 0; r < valid; ++r) {
        for (uint32_t k = 0; k < kt; ++k) {
            for (uint32_t c = 0; c < ct; ++c) {
                const float* t = &w_q[(k * ct + c) * 1024];
                for (uint32_t i = 0; i < 32; ++i) {
                    const double av = static_cast<float>(a_bf[r * K + k * 32 + i]);
                    for (uint32_t j = 0; j < 32; ++j) {
                        ref[r * N + c * 32 + j] += av * t[i * 32 + j];
                    }
                }
            }
        }
    }
    auto* dev = mesh->get_devices()[0];
    (void)dev;
    auto& cq = mesh->mesh_command_queue();
    const tt::DataFormat w_df = fmt == Fmt::Bf8 ? tt::DataFormat::Bfp8_b : fmt == Fmt::Bf4 ? tt::DataFormat::Bfp4_b : tt::DataFormat::Float16_b;
    const uint32_t act_tile = in0_tile.get_tile_size(tt::DataFormat::Float16_b);
    const uint32_t w_tile = tt::tt_metal::Tile({32, 32}).get_tile_size(w_df);
    const uint32_t out_tile = in0_tile.get_tile_size(tt::DataFormat::Float16_b);
    distributed::DeviceLocalBufferConfig dram_a{.page_size = act_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_w{.page_size = w_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    distributed::DeviceLocalBufferConfig dram_o{.page_size = out_tile, .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto buf_a = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = kt * act_tile}, dram_a, mesh.get());
    auto buf_w = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = kt * ct * w_tile}, dram_w, mesh.get());
    auto buf_o = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = ct * out_tile}, dram_o, mesh.get());
    distributed::DeviceLocalBufferConfig dram_s{.page_size = 64, .buffer_type = tt::tt_metal::BufferType::DRAM};
    auto buf_s = distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = 64}, dram_s, mesh.get());
    distributed::EnqueueWriteMeshBuffer(cq, buf_a, a_tiled, false);
    distributed::EnqueueWriteMeshBuffer(cq, buf_w, w_bf8, false);
    tt::tt_metal::Program program;
    const tt::tt_metal::CoreCoord core(0, 0);
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(kt * act_tile, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, act_tile)
            .set_tile_dims(0, in0_tile));
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(kt * ct * w_tile, {{1, w_df}}).set_page_size(1, w_tile));
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(2 * ct * out_tile, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, out_tile)
            .set_tile_dims(16, in0_tile));
    tt::tt_metal::CreateCircularBuffer(
        program, core, tt::tt_metal::CircularBufferConfig(64, {{24, tt::DataFormat::UInt32}}).set_page_size(24, 64));
    std::vector<uint32_t> rd_ct;
    tt::tt_metal::TensorAccessorArgs(*buf_a).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_w).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_o).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_s).append_to(rd_ct);
    auto rd = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_bench_reader.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = rd_ct});
    tt::tt_metal::SetRuntimeArgs(
        program, rd, core, {static_cast<uint32_t>(buf_a->address()), static_cast<uint32_t>(buf_w->address()), static_cast<uint32_t>(buf_o->address()), kt, ct, reps, static_cast<uint32_t>(buf_s->address())});
    tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_bench_compute.cpp",
        core,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = fmt == Fmt::Bf8 ? MathFidelity::HiFi2 : fmt == Fmt::Bf4 ? MathFidelity::LoFi : MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = ct > 4,
            .compile_args = {kt, ct, reps}});
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<bfloat16> out;
    distributed::EnqueueReadMeshBuffer(cq, out, buf_o, true);
    std::vector<uint32_t> st;
    distributed::EnqueueReadMeshBuffer(cq, st, buf_s, true);
    BenchResult res;
    res.ok = st.size() >= 3 && st[2] == 0x600d0000u;
    res.cycles_per_mac = static_cast<double>(st[1] - st[0]) / (static_cast<double>(reps) * kt * ct);
    const auto out_rm = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(out),
        {M, N},
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        in0_tile.get_tile_shape(),
        in0_tile.get_face_shape());
    double sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0;
    const uint32_t n = valid * N;
    for (uint32_t r = 0; r < valid; ++r) {
        for (uint32_t j = 0; j < N; ++j) {
            const double x = static_cast<float>(out_rm[r * N + j]), y = ref[r * N + j];
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
    }
    res.pcc = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return res;
}

TEST(Qwen38DataflowMm, UnitCost) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const char* out_path = std::getenv("QWEN38_MM_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_mm.txt");
    bool ok = true;
    for (uint32_t ct : {1u, 2u, 4u, 8u}) {
        for (uint32_t kt : {8u, 40u, 80u}) {
            const uint32_t reps = std::max<uint32_t>(4, 20480 / (kt * ct));
            const auto r = bench(mesh, kt, ct, reps);
            out << "HiFi2 bf8 x bf16, fp32 DEST: ct " << ct << " kt " << kt << " reps " << reps << ": "
                << r.cycles_per_mac << " cycles/tile-MAC, PCC " << r.pcc << (r.ok ? "" : " (no end stamp)") << "\n";
            out.flush();
            ok = ok && r.ok && r.pcc > 0.999;
        }
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

// I-1c-4: the unit cost against the activation's row count and the weight format (worklog 14: the limit's floor is
// max(MVMUL ceil(r/8) x 4 x phases, weight unpack) = 8 (bf8 / BF4) and 16 (bf16) cycles at r <= 8)
TEST(Qwen38DataflowMm, UnitCostRows) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const char* out_path = std::getenv("QWEN38_MM_OUT");
    std::ofstream out(std::string(out_path ? out_path : "/tmp/qwen38_mm.txt") + ".rows");
    bool ok = true;
    for (Fmt fmt : {Fmt::Bf8, Fmt::Bf4, Fmt::Bf16}) {
        for (uint32_t rows : {32u, 16u, 8u, 4u, 2u, 1u}) {
            for (uint32_t ct : {1u, 4u}) {
                const uint32_t kt = 40;
                const uint32_t reps = std::max<uint32_t>(4, 20480 / (kt * ct));
                const auto r = bench(mesh, kt, ct, reps, rows, fmt);
                out << (fmt == Fmt::Bf8 ? "bf8 HiFi2" : fmt == Fmt::Bf4 ? "BF4 LoFi" : "bf16 HiFi4") << " rows " << rows
                    << " ct " << ct << " kt " << kt << ": " << r.cycles_per_mac << " cycles/tile-MAC, PCC " << r.pcc
                    << (r.ok ? "" : " (no end stamp)") << "\n";
                out.flush();
                ok = ok && r.ok && r.pcc > 0.999;
            }
        }
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
