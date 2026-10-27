// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-3b (worklog 14): a chain of dependent projections x_{s+1} = x_s W (K = N = 80 tiles, bf8 weights, HiFi2, fp32
// DEST, 2 rows) on the 130 cores of one chip, Stream-K (80 owners: column c over K [0, k_own); 50 senders over the rest
// of the column-major stream), the owners multicasting each finished column to every core as the next step's x tile
// (kernels/mm/mm_seq_*.cpp).  One go for the whole chain.  Time per dependent step = (an owner's stamp after its last
// column - its stamp after its first column) / (steps - 1) on its own clock, the max over owners; this is the unit's
// time in a frame where every unit waits for the previous one's output (the chain's fixed costs included).
// Accuracy: the final x against a host chain on the same bf8-quantized W with bf16 rounding of x after every step.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/allocator.hpp>
#include "impl/context/metal_context.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kKt = 80, kNt = 80, kRows = 2, kCores = 130, kGridX = 13, kGridY = 10;
constexpr uint32_t kWTile = 1088, kTile = 128, kSlot = 4096 + 32, kHub = 72;
constexpr double kClockGHz = 1.35;

struct Segment {
    uint32_t col, k0, kt, slot;
};

struct CorePlan {
    uint32_t role;
    uint32_t col = 0;
    std::vector<Segment> segs;
    uint32_t n_in = 0;
    tt::tt_metal::CoreCoord core;
};

std::vector<CorePlan> make_plan(uint32_t k_own) {
    const uint32_t rest = kKt - k_own, n_send = kCores - kNt, total = kNt * rest;
    std::vector<CorePlan> owners(kNt);
    for (uint32_t c = 0; c < kNt; ++c) {
        owners[c].role = 0;
        owners[c].col = c;
        owners[c].segs.push_back({c, 0, k_own, 0});
    }
    std::vector<CorePlan> senders(n_send);
    std::vector<uint32_t> start(n_send + 1);
    for (uint32_t s = 0; s <= n_send; ++s) {
        start[s] = static_cast<uint32_t>((static_cast<uint64_t>(total) * s) / n_send) & ~1u;  // even K runs (v4 unit)
    }
    for (uint32_t s = 0; s < n_send; ++s) {
        senders[s].role = 1;
        uint32_t p = start[s];
        while (p < start[s + 1]) {
            const uint32_t c = p / rest, off = p % rest;
            const uint32_t len = std::min(start[s + 1] - p, rest - off);
            senders[s].segs.push_back({c, k_own + off, len, owners[c].n_in++});
            p += len;
        }
        TT_FATAL(senders[s].segs.size() <= 3, "sender {} has {} segments", s, senders[s].segs.size());
    }
    for (auto& o : owners) {
        TT_FATAL(o.n_in <= 3 && (o.n_in >= 1 || rest == 0), "column {} has {} partials", o.col, o.n_in);
    }
    std::vector<CorePlan> plan;
    uint32_t next = 0;
    for (uint32_t c = 0; c < kNt; ++c) {
        plan.push_back(owners[c]);
        while (rest > 0 && next < n_send && start[next] / rest <= c) {
            plan.push_back(senders[next++]);
        }
    }
    while (next < n_send) {
        plan.push_back(senders[next++]);
    }
    TT_FATAL(plan.size() == kCores, "plan has {} cores", plan.size());
    for (uint32_t i = 0; i < kCores; ++i) {
        plan[i].core = tt::tt_metal::CoreCoord(i % kGridX, i / kGridX);
    }
    return plan;
}

std::string run_seq(const std::shared_ptr<distributed::MeshDevice>& mesh, uint32_t k_own, uint32_t steps, bool& ok) {
    auto* dev = mesh->get_devices()[0];
    auto& cq = mesh->mesh_command_queue();
    const tt::tt_metal::Tile tile({kRows, 32});
    const uint32_t spart = tile.get_tile_size(tt::DataFormat::Float32);
    const auto plan = make_plan(k_own);
    std::vector<uint32_t> owner_index(kNt);
    for (uint32_t i = 0; i < kCores; ++i) {
        if (plan[i].role == 0) {
            owner_index[plan[i].col] = i;
        }
    }
    const uint32_t K = kKt * 32, N = kNt * 32;
    std::mt19937 rng(23);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(kRows * K), w(static_cast<size_t>(K) * N);
    for (auto& v : x) {
        v = dist(rng);
    }
    const float sigma = 1.0f / std::sqrt(static_cast<float>(K));
    for (auto& v : w) {
        v = dist(rng) * sigma;
    }
    std::vector<bfloat16> x_bf(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_bf[i] = bfloat16(x[i]);
    }
    // x_0 as Kt 2-row bf16 tiles of 128 B
    const auto x_t = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(x_bf), {kRows, K}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES,
        tile.get_tile_shape(), tile.get_face_shape());
    std::vector<float> w_tiles;
    w_tiles.reserve(static_cast<size_t>(K) * N);
    for (uint32_t c = 0; c < kNt; ++c) {
        for (uint32_t k = 0; k < kKt; ++k) {
            for (uint32_t i = 0; i < 32; ++i) {
                for (uint32_t j = 0; j < 32; ++j) {
                    w_tiles.push_back(w[static_cast<size_t>(k * 32 + i) * N + c * 32 + j]);
                }
            }
        }
    }
    const auto w_bf8 = pack_as_bfp8_tiles(ttsl::Span<const float>(w_tiles), true, false);
    const auto w_q = unpack_bfp8_tiles_into_float_vec(ttsl::Span<const uint32_t>(w_bf8), true, false);
    // host chain: x_{s+1} = bf16(x_s W_q) in fp64
    std::vector<double> xs(kRows * K);
    for (size_t i = 0; i < xs.size(); ++i) {
        xs[i] = static_cast<float>(x_bf[i]);
    }
    for (uint32_t s = 0; s < steps; ++s) {
        std::vector<double> y(kRows * N, 0.0);
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kNt; ++c) {
                for (uint32_t k = 0; k < kKt; ++k) {
                    const float* t = &w_q[static_cast<size_t>(c * kKt + k) * 1024];
                    for (uint32_t i = 0; i < 32; ++i) {
                        const double av = xs[r * K + k * 32 + i];
                        for (uint32_t j = 0; j < 32; ++j) {
                            y[r * N + c * 32 + j] += av * t[i * 32 + j];
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < y.size(); ++i) {
            xs[i] = static_cast<float>(bfloat16(static_cast<float>(y[i])));
        }
    }
    auto dram = [&](uint32_t page, uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            distributed::DeviceLocalBufferConfig{.page_size = page, .buffer_type = tt::tt_metal::BufferType::DRAM},
            mesh.get());
    };
    auto buf_x = dram(kTile, kKt * kTile);
    auto buf_w = dram(kWTile, kNt * kKt * kWTile);
    auto buf_o = dram(kTile, kNt * kTile);
    auto buf_st = dram(512, kCores * 512);
    auto buf_sel = dram(kTile, kTile);
    {
        std::vector<bfloat16> sel_rm(kRows * 32, bfloat16(0.0f));
        for (uint32_t r = 0; r < kRows; ++r) {
            sel_rm[r * 32 + r] = bfloat16(1.0f);
        }
        const auto sel_t = convert_layout<bfloat16>(
            ttsl::Span<const bfloat16>(sel_rm), {kRows, 32}, TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES, tile.get_tile_shape(), tile.get_face_shape());
        distributed::EnqueueWriteMeshBuffer(cq, buf_sel, sel_t, false);
    }
    distributed::EnqueueWriteMeshBuffer(cq, buf_x, x_t, false);
    distributed::EnqueueWriteMeshBuffer(cq, buf_w, w_bf8, false);
    tt::tt_metal::Program program;
    const tt::tt_metal::CoreRangeSet all(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1)));
    std::vector<tt::tt_metal::CoreRange> own_r, send_r;
    uint32_t max_w = 0;
    for (const auto& p : plan) {
        (p.role == 0 ? own_r : send_r).emplace_back(p.core);
        uint32_t t = 0;
        for (const auto& s : p.segs) {
            t += s.kt;
        }
        max_w = std::max(max_w, t);
    }
    const tt::tt_metal::CoreRangeSet owners_set(own_r), senders_set(send_r);
    const uint32_t sem_go = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_done = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_cnt = tt::tt_metal::CreateSemaphore(program, all, 0);
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t page, tt::DataFormat df, const tt::tt_metal::Tile* t) {
        auto cfg = tt::tt_metal::CircularBufferConfig(pages * page, {{index, df}}).set_page_size(index, page);
        if (t) {
            cfg.set_tile_dims(index, *t);
        }
        tt::tt_metal::CreateCircularBuffer(program, all, cfg);
    };
    const tt::tt_metal::Tile ptile({32, 32});
    cb(0, 2 * (kKt + 1), kTile, tt::DataFormat::Float16_b, &tile);
    cb(1, max_w, kWTile, tt::DataFormat::Bfp8_b, nullptr);
    cb(2, 8, kSlot, tt::DataFormat::Float32, &ptile);
    cb(5, 4, kTile, tt::DataFormat::Float16_b, &tile);
    cb(16, 2, kTile, tt::DataFormat::Float16_b, &tile);
    cb(17, 3, spart, tt::DataFormat::Float32, &tile);
    cb(24, 1, 512, tt::DataFormat::UInt32, nullptr);
    cb(25, 1, 32, tt::DataFormat::UInt32, nullptr);
    cb(26, 1, 256, tt::DataFormat::UInt32, nullptr);
    std::vector<uint32_t> rd_ct = {kKt, steps};
    tt::tt_metal::TensorAccessorArgs(*buf_x).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_w).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_o).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_st).append_to(rd_ct);
    auto rd = tt::tt_metal::CreateKernel(
        program, "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_seq_reader.cpp", all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default, .compile_args = rd_ct});
    auto comp_own = tt::tt_metal::CreateKernel(
        program, "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_seq2_compute.cpp", owners_set,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .compile_args = {0u, kKt, steps}});
    auto comp_send = tt::tt_metal::CreateKernel(
        program, "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_seq2_compute.cpp", senders_set,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .compile_args = {1u, kKt, steps}});
    const auto go_phys = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(0, 0));
    const auto mc1 = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1));
    TT_FATAL(plan[kHub].role == 0, "the hub (core {}) must be an owner", kHub);
    const auto hub = dev->worker_core_from_logical_core(plan[kHub].core);
    for (uint32_t i = 0; i < kCores; ++i) {
        const CorePlan& p = plan[i];
        std::vector<uint32_t> ra = {
            static_cast<uint32_t>(buf_x->address()), static_cast<uint32_t>(buf_w->address()),
            static_cast<uint32_t>(buf_o->address()), static_cast<uint32_t>(buf_st->address()), i, p.role, sem_go, sem_done,
            i == 0 ? 1u : 0u, kCores, static_cast<uint32_t>(go_phys.x), static_cast<uint32_t>(go_phys.y),
            static_cast<uint32_t>(mc1.x), static_cast<uint32_t>(mc1.y), static_cast<uint32_t>(go_phys.x),
            static_cast<uint32_t>(go_phys.y), p.col, p.n_in, static_cast<uint32_t>(buf_sel->address()), i == kHub ? 1u : 0u,
            static_cast<uint32_t>(hub.x), static_cast<uint32_t>(hub.y), sem_cnt, kNt, static_cast<uint32_t>(p.segs.size())};
        std::vector<uint32_t> ca = {static_cast<uint32_t>(p.segs.size())};
        for (const auto& s : p.segs) {
            const auto own = dev->worker_core_from_logical_core(plan[owner_index[s.col]].core);
            ra.insert(ra.end(), {s.col * kKt + s.k0, s.kt, static_cast<uint32_t>(own.x), static_cast<uint32_t>(own.y), s.slot});
            ca.insert(ca.end(), {s.k0, s.kt});
        }
        ca.push_back(p.n_in);
        ca.push_back(i == kHub ? 1u : 0u);
        ca.push_back(p.col);
        tt::tt_metal::SetRuntimeArgs(program, rd, p.core, ra);
        tt::tt_metal::SetRuntimeArgs(program, p.role == 0 ? comp_own : comp_send, p.core, ca);
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<bfloat16> o_t;
    distributed::EnqueueReadMeshBuffer(cq, o_t, buf_o, true);
    std::vector<uint32_t> st;
    distributed::EnqueueReadMeshBuffer(cq, st, buf_st, true);
    uint32_t bad_status = 0;
    double step_max = 0, first_max = 0;
    for (uint32_t i = 0; i < kCores; ++i) {
        const uint32_t* s = &st[i * 128];
        if (s[127] != 0x600d0000u) {
            ++bad_status;
        }
        if (plan[i].role == 0 && steps > 1) {
            const uint32_t last = std::min<uint32_t>(steps, 59);
            step_max = std::max(step_max, static_cast<double>(s[last] - s[1]) / (last - 1));
            first_max = std::max(first_max, static_cast<double>(s[1] - s[0]));
        }
    }
    if (steps == 48 && k_own == 50) {
        const char* out_path = std::getenv("QWEN38_MM_OUT");
        std::ofstream raw(std::string(out_path ? out_path : "/tmp/qwen38_mm.txt") + ".seqraw");
        for (uint32_t i = 0; i < kCores; ++i) {
            raw << i << " " << plan[i].core.x << " " << plan[i].core.y << " " << plan[i].role << " " << plan[i].segs.size()
                << " " << plan[i].n_in << " " << (i == kHub ? 1 : 0);
            for (uint32_t w = 64; w < 127; ++w) {
                raw << " " << st[i * 128 + w];
            }
            raw << "\n";
        }
    }
    const auto o_rm = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(o_t), {kRows, N}, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR,
        tile.get_tile_shape(), tile.get_face_shape());
    double sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0;
    const uint32_t n = kRows * N;
    for (uint32_t i = 0; i < n; ++i) {
        const double a = static_cast<float>(o_rm[i]), b = xs[i];
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    const double pcc = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    std::ostringstream os;
    os << "seq k_own " << k_own << " steps " << steps << ": dependent step (owner's own clock, max over owners) "
       << step_max << " cycles = " << step_max / (kClockGHz * 1000.0) << " us; first step " << first_max
       << " cycles; final-x PCC " << pcc << " vs the host chain; bad status " << bad_status;
    ok = ok && bad_status == 0 && pcc > 0.999;
    return os.str();
}

TEST(Qwen38DataflowMm, DependentChain) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const auto grid = mesh->get_devices()[0]->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= kGridX && grid.y >= kGridY, "grid {}x{} is smaller than 13x10", grid.x, grid.y);
    const char* out_path = std::getenv("QWEN38_MM_OUT");
    std::ofstream out(std::string(out_path ? out_path : "/tmp/qwen38_mm.txt") + ".seq");
    bool ok = true;
    out << run_seq(mesh, 54, 1, ok) << "\n";
    out.flush();
    for (uint32_t k_own : {50u, 54u, 58u, 80u}) {
        out << run_seq(mesh, k_own, 48, ok) << "\n";
        out.flush();
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
