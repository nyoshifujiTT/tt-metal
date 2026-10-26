// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-2 (worklog 14): the in_proj pool of one chip (README 12.4 step 4: qkv 40 + z 24 + a 1 + b 1 = 66 output column
// tiles, K = H = 2,560 = 80 tiles, bf8 weights, HiFi2, fp32 DEST) on the 130 worker cores of one chip in the Stream-K
// assignment (kernels/mm/mm_chain_*.cpp): 66 owner cores compute K [0, k_own) of their column, 64 sender cores split
// the remaining 66 x (80 - k_own) tile-MACs into contiguous ranges of the column-major stream (cut at the column
// boundaries into 1..3 segments) and send each finished fp32 partial to the column's owner, which adds it in DEST and
// packs the bf16 column.  Inputs resident in L1 (DRAM reads are I-1b' / I-3).  A round starts on a multicast go after
// every core has run its init; the round time is the earliest go -> the last owner's packed column (one chip; the
// per-core end - go is reported as well, independent of the cores' clock offsets).
// Budget: step 4, in_proj pool 1.10 us (5,280 tile-MACs / 130 x 32 cycles + one tree level of 188 cycles).
// Accuracy: PCC of the 66 columns (2 decode rows) against a host fp64 reference on the bf8-quantized weights and the
// bf16 activation.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
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
#include "impl/context/metal_context.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kNt = 66, kKt = 80, kM = 32, kCores = 130, kGridX = 13, kGridY = 10;
constexpr uint32_t kRounds = 48, kStampRounds = 40, kSkip = 8;
constexpr uint32_t kActTile = 2048, kWTile = 1088, kOutTile = 2048, kPartPage = 4096 + 32;
constexpr double kClockGHz = 1.35;

struct Segment {
    uint32_t col, k0, kt, slot;
};

struct CorePlan {
    uint32_t role;  // 0 owner, 1 sender
    uint32_t col;   // owner: its column
    std::vector<Segment> segs;
    uint32_t n_in = 0;
    tt::tt_metal::CoreCoord core;
};

// the Stream-K plan: owners and senders interleaved along the stream so a sender sits next to its owners
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
        start[s] = static_cast<uint32_t>((static_cast<uint64_t>(total) * s) / n_send);
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
        TT_FATAL(o.n_in >= 1 && o.n_in <= 3, "column {} has {} partials", o.col, o.n_in);
    }
    std::vector<CorePlan> plan;
    uint32_t next_sender = 0;
    for (uint32_t c = 0; c < kNt; ++c) {
        plan.push_back(owners[c]);
        while (next_sender < n_send && start[next_sender] / rest <= c) {
            plan.push_back(senders[next_sender++]);
        }
    }
    while (next_sender < n_send) {
        plan.push_back(senders[next_sender++]);
    }
    TT_FATAL(plan.size() == kCores, "plan has {} cores", plan.size());
    for (uint32_t i = 0; i < kCores; ++i) {
        plan[i].core = tt::tt_metal::CoreCoord(i % kGridX, i / kGridX);
    }
    return plan;
}

std::string run_chain(const std::shared_ptr<distributed::MeshDevice>& mesh, uint32_t k_own, bool& ok) {
    auto* dev = mesh->get_devices()[0];
    auto& cq = mesh->mesh_command_queue();
    const auto plan = make_plan(k_own);
    std::vector<uint32_t> owner_index(kNt);
    for (uint32_t i = 0; i < kCores; ++i) {
        if (plan[i].role == 0) {
            owner_index[plan[i].col] = i;
        }
    }
    const uint32_t K = kKt * 32, N = kNt * 32;
    std::mt19937 rng(7 + k_own);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(kM * K, 0.0f), w(static_cast<size_t>(K) * N);
    for (uint32_t k = 0; k < K; ++k) {
        a[k] = dist(rng);
        a[K + k] = dist(rng);
    }
    for (auto& v : w) {
        v = dist(rng) * 0.02f;
    }
    std::vector<bfloat16> a_bf(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a_bf[i] = bfloat16(a[i]);
    }
    const auto a_tiled = tilize_nfaces(a_bf, kM, K);
    // weight tiles column-major: tile (k, c) at c * kKt + k, every tile row-major inside
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
    std::vector<double> ref(2 * N, 0.0);
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t c = 0; c < kNt; ++c) {
            for (uint32_t k = 0; k < kKt; ++k) {
                const float* t = &w_q[static_cast<size_t>(c * kKt + k) * 1024];
                for (uint32_t i = 0; i < 32; ++i) {
                    const double av = static_cast<float>(a_bf[r * K + k * 32 + i]);
                    for (uint32_t j = 0; j < 32; ++j) {
                        ref[r * N + c * 32 + j] += av * t[i * 32 + j];
                    }
                }
            }
        }
    }
    auto dram = [&](uint32_t page, uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            distributed::DeviceLocalBufferConfig{.page_size = page, .buffer_type = tt::tt_metal::BufferType::DRAM},
            mesh.get());
    };
    auto buf_a = dram(kActTile, kKt * kActTile);
    auto buf_w = dram(kWTile, kNt * kKt * kWTile);
    auto buf_o = dram(kOutTile, kNt * kOutTile);
    auto buf_st = dram(512, kCores * 512);
    distributed::EnqueueWriteMeshBuffer(cq, buf_a, a_tiled, false);
    distributed::EnqueueWriteMeshBuffer(cq, buf_w, w_bf8, false);
    tt::tt_metal::Program program;
    const tt::tt_metal::CoreRangeSet all(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1)));
    std::vector<tt::tt_metal::CoreRange> own_r, send_r;
    for (const auto& p : plan) {
        (p.role == 0 ? own_r : send_r).emplace_back(p.core);
    }
    const tt::tt_metal::CoreRangeSet owners_set(own_r), senders_set(send_r);
    const uint32_t sem_go = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_done = tt::tt_metal::CreateSemaphore(program, all, 0);
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t page, tt::DataFormat df) {
        tt::tt_metal::CreateCircularBuffer(
            program, all, tt::tt_metal::CircularBufferConfig(pages * page, {{index, df}}).set_page_size(index, page));
    };
    cb(0, kKt, kActTile, tt::DataFormat::Float16_b);
    cb(1, std::max(k_own, 48u), kWTile, tt::DataFormat::Bfp8_b);
    cb(2, 3, kPartPage, tt::DataFormat::Float32);
    cb(3, 2, 32, tt::DataFormat::Float16_b);
    cb(4, 2, 32, tt::DataFormat::Float16_b);
    cb(16, 2, kOutTile, tt::DataFormat::Float16_b);
    cb(17, 2, kPartPage, tt::DataFormat::Float32);
    cb(24, 1, 512, tt::DataFormat::UInt32);
    std::vector<uint32_t> rd_ct = {kKt, kPartPage};
    tt::tt_metal::TensorAccessorArgs(*buf_a).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_w).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_o).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_st).append_to(rd_ct);
    auto rd = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain_reader.cpp",
        all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = rd_ct});
    auto comp_own = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain_compute.cpp",
        owners_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .compile_args = {0u, kKt, kRounds}});
    auto comp_send = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain_compute.cpp",
        senders_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .compile_args = {1u, kKt, kRounds}});
    const auto go_phys = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(0, 0));
    const auto mc1 = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1));
    for (uint32_t i = 0; i < kCores; ++i) {
        const CorePlan& p = plan[i];
        std::vector<uint32_t> ra = {
            static_cast<uint32_t>(buf_a->address()),
            static_cast<uint32_t>(buf_w->address()),
            static_cast<uint32_t>(buf_o->address()),
            static_cast<uint32_t>(buf_st->address()),
            i,
            p.role,
            kRounds,
            sem_go,
            sem_done,
            i == 0 ? 1u : 0u,
            kCores,
            static_cast<uint32_t>(go_phys.x),
            static_cast<uint32_t>(go_phys.y),
            static_cast<uint32_t>(mc1.x),
            static_cast<uint32_t>(mc1.y),
            static_cast<uint32_t>(go_phys.x),
            static_cast<uint32_t>(go_phys.y),
            static_cast<uint32_t>(p.segs.size())};
        std::vector<uint32_t> ca = {static_cast<uint32_t>(p.segs.size())};
        for (const auto& s : p.segs) {
            const auto own = dev->worker_core_from_logical_core(plan[owner_index[s.col]].core);
            ra.insert(ra.end(), {s.col * kKt + s.k0, s.kt, static_cast<uint32_t>(own.x), static_cast<uint32_t>(own.y), s.slot});
            ca.insert(ca.end(), {s.k0, s.kt});
        }
        ra.push_back(p.n_in);
        ra.push_back(p.col);
        ca.push_back(p.n_in);
        tt::tt_metal::SetRuntimeArgs(program, rd, p.core, ra);
        tt::tt_metal::SetRuntimeArgs(program, p.role == 0 ? comp_own : comp_send, p.core, ca);
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<bfloat16> out;
    distributed::EnqueueReadMeshBuffer(cq, out, buf_o, true);
    std::vector<uint32_t> st;
    distributed::EnqueueReadMeshBuffer(cq, st, buf_st, true);
    // status words
    uint32_t bad_status = 0;
    for (uint32_t i = 0; i < kCores; ++i) {
        if (st[i * 128 + 127] != 0x600d0000u) {
            ++bad_status;
        }
    }
    // timing: per round, earliest go over the cores -> last owner end; and the per-core spans
    double sum_round = 0, best_round = 1e18, sum_span = 0, sum_go_spread = 0, sum_send = 0, sum_land = 0;
    uint32_t cnt = 0, worst_core = 0;
    double worst_span = 0;
    for (uint32_t r = kSkip; r < kStampRounds; ++r) {
        uint32_t go_min = UINT32_MAX, go_max = 0, end_max = 0;
        double span_max = 0, send_max = 0, land_max = 0;
        for (uint32_t i = 0; i < kCores; ++i) {
            const uint32_t* s = &st[i * 128 + 3 * r];
            go_min = std::min(go_min, s[0]);
            go_max = std::max(go_max, s[0]);
            if (plan[i].role == 0) {
                end_max = std::max(end_max, s[2]);
                const double span = static_cast<double>(s[2] - s[0]);
                if (span > span_max) {
                    span_max = span;
                    if (span > worst_span) {
                        worst_span = span;
                        worst_core = i;
                    }
                }
                land_max = std::max(land_max, static_cast<double>(s[1] - s[0]));
            } else {
                send_max = std::max(send_max, static_cast<double>(s[2] - s[0]));
            }
        }
        const double us = static_cast<double>(end_max - go_min) / (kClockGHz * 1000.0);
        sum_round += us;
        best_round = std::min(best_round, us);
        sum_span += span_max / (kClockGHz * 1000.0);
        sum_go_spread += static_cast<double>(go_max - go_min);
        sum_send += send_max;
        sum_land += land_max;
        ++cnt;
    }
    // accuracy
    const auto out_rm = untilize_nfaces(out, kM, N);
    double sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0, max_err = 0, max_ref = 0;
    const uint32_t n = 2 * N;
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t j = 0; j < N; ++j) {
            const double x = static_cast<float>(out_rm[r * N + j]), y = ref[r * N + j];
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            max_err = std::max(max_err, std::abs(x - y));
            max_ref = std::max(max_ref, std::abs(y));
        }
    }
    const double pcc = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    uint32_t max_seg = 0, max_in = 0, max_send_k = 0;
    for (const auto& p : plan) {
        max_seg = std::max<uint32_t>(max_seg, p.segs.size());
        max_in = std::max(max_in, p.n_in);
        if (p.role == 1) {
            uint32_t k = 0;
            for (const auto& s : p.segs) {
                k += s.kt;
            }
            max_send_k = std::max(max_send_k, k);
        }
    }
    std::ostringstream os;
    os << "k_own " << k_own << " (sender max " << max_send_k << " K tiles, max " << max_seg << " segments, owner max "
       << max_in << " partials): round go -> last column mean " << sum_round / cnt << " us, best " << best_round
       << " us; owner end - own go max mean " << sum_span / cnt << " us (worst core " << worst_core << " "
       << worst_span / (kClockGHz * 1000.0) << " us); go spread mean " << sum_go_spread / cnt
       << " cycles; sender go -> last sent max mean " << sum_send / cnt << " cycles; owner go -> partials landed max mean "
       << sum_land / cnt << " cycles; PCC " << pcc << ", max abs err " << max_err << " (max |ref| " << max_ref
       << "); bad status " << bad_status << " over " << cnt << " rounds";
    ok = ok && bad_status == 0 && pcc > 0.999 && cnt > 0;
    return os.str();
}

TEST(Qwen38DataflowMm, InProjPoolChain) {
    // slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1, jobs/mm_chain.sh): no Tensix column is taken by dispatch, so the
    // 13 x 10 worker grid (130 cores, README 11.2.3) is whole (fast dispatch: 12 x 10, MCP 3fe17085; dispatch on
    // Ethernet cores: no ETH dispatch core on this chip, MCP 37fd5309)
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const auto grid = mesh->get_devices()[0]->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= kGridX && grid.y >= kGridY, "grid {}x{} is smaller than 13x10", grid.x, grid.y);
    const char* out_path = std::getenv("QWEN38_MM_OUT");
    std::ofstream out(std::string(out_path ? out_path : "/tmp/qwen38_mm.txt") + ".chain");
    out << "grid " << grid.x << "x" << grid.y << "; in_proj pool on 130 cores (66 cols x 80 K tiles, Stream-K, HiFi2 bf8, "
        << "fp32 partials over the NoC; step 4 budget 1.10 us)\n";
    bool ok = true;
    std::vector<uint32_t> k_owns = {40, 44, 48};
    if (const char* e = std::getenv("QWEN38_MM_KOWN")) {
        k_owns = {static_cast<uint32_t>(std::atoi(e))};
    }
    for (uint32_t k_own : k_owns) {
        out << run_chain(mesh, k_own, ok) << "\n";
        out.flush();
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
