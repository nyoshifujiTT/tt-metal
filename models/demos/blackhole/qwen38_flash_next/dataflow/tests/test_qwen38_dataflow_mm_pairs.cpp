// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1c-6 (worklog 14): the in_proj pool of one chip (66 output column tiles = 33 pairs, K = 80 tiles, bf8 HiFi2, fp32
// DEST, 2-row activation / output tiny tiles) on the 130 cores of a chip in the column-pair Stream-K assignment
// (kernels/mm/mm_chain6_*.cpp): 33 owner cores hold a pair over K [0, k_own) with ct = 2; the 97 sender cores split
// the remaining 33 x (80 - k_own) pair-K units of the pair-major stream (cut at pair boundaries into 1..3 segments) and
// send each segment's two 2-row fp32 partials (two 128 B chunks each) into the owner's 32-row slots; the owner adds
// them by the matmul with the 2-row selector and packs the two bf16 columns.  Round time measured as in I-1c-5:
// per owner, its end - its go on its own clock (no clock offset).  Accuracy: PCC of the 66 columns (2 rows).
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

constexpr uint32_t kNt = 66, kNp = 33, kKt = 80, kRows = 2, kCores = 130, kGridX = 13, kGridY = 10;
constexpr uint32_t kRounds = 48, kStampRounds = 40, kSkip = 8, kPerCol = 4;
constexpr uint32_t kWTile = 1088, kSlot = 4096;
constexpr double kClockGHz = 1.35;

struct Segment {
    uint32_t pair, k0, kt, slot;
};

struct CorePlan {
    uint32_t role;
    uint32_t pair;
    std::vector<Segment> segs;
    uint32_t n_in = 0;
    tt::tt_metal::CoreCoord core;
};

std::vector<CorePlan> make_plan(uint32_t k_own) {
    const uint32_t rest = kKt - k_own, n_send = kCores - kNp, total = kNp * rest;
    std::vector<CorePlan> owners(kNp);
    for (uint32_t p = 0; p < kNp; ++p) {
        owners[p].role = 0;
        owners[p].pair = p;
        owners[p].segs.push_back({p, 0, k_own, 0});
    }
    std::vector<CorePlan> senders(n_send);
    std::vector<uint32_t> start(n_send + 1);
    for (uint32_t s = 0; s <= n_send; ++s) {
        start[s] = static_cast<uint32_t>((static_cast<uint64_t>(total) * s) / n_send);
    }
    for (uint32_t s = 0; s < n_send; ++s) {
        senders[s].role = 1;
        uint32_t q = start[s];
        while (q < start[s + 1]) {
            const uint32_t p = q / rest, off = q % rest;
            const uint32_t len = std::min(start[s + 1] - q, rest - off);
            senders[s].segs.push_back({p, k_own + off, len, owners[p].n_in++});
            q += len;
        }
        TT_FATAL(senders[s].segs.size() <= 3, "sender {} has {} segments", s, senders[s].segs.size());
    }
    for (auto& o : owners) {
        TT_FATAL(o.n_in <= kPerCol, "pair {} has {} partials", o.pair, o.n_in);
    }
    std::vector<CorePlan> plan;
    uint32_t next = 0;
    for (uint32_t p = 0; p < kNp; ++p) {
        plan.push_back(owners[p]);
        while (next < n_send && start[next] / rest <= p) {
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

std::string run_pairs(const std::shared_ptr<distributed::MeshDevice>& mesh, uint32_t k_own, bool& ok) {
    auto* dev = mesh->get_devices()[0];
    auto& cq = mesh->mesh_command_queue();
    const tt::tt_metal::Tile tile({kRows, 32});
    const tt::tt_metal::Tile ptile({32, 32});
    const uint32_t act_tile = tile.get_tile_size(tt::DataFormat::Float16_b);
    const uint32_t out_tile = act_tile;
    const uint32_t spart = tile.get_tile_size(tt::DataFormat::Float32);
    const auto plan = make_plan(k_own);
    std::vector<uint32_t> owner_index(kNp);
    for (uint32_t i = 0; i < kCores; ++i) {
        if (plan[i].role == 0) {
            owner_index[plan[i].pair] = i;
        }
    }
    const uint32_t K = kKt * 32, N = kNt * 32;
    std::mt19937 rng(11 + k_own);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a(kRows * K), w(static_cast<size_t>(K) * N);
    for (auto& v : a) {
        v = dist(rng);
    }
    for (auto& v : w) {
        v = dist(rng) * 0.02f;
    }
    std::vector<bfloat16> a_bf(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        a_bf[i] = bfloat16(a[i]);
    }
    const auto a_tiled = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(a_bf),
        {kRows, K},
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile.get_tile_shape(),
        tile.get_face_shape());
    // weight tiles pair-major, K-row-major within a pair, columns interleaved: tile (k, c) of pair p at
    // (p * kKt + k) * 2 + (c - 2p); every tile row-major inside
    std::vector<float> w_tiles;
    w_tiles.reserve(static_cast<size_t>(K) * N);
    for (uint32_t p = 0; p < kNp; ++p) {
        for (uint32_t k = 0; k < kKt; ++k) {
            for (uint32_t c = 2 * p; c < 2 * p + 2; ++c) {
                for (uint32_t i = 0; i < 32; ++i) {
                    for (uint32_t j = 0; j < 32; ++j) {
                        w_tiles.push_back(w[static_cast<size_t>(k * 32 + i) * N + c * 32 + j]);
                    }
                }
            }
        }
    }
    const auto w_bf8 = pack_as_bfp8_tiles(ttsl::Span<const float>(w_tiles), true, false);
    const auto w_q = unpack_bfp8_tiles_into_float_vec(ttsl::Span<const uint32_t>(w_bf8), true, false);
    std::vector<double> ref(kRows * N, 0.0);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t p = 0; p < kNp; ++p) {
            for (uint32_t k = 0; k < kKt; ++k) {
                for (uint32_t cc = 0; cc < 2; ++cc) {
                    const float* t = &w_q[static_cast<size_t>((p * kKt + k) * 2 + cc) * 1024];
                    const uint32_t c = 2 * p + cc;
                    for (uint32_t i = 0; i < 32; ++i) {
                        const double av = static_cast<float>(a_bf[r * K + k * 32 + i]);
                        for (uint32_t j = 0; j < 32; ++j) {
                            ref[r * N + c * 32 + j] += av * t[i * 32 + j];
                        }
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
    auto buf_a = dram(act_tile, kKt * act_tile);
    auto buf_w = dram(kWTile, kNt * kKt * kWTile);
    auto buf_o = dram(out_tile, kNt * out_tile);
    auto buf_st = dram(512, kCores * 512);
    auto buf_sel = dram(act_tile, act_tile);
    {
        std::vector<bfloat16> sel_rm(kRows * 32, bfloat16(0.0f));
        for (uint32_t r = 0; r < kRows; ++r) {
            sel_rm[r * 32 + r] = bfloat16(1.0f);
        }
        const auto sel_t = convert_layout<bfloat16>(
            ttsl::Span<const bfloat16>(sel_rm),
            {kRows, 32},
            TensorLayoutType::LIN_ROW_MAJOR,
            TensorLayoutType::TILED_NFACES,
            tile.get_tile_shape(),
            tile.get_face_shape());
        distributed::EnqueueWriteMeshBuffer(cq, buf_sel, sel_t, false);
    }
    distributed::EnqueueWriteMeshBuffer(cq, buf_a, a_tiled, false);
    distributed::EnqueueWriteMeshBuffer(cq, buf_w, w_bf8, false);
    tt::tt_metal::Program program;
    const tt::tt_metal::CoreRangeSet all(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1)));
    std::vector<tt::tt_metal::CoreRange> own_r, send_r;
    uint32_t max_w = 0;
    for (const auto& p : plan) {
        (p.role == 0 ? own_r : send_r).emplace_back(p.core);
        uint32_t wt = 0;
        for (const auto& s : p.segs) {
            wt += 2 * s.kt;
        }
        max_w = std::max(max_w, wt);
    }
    const tt::tt_metal::CoreRangeSet owners_set(own_r), senders_set(send_r);
    const uint32_t sem_go = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_done = tt::tt_metal::CreateSemaphore(program, all, 0);
    auto cb = [&](uint32_t index, uint32_t pages, uint32_t page, tt::DataFormat df, const tt::tt_metal::Tile* t) {
        auto cfg = tt::tt_metal::CircularBufferConfig(pages * page, {{index, df}}).set_page_size(index, page);
        if (t) {
            cfg.set_tile_dims(index, *t);
        }
        tt::tt_metal::CreateCircularBuffer(program, all, cfg);
    };
    cb(0, kKt, act_tile, tt::DataFormat::Float16_b, &tile);
    cb(1, max_w, kWTile, tt::DataFormat::Bfp8_b, nullptr);
    cb(2, 2 * kPerCol, kSlot, tt::DataFormat::Float32, &ptile);
    cb(3, 2, 32, tt::DataFormat::Float16_b, nullptr);
    cb(4, 2, 32, tt::DataFormat::Float16_b, nullptr);
    cb(5, 1, act_tile, tt::DataFormat::Float16_b, &tile);
    cb(16, 2, out_tile, tt::DataFormat::Float16_b, &tile);
    cb(17, 2, spart, tt::DataFormat::Float32, &tile);
    cb(24, 1, 512, tt::DataFormat::UInt32, nullptr);
    std::vector<uint32_t> rd_ct = {kKt};
    tt::tt_metal::TensorAccessorArgs(*buf_a).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_w).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_o).append_to(rd_ct);
    tt::tt_metal::TensorAccessorArgs(*buf_st).append_to(rd_ct);
    auto rd = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain6_reader.cpp",
        all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = rd_ct});
    auto comp_own = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain6_compute.cpp",
        owners_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2, .fp32_dest_acc_en = true, .compile_args = {0u, kKt, kRounds}});
    auto comp_send = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/mm/mm_chain6_compute.cpp",
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
            const auto own = dev->worker_core_from_logical_core(plan[owner_index[s.pair]].core);
            ra.insert(ra.end(), {(s.pair * kKt + s.k0) * 2, s.kt, static_cast<uint32_t>(own.x), static_cast<uint32_t>(own.y), s.slot});
            ca.insert(ca.end(), {s.k0, s.kt});
        }
        ra.push_back(p.n_in);
        ra.push_back(p.pair);
        ra.push_back(static_cast<uint32_t>(buf_sel->address()));
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
    uint32_t bad_status = 0;
    for (uint32_t i = 0; i < kCores; ++i) {
        if (st[i * 128 + 127] != 0x600d0000u) {
            ++bad_status;
        }
    }
    double sum_span = 0, sum_send = 0, sum_land = 0;
    uint32_t cnt = 0;
    for (uint32_t r = kSkip; r < kStampRounds; ++r) {
        double span_max = 0, send_max = 0, land_max = 0;
        for (uint32_t i = 0; i < kCores; ++i) {
            const uint32_t* s = &st[i * 128 + 3 * r];
            if (plan[i].role == 0) {
                span_max = std::max(span_max, static_cast<double>(s[2] - s[0]));
                land_max = std::max(land_max, static_cast<double>(s[1] - s[0]));
            } else {
                send_max = std::max(send_max, static_cast<double>(s[2] - s[0]));
            }
        }
        sum_span += span_max;
        sum_send += send_max;
        sum_land += land_max;
        ++cnt;
    }
    // the output: page 2 x pair + c (pair-major, the pair's two column tiles)
    const auto out_rm = convert_layout<bfloat16>(
        ttsl::Span<const bfloat16>(out),
        {kRows, N},
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        tile.get_tile_shape(),
        tile.get_face_shape());
    double sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0, max_err = 0;
    const uint32_t n = kRows * N;
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t j = 0; j < N; ++j) {
            const double x = static_cast<float>(out_rm[r * N + j]), y = ref[r * N + j];
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            max_err = std::max(max_err, std::abs(x - y));
        }
    }
    const double pcc = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    uint32_t max_k = 0, max_seg = 0;
    for (const auto& p : plan) {
        uint32_t k = 0;
        for (const auto& s : p.segs) {
            k += s.kt;
        }
        if (p.role == 1) {
            max_k = std::max(max_k, k);
        }
        max_seg = std::max<uint32_t>(max_seg, p.segs.size());
    }
    std::ostringstream os;
    os << "pairs k_own " << k_own << " (sender max " << max_k << " pair-K, max " << max_seg
       << " segments): owner end - own go max mean " << sum_span / cnt << " cycles = " << sum_span / cnt / (kClockGHz * 1000.0)
       << " us; sender go -> last sent max mean " << sum_send / cnt << "; owner go -> partials landed max mean "
       << sum_land / cnt << "; PCC " << pcc << ", max abs err " << max_err << "; bad status " << bad_status << " over " << cnt
       << " rounds";
    ok = ok && bad_status == 0 && pcc > 0.999;
    return os.str();
}

TEST(Qwen38DataflowMm, InProjPairs) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const auto grid = mesh->get_devices()[0]->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= kGridX && grid.y >= kGridY, "grid {}x{} is smaller than 13x10", grid.x, grid.y);
    const char* out_path = std::getenv("QWEN38_MM_OUT");
    std::ofstream out(std::string(out_path ? out_path : "/tmp/qwen38_mm.txt") + ".pairs");
    bool ok = true;
    for (uint32_t k_own : {20u, 24u, 28u, 32u}) {
        out << run_pairs(mesh, k_own, ok) << "\n";
        out.flush();
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
