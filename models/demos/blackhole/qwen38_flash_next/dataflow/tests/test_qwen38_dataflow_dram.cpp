// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1b' (worklog 14): the DRAM lane of one chip in the design's form -- the cores read their own weight slices (each
// contiguous in one bank: page i of a buffer whose page is the slice lives in bank i mod 8) straight into their L1,
// all at once (kernels/dram/dram_slice_reader.cpp).  A round is timed on the go core's clock alone (go multicast ->
// all done), so no clock offset enters.  Cases: (readers, slice) with the 130 cores of the design and the 8
// bank-adjacent readers of 27B section 601 for reference.  Efficiency = readers x slice / (512 GB/s x round time).
// Correctness: every reader's 32-bit sum of its slice against the host's.
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <map>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kGridX = 13, kGridY = 10, kCores = kGridX * kGridY, kRounds = 24, kSkip = 4;
constexpr double kClockGHz = 1.35, kDramGBs = 512.0;

struct Case {
    uint32_t readers, slice, mode, page, group;
    std::string assign;  // mode 2: the LP's assignment file (per core: x y nseg {bank offset bytes})
    bool optimal;  // the 8 bank-adjacent cores of get_optimal_dram_bank_to_logical_worker_assignment (readers == banks)
    bool vc_spread;
    uint32_t chain = 1;   // mode 2: reads of the slice per round (back to back, one go)
    uint32_t period = 0;  // mode 2: pace the reads of a round at go + q x period cycles (0: back to back)
};

std::string run_case(const std::shared_ptr<distributed::MeshDevice>& mesh, const Case& c, bool& ok) {
    const uint32_t readers = c.readers, slice = c.slice;
    auto* dev = mesh->get_devices()[0];
    auto& cq = mesh->mesh_command_queue();
    const uint32_t nb = dev->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    // mode 2: segments per logical core from the assignment file; the buffer is pages of `page_bytes` (interleaved over
    // the banks), so bank b's region is the concatenation of its pages and a byte offset o in bank b is page
    // (o / page_bytes) * nb + b, at o % page_bytes
    struct Seg {
        uint32_t bank, off, len, tx = 0, ty = 0, noc = 0;
    };
    std::map<std::pair<uint32_t, uint32_t>, std::vector<Seg>> segs;
    uint32_t bank_bytes = 0;
    if (c.mode == 2) {
        std::ifstream f(c.assign);
        TT_FATAL(f.good(), "no assignment file {}", c.assign);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ls(line);
            uint32_t x, y, n;
            if (!(ls >> x >> y >> n)) {
                continue;
            }
            // per segment: bank offset bytes [tx ty [noc]] (rowlocal: one segment with tx ty; balanced: all six)
            auto& v = segs[{x, y}];
            std::vector<uint32_t> nums;
            uint32_t t;
            while (ls >> t) {
                nums.push_back(t);
            }
            const uint32_t per = static_cast<uint32_t>(nums.size()) / n;
            for (uint32_t i = 0; i < n; ++i) {
                Seg sg{};
                sg.bank = nums[i * per];
                sg.off = nums[i * per + 1];
                sg.len = nums[i * per + 2];
                if (per >= 5) {
                    sg.tx = nums[i * per + 3];
                    sg.ty = nums[i * per + 4];
                }
                if (per >= 6) {
                    sg.noc = nums[i * per + 5];
                }
                v.push_back(sg);
                bank_bytes = std::max(bank_bytes, sg.off + sg.len);
            }
        }
    }
    const uint32_t page_bytes = c.mode == 2 ? 65536u : slice;
    const uint32_t pages = c.mode == 2 ? ((bank_bytes + page_bytes - 1) / page_bytes) * nb : ((readers + nb - 1) / nb) * nb;
    std::vector<uint32_t> data(static_cast<size_t>(pages) * page_bytes / 4);
    std::mt19937 rng(slice + readers);
    for (auto& v : data) {
        v = rng();
    }
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = static_cast<uint64_t>(pages) * page_bytes},
        distributed::DeviceLocalBufferConfig{.page_size = page_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh.get());
    auto out = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = kCores * 256},
        distributed::DeviceLocalBufferConfig{.page_size = 256, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh.get());
    distributed::EnqueueWriteMeshBuffer(cq, buf, data, true);
    tt::tt_metal::Program program;
    const tt::tt_metal::CoreRangeSet all(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1)));
    const uint32_t sem_go = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_done = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t sem_local = tt::tt_metal::CreateSemaphore(program, all, 0);
    const uint32_t l1_base = dev->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t dst = l1_base;
    // the largest slice any core reads (mode 2: from the assignment), then the result scratch
    uint32_t max_slice = slice;
    for (const auto& [k2, v] : segs) {
        uint32_t t = 0;
        for (const auto& sg : v) {
            t += sg.len;
        }
        max_slice = std::max(max_slice, t);
    }
    TT_FATAL(l1_base + max_slice + 4096 < 1400 * 1024, "slice {} B does not fit the L1", max_slice);
    const uint32_t scratch = l1_base + ((max_slice + 1023) / 1024) * 1024;
    std::vector<uint32_t> ct0 = {0}, ct1 = {1};
    tt::tt_metal::TensorAccessorArgs(*out).append_to(ct0);
    tt::tt_metal::TensorAccessorArgs(*out).append_to(ct1);
    auto k = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/dram/dram_slice_reader.cpp",
        all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct0});
    auto k1 = tt::tt_metal::CreateKernel(
        program,
        "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/dram/dram_slice_reader.cpp",
        all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = ct1});
    const auto go_phys = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(0, 0));
    const auto mc1 = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(kGridX - 1, kGridY - 1));
    std::vector<uint32_t> expect(kCores, 0);
    uint64_t total_bytes = 0;
    // readers: the first `readers` cores in a spread order (every 130/readers-th core) so 8 readers do not share a row
    std::vector<uint32_t> reader_of(kCores, UINT32_MAX);
    if (c.optimal) {
        const auto opt = dev->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
        TT_FATAL(readers <= opt.size(), "{} readers but {} bank-adjacent cores", readers, opt.size());
        for (uint32_t j = 0; j < readers; ++j) {
            TT_FATAL(opt[j].x < kGridX && opt[j].y < kGridY, "bank core {} outside 13x10", opt[j].str());
            reader_of[opt[j].y * kGridX + opt[j].x] = j;
        }
    } else {
        for (uint32_t j = 0; j < readers; ++j) {
            reader_of[(j * kCores) / readers] = j;
        }
    }
    for (uint32_t i = 0; i < kCores; ++i) {
        const tt::tt_metal::CoreCoord core(i % kGridX, i / kGridX);
        uint32_t j = reader_of[i];
        std::vector<Seg> my;
        if (c.mode == 2) {
            const auto it = segs.find({core.x, core.y});
            j = it == segs.end() ? UINT32_MAX : i;
            reader_of[i] = j;
            if (it != segs.end()) {
                my = it->second;
            }
        }
        uint32_t s = j == UINT32_MAX ? 0 : slice;
        if (c.mode == 2) {
            s = 0;
            for (const auto& sg : my) {
                s += sg.len;
            }
            total_bytes += s;
        }
        if (j != UINT32_MAX) {
            uint32_t sum = 0;
            if (c.mode == 2) {
                for (const auto& sg : my) {
                    for (uint32_t o = sg.off; o < sg.off + sg.len; o += 4) {
                        const size_t page = static_cast<size_t>(o / page_bytes) * nb + sg.bank;
                        sum += data[(page * page_bytes + o % page_bytes) / 4];
                    }
                }
            } else {
                for (uint32_t w = 0; w < slice / 4; ++w) {
                    sum += data[static_cast<size_t>(j) * slice / 4 + w];
                }
            }
            expect[i] = sum;
        }
        TT_FATAL(my.size() <= 2, "core {} has {} segments", i, my.size());

        std::vector<uint32_t> args = {
             static_cast<uint32_t>(buf->address()),
             j == UINT32_MAX ? 0u : j % nb,
             j == UINT32_MAX ? 0u : (j / nb) * slice,
             s,
             dst,
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
             static_cast<uint32_t>(out->address()),
             i,
             scratch,
             c.mode,
             c.page,
             c.group,
             j == UINT32_MAX ? 0u : (c.vc_spread ? j % 4 : 0u),
             static_cast<uint32_t>(my.size())};
        for (uint32_t q = 0; q < 2; ++q) {
            const Seg z{};
            const Seg& sg = q < my.size() ? my[q] : z;
            args.insert(args.end(), {sg.bank, sg.off, sg.len, sg.tx, sg.ty, sg.noc});
        }
        args.push_back(sem_local);
        args.push_back(c.chain);
        args.push_back(c.period);
        SetRuntimeArgs(program, k, core, args);
        SetRuntimeArgs(program, k1, core, args);
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(mesh->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<uint32_t> res;
    distributed::EnqueueReadMeshBuffer(cq, res, out, true);
    uint32_t bad_sum = 0, bad_status = 0;
    for (uint32_t i = 0; i < kCores; ++i) {
        const uint32_t* r = &res[i * 64];
        if (r[63] != 0x600d0000u || r[61] != 0) {
            ++bad_status;
        }
        if (reader_of[i] != UINT32_MAX && r[60] != expect[i]) {
            ++bad_sum;
        }
    }
    const uint32_t* g = &res[0];
    double sum = 0, best = 1e18, worst = 0;
    uint32_t n = 0;
    for (uint32_t r = kSkip; r < kRounds && 2 * r + 1 < 60; ++r) {
        const double us = static_cast<double>(g[2 * r + 1] - g[2 * r]) / (kClockGHz * 1000.0);
        sum += us;
        best = std::min(best, us);
        worst = std::max(worst, us);
        ++n;
    }
    const double mean = sum / n;
    const double bytes = c.mode == 2 ? static_cast<double>(total_bytes) * c.chain : static_cast<double>(readers) * slice;
    std::ostringstream os;
    os << "chain " << c.chain << (c.period ? " period " + std::to_string(c.period) : std::string()) << ": ";
    os << (c.mode ? "TRID pipeline (page " + std::to_string(c.page) + ", " + std::to_string(c.group) + " a trid)" : "one read")
       << (c.optimal ? ", bank-adjacent" : ", spread") << (c.vc_spread ? ", VC j%4" : ", VC 0") << ": readers " << readers << " x slice " << slice << " B = " << bytes / 1e6 << " MB: round mean " << mean
       << " us (best " << best << ", worst " << worst << ") = " << bytes / (mean * 1e3) << " GB/s = "
       << 100.0 * bytes / (mean * 1e3) / kDramGBs << "% of 512 (best " << 100.0 * bytes / (best * 1e3) / kDramGBs
       << "%); bad sums " << bad_sum << ", bad status " << bad_status << " (banks " << nb << ")";
    ok = ok && bad_sum == 0 && bad_status == 0;
    return os.str();
}

TEST(Qwen38DataflowDram, SliceLane) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    const auto grid = mesh->get_devices()[0]->compute_with_storage_grid_size();
    TT_FATAL(grid.x >= kGridX && grid.y >= kGridY, "grid {}x{} is smaller than 13x10", grid.x, grid.y);
    const char* out_path = std::getenv("QWEN38_DRAM_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_dram.txt");
    bool ok = true;
    [[maybe_unused]] constexpr uint32_t P = 16384;
    const std::string A = std::string(std::getenv("HOME")) + "/work/noc/";
    // the MILP NOC1 slice (256 KiB a core, 34.1 MB in all) at the bank bound takes 34.1 MB / (8 x 47.4 B/cycle) =
    // 89,856 cycles; paced streams of 4 reads at periods of 1.00 / 1.02 / 1.05 x that
    std::vector<Case> cases = {
        {130, 0, 2, 0, 0, A + "milp_262144_noc1_u50.txt", false, false},  // one read a round (run 9: 98.4%)
        {130, 0, 2, 0, 0, A + "milp_262144_noc1_u50.txt", false, false},  // 4 reads back to back
        {130, 0, 2, 0, 0, A + "milp_262144_noc1_u50.txt", false, false},  // 4 reads paced 1.00
        {130, 0, 2, 0, 0, A + "milp_262144_noc1_u50.txt", false, false},  // 4 reads paced 1.02
        {130, 0, 2, 0, 0, A + "milp_262144_noc1_u50.txt", false, false},  // 4 reads paced 1.05
    };
    cases[1].chain = 4;
    cases[2].chain = 4;
    cases[2].period = 89856;
    cases[3].chain = 4;
    cases[3].period = 91653;
    cases[4].chain = 4;
    cases[4].period = 94349;
    for (const auto& c : cases) {
        out << run_case(mesh, c, ok) << "\n";
        out.flush();
    }
    mesh->close();
    EXPECT_TRUE(ok);
}

// the chip's coordinates for the NoC link-load model (worklog 14 I-1b'): every logical worker's NOC0 physical and
// translated coordinates, and every DRAM bank's endpoint tiles (translated, NOC0 physical) with the NOC0 / NOC1
// preferred worker endpoints
TEST(Qwen38DataflowDram, Map) {
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto* dev = mesh->get_devices()[0];
    const auto& soc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(dev->id());
    const char* out_path = std::getenv("QWEN38_DRAM_OUT");
    std::ofstream out(std::string(out_path ? out_path : "/tmp/qwen38_dram.txt") + ".map");
    const auto grid = dev->compute_with_storage_grid_size();
    out << "grid " << grid.x << " " << grid.y << "\n";
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            const auto t = dev->worker_core_from_logical_core(tt::tt_metal::CoreCoord(x, y));
            const auto p = soc.translate_coord_to(tt::umd::CoreCoord(t.x, t.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED), tt::CoordSystem::NOC0);
            out << "worker " << x << " " << y << " translated " << t.x << " " << t.y << " noc0 " << p.x << " " << p.y << "\n";
        }
    }
    const uint32_t nb = soc.get_num_dram_views();
    for (uint32_t b = 0; b < nb; ++b) {
        const auto n0 = soc.get_preferred_worker_core_for_dram_view(static_cast<int>(b), 0);
        const auto n1 = soc.get_preferred_worker_core_for_dram_view(static_cast<int>(b), 1);
        out << "bank " << b << " pref0 " << n0.x << " " << n0.y << " pref1 " << n1.x << " " << n1.y;
        for (const auto& e : soc.dram_bank_endpoint_coords.at(b)) {
            const auto p = soc.translate_coord_to(tt::umd::CoreCoord(e.x, e.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED), tt::CoordSystem::NOC0);
            out << " ep " << e.x << " " << e.y << " noc0 " << p.x << " " << p.y;
        }
        out << "\n";
    }
    for (int noc = 0; noc < 2; ++noc) {
        const auto opt = dev->get_optimal_dram_bank_to_logical_worker_assignment(noc == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1);
        out << "optimal noc" << noc;
        for (const auto& c : opt) {
            out << " " << c.x << "," << c.y;
        }
        out << "\n";
    }
    mesh->close();
}

}  // namespace
}  // namespace qwen38_dataflow
