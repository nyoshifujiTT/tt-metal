// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1d (worklog 14): how many channels a direction toward a neighbour has.  Physical claim to verify (27B worklog
// 10-23 23:11Z, Confluence SYS-4055 "Blackhole Galaxy: Ethernet 200G", ccl_common.cpp lookup_fabric_link_bw: 25 GB/s
// is one link = one channel; the mesh descriptor has channels { count: 2 }): a direction is 2 channels x 25 GB/s.
// For every neighbour pair of chip 0 the test streams packets from chip 0 to the neighbour over one Ethernet core, then
// over both cores of the link at once (kernels/link_stream_erisc.cpp), and reports each channel's sustained rate (the
// receiver's first-to-last arrival) and the sum.  Accuracy: every packet's sequence word arrives in order.
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <map>
#include <thread>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kResult = 64;

struct Channel {
    tt::tt_metal::CoreCoord core;
    int peer_chip;
    tt::tt_metal::CoreCoord peer_core;
};

double run_stream(
    const std::map<int, std::shared_ptr<distributed::MeshDevice>>& meshes,
    int chip,
    const std::vector<Channel>& active,
    uint32_t bytes,
    uint32_t n,
    uint32_t run_id,
    std::vector<double>& per_channel,
    std::string& error) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t slots = 8;
    std::map<int, tt::tt_metal::Program> programs;
    std::map<int, std::vector<std::pair<tt::tt_metal::CoreCoord, uint32_t>>> cores;
    for (const auto& c : active) {
        cores[chip].push_back({c.core, 1u});
        cores[c.peer_chip].push_back({c.peer_core, 2u});
    }
    std::map<int, distributed::MeshWorkload> workloads;
    for (auto& [ch, list] : cores) {
        auto* dev = meshes.at(ch)->get_devices()[0];
        tt::tt_metal::Program program;
        for (auto& [core, mode] : list) {
            auto k = tt::tt_metal::CreateKernel(
                program,
                "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/link_stream_erisc.cpp",
                core,
                tt::tt_metal::EthernetConfig{
                    .noc = tt::tt_metal::NOC::NOC_0, .processor = tt::tt_metal::DataMovementProcessor::RISCV_0});
            tt::tt_metal::SetRuntimeArgs(program, k, core, {base, bytes, slots, n, run_id, mode});
            std::vector<uint32_t> zero((4096 + slots * bytes) / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, base, zero, tt::CoreType::ETH);
        }
        distributed::MeshWorkload wl;
        wl.add_program(
            distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0)),
            std::move(program));
        workloads.emplace(ch, std::move(wl));
    }
    std::vector<std::thread> threads;
    for (auto& [ch, wl] : workloads) {
        threads.emplace_back(
            [&, ch] { distributed::EnqueueMeshWorkload(meshes.at(ch)->mesh_command_queue(), workloads.at(ch), true); });
    }
    for (auto& th : threads) {
        th.join();
    }
    double total = 0;
    per_channel.clear();
    for (const auto& c : active) {
        auto* sdev = meshes.at(chip)->get_devices()[0];
        auto* rdev = meshes.at(c.peer_chip)->get_devices()[0];
        std::vector<uint32_t> s, r;
        tt::tt_metal::detail::ReadFromDeviceL1(sdev, c.core, base + kResult, 32, s, tt::CoreType::ETH);
        tt::tt_metal::detail::ReadFromDeviceL1(rdev, c.peer_core, base + kResult, 32, r, tt::CoreType::ETH);
        if (s[0] != 0x600d0000u || r[4] != 0x600d0000u) {
            error += "core (" + std::to_string(c.core.x) + "," + std::to_string(c.core.y) + ") sender state " +
                     std::to_string(s[0]) + " at " + std::to_string(s[1]) + ", receiver state " + std::to_string(r[4]) +
                     " at " + std::to_string(r[5]) + "; ";
            per_channel.push_back(0);
            continue;
        }
        const double cycles = static_cast<double>(r[6] - r[5]);
        const double gbps = (static_cast<double>(n - 1) * bytes) / (cycles / 1.35);
        per_channel.push_back(gbps);
        total += gbps;
    }
    return total;
}

TEST(Qwen38DataflowLink, ChannelsPerDirection) {
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i) {
        ids[i] = i;
    }
    auto meshes = distributed::MeshDevice::create_unit_meshes(ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
    std::map<int, std::shared_ptr<distributed::MeshDevice>> by_id;
    for (auto& [id, m] : meshes) {
        by_id[id] = m;
    }
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const int chip = 0;
    std::map<int, std::vector<Channel>> toward;
    for (const auto& [chan, remote] : cluster.get_ethernet_connections().at(chip)) {
        const auto& [rchip, rchan] = remote;
        const auto core = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
        const auto rcore = cluster.get_soc_desc(rchip).get_eth_core_for_channel(rchan, tt::CoordSystem::LOGICAL);
        toward[rchip].push_back(
            {tt::tt_metal::CoreCoord(core.x, core.y), rchip, tt::tt_metal::CoreCoord(rcore.x, rcore.y)});
    }
    const char* out_path = std::getenv("QWEN38_LINK_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_link.txt");
    uint32_t run_id = 0x11c40000;
    bool ok = true;
    for (auto& [peer, chans] : toward) {
        for (uint32_t bytes : {2048u + 32u, 4096u + 32u}) {
            const uint32_t n =
                100000;  // the stream register credit is 17 bits wide (REMOTE_DEST_WORDS_FREE_WIDTH): stay below 131072
            std::vector<double> per;
            for (size_t i = 0; i < chans.size(); ++i) {
                std::string err;
                const double one = run_stream(by_id, chip, {chans[i]}, bytes, n, ++run_id, per, err);
                out << "chip " << chip << " -> " << peer << " core (" << chans[i].core.x << "," << chans[i].core.y
                    << ") alone, packet " << bytes << " B: " << one << " GB/s" << (err.empty() ? "" : " ERROR " + err)
                    << "\n";
                ok = ok && err.empty();
            }
            std::string err;
            const double both = run_stream(by_id, chip, chans, bytes, n, ++run_id, per, err);
            out << "chip " << chip << " -> " << peer << " both cores at once, packet " << bytes << " B: sum " << both
                << " GB/s (";
            for (double v : per) {
                out << v << " ";
            }
            out << ")" << (err.empty() ? "" : " ERROR " + err) << "\n";
            out.flush();
            ok = ok && err.empty();
        }
    }
    for (auto& [id, m] : meshes) {
        m->close();
    }
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
