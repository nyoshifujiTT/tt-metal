// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1e (worklog 14): the cycle costs of the relay steps on an Ethernet core's RISC (kernels/costs_erisc.cpp), on chip
// 0, every Ethernet core of a link to chip 1 (ERISC1, NoC 1), the peer being the other core of the same link.
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <map>
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

TEST(Qwen38DataflowCosts, RelaySteps) {
    auto meshes = distributed::MeshDevice::create_unit_meshes({0}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
    auto* dev = meshes.at(0)->get_devices()[0];
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t n = 4096;
    const tt::tt_metal::CoreCoord a(0, 4), b(0, 5);
    tt::tt_metal::Program program;
    for (auto [core, peer] : {std::pair{a, b}, std::pair{b, a}}) {
        auto k = tt::tt_metal::CreateKernel(
            program,
            "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/costs_erisc.cpp",
            core,
            tt::tt_metal::EthernetConfig{
                .noc = tt::tt_metal::NOC::NOC_1, .processor = tt::tt_metal::DataMovementProcessor::RISCV_1});
        const auto pv = dev->ethernet_core_from_logical_core(peer);
        tt::tt_metal::SetRuntimeArgs(program, k, core, {base, n, static_cast<uint32_t>(pv.x), static_cast<uint32_t>(pv.y)});
        std::vector<uint32_t> zero(4096, 0);
        tt::tt_metal::detail::WriteToDeviceL1(dev, core, base, zero, tt::CoreType::ETH);
    }
    distributed::MeshWorkload wl;
    wl.add_program(
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0)),
        std::move(program));
    distributed::EnqueueMeshWorkload(meshes.at(0)->mesh_command_queue(), wl, true);
    const char* out_path = std::getenv("QWEN38_COSTS_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_costs.txt");
    const char* names[] = {
        "empty loop",
        "wall clock read",
        "L1 load (dependent)",
        "L1 store + load back + use",
        "L1 store + fence",
        "cmd buf ready poll",
        "NoC write issue (poll + 5 regs, 64 B to self)",
        "NIU status read",
        "semaphore inc (to self)",
        "704 B NoC write to the other core of the link, until sent",
        "704 B NoC write to the other core of the link, until acked"};
    bool ok = true;
    for (auto core : {a, b}) {
        std::vector<uint32_t> w;
        tt::tt_metal::detail::ReadFromDeviceL1(dev, core, base, 128, w, tt::CoreType::ETH);
        ok = ok && w[30] == 0x600d0000u;
        out << "core (" << core.x << "," << core.y << ") ERISC1 NoC1, cycles per repetition (n " << n << "):\n";
        for (int i = 0; i < 11; ++i) {
            out << "  " << names[i] << ": " << static_cast<double>(w[i]) / n << "\n";
        }
    }
    meshes.at(0)->close();
    EXPECT_TRUE(ok);
}

}  // namespace
}  // namespace qwen38_dataflow
