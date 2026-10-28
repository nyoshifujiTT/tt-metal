// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The machine as the decode design (README section 12) needs it: the 8 x 4 torus of the Galaxy as the Ethernet links
// see it.  For every physical chip: its Ethernet coordinate, and per remote chip the active Ethernet cores that reach
// it (the design sends a meeting's packets directly from these cores, no fabric router).  The test fails when the
// links do not form an 8 x 4 torus (every chip with exactly four distinct neighbours, each over at least two cores).
// It prints the torus position of every chip (row, column) and the link table, which the group placement reads.
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"

namespace qwen38_dataflow {
namespace {

TEST(Qwen38DataflowProbe, TorusLinks) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto connections = cluster.get_ethernet_connections();
    const auto coords = cluster.get_all_chip_ethernet_coordinates();
    std::map<tt::ChipId, std::map<tt::ChipId, std::vector<std::string>>> links;
    for (const auto& [chip, chans] : connections) {
        for (const auto& [chan, remote] : chans) {
            const auto& [rchip, rchan] = remote;
            if (rchip == chip) {
                continue;
            }
            const auto core = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
            links[chip][rchip].push_back(
                "ch" + std::to_string(chan) + "(" + std::to_string(core.x) + "," + std::to_string(core.y) + ")->ch" +
                std::to_string(rchan));
        }
    }
    bool torus = links.size() == 32;
    const char* out_path = std::getenv("QWEN38_PROBE_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_probe_links.txt");
    for (const auto& [chip, peers] : links) {
        const auto it = coords.find(chip);
        std::string row = "chip " + std::to_string(chip);
        if (it != coords.end()) {
            row += " eth_coord (x " + std::to_string(it->second.x) + ", y " + std::to_string(it->second.y) + ", rack " +
                   std::to_string(it->second.rack) + ", shelf " + std::to_string(it->second.shelf) + ")";
        }
        row += " peers " + std::to_string(peers.size()) + ":";
        for (const auto& [rchip, cores] : peers) {
            row += " [" + std::to_string(rchip) + ":";
            for (const auto& c : cores) {
                row += " " + c;
            }
            row += "]";
            if (cores.size() < 2) {
                torus = false;
            }
        }
        if (peers.size() != 4) {
            torus = false;
        }
        out << row << "\n";
    }
    EXPECT_TRUE(torus) << "the Ethernet links are not an 8 x 4 torus with two cores per neighbour";
}

}  // namespace
}  // namespace qwen38_dataflow
