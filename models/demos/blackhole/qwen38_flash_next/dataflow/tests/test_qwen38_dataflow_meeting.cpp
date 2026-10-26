// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1b (README 12.5, worklog 14): the all-gather of one 2 x 4 group (TP8) sent directly from the Ethernet cores along
// the BFS trees of the block (no fabric router), self-timed: each chip starts iteration k of its own chunk when it
// holds every chunk of iteration k-1, so the steady period is the meeting's time (the last byte at the farthest chip).
// The group is torus rows {2g, 2g+1}; the torus (row, column) of every physical chip comes from the Ethernet links (the
// probe's table, recomputed here).  Payload sizes are the design's: 1,280 B (x 1/8), 3,840 B, 10,400 B (AR_down as
// AG+local).  Accuracy: every chip ends with every root's chunk bit-exact (the host fills the roots' slots).
// Budget: the packet model's AG time (comm_model.py, AG_tree) for the same payload.
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <map>
#include <algorithm>
#include <array>
#include <set>
#include <thread>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/hal.hpp"
#include "qwen38_torus.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kRoots = 8;
constexpr uint32_t kStamps = 1024, kProbe = 4096, kState = 4608, kSeen = 6144, kRelayProbe = 14336,
                   kData = 16384;  // meeting_erisc.cpp: words 16 .. 288 flags, counters, arrivals; stamps; data slots

struct MeetingRun {
    double period_us = 0;
    std::vector<double> periods;
    bool exact = true;
    double arrive[8][8] = {};  // [n][r]: chip n's arrival of root r's probe chunk minus chip n's own start (us)
    double restart[8] = {};    // chip n's next own start minus its own start (us)
    int hops[8][8] = {};
    std::vector<std::string> stalls;
    uint32_t bad_iterations = 0;
    std::vector<double> relay_us, loop_ns, loop_ns_rx, stage_rx, stage_noc, stage_tx;
    std::vector<std::string> relay_lines;
};

MeetingRun run_meeting(
    const std::map<int, std::shared_ptr<distributed::MeshDevice>>& meshes,
    const Torus& t,
    int group,
    uint32_t payload,
    uint32_t iters,
    uint32_t run_id) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t limit =
        base +
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t packet_max = 2048;
    const uint32_t packets = (payload + packet_max - 1) / packet_max;
    const uint32_t packet = (((payload + packets - 1) / packets) + 31) & ~31u;  // equal packets, 32 B multiples
    const uint32_t frame = packet + 32;                                         // + the trailer (meeting_erisc.cpp)
    const uint32_t slot_bytes = packets * frame;
    TT_FATAL(
        base + kData + 2 * kRoots * slot_bytes <= limit,
        "payload {} B does not fit the eth L1 ({} B)",
        payload,
        limit - base);
    std::array<int, 8> chip_of{};
    for (int n = 0; n < 8; ++n) {
        chip_of[n] = t.at[2 * group + n / 4][n % 4];
    }
    // per chip: the eth core toward each neighbour inside the block (the first of the two link cores)
    // the root's channel: the two Ethernet cores of a link carry alternate roots (each link 25 GB/s a direction = 2
    // channels)
    auto core_toward = [&](int from, int to, int root) {
        const auto& v = t.link_cores.at({chip_of[from], chip_of[to]});
        return v[root % v.size()];
    };
    struct TxSpec {
        uint32_t root, relayed;
        tt::tt_metal::CoreCoord up;
        uint32_t up_index;
    };
    struct CoreJobs {
        std::vector<TxSpec> tx;
        std::vector<std::pair<uint32_t, std::vector<tt::tt_metal::CoreCoord>>>
            rx;  // (root, forward-to cores of this chip)
    };
    std::array<std::map<tt::tt_metal::CoreCoord, CoreJobs>, 8> jobs;
    std::array<std::map<std::pair<uint32_t, tt::tt_metal::CoreCoord>, std::pair<tt::tt_metal::CoreCoord, uint32_t>>, 8>
        relay_of;  // node: (root, forwarder core) -> (its rx core, target index)
    for (int root = 0; root < 8; ++root) {
        const auto parent = bfs_parent(root);
        for (int n = 0; n < 8; ++n) {
            if (parent[n] < 0) {
                continue;
            }
            const int p = parent[n];
            const tt::tt_metal::CoreCoord send = core_toward(p, n, root);
            const auto [peer_chip, recv] = t.peer.at({chip_of[p], send});
            TT_FATAL(peer_chip == chip_of[n], "the link core toward a neighbour does not reach it");
            std::vector<tt::tt_metal::CoreCoord> forward;
            for (int k = 0; k < 8; ++k) {
                if (parent[k] == n) {
                    forward.push_back(core_toward(n, k, root));
                }
            }
            jobs[n][recv].rx.push_back({static_cast<uint32_t>(root), forward});
            for (uint32_t idx = 0; idx < forward.size(); ++idx) {
                relay_of[n][{static_cast<uint32_t>(root), forward[idx]}] = {recv, idx};
            }
        }
    }
    for (int root = 0; root < 8; ++root) {
        const auto parent = bfs_parent(root);
        for (int n = 0; n < 8; ++n) {
            if (parent[n] < 0) {
                continue;
            }
            const int p = parent[n];
            const tt::tt_metal::CoreCoord send = core_toward(p, n, root);
            if (p == root) {
                jobs[p][send].tx.push_back({static_cast<uint32_t>(root), 0u, tt::tt_metal::CoreCoord(0, 0), 0u});
            } else {
                const auto& [up, idx] = relay_of[p].at({static_cast<uint32_t>(root), send});
                jobs[p][send].tx.push_back({static_cast<uint32_t>(root), 1u, up, idx});
            }
        }
    }
    // every core of a chip that has jobs counts arrivals; each chip's own chunk is sent once all 7 others arrived
    MeetingRun run;
    tt::tt_metal::CoreCoord stamp_core_of_chip0(0, 0);
    std::map<int, distributed::MeshWorkload> workloads;
    std::map<int, std::vector<tt::tt_metal::CoreCoord>> chip_cores;
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        tt::tt_metal::Program program;
        std::vector<tt::tt_metal::CoreCoord> cores;
        for (auto& [core, cj] : jobs[n]) {
            cores.push_back(core);
        }
        chip_cores[n] = cores;
        const bool stamp_chip = n == 0;
        tt::tt_metal::CoreCoord stamp_core = cores.front();
        for (auto& [core, cj] : jobs[n]) {
            for (auto& tj : cj.tx) {
                if (!tj.relayed) {
                    stamp_core = core;
                }
            }
        }
        if (stamp_chip) {
            stamp_core_of_chip0 = stamp_core;
        }
        for (auto& [core, cj] : jobs[n]) {
            auto k = tt::tt_metal::CreateKernel(
                program,
                "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/meeting_erisc.cpp",
                core,
                tt::tt_metal::EthernetConfig{
                    .noc = tt::tt_metal::NOC::NOC_0,
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .compile_args = {0u}});
            auto k1 = tt::tt_metal::CreateKernel(
                program,
                "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/meeting_erisc.cpp",
                core,
                tt::tt_metal::EthernetConfig{
                    .noc = tt::tt_metal::NOC::NOC_1,
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                    .compile_args = {1u}});
            std::vector<uint32_t> args{
                base,
                iters,
                payload,
                packet,
                packets,
                slot_bytes,
                7u,
                run_id,
                stamp_chip && core == stamp_core ? 1u : 0u,
                2000000u,
                iters / 2};
            std::vector<tt::tt_metal::CoreCoord> own_cores;
            for (auto& [oc, ocj] : jobs[n]) {
                for (auto& tj : ocj.tx) {
                    if (!tj.relayed) {
                        own_cores.push_back(oc);
                        break;
                    }
                }
            }
            args.push_back(static_cast<uint32_t>(own_cores.size()));
            for (auto& c : own_cores) {
                const auto v = dev->ethernet_core_from_logical_core(c);
                args.push_back(v.x);
                args.push_back(v.y);
            }
            std::stable_sort(
                cj.tx.begin(), cj.tx.end(), [](const TxSpec& a, const TxSpec& b) { return a.relayed > b.relayed; });
            args.push_back(static_cast<uint32_t>(cj.tx.size()));
            for (auto& tj : cj.tx) {
                const auto v = dev->ethernet_core_from_logical_core(tj.up);
                args.push_back(tj.root);
                args.push_back(tj.relayed);
                args.push_back(tj.relayed ? v.x : 0u);
                args.push_back(tj.relayed ? v.y : 0u);
                args.push_back(tj.up_index);
            }
            args.push_back(static_cast<uint32_t>(cj.rx.size()));
            for (auto& [root, fwd] : cj.rx) {
                args.push_back(root);
                args.push_back(static_cast<uint32_t>(fwd.size()));
                for (auto& c : fwd) {
                    const auto v = dev->ethernet_core_from_logical_core(c);
                    args.push_back(v.x);
                    args.push_back(v.y);
                }
            }
            tt::tt_metal::SetRuntimeArgs(program, k, core, args);
            tt::tt_metal::SetRuntimeArgs(program, k1, core, args);
            // zero the words, fill the own chunk's slots (both parities) with a pattern of (chip, root, word)
            std::vector<uint32_t> zero((kData) / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, base, zero, tt::CoreType::ETH);
            std::vector<uint32_t> data(2 * kRoots * slot_bytes / 4, 0);
            for (uint32_t parity = 0; parity < 2; ++parity) {
                for (uint32_t w = 0; w < payload / 4; ++w) {
                    data[((parity * kRoots + n) * slot_bytes + (4 * w / packet) * frame + (4 * w) % packet) / 4] =
                        (static_cast<uint32_t>(n) << 24) ^ w ^ 0x5a5a0000u;
                }
            }
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, base + kData, data, tt::CoreType::ETH);
        }
        distributed::MeshWorkload wl;
        wl.add_program(
            distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(0, 0)),
            std::move(program));
        workloads.emplace(n, std::move(wl));
    }
    std::vector<std::thread> threads;
    for (int n = 0; n < 8; ++n) {
        threads.emplace_back([&, n] {
            distributed::EnqueueMeshWorkload(meshes.at(chip_of[n])->mesh_command_queue(), workloads.at(n), true);
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    // the end state of every core: a stalled core (watchdog) reports its jobs' counts and the trailer it waits for
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        for (auto& [core, cj] : jobs[n]) {
            std::vector<uint32_t> st;
            tt::tt_metal::detail::ReadFromDeviceL1(dev, core, base + kState, 320, st, tt::CoreType::ETH);
            if (st[1] != 0x600d0000u && !cj.rx.empty()) {
                run.exact = false;
                run.stalls.push_back(
                    "STALL rx role chip " + std::to_string(n) + " core (" + std::to_string(core.x) + "," +
                    std::to_string(core.y) + ") state " + std::to_string(st[1]));
            }
            if (cj.tx.empty()) {
                continue;
            }
            if (st[0] == 0xdead0001u) {
                run.exact = false;
                run.stalls.push_back(
                    "STALL chip " + std::to_string(n) + " core (" + std::to_string(core.x) + "," +
                    std::to_string(core.y) + ") never got its peer's hello");
            } else if (st[0] != 0x600d0000u) {
                run.exact = false;
                std::string line = "STALL chip " + std::to_string(n) + " core (" + std::to_string(core.x) + "," +
                                   std::to_string(core.y) + ") arrivals " + std::to_string(st[2]);
                for (size_t i = 0; i < cj.rx.size(); ++i) {
                    line += " rx[root " + std::to_string(st[4 + 4 * i]) + " done " + std::to_string(st[5 + 4 * i]) +
                            " flag " + std::to_string(st[6 + 4 * i]) + "]";
                }
                for (size_t i = 0; i < cj.tx.size(); ++i) {
                    line += " tx[root " + std::to_string(st[40 + 4 * i] & 0xffff) +
                            (st[40 + 4 * i] >> 16 ? " relay" : " own") + " sent " + std::to_string(st[41 + 4 * i]) +
                            " trailer " + std::to_string(st[42 + 4 * i]) + "/" + std::to_string(st[43 + 4 * i]) + "]";
                }
                run.stalls.push_back(line);
            }
        }
    }
    // the stamps on chip 0's first core: period = mean of the steady iterations
    auto* dev0 = meshes.at(chip_of[0])->get_devices()[0];
    std::vector<uint32_t> st;
    tt::tt_metal::detail::ReadFromDeviceL1(dev0, stamp_core_of_chip0, base + kStamps, 8 * iters, st, tt::CoreType::ETH);
    std::vector<uint64_t> ts;
    for (uint32_t i = 0; i < iters; ++i) {
        ts.push_back((static_cast<uint64_t>(st[2 * i + 1]) << 32) | st[2 * i]);
    }
    const uint32_t warm = iters / 4;
    for (uint32_t i = warm + 1; i < iters; ++i) {
        run.periods.push_back((ts[i] - ts[i - 1]) / 1350.0);
    }
    double sum = 0;
    for (double p : run.periods) {
        sum += p;
    }
    run.period_us = run.periods.empty() ? 0 : sum / run.periods.size();
    // the probe: per chip, the own-tx start stamps (iterations probe, probe+1) and each rx job's completion stamp
    auto read64 = [&](tt::tt_metal::IDevice* dev, tt::tt_metal::CoreCoord core, uint32_t addr) {
        std::vector<uint32_t> w;
        tt::tt_metal::detail::ReadFromDeviceL1(dev, core, addr, 8, w, tt::CoreType::ETH);
        return (static_cast<uint64_t>(w[1]) << 32) | w[0];
    };
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        uint64_t s0 = 0, s1 = 0;
        for (auto& [core, cj] : jobs[n]) {
            for (auto& tj : cj.tx) {
                if (!tj.relayed) {
                    s0 = read64(dev, core, base + kProbe + 8 * kRoots);
                    s1 = read64(dev, core, base + kProbe + 8 * (kRoots + 1));
                }
            }
        }
        run.restart[n] = (static_cast<double>(s1) - static_cast<double>(s0)) / 1350.0;
        for (auto& [core, cj] : jobs[n]) {
            for (auto& [root, fwd] : cj.rx) {
                run.arrive[n][root] =
                    (static_cast<double>(read64(dev, core, base + kProbe + 8 * root)) - static_cast<double>(s0)) /
                    1350.0;
            }
        }
        for (int r = 0; r < 8; ++r) {
            const int rr = r / 4, rc = r % 4, nr = n / 4, nc = n % 4;
            int dc = std::abs(nc - rc);
            dc = std::min(dc, 4 - dc);
            run.hops[n][r] = (rr != nr) + dc;
        }
    }
    // the relay inside a chip: forward core's first send of root r (probe iteration) minus the receiving core's first
    // flag seen for root r, on the same chip's wall clock; and each core's loop pass time
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        std::map<uint32_t, uint32_t> rx_seen, tx_first, rx_issued, tx_seen;
        std::map<uint32_t, tt::tt_metal::CoreCoord> rx_core, tx_core;
        for (auto& [core, cj] : jobs[n]) {
            std::vector<uint32_t> pr;
            tt::tt_metal::detail::ReadFromDeviceL1(dev, core, base + kRelayProbe, 160, pr, tt::CoreType::ETH);
            for (auto& [root, fwd] : cj.rx) {
                rx_seen[root] = pr[root];
                rx_issued[root] = pr[24 + root];
                rx_core[root] = dev->ethernet_core_from_logical_core(core);
            }
            for (auto& tj : cj.tx) {
                if (tj.relayed) {
                    tx_first[tj.root] = std::max(tx_first[tj.root], pr[8 + tj.root]);
                    if (pr[32 + tj.root] >= tx_seen[tj.root]) {
                        tx_seen[tj.root] = pr[32 + tj.root];
                        tx_core[tj.root] = dev->ethernet_core_from_logical_core(core);
                    }
                }
            }
            if (!cj.tx.empty()) {
                run.loop_ns.push_back(pr[16] ? pr[17] / 1.35 / pr[16] : 0);
            }
            if (!cj.rx.empty()) {
                run.loop_ns_rx.push_back(pr[18] ? pr[19] / 1.35 / pr[18] : 0);
            }
        }
        for (auto& [root, t1] : tx_first) {
            if (rx_seen.count(root)) {
                run.relay_us.push_back(static_cast<int32_t>(t1 - rx_seen[root]) / 1350.0);
                run.stage_rx.push_back(static_cast<int32_t>(rx_issued[root] - rx_seen[root]) / 1350.0);
                run.stage_noc.push_back(static_cast<int32_t>(tx_seen[root] - rx_issued[root]) / 1350.0);
                run.stage_tx.push_back(static_cast<int32_t>(t1 - tx_seen[root]) / 1350.0);
                run.relay_lines.push_back(
                    "    chip " + std::to_string(n) + " root " + std::to_string(root) + " rx (" +
                    std::to_string(rx_core[root].x) + "," + std::to_string(rx_core[root].y) + ") -> tx (" +
                    std::to_string(tx_core[root].x) + "," + std::to_string(tx_core[root].y) + "): seen->issued " +
                    std::to_string(static_cast<int32_t>(rx_issued[root] - rx_seen[root])) + " cyc, issued->tx sees " +
                    std::to_string(static_cast<int32_t>(tx_seen[root] - rx_issued[root])) + " cyc, tx sees->sent " +
                    std::to_string(static_cast<int32_t>(t1 - tx_seen[root])) + " cyc");
            }
        }
    }
    // accuracy, every iteration: the first word of each received chunk (the root's pattern, word 0)
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        for (auto& [core, cj] : jobs[n]) {
            for (auto& [root, fwd] : cj.rx) {
                std::vector<uint32_t> seen;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    dev,
                    core,
                    base + kSeen + 4 * root * 256,
                    4 * std::min<uint32_t>(iters, 256),
                    seen,
                    tt::CoreType::ETH);
                for (uint32_t i = 0; i < std::min<uint32_t>(iters, 256); ++i) {
                    if (seen[i] != ((root << 24) ^ 0x5a5a0000u)) {
                        run.exact = false;
                        run.bad_iterations++;
                    }
                }
            }
        }
    }
    // accuracy: on every chip, the rx core of each root holds the root's chunk in the last iteration's slot
    const uint32_t last_parity = (iters - 1) & 1;
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        for (auto& [core, cj] : jobs[n]) {
            for (auto& [root, fwd] : cj.rx) {
                std::vector<uint32_t> got;
                tt::tt_metal::detail::ReadFromDeviceL1(
                    dev,
                    core,
                    base + kData + (last_parity * kRoots + root) * slot_bytes,
                    slot_bytes,
                    got,
                    tt::CoreType::ETH);
                for (uint32_t w = 0; w < payload / 4; ++w) {
                    if (got[((4 * w / packet) * frame + (4 * w) % packet) / 4] != ((root << 24) ^ w ^ 0x5a5a0000u)) {
                        run.exact = false;
                        break;
                    }
                }
            }
        }
    }
    return run;
}

TEST(Qwen38DataflowMeeting, GroupAllGather) {
    const Torus t = map_torus();
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i) {
        ids[i] = i;
    }
    auto meshes = distributed::MeshDevice::create_unit_meshes(ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
    std::map<int, std::shared_ptr<distributed::MeshDevice>> by_id;
    for (auto& [id, m] : meshes) {
        by_id[id] = m;
    }
    const char* out_path = std::getenv("QWEN38_MEETING_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_meeting.txt");
    const int group = std::getenv("QWEN38_GROUP") ? std::atoi(std::getenv("QWEN38_GROUP")) : 0;
    const uint32_t iters = 256;
    uint32_t run_id = 0x51e10000;
    for (uint32_t payload : {1280u, 3840u, 10400u}) {
        const auto r = run_meeting(by_id, t, group, payload, iters, ++run_id);
        auto r_hops = [&](int a, int b) { return r.hops[a][b]; };
        auto r_arrive = [&](int a, int b) { return r.arrive[a][b]; };
        double lo = 1e9, hi = 0;
        for (double p : r.periods) {
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        for (auto& line : r.stalls) {
            out << line << "\n";
        }
        out << "group " << group << " payload " << payload << " B: period " << r.period_us << " us (min " << lo
            << " max " << hi << ", " << r.periods.size() << " steady iterations), exact " << r.exact
            << " (iterations with a wrong chunk start: " << r.bad_iterations << ")\n";
        // pair latency with the clock offsets cancelled: (arrive[n][r] + arrive[r][n]) / 2 by hop count
        double sum[4] = {};
        int cnt[4] = {};
        for (int n = 0; n < 8; ++n) {
            for (int r = n + 1; r < 8; ++r) {
                const int h = r_hops(r, n);
                sum[h] += (r_arrive(n, r) + r_arrive(r, n)) / 2;
                ++cnt[h];
            }
        }
        out << "  pair latency (offsets cancelled) by hops:";
        for (int h = 1; h <= 3; ++h) {
            out << " " << h << ":" << (cnt[h] ? sum[h] / cnt[h] : 0) << " us (" << cnt[h] << " pairs)";
        }
        double gap = 0;
        for (int n = 0; n < 8; ++n) {
            double last = -1e9;
            for (int q = 0; q < 8; ++q) {
                if (q != n) {
                    last = std::max(last, r.arrive[n][q]);
                }
            }
            gap += r.restart[n] - last;
        }
        out << "; restart after the last arrival " << gap / 8 << " us (mean over chips)\n";
        {
            double a = 0, lo = 1e9, hi = -1e9;
            for (double v : r.relay_us) {
                a += v;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            double l = 0;
            for (double v : r.loop_ns) {
                l += v;
            }
            double lr = 0;
            for (double v : r.loop_ns_rx) {
                lr += v;
            }
            out << "  relay inside a chip (first flag seen -> last forward send): mean "
                << (r.relay_us.empty() ? 0 : a / r.relay_us.size()) << " us (min " << lo << " max " << hi << ", "
                << r.relay_us.size() << " relays); loop pass tx " << (r.loop_ns.empty() ? 0 : l / r.loop_ns.size())
                << " ns, rx " << (r.loop_ns_rx.empty() ? 0 : lr / r.loop_ns_rx.size()) << " ns\n";
        }
        {
            auto mean = [](const std::vector<double>& v) {
                double a = 0;
                for (double x : v) {
                    a += x;
                }
                return v.empty() ? 0 : a / v.size();
            };
            out << "  relay stages: rx seen -> NoC issued " << mean(r.stage_rx) << " us, -> tx core sees the trailer "
                << mean(r.stage_noc) << " us, -> eth send issued " << mean(r.stage_tx) << " us (" << r.stage_rx.size()
                << " relays)\n";
        }
        for (auto& line : r.relay_lines) {
            out << line << "\n";
        }
        out << "  raw arrive[n][r] (us):\n";
        for (int n = 0; n < 8; ++n) {
            out << "   n" << n << ":";
            for (int q = 0; q < 8; ++q) {
                out << " " << (q == n ? 0.0 : r.arrive[n][q]);
            }
            out << " | restart " << r.restart[n] << "\n";
        }
        out.flush();
        EXPECT_TRUE(r.exact) << "payload " << payload;
    }
    for (auto& [id, m] : meshes) {
        m->close();
    }
}

}  // namespace
}  // namespace qwen38_dataflow
