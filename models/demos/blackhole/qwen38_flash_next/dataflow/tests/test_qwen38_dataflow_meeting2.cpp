// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1b' (worklog 14): the group meeting, version 2 (kernels/meeting2_erisc.cpp): the all-gather of one 2 x 4 group
// (TP8) along the BFS trees of the block in two planes (channel parity), chunk half p in plane p, receive FIFO rings,
// relays through staging rings of the same plane, self-timed per plane (iteration k of the own half starts when the
// chip holds every other root's half of iteration k-1 in the plane).  Budget: comm_model.py revision 5 (2 channels per
// direction) AG_tree of the same payload.  Accuracy: a timed run checks the first and last word of every packet on
// arrival; a second run checks every word of every packet; both must report zero bad packets and every core a clean end.
#include <gtest/gtest.h>
#include <algorithm>
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
#include "qwen38_torus.hpp"

namespace qwen38_dataflow {
namespace {

namespace distributed = tt::tt_metal::distributed;

constexpr uint32_t kState = 256, kBad = 512, kStamps = 1024, kProbe = 4096, kRegion = 16384;
constexpr uint32_t kMaxSrc = 2, kMaxChildren = 2, kMaxOwnCores = 3, kMaxRing = 64;

uint32_t pattern(uint32_t root, uint32_t plane, uint32_t word) {
    return (root << 24) ^ (plane << 20) ^ word ^ 0x5a5a0000u;
}

struct CoreJob {
    int chip = -1;              // block position
    uint32_t plane = 0;
    tt::tt_metal::CoreCoord core;
    int peer = -1;              // block position of the neighbour this core's link reaches
    std::vector<tt::tt_metal::CoreCoord> sources;   // rx cores (same chip) feeding this core's staging rings
    std::vector<uint32_t> src_roots;                // per source: roots relayed
    std::vector<uint32_t> src_child_index;          // per source: this core's index among the source's children
    std::vector<tt::tt_metal::CoreCoord> children;  // tx cores (same chip) this core's rx relays to
    std::vector<uint32_t> child_src_index;          // per child: this core's index among the child's sources
    uint32_t child_mask = 0;                        // 2 bits per root
    uint32_t rx_roots = 0;                          // roots arriving on this core's channel
    bool own = false;
};

struct Meeting2Run {
    double period_us = 0;
    std::vector<double> periods;
    bool exact = true;
    uint32_t bad = 0;
    std::vector<std::string> stalls;
    double arrive[2][8][8] = {};
    int hops[8][8] = {};
    double restart[2][8] = {};
    std::vector<double> st_rx, st_noc, st_tx, relay;
    std::vector<double> loop_tx, loop_rx;
};

Meeting2Run run_meeting2(
    const std::map<int, std::shared_ptr<distributed::MeshDevice>>& meshes,
    const Torus& t,
    int group,
    uint32_t chunk,
    uint32_t iters,
    uint32_t run_id,
    bool full_check) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t limit =
        base +
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t half = chunk / 2;
    TT_FATAL(chunk % 32 == 0, "chunk {} B is not a multiple of 32", chunk);
    const uint32_t packets = (half + 2047) / 2048;
    const uint32_t payload = (((half + packets - 1) / packets) + 31) & ~31u;
    const uint32_t last_payload = half - (packets - 1) * payload;
    TT_FATAL(last_payload > 0 && last_payload % 16 == 0 && last_payload <= payload, "bad cut of {} B", half);
    const uint32_t stride = payload + 32;
    // the receive ring and the staging rings (a power of two each: the kernel masks the counts); the staging rings of a
    // core are only needed if it relays, but every core gets them (same layout everywhere).  Eth L1 for kernels is
    // 366,368 B (UNRESERVED, this build): 10,400 B (packet 1,792 B slot) fits a 64-slot ring and two 16-slot staging
    // rings, not three 64-slot rings.
    uint32_t slots = kMaxRing;
    while (slots > 16 && slots / 2 >= 3 * 4 * packets && (slots / 2) * stride >= 65536) {
        slots /= 2;
    }
    // a staging ring carries at most 2 roots (the block's BFS trees, checked below): 3 iterations x 2 roots x packets
    uint32_t stage_slots = slots;
    while (base + kRegion + (slots + kMaxSrc * stage_slots + 2 * packets) * stride > limit &&
           stage_slots / 2 >= 3 * 2 * packets) {
        stage_slots /= 2;
    }
    while (base + kRegion + (slots + kMaxSrc * stage_slots + 2 * packets) * stride > limit) {
        slots /= 2;
    }
    // in flight on a channel: at most three iterations (a chip starts k+1 once it holds every half of k, so every chip
    // has started k and holds all of k-1, while frames of k-1 toward other chips may still be on their way) x at most 4
    // roots per channel x the packets of a half.  A receive ring and a staging ring at least that deep never fill, so
    // no receiver stops and the cyclic relay dependencies around the 4-rings cannot deadlock.
    TT_FATAL(slots >= 3 * 4 * packets, "chunk {} B leaves only {} ring slots ({} needed)", chunk, slots, 3 * 4 * packets);
    const uint32_t ring = base + kRegion;
    const uint32_t stage = ring + slots * stride;
    const uint32_t own_buf = stage + kMaxSrc * stage_slots * stride;
    std::array<int, 8> chip_of{};
    for (int n = 0; n < 8; ++n) {
        chip_of[n] = t.at[2 * group + n / 4][n % 4];
    }
    // the core of chip a toward chip b in plane p, and its peer
    auto core_toward = [&](int a, int b, uint32_t p) {
        for (const auto& c : t.link_cores.at({chip_of[a], chip_of[b]})) {
            if ((c.y & 1) == p) {
                return c;
            }
        }
        TT_THROW("no plane {} core from {} to {}", p, a, b);
    };
    std::map<std::tuple<int, uint32_t, tt::tt_metal::CoreCoord>, CoreJob> jobs;
    auto job_of = [&](int n, uint32_t p, tt::tt_metal::CoreCoord c) -> CoreJob& {
        auto& j = jobs[{n, p, c}];
        j.chip = n;
        j.plane = p;
        j.core = c;
        return j;
    };
    for (uint32_t p = 0; p < 2; ++p) {
        for (int root = 0; root < 8; ++root) {
            const auto parent = bfs_parent(root);
            for (int n = 0; n < 8; ++n) {
                if (parent[n] < 0) {
                    continue;
                }
                const int up = parent[n];
                const auto send = core_toward(up, n, p);
                const auto [peer_chip, recv] = t.peer.at({chip_of[up], send});
                TT_FATAL(peer_chip == chip_of[n] && (recv.y & 1) == p, "plane {} link {}->{} breaks parity", p, up, n);
                auto& sj = job_of(up, p, send);
                sj.peer = n;
                auto& rj = job_of(n, p, recv);
                rj.peer = up;
                rj.rx_roots |= 1u << root;
                if (up == root) {
                    sj.own = true;
                } else {
                    // relayed at chip `up`: the rx core of `up` for this root feeds `send`
                    const int upup = parent[up];
                    const auto in_send = core_toward(upup, up, p);
                    const auto [c2, in_recv] = t.peer.at({chip_of[upup], in_send});
                    auto& in_rx = job_of(up, p, in_recv);
                    auto& out_tx = job_of(up, p, send);
                    auto it = std::find(in_rx.children.begin(), in_rx.children.end(), send);
                    uint32_t ci = it - in_rx.children.begin();
                    if (it == in_rx.children.end()) {
                        in_rx.children.push_back(send);
                    }
                    auto is = std::find(out_tx.sources.begin(), out_tx.sources.end(), in_recv);
                    uint32_t si = is - out_tx.sources.begin();
                    if (is == out_tx.sources.end()) {
                        out_tx.sources.push_back(in_recv);
                        out_tx.src_roots.push_back(0);
                        out_tx.src_child_index.push_back(ci);
                    }
                    out_tx.src_roots[si] |= 1u << root;
                    if (in_rx.child_src_index.size() < in_rx.children.size()) {
                        in_rx.child_src_index.push_back(si);
                    }
                    in_rx.child_mask |= 1u << (2 * root + ci);
                }
            }
        }
    }
    for (auto& [key, j] : jobs) {
        TT_FATAL(j.sources.size() <= kMaxSrc && j.children.size() <= kMaxChildren, "relay fan-in/out too large");
        for (uint32_t roots : j.src_roots) {
            TT_FATAL(
                stage_slots >= 3 * static_cast<uint32_t>(__builtin_popcount(roots)) * packets,
                "staging ring of {} slots for {} roots of {} packets",
                stage_slots,
                __builtin_popcount(roots),
                packets);
        }
    }
    // the own-half tx cores per (chip, plane)
    std::map<std::pair<int, uint32_t>, std::vector<tt::tt_metal::CoreCoord>> own_cores;
    for (auto& [key, j] : jobs) {
        if (j.own) {
            own_cores[{j.chip, j.plane}].push_back(j.core);
        }
    }
    const uint32_t probe_iter = iters / 2;
    std::map<int, distributed::MeshWorkload> workloads;
    tt::tt_metal::CoreCoord stamp_core;
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        tt::tt_metal::Program program;
        std::set<tt::tt_metal::CoreCoord> cores;
        for (auto& [key, j] : jobs) {
            if (j.chip == n) {
                cores.insert(j.core);
            }
        }
        for (const auto& core : cores) {
            const CoreJob* jp = nullptr;
            for (auto& [key, j] : jobs) {
                if (j.chip == n && j.core == core) {
                    jp = &j;
                }
            }
            const CoreJob& j = *jp;
            const bool stamp = n == 0 && j.plane == 0 && own_cores[{0, 0}].front() == core;
            if (stamp) {
                stamp_core = core;
            }
            auto v = [&](tt::tt_metal::CoreCoord c) { return dev->ethernet_core_from_logical_core(c); };
            std::vector<uint32_t> args{
                base,
                iters,
                packets,
                payload,
                last_payload,
                stride,
                ring,
                slots,
                stage,
                stage_slots,
                own_buf,
                j.plane,
                static_cast<uint32_t>(n),
                run_id,
                probe_iter,
                stamp ? 1u : 0u,
                2000000u,
                full_check ? 1u : 0u,
                j.own ? 1u : 0u,
                7u,
                static_cast<uint32_t>(j.sources.size())};
            for (uint32_t s = 0; s < kMaxSrc; ++s) {
                if (s < j.sources.size()) {
                    const auto c = v(j.sources[s]);
                    args.insert(
                        args.end(),
                        {static_cast<uint32_t>(c.x),
                         static_cast<uint32_t>(c.y),
                         j.src_child_index[s],
                         iters * packets * static_cast<uint32_t>(__builtin_popcount(j.src_roots[s]))});
                } else {
                    args.insert(args.end(), {0u, 0u, 0u, 0u});
                }
            }
            args.push_back(iters * packets * static_cast<uint32_t>(__builtin_popcount(j.rx_roots)));
            args.push_back(j.child_mask);
            args.push_back(static_cast<uint32_t>(j.children.size()));
            for (uint32_t c = 0; c < kMaxChildren; ++c) {
                if (c < j.children.size()) {
                    const auto cc = v(j.children[c]);
                    args.insert(args.end(), {static_cast<uint32_t>(cc.x), static_cast<uint32_t>(cc.y), j.child_src_index[c]});
                } else {
                    args.insert(args.end(), {0u, 0u, 0u});
                }
            }
            const auto& oc = own_cores[{n, j.plane}];
            args.push_back(static_cast<uint32_t>(oc.size()));
            for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
                if (c < oc.size()) {
                    const auto cc = v(oc[c]);
                    args.insert(args.end(), {static_cast<uint32_t>(cc.x), static_cast<uint32_t>(cc.y)});
                } else {
                    args.insert(args.end(), {0u, 0u});
                }
            }
            for (uint32_t role = 0; role < 2; ++role) {
                auto k = tt::tt_metal::CreateKernel(
                    program,
                    "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/meeting2_erisc.cpp",
                    core,
                    tt::tt_metal::EthernetConfig{
                        .noc = role == 0 ? tt::tt_metal::NOC::NOC_0 : tt::tt_metal::NOC::NOC_1,
                        .processor = role == 0 ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                               : tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .compile_args = {role}});
                tt::tt_metal::SetRuntimeArgs(program, k, core, args);
            }
            std::vector<uint32_t> zero((own_buf - base) / 4, 0);
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, base, zero, tt::CoreType::ETH);
            std::vector<uint32_t> own(2 * packets * stride / 4, 0);
            for (uint32_t parity = 0; parity < 2; ++parity) {
                for (uint32_t pk = 0; pk < packets; ++pk) {
                    const uint32_t len = pk == packets - 1 ? last_payload : payload;
                    const uint32_t slot = (parity * packets + pk) * stride + (payload - len);
                    for (uint32_t w = 0; w < len / 4; ++w) {
                        own[slot / 4 + w] = pattern(n, j.plane, pk * (payload / 4) + w);
                    }
                }
            }
            tt::tt_metal::detail::WriteToDeviceL1(dev, core, own_buf, own, tt::CoreType::ETH);
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
    Meeting2Run run;
    auto read = [&](int n, tt::tt_metal::CoreCoord core, uint32_t addr, uint32_t bytes) {
        std::vector<uint32_t> w;
        tt::tt_metal::detail::ReadFromDeviceL1(
            meshes.at(chip_of[n])->get_devices()[0], core, addr, bytes, w, tt::CoreType::ETH);
        return w;
    };
    auto read64 = [&](int n, tt::tt_metal::CoreCoord core, uint32_t addr) {
        const auto w = read(n, core, addr, 8);
        return (static_cast<uint64_t>(w[1]) << 32) | w[0];
    };
    for (auto& [key, j] : jobs) {
        const auto st = read(j.chip, j.core, base + kState, 64);
        const auto bad = read(j.chip, j.core, base + kBad, 4)[0];
        run.bad += bad;
        if (st[0] != 0x600d0000u || st[8] != 0x600d0000u || bad != 0) {
            run.exact = false;
            run.stalls.push_back(
                "chip " + std::to_string(j.chip) + " plane " + std::to_string(j.plane) + " core (" +
                std::to_string(j.core.x) + "," + std::to_string(j.core.y) + "): tx state " + std::to_string(st[0]) +
                " sent " + std::to_string(st[1]) + " credits " + std::to_string(st[2]) + " own " +
                std::to_string(st[3]) + " src " + std::to_string(st[4]) + "/" + std::to_string(st[5]) + " arrivals " +
                std::to_string(st[6]) + " fwd " + std::to_string(st[7]) + "; rx state " + std::to_string(st[8]) +
                " received " + std::to_string(st[9]) + " released " + std::to_string(st[10]) + " staged " +
                std::to_string(st[11] & 0xffff) + "/" + std::to_string(st[11] >> 16) + " head seq " +
                std::to_string(st[12]) + "; bad " + std::to_string(bad));
        }
        const auto pr = read(j.chip, j.core, base + kProbe + 256, 16);
        if (pr[0]) {
            run.loop_tx.push_back(pr[1] / 1.35 / pr[0]);
        }
        if (pr[2]) {
            run.loop_rx.push_back(pr[3] / 1.35 / pr[2]);
        }
    }
    const auto st = read(0, stamp_core, base + kStamps, 8 * iters);
    for (uint32_t i = iters / 4 + 1; i < iters; ++i) {
        const uint64_t a = (static_cast<uint64_t>(st[2 * i - 1]) << 32) | st[2 * i - 2];
        const uint64_t b = (static_cast<uint64_t>(st[2 * i + 1]) << 32) | st[2 * i];
        run.periods.push_back((static_cast<double>(b) - static_cast<double>(a)) / 1350.0);
    }
    for (double p : run.periods) {
        run.period_us += p / run.periods.size();
    }
    // arrivals per plane: chip n's rx core for root r, relative to chip n's own start in that plane
    for (uint32_t p = 0; p < 2; ++p) {
        for (int n = 0; n < 8; ++n) {
            const auto oc = own_cores[{n, p}].front();
            const double s0 = static_cast<double>(read64(n, oc, base + kProbe));
            const double s1 = static_cast<double>(read64(n, oc, base + kProbe + 8));
            run.restart[p][n] = (s1 - s0) / 1350.0;
            for (auto& [key, j] : jobs) {
                if (j.chip != n || j.plane != p) {
                    continue;
                }
                for (int r = 0; r < 8; ++r) {
                    if ((j.rx_roots >> r) & 1) {
                        run.arrive[p][n][r] = (static_cast<double>(read64(n, j.core, base + kProbe + 128 + 8 * r)) - s0) / 1350.0;
                    }
                }
            }
        }
    }
    for (int n = 0; n < 8; ++n) {
        for (int r = 0; r < 8; ++r) {
            int dc = std::abs(n % 4 - r % 4);
            dc = std::min(dc, 4 - dc);
            run.hops[n][r] = (n / 4 != r / 4) + dc;
        }
    }
    // relay stages: rx core (seen, issued) of root r at chip n, the child tx core (saw, sent)
    for (auto& [key, j] : jobs) {
        if (j.children.empty()) {
            continue;
        }
        const auto rx = read(j.chip, j.core, base + kProbe + 192, 64);
        for (uint32_t c = 0; c < j.children.size(); ++c) {
            const auto tx = read(j.chip, j.children[c], base + kProbe + 64, 64);
            for (int r = 0; r < 8; ++r) {
                if (((j.child_mask >> (2 * r + c)) & 1) == 0) {
                    continue;
                }
                const uint32_t seen = rx[r], issued = rx[8 + r], saw = tx[r], sent = tx[8 + r];
                run.st_rx.push_back(static_cast<int32_t>(issued - seen) / 1350.0);
                run.st_noc.push_back(static_cast<int32_t>(saw - issued) / 1350.0);
                run.st_tx.push_back(static_cast<int32_t>(sent - saw) / 1350.0);
                run.relay.push_back(static_cast<int32_t>(sent - seen) / 1350.0);
            }
        }
    }
    return run;
}

double mean(const std::vector<double>& v) {
    double a = 0;
    for (double x : v) {
        a += x;
    }
    return v.empty() ? 0 : a / v.size();
}

TEST(Qwen38DataflowMeeting2, GroupAllGatherTwoPlanes) {
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
    const char* out_path = std::getenv("QWEN38_MEETING2_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_meeting2.txt");
    const int group = std::getenv("QWEN38_GROUP") ? std::atoi(std::getenv("QWEN38_GROUP")) : 0;
    const uint32_t iters = 256;
    uint32_t run_id = 0x5e200000;
    bool all_exact = true;
    for (uint32_t chunk : {1280u, 3840u, 10400u}) {
        const Meeting2Run check = run_meeting2(by_id, t, group, chunk, iters, ++run_id, true);
        const Meeting2Run r = run_meeting2(by_id, t, group, chunk, iters, ++run_id, false);
        for (auto& line : check.stalls) {
            out << "CHECK RUN " << line << "\n";
        }
        for (auto& line : r.stalls) {
            out << "TIMED RUN " << line << "\n";
        }
        const bool exact = check.exact && r.exact;
        all_exact = all_exact && exact;
        double lo = 1e9, hi = 0;
        for (double p : r.periods) {
            lo = std::min(lo, p);
            hi = std::max(hi, p);
        }
        out << "group " << group << " chunk " << chunk << " B (2 planes x " << chunk / 2 << " B): period " << r.period_us
            << " us (min " << lo << " max " << hi << ", " << r.periods.size() << " steady), exact " << exact
            << " (bad packets: full check " << check.bad << ", timed " << r.bad << ")\n";
        for (uint32_t p = 0; p < 2; ++p) {
            double sum[4] = {};
            int cnt[4] = {};
            for (int n = 0; n < 8; ++n) {
                for (int q = n + 1; q < 8; ++q) {
                    const int h = r.hops[n][q];
                    sum[h] += (r.arrive[p][n][q] + r.arrive[p][q][n]) / 2;
                    ++cnt[h];
                }
            }
            double gap = 0;
            for (int n = 0; n < 8; ++n) {
                double last = -1e9;
                for (int q = 0; q < 8; ++q) {
                    if (q != n) {
                        last = std::max(last, r.arrive[p][n][q]);
                    }
                }
                gap += r.restart[p][n] - last;
            }
            out << "  plane " << p << " pair latency by hops:";
            for (int h = 1; h <= 3; ++h) {
                out << " " << h << ":" << (cnt[h] ? sum[h] / cnt[h] : 0) << " us";
            }
            out << "; restart after the last arrival " << gap / 8 << " us\n";
        }
        out << "  relay: mean " << mean(r.relay) << " us (" << r.relay.size() << "): rx seen -> issued " << mean(r.st_rx)
            << ", issued -> tx sees " << mean(r.st_noc) << ", tx sees -> sent " << mean(r.st_tx) << "; loop pass tx "
            << mean(r.loop_tx) << " ns, rx " << mean(r.loop_rx) << " ns\n";
        out.flush();
    }
    for (auto& [id, m] : meshes) {
        m->close();
    }
    EXPECT_TRUE(all_exact);
}

}  // namespace
}  // namespace qwen38_dataflow
