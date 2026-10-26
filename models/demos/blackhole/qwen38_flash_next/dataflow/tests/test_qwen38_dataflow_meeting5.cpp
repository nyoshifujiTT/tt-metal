// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// I-1b v5 (worklog 14): the group collectives on one table-driven kernel (kernels/meeting5_erisc.cpp): AG (all-gather,
// BFS trees: flow = root) and RS (reduce-scatter, direct: flow = (source, owner), the owner's next iteration waits for
// its 7 slices).  Two planes by channel parity (half of every flow's payload per plane), receive and staging rings
// sized by the causality bound, no flow control, frames untouched after the source.  Self-timed per plane.
// Meeting time, free of clock offsets: per iteration and pair of chips (a, b) with flows a->b and b->a, the mean of
// (arrival at b of a's flow - a's start) and (arrival at a of b's flow - b's start); the meeting = the largest pair
// value, averaged over 64 iterations.  Budget: comm_model.py revision 5.  Accuracy: every packet's tag order and its
// payload words at its destination (all words in a second run); zero bad packets and a clean end on every core.
#include <gtest/gtest.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <map>
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

constexpr uint32_t kState = 256, kBad = 512, kStamps = 1024, kArrArr = 8192, kOwnArr = 16384,
                   kRegion = 16640, kWindow = 64;
constexpr uint32_t kMaxFlows = 64, kMaxOwnFlows = 7, kMaxInputs = 8, kMaxChildren = 2, kMaxOwnCores = 3,
                   kMaxRing = 64;

uint32_t pattern(uint32_t flow, uint32_t plane, uint32_t word) {
    return (flow << 24) ^ (plane << 20) ^ word ^ 0x5a5a0000u;
}

enum class Kind { AG, RS };

struct Flow {
    int src;
    std::vector<int> dsts;
    std::map<int, int> parent;  // node -> parent on the route tree (dsts and relays)
};

// route src -> dst in the 2 x 4 block: vertical first, then the shorter ring direction (bfs_parent's rule)
std::vector<int> route(int src, int dst) {
    std::vector<int> path{src};
    int cr = src / 4, cc = src % 4;
    const int nr = dst / 4, nc = dst % 4;
    if (nr != cr) {
        cr = nr;
        path.push_back(cr * 4 + cc);
    }
    int dc = nc - cc;
    if (dc > 2) {
        dc -= 4;
    }
    if (dc < -2) {
        dc += 4;
    }
    while (dc != 0) {
        const int s = dc > 0 ? 1 : -1;
        cc = (cc + s + 4) % 4;
        dc -= s;
        path.push_back(cr * 4 + cc);
    }
    return path;
}

std::vector<Flow> make_flows(Kind kind) {
    std::vector<Flow> flows;
    if (kind == Kind::AG) {
        for (int r = 0; r < 8; ++r) {
            Flow f{r, {}, {}};
            const auto par = bfs_parent(r);
            for (int n = 0; n < 8; ++n) {
                if (par[n] >= 0) {
                    f.dsts.push_back(n);
                    f.parent[n] = par[n];
                }
            }
            flows.push_back(f);
        }
    } else {
        for (int s = 0; s < 8; ++s) {
            for (int d = 0; d < 8; ++d) {
                if (s == d) {
                    continue;
                }
                Flow f{s, {d}, {}};
                const auto p = route(s, d);
                for (size_t i = 1; i < p.size(); ++i) {
                    f.parent[p[i]] = p[i - 1];
                }
                flows.push_back(f);
            }
        }
    }
    return flows;
}

struct CoreJob {
    int chip = -1;
    uint32_t plane = 0;
    tt::tt_metal::CoreCoord core;
    std::vector<uint32_t> own_flows;                   // flows this core sends as source
    std::vector<tt::tt_metal::CoreCoord> sources;      // rx cores feeding this core's staging rings
    std::vector<uint32_t> src_frames_per_iter;         // per source: flows relayed
    std::vector<tt::tt_metal::CoreCoord> children;     // tx cores this core's rx relays to
    std::vector<uint32_t> child_src_index;
    std::map<uint32_t, uint32_t> action;               // flow -> bits (relay child 0 / 1, destination)
    uint32_t rx_flows = 0;                              // flows arriving on this core's channel
};

struct Result {
    bool exact = true;
    uint32_t bad = 0;
    std::vector<std::string> stalls;
    double period_us = 0;
    double meeting_us[2] = {0, 0};
    double pair_mean_us[2] = {0, 0};
};

Result run(
    const std::map<int, std::shared_ptr<distributed::MeshDevice>>& meshes,
    const Torus& t,
    int group,
    Kind kind,
    uint32_t chunk,  // AG: bytes per chip (the chip's chunk); RS: bytes per slice (one flow)
    uint32_t iters,
    uint32_t run_id,
    bool full_check) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t base =
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const uint32_t limit =
        base +
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    const std::vector<Flow> flows = make_flows(kind);
    TT_FATAL(flows.size() <= kMaxFlows, "too many flows");
    const uint32_t half = chunk / 2;
    TT_FATAL(chunk % 32 == 0, "payload {} B is not a multiple of 32", chunk);
    const uint32_t packets = (half + 2047) / 2048;
    const uint32_t payload = (((half + packets - 1) / packets) + 31) & ~31u;
    const uint32_t last_payload = half - (packets - 1) * payload;
    TT_FATAL(last_payload > 0 && last_payload % 16 == 0 && last_payload <= payload, "bad cut of {} B", half);
    const uint32_t stride = payload + 32;
    std::array<int, 8> chip_of{};
    for (int n = 0; n < 8; ++n) {
        chip_of[n] = t.at[2 * group + n / 4][n % 4];
    }
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
        for (uint32_t fi = 0; fi < flows.size(); ++fi) {
            const Flow& f = flows[fi];
            std::set<int> dset(f.dsts.begin(), f.dsts.end());
            for (const auto& [node, up] : f.parent) {
                const auto send = core_toward(up, node, p);
                const auto [peer_chip, recv] = t.peer.at({chip_of[up], send});
                TT_FATAL(peer_chip == chip_of[node] && (recv.y & 1) == p, "plane {} breaks parity", p);
                job_of(up, p, send);
                auto& rj = job_of(node, p, recv);
                rj.rx_flows += 1;
                if (dset.count(node)) {
                    rj.action[fi] |= 4;
                }
                if (up == f.src) {
                    auto& sj = jobs.at({up, p, send});
                    sj.own_flows.push_back(fi);
                } else {
                    const int uu = f.parent.at(up);
                    const auto in_send = core_toward(uu, up, p);
                    const auto [c2, in_recv] = t.peer.at({chip_of[uu], in_send});
                    auto& in_rx = job_of(up, p, in_recv);
                    auto& out_tx = job_of(up, p, send);
                    auto it = std::find(in_rx.children.begin(), in_rx.children.end(), send);
                    const uint32_t ci = it - in_rx.children.begin();
                    if (it == in_rx.children.end()) {
                        in_rx.children.push_back(send);
                    }
                    auto is = std::find(out_tx.sources.begin(), out_tx.sources.end(), in_recv);
                    const uint32_t si = is - out_tx.sources.begin();
                    if (is == out_tx.sources.end()) {
                        out_tx.sources.push_back(in_recv);
                        out_tx.src_frames_per_iter.push_back(0);
                    }
                    out_tx.src_frames_per_iter[si] += 1;
                    if (in_rx.child_src_index.size() < in_rx.children.size()) {
                        in_rx.child_src_index.push_back(si);
                    }
                    in_rx.action[fi] |= 1u << ci;
                }
            }
        }
    }
    // the inputs of a chip's next iteration: AG every flow not its own; RS the flows that end at it
    std::array<std::vector<uint32_t>, 8> inputs;
    for (int n = 0; n < 8; ++n) {
        for (uint32_t fi = 0; fi < flows.size(); ++fi) {
            const Flow& f = flows[fi];
            if (f.src != n && std::find(f.dsts.begin(), f.dsts.end(), n) != f.dsts.end()) {
                inputs[n].push_back(fi);
            }
        }
        TT_FATAL(inputs[n].size() <= kMaxInputs, "too many inputs");
    }
    // per (chip, plane): the cores that send own flows (they wait for the inputs)
    std::map<std::pair<int, uint32_t>, std::vector<tt::tt_metal::CoreCoord>> wait_cores;
    uint32_t max_link_frames = 0, max_stage_frames = 0, max_own = 0;
    for (auto& [key, j] : jobs) {
        TT_FATAL(j.sources.size() <= 2 && j.children.size() <= kMaxChildren, "relay fan-in/out too large");
        TT_FATAL(j.own_flows.size() <= kMaxOwnFlows, "too many own flows on a link");
        if (!j.own_flows.empty()) {
            wait_cores[{j.chip, j.plane}].push_back(j.core);
        }
        max_link_frames = std::max(max_link_frames, j.rx_flows * packets);
        for (uint32_t fpi : j.src_frames_per_iter) {
            max_stage_frames = std::max(max_stage_frames, fpi * packets);
        }
        max_own = std::max<uint32_t>(max_own, j.own_flows.size());
    }
    // the causality bound: a link carries frames of at most three iterations
    uint32_t slots = kMaxRing;
    while (slots / 2 >= 3 * max_link_frames && slots > 16) {
        slots /= 2;
    }
    uint32_t stage_slots = slots;
    while (stage_slots / 2 >= 3 * max_stage_frames && stage_slots > 8) {
        stage_slots /= 2;
    }
    const uint32_t own_bytes = max_own * 2 * packets * stride;
    while (base + kRegion + (slots + 2 * stage_slots) * stride + own_bytes > limit && stage_slots / 2 >= 3 * max_stage_frames) {
        stage_slots /= 2;
    }
    TT_FATAL(
        slots >= 3 * max_link_frames && stage_slots >= 3 * max_stage_frames &&
            base + kRegion + (slots + 2 * stage_slots) * stride + own_bytes <= limit,
        "rings do not fit: ring {} (needs {}), staging {} (needs {}), {} B",
        slots,
        3 * max_link_frames,
        stage_slots,
        3 * max_stage_frames,
        kRegion + (slots + 2 * stage_slots) * stride + own_bytes);
    const uint32_t ring = base + kRegion;
    const uint32_t stage = ring + slots * stride;
    const uint32_t own_buf = stage + 2 * stage_slots * stride;
    const uint32_t probe_iter = iters / 2;
    // arrival stamp slots (rx core): one per arriving flow that ends at this chip
    std::map<std::pair<int, uint32_t>, std::map<uint32_t, uint32_t>> arrival_slot;  // (chip, plane) -> flow -> slot
    std::map<std::tuple<int, uint32_t, uint32_t>, tt::tt_metal::CoreCoord> arrival_core;  // (chip, plane, flow)
    for (auto& [key, j] : jobs) {
        for (auto& [fi, a] : j.action) {
            if (a & 4) {
                auto& m = arrival_slot[{j.chip, j.plane}];
                const uint32_t slot = m.size();
                m[fi] = slot;
                arrival_core[{j.chip, j.plane, fi}] = j.core;
            }
        }
    }
    std::map<int, distributed::MeshWorkload> workloads;
    tt::tt_metal::CoreCoord stamp_core;
    for (int n = 0; n < 8; ++n) {
        auto* dev = meshes.at(chip_of[n])->get_devices()[0];
        tt::tt_metal::Program program;
        std::map<tt::tt_metal::CoreCoord, const CoreJob*> by_core;
        for (auto& [key, j] : jobs) {
            if (j.chip == n) {
                by_core[j.core] = &j;
            }
        }
        for (const auto& [core, jp] : by_core) {
            const CoreJob& j = *jp;
            const bool stamp = n == 0 && j.plane == 0 && !wait_cores[{0, 0}].empty() && wait_cores[{0, 0}].front() == core;
            if (stamp) {
                stamp_core = core;
            }
            auto v = [&](tt::tt_metal::CoreCoord c) { return dev->ethernet_core_from_logical_core(c); };
            std::vector<uint32_t> args{
                base, iters, packets, payload, last_payload, stride, ring, slots, stage, stage_slots, own_buf, j.plane,
                run_id, probe_iter, stamp ? 1u : 0u, 2000000u, full_check ? 1u : 0u};
            args.push_back(static_cast<uint32_t>(j.own_flows.size()));
            for (uint32_t i = 0; i < kMaxOwnFlows; ++i) {
                args.push_back(i < j.own_flows.size() ? j.own_flows[i] : 0u);
            }
            const auto& in = j.own_flows.empty() ? std::vector<uint32_t>{} : inputs[n];
            args.push_back(static_cast<uint32_t>(in.size()));
            for (uint32_t i = 0; i < kMaxInputs; ++i) {
                args.push_back(i < in.size() ? in[i] : 0u);
            }
            for (uint32_t s = 0; s < 2; ++s) {
                args.push_back(s < j.src_frames_per_iter.size() ? iters * packets * j.src_frames_per_iter[s] : 0u);
            }
            args.push_back(iters * packets * j.rx_flows);
            std::array<uint32_t, kMaxFlows / 8> act{};
            for (auto& [fi, a] : j.action) {
                act[fi / 8] |= a << (4 * (fi % 8));
            }
            args.insert(args.end(), act.begin(), act.end());
            args.push_back(static_cast<uint32_t>(j.children.size()));
            for (uint32_t c = 0; c < kMaxChildren; ++c) {
                if (c < j.children.size()) {
                    const auto cc = v(j.children[c]);
                    args.insert(args.end(), {static_cast<uint32_t>(cc.x), static_cast<uint32_t>(cc.y), j.child_src_index[c]});
                } else {
                    args.insert(args.end(), {0u, 0u, 0u});
                }
            }
            const auto& wc = wait_cores[{n, j.plane}];
            args.push_back(static_cast<uint32_t>(wc.size()));
            TT_FATAL(wc.size() <= kMaxOwnCores, "too many waiting cores");
            for (uint32_t c = 0; c < kMaxOwnCores; ++c) {
                if (c < wc.size()) {
                    const auto cc = v(wc[c]);
                    args.insert(args.end(), {static_cast<uint32_t>(cc.x), static_cast<uint32_t>(cc.y)});
                } else {
                    args.insert(args.end(), {0u, 0u});
                }
            }
            const auto& as = arrival_slot[{n, j.plane}];
            args.push_back(static_cast<uint32_t>(as.size()));
            std::array<uint32_t, kMaxFlows / 4> slot_of{};
            slot_of.fill(0xffffffffu);
            for (auto& [fi, sl] : as) {
                if (arrival_core.at({n, j.plane, fi}) == core) {
                    slot_of[fi / 4] = (slot_of[fi / 4] & ~(0xffu << (8 * (fi % 4)))) | (sl << (8 * (fi % 4)));
                }
            }
            args.insert(args.end(), slot_of.begin(), slot_of.end());
            for (uint32_t role = 0; role < 2; ++role) {
                auto k = tt::tt_metal::CreateKernel(
                    program,
                    "models/demos/blackhole/qwen38_flash_next/dataflow/kernels/meeting5_erisc.cpp",
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
            if (!j.own_flows.empty()) {
                std::vector<uint32_t> own(j.own_flows.size() * 2 * packets * stride / 4, 0);
                for (uint32_t i = 0; i < j.own_flows.size(); ++i) {
                    for (uint32_t parity = 0; parity < 2; ++parity) {
                        for (uint32_t pk = 0; pk < packets; ++pk) {
                            const uint32_t len = pk == packets - 1 ? last_payload : payload;
                            const uint32_t slot = ((i * 2 + parity) * packets + pk) * stride + (payload - len);
                            for (uint32_t w = 0; w < len / 4; ++w) {
                                own[slot / 4 + w] = pattern(j.own_flows[i], j.plane, pk * (payload / 4) + w);
                            }
                        }
                    }
                }
                tt::tt_metal::detail::WriteToDeviceL1(dev, core, own_buf, own, tt::CoreType::ETH);
            }
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
    Result res;
    auto read = [&](int n, tt::tt_metal::CoreCoord core, uint32_t addr, uint32_t bytes) {
        std::vector<uint32_t> w;
        tt::tt_metal::detail::ReadFromDeviceL1(
            meshes.at(chip_of[n])->get_devices()[0], core, addr, bytes, w, tt::CoreType::ETH);
        return w;
    };
    for (auto& [key, j] : jobs) {
        const auto st = read(j.chip, j.core, base + kState, 64);
        const auto bad = read(j.chip, j.core, base + kBad, 4)[0];
        res.bad += bad;
        if (st[0] != 0x600d0000u || st[8] != 0x600d0000u || st[13] != 0x600d0000u || st[15] != 0x600d0000u || bad) {
            res.exact = false;
            res.stalls.push_back(
                "chip " + std::to_string(j.chip) + " plane " + std::to_string(j.plane) + " core (" +
                std::to_string(j.core.x) + "," + std::to_string(j.core.y) + "): tx " + std::to_string(st[0]) + " sent " +
                std::to_string(st[1]) + " own " + std::to_string(st[3]) + " src " + std::to_string(st[4]) + "/" +
                std::to_string(st[5]) + " wait " + std::to_string(st[6]) + "; rx " + std::to_string(st[8]) +
                " received " + std::to_string(st[9]) + " head " + std::to_string(st[12]) + "; drain " +
                std::to_string(st[13]) + "/" + std::to_string(st[15]) + "; bad " + std::to_string(bad));
        }
    }
    const auto stv = read(0, stamp_core, base + kStamps, 8 * iters);
    std::vector<double> periods;
    for (uint32_t i = iters / 4 + 1; i < iters; ++i) {
        const uint64_t a = (static_cast<uint64_t>(stv[2 * i - 1]) << 32) | stv[2 * i - 2];
        const uint64_t b = (static_cast<uint64_t>(stv[2 * i + 1]) << 32) | stv[2 * i];
        periods.push_back((static_cast<double>(b) - static_cast<double>(a)) / 1350.0);
    }
    for (double p : periods) {
        res.period_us += p / periods.size();
    }
    // offset-free pair meeting time
    for (uint32_t p = 0; p < 2; ++p) {
        std::array<std::vector<uint32_t>, 8> own_t;
        for (int n = 0; n < 8; ++n) {
            own_t[n] = read(n, wait_cores[{n, p}].front(), base + kOwnArr, 4 * kWindow);
        }
        // arrival of flow fi at chip n (fi ends at n): from its rx core
        std::map<std::pair<int, uint32_t>, std::vector<uint32_t>> arr;
        for (auto& [k2, m] : arrival_slot) {
            if (k2.second != p) {
                continue;
            }
            const int n = k2.first;
            for (auto& [fi, sl] : m) {
                const auto core = arrival_core.at({n, p, fi});
                arr[{n, fi}] = read(n, core, base + kArrArr + 4 * sl * kWindow, 4 * kWindow);
            }
        }
        auto flow_of = [&](int a, int b) -> int {  // a flow from a that ends at b
            for (uint32_t fi = 0; fi < flows.size(); ++fi) {
                if (flows[fi].src == a &&
                    std::find(flows[fi].dsts.begin(), flows[fi].dsts.end(), b) != flows[fi].dsts.end()) {
                    return fi;
                }
            }
            return -1;
        };
        double sum = 0, sum_max = 0;
        uint32_t cnt = 0;
        for (uint32_t w = 0; w < kWindow; ++w) {
            double worst = 0;
            for (int a = 0; a < 8; ++a) {
                for (int b = a + 1; b < 8; ++b) {
                    const int fab = flow_of(a, b), fba = flow_of(b, a);
                    const double x = static_cast<int32_t>(arr.at({b, fab})[w] - own_t[a][w]) / 1350.0;
                    const double y = static_cast<int32_t>(arr.at({a, fba})[w] - own_t[b][w]) / 1350.0;
                    worst = std::max(worst, (x + y) / 2);
                    sum += (x + y) / 2;
                    ++cnt;
                }
            }
            sum_max += worst;
        }
        res.meeting_us[p] = sum_max / kWindow;
        res.pair_mean_us[p] = sum / cnt;
    }
    return res;
}

TEST(Qwen38DataflowMeeting5, GroupCollectives) {
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
    const char* out_path = std::getenv("QWEN38_MEETING5_OUT");
    std::ofstream out(out_path ? out_path : "/tmp/qwen38_meeting5.txt");
    const int group = std::getenv("QWEN38_GROUP") ? std::atoi(std::getenv("QWEN38_GROUP")) : 0;
    const uint32_t iters = 256;
    uint32_t run_id = 0x5e500000;
    bool all = true;
    // AG of x (1,280 B), RS_y slices (10,240 B / 8 = 1,280 B per owner), AG of 10,400 B (AR_down as AG + local)
    const std::vector<std::tuple<const char*, Kind, uint32_t>> cases{
        {"AG_x 1280 B", Kind::AG, 1280u}, {"RS_y 10240 B (1280 B slices)", Kind::RS, 1280u},
        {"AR_down 10400 B (AG + local)", Kind::AG, 10400u}};
    for (const auto& [name, kind, bytes] : cases) {
        const Result check = run(by_id, t, group, kind, bytes, iters, ++run_id, true);
        const Result r = run(by_id, t, group, kind, bytes, iters, ++run_id, false);
        for (auto& line : check.stalls) {
            out << "CHECK RUN " << line << "\n";
        }
        for (auto& line : r.stalls) {
            out << "TIMED RUN " << line << "\n";
        }
        const bool exact = check.exact && r.exact;
        all = all && exact;
        out << "group " << group << " " << name << ": period " << r.period_us << " us, meeting (farthest pair, offset-free, 64 iterations) plane 0 "
            << r.meeting_us[0] << " / plane 1 " << r.meeting_us[1] << " us (pair mean " << r.pair_mean_us[0] << " / "
            << r.pair_mean_us[1] << "), exact " << exact << " (bad: full check " << check.bad << ", timed " << r.bad
            << ")\n";
        out.flush();
    }
    for (auto& [id, m] : meshes) {
        m->close();
    }
    EXPECT_TRUE(all);
}

}  // namespace
}  // namespace qwen38_dataflow
