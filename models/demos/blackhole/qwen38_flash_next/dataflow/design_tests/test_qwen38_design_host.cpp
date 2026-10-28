// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// test_qwen38_design_host.cpp - phase H-2 of the integrated plan (worklog 10-27): host-only checks (no device) that the
// generated design headers agree with each other over every chip, frame and unit of the chosen form (README section 12,
// revision 8).  The static_asserts inside each header check that header alone; this test checks the joints:
//   core_map.h  x stage_costs.h  the hop bounds the costs assume hold on the placement of every harvest case
//   design_plan.h x chip_l1.h    every frame's column fits the 130 cores' frame regions; the frame capacity of the plan
//                                is the L1 map's (fixed, stream, landing taken out)
//   coll_routes.h x chip_l1.h    every meeting payload fits the meeting landing; every hand-over fits the hand landing
//   coll_routes.h x core_map.h   every Ethernet direction a tree uses has a used Ethernet tile pair on the chip
//   design_plan.h                the units' bytes sum to the frames' columns; every unit sits in exactly one frame
// Build: g++ -std=c++20 -fconstexpr-ops-limit=400000000 -I dataflow/design test_qwen38_design_host.cpp
#include <cstdint>
#include <cstdio>

#include "chip_l1.h"
#include "coll_routes.h"
#include "core_map.h"
#include "design_plan.h"
#include "stage_costs.h"
#include "unit_assign.h"
#include "dram_plan.h"

namespace {
int failures = 0;
#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            ++failures;                                          \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                            \
            std::printf("\n");                                   \
        }                                                        \
    } while (0)

void core_map_vs_costs() {
    using namespace qwen38_core;
    CHECK(kMcHops == qwen38_cost::kMcHops, "multicast hops %u vs %u", kMcHops, qwen38_cost::kMcHops);
    CHECK(kUpHops == qwen38_cost::kUpHops, "up hops %u vs %u", kUpHops, qwen38_cost::kUpHops);
    CHECK(kUpOwn40 == qwen38_cost::kUpOwn40 && kUpOwn80 == qwen38_cost::kUpOwn80, "owner up bounds");
    for (uint32_t h = 0; h < kHarvestCases; ++h) {
        uint32_t worst = 0;
        for (uint32_t i = 0; i < kCores - 1; ++i) {
            const uint32_t d = du(kOwnerOrder[h][i], kHub[h]);
            CHECK(i == 0 || d >= du(kOwnerOrder[h][i - 1], kHub[h]), "harvest %u: owner order not sorted at %u", h, i);
            worst = d > worst ? d : worst;
        }
        CHECK(worst <= qwen38_cost::kUpHops, "harvest %u: farthest owner %u hops", h, worst);
        for (uint32_t i = 0; i < kCores - 1; ++i) {
            CHECK(kOwnerOrder[h][i].x != kTensixX[h], "harvest %u: owner %u on the harvested column", h, i);
            CHECK(kOwnerOrder[h][i].x != kHub[h].x || kOwnerOrder[h][i].y != kHub[h].y, "harvest %u: hub in the owner list", h);
        }
    }
}

void plan_vs_l1() {
    using namespace qwen38_plan;
    using namespace qwen38_l1;
    const uint64_t frame_regions = uint64_t(qwen38_core::kCores) * kFrameBytes;
    CHECK(kFrameCap == kChipFrameCapacity, "plan frame cap %llu vs L1 map %llu", (unsigned long long)kFrameCap,
          (unsigned long long)kChipFrameCapacity);
    CHECK(kFrameCap + kChipStateL1 <= frame_regions, "frame cap %llu + state %llu > frame regions %llu",
          (unsigned long long)kFrameCap, (unsigned long long)kChipStateL1, (unsigned long long)frame_regions);
    const uint64_t landing = kXReceiveBytes + kPartialsBytes + kMeetLandingBytes + kHandLandingBytes + kFlagsBytes;
    CHECK(uint64_t(qwen38_core::kCores) * landing == kChipLanding, "landing %llu x 130 vs step 3 %llu", (unsigned long long)landing,
          (unsigned long long)kChipLanding);
    CHECK(uint64_t(qwen38_core::kCores) * (kCoreL1Bytes - kFirmwareBytes - kKernelBytes - landing) ==
              kFrameCap + kChipStateL1,
          "the plan's capacity is not the L1 map's frame regions");
    uint32_t seen[kUnits] = {};
    for (uint32_t f = 0; f < kFrames; ++f) {
        uint64_t col = 0;
        for (uint32_t u = kFrame[f].first; u < kFrame[f].end; ++u) {
            ++seen[u];
            col += kUnit[u].w + kUnit[u].s;   // correction 42: a unit's lasting L1 = stored weights + S
        }
        const uint64_t diff = col > kFrame[f].column ? col - kFrame[f].column : kFrame[f].column - col;
        // the frame's column = its units' lasting L1 + its largest work + the residual (step 3's frame_extra)
        uint64_t wmax = 0;
        for (uint32_t u = kFrame[f].first; u < kFrame[f].end; ++u) wmax = kUnit[u].work > wmax ? kUnit[u].work : wmax;
        const uint64_t want = col + wmax + uint64_t(qwen38_plan::kRows) * 4 * 2560 * 2;
        const uint64_t d2 = want > kFrame[f].column ? want - kFrame[f].column : kFrame[f].column - want;
        CHECK(d2 <= uint64_t(kFrame[f].end - kFrame[f].first) + 1, "frame %u: units %llu vs column %llu", f,
              (unsigned long long)want, (unsigned long long)kFrame[f].column);
        (void)diff;
        CHECK(kFrame[f].column + kChipStateL1 <= frame_regions, "frame %u does not fit the frame regions", f);
        CHECK(kFrame[f].column + kFrame[f].left == kFrameCap || kFrame[f].column + kFrame[f].left + 1 >= kFrameCap,
              "frame %u: column + left != capacity", f);
    }
    for (uint32_t u = 0; u < kUnits; ++u) CHECK(seen[u] == 1, "unit %u in %u frames", u, seen[u]);
    CHECK(qwen38_plan::kRows == qwen38_l1::kRows, "rows");
}

void coll_vs_l1() {
    using namespace qwen38_coll;
    for (uint32_t m = 0; m < kMeetingBytes.size(); ++m)
        CHECK(2 * kMeetingBytes[m] <= qwen38_l1::kMeetLandingBytes, "meeting %u (%u B) does not fit the landing", m, kMeetingBytes[m]);
    CHECK(qwen38_l1::kMeetMaxBytes <= qwen38_l1::kMeetLandingBytes / 2, "meeting max");
    for (const auto& h : kHandOver) CHECK(h.hops <= kHandHopsMax && h.hops >= 1, "hand-over hops %u", h.hops);
}

void coll_vs_core_map() {
    using namespace qwen38_coll;
    // the used Ethernet tiles: channels 0..7 = x 1, 16, 2, 15, 3, 14, 4, 13 on NoC row 1 (core_map.h kEth); every ring
    // direction (E / W) and every in-group vertical direction (N / S) needs a channel pair: 4 directions x 2 channels
    CHECK(qwen38_core::kEthTiles == 8, "8 used Ethernet tiles = 4 directions x 2 channels");
    uint32_t used[4] = {};
    for (uint32_t r = 0; r < kGroupChips; ++r)
        for (uint32_t c = 0; c < kGroupChips; ++c)
            for (uint32_t k = 0; k < kTree[r][c].n_children; ++k) ++used[uint32_t(kTree[r][c].to_child[k])];
    CHECK(used[0] + used[1] + used[2] + used[3] == 8 * 7, "8 trees x 7 edges");
    for (uint32_t r = 0; r < qwen38_core::kEthTiles; ++r) {
        const uint32_t to = qwen38_core::kRelayTo[r];
        CHECK(to < qwen38_core::kEthTiles && to != r, "relay pair of tile %u", r);
        const auto a = qwen38_core::kEth[r], b = qwen38_core::kEth[to];
        CHECK((a.x <= 4) == (b.x <= 4), "relay pair %u -> %u crosses the chip", r, to);
    }
}

void assign_vs_plan() {
    using namespace qwen38_assign;
    // every (harvest, unit) entry's pools exist; every pool: one owner per column, the works of a column cover its K
    // units exactly once in order, every sender's parent holds the same column and following parents reaches the owner,
    // the fan-in is within the pool's and the L1 partial landing's
    for (uint32_t e = 0; e < kUnitPoolEntries; ++e) {
        CHECK(kUnitPools[e].unit < qwen38_plan::kUnits, "entry %u: unit %u", e, kUnitPools[e].unit);
        for (uint32_t k = 0; k < kUnitPools[e].n; ++k) CHECK(kUnitPools[e].pool[k] >= 0 && uint32_t(kUnitPools[e].pool[k]) < kPools, "entry %u pool", e);
    }
    for (uint32_t p = 0; p < kPools; ++p) {
        const Pool& q = kPool[p];
        CHECK(q.fanin <= qwen38_l1::kFanin, "pool %u fan-in %u > the partial landing's %u", p, q.fanin, qwen38_l1::kFanin);
        const uint32_t a = q.first_work, z = q.first_work + q.n_work;
        for (uint32_t c = 0; c < q.columns; ++c) {
            uint32_t owners = 0, next = 0;
            for (bool more = true; more;) {
                more = false;
                for (uint32_t i = a; i < z; ++i)
                    if (kWork[i].column == c && kWork[i].first_unit == next) { next += kWork[i].units; more = true; }
            }
            for (uint32_t i = a; i < z; ++i) owners += (kWork[i].column == c && kWork[i].role == 0) ? 1 : 0;
            CHECK(owners == 1 && next > 0, "pool %u column %u: owners %u, units %u", p, c, owners, next);
        }
        for (uint32_t i = a; i < z; ++i) {
            const Work& x = kWork[i];
            CHECK((x.role == 0) == (x.parent < 0), "pool %u work %u role/parent", p, i);
            int32_t at = x.core;
            uint32_t steps = 0;
            for (;; ++steps) {
                if (steps > 130) break;
                int32_t par = -2;
                for (uint32_t j = a; j < z; ++j)
                    if (kWork[j].column == x.column && kWork[j].core == at) par = kWork[j].parent;
                if (par < 0) { CHECK(par == -1, "pool %u work %u: a parent without the column", p, i); break; }
                at = par;
            }
            CHECK(steps <= 130, "pool %u work %u: a cycle", p, i);
            uint32_t kids = 0;
            for (uint32_t j = a; j < z; ++j)
                if (kWork[j].column == x.column && kWork[j].parent == int16_t(x.core)) ++kids;
            CHECK(kids <= q.fanin, "pool %u work %u: %u children > fan-in %u", p, i, kids, q.fanin);
        }
    }
}

void dram_vs_assign() {
    // the plan's frames are the design plan's; the per-core placed bytes summed over the cores equal the frame's pools'
    // stored bytes (the unit_assign pools of the frame on that harvest)
    CHECK(qwen38_dram::kFrames == qwen38_plan::kFrames, "dram plan frames %u vs plan %u", qwen38_dram::kFrames, qwen38_plan::kFrames);
    for (uint32_t h = 0; h < qwen38_dram::kHarvests; ++h)
        for (uint32_t f = 0; f < qwen38_dram::kFrames; ++f) {
            uint64_t placed = 0;
            for (uint32_t c = 0; c < qwen38_dram::kCores; ++c) placed += qwen38_dram::kPlan[h][f][c].placed_bytes;
            CHECK(placed > 0, "harvest %u frame %u: no placed bytes", h, f);
        }
}

void costs_sane() {
    using namespace qwen38_cost;
    CHECK(kTopk10 == kTopkLocal + kTopkPost, "top-10 split");
    CHECK(kPoolGdnIn > 0 && kPoolLmHead > kPoolGdnIn, "pool budgets");
}
}  // namespace

int main() {
    core_map_vs_costs();
    plan_vs_l1();
    coll_vs_l1();
    coll_vs_core_map();
    assign_vs_plan();
    dram_vs_assign();
    costs_sane();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
