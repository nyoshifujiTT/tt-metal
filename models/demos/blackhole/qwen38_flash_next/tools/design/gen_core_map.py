#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""gen_core_map.py - the roles of one chip's 130 Tensix cores and 8 used Ethernet tiles, as constexpr C++ tables
(phase H-1 of the integrated plan, worklog 10-27; the design is README section 12, revision 8).

Every hop bound the stage model uses (design/stages.py, design/comm_model.py) must hold on the real placement, for each
of the 14 harvest cases (one of the 14 Tensix columns x 1-7, 10-16 is fused off; NoC0 runs right/down and NoC1 left/up
on the 17 x 12 torus: tt-isa-documentation NoC/RoutingPaths.md, Coordinates.md):
  hub:        one core per chip, within NOC_HUB_ETH hops (both directions) of an Ethernet tile; its multicast to all
              130 cores as two rectangles (one on each NoC) reaches every core within MC_HOPS hops
  owners:     the cores sorted by their hop distance to the hub (the better NoC); the first 40 within UP_OWN[40] hops,
              the first 80 within UP_OWN[80], all within UP_HOPS
  near_eth:   the cores within NOC_ETH_CORE hops of an Ethernet tile in both directions (owners of additive meetings);
              at least NEAR_ETH_MIN of them
  relay:      for each used Ethernet tile, the tile on the same side of the chip (x <= 4 or x >= 13) it hands relayed
              packets to, within NOC_ETH_ETH hops
The generator searches the placement, checks every bound and writes kernels' header core_map.h with the tables (as
physical NoC0 coordinates) and static_asserts that re-check the bounds from the tables at compile time.

usage: gen_core_map.py <out core_map.h>
"""
import sys

XS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
YS = list(range(2, 12))
ETH = [(1, 1), (16, 1), (2, 1), (15, 1), (3, 1), (14, 1), (4, 1), (13, 1)]   # channels 0..7 (blackhole_140_arch.yaml eth order)
NOC_HUB_ETH, MC_HOPS, UP_OWN, UP_HOPS, NOC_ETH_CORE, NEAR_ETH_MIN, NOC_ETH_ETH = 2, 17, {40: 7, 80: 11}, 14, 6, 91, 3


def d0(a, b):
    return (b[0] - a[0]) % 17 + (b[1] - a[1]) % 12


def d1(a, b):
    return (a[0] - b[0]) % 17 + (a[1] - b[1]) % 12


def du(a, b):
    return min(d0(a, b), d1(a, b))


def mc_rects(src, cores):
    """the best split of the cores into two rectangles (by a column or a row cut), one multicast on each NoC:
    returns (worst hops, (noc of rectangle A, xa0, xa1, ya0, ya1), (noc of B, ...)) with NoC0 coordinates"""
    xs = sorted({c[0] for c in cores})
    best = None

    def cand(A, B, fa, fb, na, nb):
        nonlocal best
        h = max(max(fa(src, c) for c in A), max(fb(src, c) for c in B))
        ra = (na, min(c[0] for c in A), max(c[0] for c in A), min(c[1] for c in A), max(c[1] for c in A))
        rb = (nb, min(c[0] for c in B), max(c[0] for c in B), min(c[1] for c in B), max(c[1] for c in B))
        if best is None or h < best[0]:
            best = (h, ra, rb)

    for i in range(1, len(xs)):
        A = [c for c in cores if c[0] in xs[:i]]
        B = [c for c in cores if c[0] in xs[i:]]
        cand(A, B, d0, d1, 0, 1)
        cand(A, B, d1, d0, 1, 0)
    for j in range(1, len(YS)):
        A = [c for c in cores if c[1] in YS[:j]]
        B = [c for c in cores if c[1] in YS[j:]]
        cand(A, B, d0, d1, 0, 1)
        cand(A, B, d1, d0, 1, 0)
    return best


def place(harv):
    cores = [(x, y) for x in XS if x != harv for y in YS]
    assert len(cores) == 130
    best = None
    for hub in cores:
        he = min(max(du(hub, e), du(e, hub)) for e in ETH)
        if he > NOC_HUB_ETH:
            continue
        mc = mc_rects(hub, cores)
        order = sorted((c for c in cores if c != hub), key=lambda c: (du(c, hub), c))
        ups = [du(c, hub) for c in order]
        key = (mc[0], ups[39], ups[79], ups[-1], hub)
        if best is None or key < best[0]:
            best = (key, hub, mc, order)
    key, hub, mc, order = best
    near = sorted(c for c in cores if min(max(du(c, e), du(e, c)) for e in ETH) <= NOC_ETH_CORE)
    relay = []
    for i, e in enumerate(ETH):
        side = [j for j, f in enumerate(ETH) if j != i and (f[0] <= 4) == (e[0] <= 4)]
        j = min(side, key=lambda j: (du(e, ETH[j]), j))
        relay.append(j)
    ok = dict(
        hub_eth=min(max(du(hub, e), du(e, hub)) for e in ETH) <= NOC_HUB_ETH,
        mc=mc[0] <= MC_HOPS,
        up40=du(order[39], hub) <= UP_OWN[40],
        up80=du(order[79], hub) <= UP_OWN[80],
        up_all=du(order[-1], hub) <= UP_HOPS,
        near=len(near) >= NEAR_ETH_MIN,
        relay=all(du(ETH[i], ETH[relay[i]]) <= NOC_ETH_ETH for i in range(8)),
    )
    return dict(harv=harv, cores=cores, hub=hub, mc=mc, order=order, near=near, relay=relay, ok=ok)


def main():
    out = sys.argv[1]
    P = [place(h) for h in XS]
    bad = [(p["harv"], k) for p in P for k, v in p["ok"].items() if not v]
    if bad:
        raise SystemExit(f"a bound of the stage model does not hold on the placement: {bad}")
    L = []
    w = L.append
    w("// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.")
    w("//")
    w("// SPDX-License-Identifier: Apache-2.0")
    w("//")
    w("// core_map.h - GENERATED by tools/gen_core_map.py; do not edit.  The roles of one chip's 130 Tensix cores and 8 used Ethernet")
    w("// tiles for each of the 14 harvest cases (README section 12, revision 8; worklog 10-27 phase H-1).  Coordinates are physical NoC0")
    w("// (x, y); hop counts are on the better of NoC0 (right/down) and NoC1 (left/up) of the 17 x 12 torus.")
    w("#pragma once")
    w("#include <cstdint>")
    w("")
    w("namespace qwen38_core {")
    w("")
    w("struct Xy { uint8_t x, y; };")
    w("struct Rect { uint8_t noc, x0, x1, y0, y1; };")
    w("inline constexpr uint32_t kCores = 130, kHarvestCases = 14, kEthTiles = 8;")
    w("inline constexpr uint8_t kTensixX[kHarvestCases] = {%s};   // the harvested column of each case" % ", ".join(str(h) for h in XS))
    w("inline constexpr Xy kEth[kEthTiles] = {%s};" % ", ".join("{%d, %d}" % e for e in ETH))
    w("// the bounds the stage model uses (design/stages.py, design/comm_model.py)")
    w("inline constexpr uint32_t kNocHubEth = %d, kMcHops = %d, kUpOwn40 = %d, kUpOwn80 = %d, kUpHops = %d, kNocEthCore = %d, kNearEthMin = %d, kNocEthEth = %d;"
      % (NOC_HUB_ETH, MC_HOPS, UP_OWN[40], UP_OWN[80], UP_HOPS, NOC_ETH_CORE, NEAR_ETH_MIN, NOC_ETH_ETH))
    w("")
    w("constexpr uint32_t d0(Xy a, Xy b) { return (uint32_t(b.x) + 17 - a.x) % 17 + (uint32_t(b.y) + 12 - a.y) % 12; }")
    w("constexpr uint32_t d1(Xy a, Xy b) { return (uint32_t(a.x) + 17 - b.x) % 17 + (uint32_t(a.y) + 12 - b.y) % 12; }")
    w("constexpr uint32_t du(Xy a, Xy b) { return d0(a, b) < d1(a, b) ? d0(a, b) : d1(a, b); }")
    w("")
    w("inline constexpr Xy kHub[kHarvestCases] = {%s};" % ", ".join("{%d, %d}" % p["hub"] for p in P))
    w("inline constexpr Rect kHubMc[kHarvestCases][2] = {")
    for p in P:
        w("    {{%d, %d, %d, %d, %d}, {%d, %d, %d, %d, %d}},   // harvest x %d" % (*p["mc"][1], *p["mc"][2], p["harv"]))
    w("};")
    w("// the 129 non-hub cores of each case in order of their distance to the hub (owners take them from the front)")
    w("inline constexpr Xy kOwnerOrder[kHarvestCases][kCores - 1] = {")
    for p in P:
        w("    {%s},   // harvest x %d" % (", ".join("{%d, %d}" % c for c in p["order"]), p["harv"]))
    w("};")
    w("inline constexpr uint32_t kNearEthCount[kHarvestCases] = {%s};" % ", ".join(str(len(p["near"])) for p in P))
    w("inline constexpr uint8_t kRelayTo[kEthTiles] = {%s};   // the same-side tile each Ethernet tile hands relayed packets to" % ", ".join(str(j) for j in P[0]["relay"]))
    w("")
    w("// compile-time re-checks of the bounds from the tables")
    w("constexpr bool check_all() {")
    w("    for (uint32_t h = 0; h < kHarvestCases; ++h) {")
    w("        uint32_t he = 99;")
    w("        for (uint32_t e = 0; e < kEthTiles; ++e) {")
    w("            const uint32_t a = du(kHub[h], kEth[e]), b = du(kEth[e], kHub[h]);")
    w("            const uint32_t m = a > b ? a : b;")
    w("            he = m < he ? m : he;")
    w("        }")
    w("        if (he > kNocHubEth) return false;")
    w("        if (du(kOwnerOrder[h][39], kHub[h]) > kUpOwn40 || du(kOwnerOrder[h][79], kHub[h]) > kUpOwn80 || du(kOwnerOrder[h][kCores - 2], kHub[h]) > kUpHops) return false;")
    w("        if (kNearEthCount[h] < kNearEthMin) return false;")
    w("        for (uint32_t r = 0; r < 2; ++r) {")
    w("            const Rect q = kHubMc[h][r];")
    w("            for (uint32_t i = 0; i < kCores - 1; ++i) {")
    w("                const Xy c = kOwnerOrder[h][i];")
    w("                if (c.x >= q.x0 && c.x <= q.x1 && c.y >= q.y0 && c.y <= q.y1 && (q.noc == 0 ? d0(kHub[h], c) : d1(kHub[h], c)) > kMcHops) return false;")
    w("            }")
    w("        }")
    w("    }")
    w("    for (uint32_t e = 0; e < kEthTiles; ++e) if (du(kEth[e], kEth[kRelayTo[e]]) > kNocEthEth) return false;")
    w("    return true;")
    w("}")
    w("static_assert(check_all(), \"a hop bound of the stage model does not hold on the core map\");")
    w("")
    w("}  // namespace qwen38_core")
    open(out, "w").write("\n".join(L) + "\n")
    for p in P:
        print(f"harvest x {p['harv']:2d}: hub {p['hub']}, multicast {p['mc'][0]} hops, up 40/80/all {du(p['order'][39], p['hub'])}/{du(p['order'][79], p['hub'])}/{du(p['order'][-1], p['hub'])}, near-Ethernet cores {len(p['near'])}")


if __name__ == "__main__":
    main()
