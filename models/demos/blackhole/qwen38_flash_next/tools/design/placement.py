"""Corrections 30-34 (worklog 10-27 19:xx): the physical placement of every matmul pool of the chosen form, per harvest
case, and the pool times on it.
- The cores of a chip are the raster path P (raster.py: columns x ascending, y 2 -> 11); a pool's logical core k sits at
  P[(k + o) % 130]; a later logical core is right / below an earlier one, so a child -> parent partial (later ->
  earlier in a column's run of cores) takes NoC1.
- A pool's time on (harvest, o) = pool_plan.plan with every tree edge's real hop count and the owners' real hop count
  to the harvest's hub (core_map.h); the scheme (A / B / C) is re-chosen on the placement.
- Per frame (step 3) and harvest, the pools of the frame's units are placed in unit order: o is the rotation with the
  least time among those whose per-core stored bytes (the frame so far + this pool) fit the frame region, ties by the
  smaller per-core maximum.  Shared weights (the speculative router, mtp.lm_head) are not stored twice.
- The design value of a pool is its worst time over the 14 harvest cases (the chips of a group meet each pool's output
  in a meeting or the next unit: the slowest chip sets the pace)."""
import functools
import json

import pool_plan as PP
import raster as R
import stages as SG

XS = R.XS


@functools.lru_cache(maxsize=None)
def hub_of(harv):
    import gen_core_map_shim as CM
    return CM.hub_of(harv)


def tup(x):
    return tuple(tup(y) for y in x) if isinstance(x, list) else x


def pool_bytes(mms):
    import pool_plan as PP
    return mms[1] if is_state(mms) else PP.stored_bytes(mms)


def is_state(mms):
    """correction 54: a pseudo-pool ("S", bytes, units) for a GDN layer's state S on the cores that run its GDR units:
    no matmul time (the GDR unit's time does not depend on the placement), units logical cores of bytes / units each"""
    return len(mms) == 3 and mms[0] == "S"


@functools.lru_cache(maxsize=None)
def timed(mms, harv, o):
    if is_state(mms):
        _, nb, nu = mms
        return 0.0, "S", 0, tuple(sorted(((k + o) % 130, nb / nu) for k in range(int(nu)))), 0
    P = R.path(harv)
    H = R.hop_table(harv)
    hub = hub_of(harv)
    b = PP.plan(mms, 130, hopf=lambda a, c: H[(a + o) % 130][(c + o) % 130],
                uphop=lambda c: R.du(P[(c + o) % 130], hub))
    per = PP.core_bytes(mms, b["items"], b["cols"])
    phys = {}
    for c, by in per.items():
        q = (c + o) % 130
        phys[q] = phys.get(q, 0) + by
    return b["t"], b["scheme"], b["fan"], tuple(sorted(phys.items())), b["up"]


def place_frame(pools, harv, region, base):
    """pools: [(unit index, mms)]; base: per-core bytes already there (rows / work / S share); returns (per-core load,
    [(unit, mms, o, t, scheme, fan)], fits)"""
    load = [base] * 130
    out = []
    fits = True
    for i, mms in pools:
        best = None
        for o in range(130):
            t, sch, fan, phys, up = timed(mms, harv, o)
            mx = max(load[q] + b for q, b in phys)
            ok = mx <= region
            key = (0 if ok else 1, t if ok else mx, mx, o)
            if best is None or key < best[0]:
                best = (key, o, t, sch, fan, phys, ok)
        key, o, t, sch, fan, phys, ok = best
        fits = fits and ok
        for q, b in phys:
            load[q] += b
        out.append((i, mms, o, t, sch, fan))
    return load, out, fits


def frame_pools(T4, U, a, z):
    pools = []
    for i in range(a, z):
        u = U[i]["unit"]
        if "speculative router" in u or u.startswith("mtp.lm_head"):
            continue
        for p in T4[i].get("pools", []):
            pools.append((i, tup(p)))
    return pools


def place_frame_opt(pools, harv, region, base):
    """corrections 30-34: the frame's pools placed to minimize the sum of their times subject to every core's stored bytes
    <= region: (1) every pool at its fastest rotation; (2) while a core overflows, move the pool (among those storing on
    an overflowing core) whose move to another rotation reduces the overflow at the least time increase per byte of
    overflow removed.  Returns (load, [(unit, mms, o, t, scheme, fan)], fits)."""
    n = len(pools)
    cand = []
    for i, mms in pools:
        rs = []
        for o in range(130):
            t, sch, fan, phys, up = timed(mms, harv, o)
            rs.append((t, o, sch, fan, dict(phys)))
        cand.append(rs)
    choice = [min(range(130), key=lambda o: (cand[k][o][0], o)) for k in range(n)]

    def loads():
        L = [base] * 130
        for k in range(n):
            for q, b in cand[k][choice[k]][4].items():
                L[q] += b
        return L

    L = loads()
    for _ in range(20 * n):
        over = [max(0.0, L[q] - region) for q in range(130)]
        tot = sum(over)
        if tot <= 0:
            break
        best = None
        for k in range(n):
            cur = cand[k][choice[k]]
            if not any(over[q] > 0 for q in cur[4]):
                continue
            for o in range(130):
                if o == choice[k]:
                    continue
                new = cand[k][o]
                L2 = list(L)
                for q, b in cur[4].items():
                    L2[q] -= b
                for q, b in new[4].items():
                    L2[q] += b
                tot2 = sum(max(0.0, x - region) for x in L2)
                gain = tot - tot2
                if gain <= 0:
                    continue
                cost = new[0] - cur[0]
                key = (cost / gain, cost, k, o)
                if best is None or key < best[0]:
                    best = (key, k, o, L2)
        if best is None:
            break
        _, k, o, L = best
        choice[k] = o
    fits = max(L) <= region
    out = [(pools[k][0], pools[k][1], choice[k], cand[k][choice[k]][0], cand[k][choice[k]][2], cand[k][choice[k]][3]) for k in range(n)]
    return L, out, fits


def place_frame_greedy(pools, harv, region, base):
    """the frame's pools in decreasing stored bytes: each at its fastest rotation among those that keep every core within
    the region (ties: the smaller per-core maximum); if none fits, the rotation with the least maximum"""
    import pool_plan as PP
    order = sorted(range(len(pools)), key=lambda k: (-pool_bytes(pools[k][1]), k))
    load = [base] * 130
    res = {}
    fits = True
    for k in order:
        i, mms = pools[k]
        best = None
        for o in range(130):
            t, sch, fan, phys, up = timed(mms, harv, o)
            mx = max(load[q] + b for q, b in phys)
            key = (0, t, mx, o) if mx <= region else (1, mx, t, o)
            if best is None or key < best[0]:
                best = (key, o, t, sch, fan, phys, up)
        key, o, t, sch, fan, phys, up = best
        fits = fits and key[0] == 0
        for q, b in phys:
            load[q] += b
        res[k] = (i, mms, o, t, sch, fan, up)
    return load, [res[k] for k in range(len(pools))], fits


def place_frame_fit(pools, harv, region, base, max_rounds=40):
    """corrections 30-35: place_frame_opt, then while a core still overflows, a balancing pass: the pools in decreasing
    stored bytes are re-placed one at a time at the rotation with the least per-core maximum of (everything else + the
    pool) -- i.e. a min-max bin balancing with the time as the tie-break -- and the pass repeats until the maximum fits
    or stops decreasing.  Then the time is recovered greedily: each pool moves to its fastest rotation that keeps every
    core within the region.  Returns (load, out, fits)."""
    import pool_plan as PP
    L, out, fits = place_frame_opt(pools, harv, region, base)
    if fits:
        return L, out, fits
    n = len(pools)
    choice = {k: out[k][2] for k in range(n)}
    phys = {}

    def ph(k, o):
        key = (k, o)
        if key not in phys:
            phys[key] = dict(timed(pools[k][1], harv, o)[3])
        return phys[key]

    def loads():
        L = [base] * 130
        for k in range(n):
            for q, b in ph(k, choice[k]).items():
                L[q] += b
        return L

    order = sorted(range(n), key=lambda k: (-pool_bytes(pools[k][1]), k))
    L = loads()
    prev = max(L)
    for _ in range(max_rounds):
        for k in order:
            for q, b in ph(k, choice[k]).items():
                L[q] -= b
            best = None
            for o in range(130):
                p = ph(k, o)
                mx = max(L[q] + b for q, b in p.items())
                t = timed(pools[k][1], harv, o)[0]
                key = (max(mx, region), t, mx, o)
                if best is None or key < best[0]:
                    best = (key, o)
            choice[k] = best[1]
            for q, b in ph(k, choice[k]).items():
                L[q] += b
        if max(L) <= region or max(L) >= prev - 1:
            break
        prev = max(L)
    # recover time
    for k in sorted(range(n), key=lambda k: -max(timed(pools[k][1], harv, choice[k])[0] - min(timed(pools[k][1], harv, o)[0] for o in range(130)), 0)):
        for q, b in ph(k, choice[k]).items():
            L[q] -= b
        best = None
        for o in range(130):
            p = ph(k, o)
            mx = max(L[q] + b for q, b in p.items())
            if mx > max(region, max(L)):
                continue
            t = timed(pools[k][1], harv, o)[0]
            if best is None or (t, mx, o) < best[0]:
                best = ((t, mx, o), o)
        if best is not None:
            choice[k] = best[1]
        for q, b in ph(k, choice[k]).items():
            L[q] += b
    fits = max(L) <= region
    out = []
    for k in range(n):
        t, sch, fan, p, up = timed(pools[k][1], harv, choice[k])
        out.append((pools[k][0], pools[k][1], choice[k], t, sch, fan))
    return L, out, fits
