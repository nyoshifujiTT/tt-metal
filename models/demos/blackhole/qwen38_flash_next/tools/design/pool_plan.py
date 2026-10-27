"""Realizable core assignments of a matmul pool and their exact time (corrections 18, 30-32; worklog 10-27).
A pool = matmuls reading the same input.  Its output columns (32 columns of one 32-row tile; the members of the same
weight format, K and width are packed column after column, each member rounded up to 8 columns (one SFPU load group of
its epilogue), the last column of a member group may be a 16-column half tile: MVMUL works on SrcA 16 x 16 faces,
MVMUL.md, and the unpacker's tile descriptor takes the face count, UNPACR_Regular.md) each need their K units.
Schemes (all evaluated, the fastest kept):
  A  per-column core counts q_c (greedy on the slowest column), a column's cores consecutive (owner last)
  B  linear Stream-K: the (column, K unit) sequence cut into P equal-cycle core ranges at unit boundaries; a column
     split over cores is owned by the core whose range ends in it (its first core); a core whose range continues is a leaf
  C  whole columns, longest first onto the least loaded core (no K split)
A column's partials go up a heap tree (owner, free senders by finish, busy senders last as leaves; fan-in m chosen) and
every edge's flight takes its hop count from hopf(child, parent) (the stage model: 2; a placement: the real hops); the
owner's column goes to the hub (the model: up_hops(columns); a placement: the owner's real hops); the hub multicasts.
plan() returns the time, the scheme, the fan-in and the per-core items (core, column, first unit, units, role, parent)."""
import heapq
import math

import stages as S

TB = {"bf8": 1088, "bf4": 576, "bf16": 2048}


def unit_mac(fmt, rows, half):
    phases = {"bf8": 2, "bf4": 1, "bf16": 4}[fmt]
    unp = {"bf8": 8, "bf4": 8, "bf16": 16}[fmt]
    if half:
        return max(math.ceil(rows / 8) * 2 * phases, unp // 2)
    return S.mac(fmt, rows)


def norm_mm(mm):
    """(M, N, K, fmt[, width[, epi]]) or (M, N, ((K1, fmt1), ...), "cat"[, width[, epi]]) -> (M, N, runs, width, epi)"""
    M, N, K, fmt = mm[:4]
    width = mm[4] if len(mm) > 4 else 1
    epi = mm[5] if len(mm) > 5 else 0
    runs = tuple((math.ceil(k / 32), f) for k, f in K) if fmt == "cat" else ((math.ceil(K / 32), fmt),)
    return M, N, runs, width, epi


def columns(mms):
    """the pool's output columns: dict(units = cycles per K unit, bytes = stored bytes per K unit, width, epi, half)"""
    groups = {}
    order = []
    for mm in mms:
        M, N, runs, width, epi = norm_mm(mm)
        key = (M, runs, width)
        if key not in groups:
            groups[key] = [0, 0]
            order.append(key)
        groups[key][0] += math.ceil(N / 8) * 8
        groups[key][1] = max(groups[key][1], epi)
    cols = []
    for key in order:
        M, runs, width = key
        ncol, epi = groups[key]
        r = min(M, 32)
        full, rem = divmod(ncol, 32)
        kinds = [False] * full + ([rem <= 16] if rem else [])
        for mt in range(math.ceil(M / 32)):
            for half in kinds:
                units = tuple(width * unit_mac(f, r, half) for kt, f in runs for _ in range(kt))
                byts = tuple(width * TB[f] // (2 if half else 1) for kt, f in runs for _ in range(kt))
                cols.append(dict(units=units, bytes=byts, width=width, epi=epi, half=half, rows=r))
    return cols


def _send(rows, h):
    return S.finalize_partial(rows) + S.pack(rows * 2) + S.issue() + S.N_RD + S.flight(h, rows * 32 * 4)


def _level(rows, m):
    part = rows * 32 * 4
    return S.poll() + m * (S.UNP_ADDR + math.ceil(part / S.UNP_BYTES)) + S.P_UNP + S.add_partials(rows, m)


def _ownend(rows):
    return S.finalize_partial(rows) + S.pack(rows * 2) + S.issue() + S.N_RD


def col_tree(rows, owner, free, busy, epi, hopf):
    """owner = (core, finish); free / busy = [(core, finish)]: the best heap tree; returns (t, m, {core: parent})"""
    if not free and not busy:
        return owner[1] + _ownend(rows) + epi, 0, {owner[0]: -1}
    nodes = [owner] + sorted(free, key=lambda x: (x[1], x[0])) + sorted(busy, key=lambda x: (x[1], x[0]))
    n = len(nodes)
    nb = len(busy)
    best = None
    for m in range(1, n):
        if nb and any(m * i + 1 < n for i in range(n - nb, n)):
            continue
        ready = [0.0] * n
        for i in range(n - 1, -1, -1):
            kids = range(m * i + 1, min(m * i + m + 1, n))
            t = nodes[i][1]
            if len(kids):
                t = max([t] + [ready[j] for j in kids]) + _level(rows, len(kids))
            if i:
                ready[i] = t + _send(rows, hopf(nodes[i][0], nodes[(i - 1) // m][0]))
            else:
                ready[i] = t + _ownend(rows) + epi
        if best is None or ready[0] < best[0]:
            best = (ready[0], m)
    t, m = best
    par = {nodes[0][0]: -1}
    for i in range(1, n):
        par[nodes[i][0]] = nodes[(i - 1) // m][0]
    return t, m, par


def _split(units, q):
    tot = sum(units)
    out = []
    a = 0
    acc = 0.0
    k = 1
    part = 0.0
    n = 0
    for u in units:
        if k < q and acc + u / 2 > k * tot / q and n > 0:
            out.append((a, n, part))
            a += n
            n = 0
            part = 0.0
            k += 1
        part += u
        acc += u
        n += 1
    out.append((a, n, part))
    return out


def scheme_A(cols, rows, P, hopf):
    out = len(cols)
    if out > P:
        return None
    model = lambda a, b: 2

    def t_of(c, q):
        sp = _split(c["units"], q)
        fin = [p + S.MVMUL_LAT for _, _, p in sp]
        return col_tree(rows, (q - 1, fin[-1]), [(j, fin[j]) for j in range(q - 1)], [], c["epi"], model)[0]

    q = [1] * out
    times = [t_of(c, 1) for c in cols]
    spare = P - out
    while spare > 0:
        j = max(range(out), key=lambda x: (times[x], -x))
        if q[j] >= len(cols[j]["units"]):
            break
        nt = t_of(cols[j], q[j] + 1)
        if nt >= times[j]:
            break
        q[j] += 1
        times[j] = nt
        spare -= 1
    items = []
    worst = 0.0
    fan = 0
    owners = []
    core = 0
    for ci, c in enumerate(cols):
        sp = _split(c["units"], q[ci])
        cores = list(range(core, core + len(sp)))
        fin = [p + S.MVMUL_LAT for _, _, p in sp]
        t, m, par = col_tree(rows, (cores[-1], fin[-1]), [(cores[j], fin[j]) for j in range(len(sp) - 1)], [], c["epi"], hopf)
        worst = max(worst, t)
        fan = max(fan, m)
        owners.append(cores[-1])
        for j, (a, n, _) in enumerate(sp):
            items.append((cores[j], ci, a, n, 0 if j == len(sp) - 1 else 1, par[cores[j]]))
        core += len(sp)
    return worst, fan, items, owners


def scheme_B(cols, rows, P, hopf):
    runs = []
    for ci, c in enumerate(cols):
        u = c["units"]
        k = 0
        while k < len(u):
            m = u[k]
            n = 1
            while k + n < len(u) and u[k + n] == m:
                n += 1
            runs.append((ci, k, m, n))
            k += n
    total = sum(m * n for _, _, m, n in runs)
    fin = {}
    first = {}
    last = {}
    core = 0
    cum = 0.0
    t_core = 0.0
    prev = None
    for ci, k0, m, n in runs:
        left = n
        kk = k0
        while left > 0:
            if core < P - 1:
                room = (core + 1) * total / P - cum - m / 2
                take = max(0, min(left, math.floor(room / m) + 1)) if room >= 0 else 0
            else:
                take = left
            if take == 0:
                core += 1
                t_core = 0.0
                prev = None
                continue
            if prev is not None and prev != ci:
                t_core += S.finalize_partial(rows)
            t_core += take * m
            cum += take * m
            fin.setdefault(ci, {})[core] = t_core + S.MVMUL_LAT
            first.setdefault((ci, core), kk)
            last[core] = ci
            prev = ci
            kk += take
            left -= take
    cnt = {}
    for (ci, c), k in first.items():
        pass
    units_of = {}
    for ci, d in fin.items():
        for c in d:
            units_of[(ci, c)] = 0
    for ci, k0, m, n in runs:
        pass
    # unit counts per (column, core) from the first-unit positions
    for ci, d in fin.items():
        cs = sorted(d)
        starts = [first[(ci, c)] for c in cs] + [len(cols[ci]["units"])]
        for j, c in enumerate(cs):
            units_of[(ci, c)] = starts[j + 1] - starts[j]
    items = []
    worst = 0.0
    fan = 0
    owners = []
    for ci, d in fin.items():
        cs = sorted(d)
        owner = cs[0]
        free = [(c, d[c]) for c in cs[1:] if last[c] == ci]
        busy = [(c, d[c]) for c in cs[1:] if last[c] != ci]
        if len(cs) > 1:
            assert last[owner] == ci
        t, m, par = col_tree(rows, (owner, d[owner]), free, busy, cols[ci]["epi"], hopf)
        worst = max(worst, t)
        fan = max(fan, m)
        owners.append(owner)
        for c in cs:
            items.append((c, ci, first[(ci, c)], units_of[(ci, c)], 0 if c == owner else 1, par[c]))
    return worst, fan, sorted(items), owners


def scheme_C(cols, rows, P, hopf):
    heap = [(0.0, k) for k in range(P)]
    worst = 0.0
    fz = S.finalize_partial(rows)
    items = []
    owners = []
    for ci in sorted(range(len(cols)), key=lambda ci: (-sum(cols[ci]["units"]), ci)):
        c = cols[ci]
        l, k = heapq.heappop(heap)
        l += sum(c["units"]) + fz + c["epi"]
        worst = max(worst, l + S.MVMUL_LAT + _ownend(rows) - fz)
        items.append((k, ci, 0, len(c["units"]), 0, -1))
        owners.append(k)
        heapq.heappush(heap, (l, k))
    return worst, 0, sorted(items), owners


def plan(mms, P=130, hopf=None, uphop=None):
    """the fastest scheme on a placement (hopf(child, parent), uphop(owner) over logical cores; None = the stage model:
    2 hops a tree edge, up_hops(columns) to the hub).  Returns dict(t, scheme, fan, items, cols)."""
    cols = columns(mms)
    rows = max(c["rows"] for c in cols)
    hf = hopf or (lambda a, b: 2)
    best = None
    for name, fn in (("A", scheme_A), ("B", scheme_B), ("C", scheme_C)):
        r = fn(cols, rows, P, hf)
        if r is None:
            continue
        t, fan, items, owners = r
        up = S.up_hops(min(len(cols), P)) if uphop is None else max(uphop(o) for o in owners)
        tail = S.flight(up, rows * 64) + S.poll() + S.issue() + S.N_RD + S.flight(S.MC_HOPS, 8 * rows * 64) + S.poll()
        tot = t + tail
        if best is None or tot < best["t"]:
            best = dict(t=tot, scheme=name, fan=fan, items=items, cols=cols, up=up)
    return best


def pool_plan(mms, P=130):
    """compatibility: (time, scheme, {}, fan) of the stage model"""
    b = plan(mms, P)
    return b["t"], b["scheme"], {}, b["fan"]


def assignment(mms, P=130):
    b = plan(mms, P)
    return b["scheme"], b["items"]


def core_bytes(mms, items=None, cols=None):
    """stored weight bytes per logical core"""
    if items is None:
        b = plan(mms)
        items, cols = b["items"], b["cols"]
    cols = cols or columns(mms)
    per = {}
    for c, ci, u, n, r, par in items:
        per[c] = per.get(c, 0) + sum(cols[ci]["bytes"][u:u + n])
    return per


def stored_bytes(mms):
    return sum(sum(c["bytes"]) for c in columns(mms))
