"""H-1 (b) (worklog 10-27 20:3x): the DRAM read plan of a frame on one harvest case.  The frame's per-core weight bytes
come from the placement (step 3p); every core reads its bytes from the 8 banks.  The weights are laid out by us, so a
core's bytes may come from any bank; the plan (I-1b', worklog 10-26 21:4x - 22:0x: the integer program with equal banks,
at most 2 flows a core, the per-link load bounded, NOC1 responses, then a common pace for the reads: 99.35%) chooses per
core at most 2 flows (bank, endpoint tile, NoC) and their bytes so that every bank delivers exactly 1/8 of the frame and
the largest link / endpoint / core load is at most U x the bank time; objective: the total link load (bytes x hops).
Physical rules (tt-isa-documentation NoC/RoutingPaths.md): NOC0 right then down, NOC1 up then left (one-way tori 17 x 12);
a router link and an endpoint carry 64 B a cycle a NoC; a bank delivers 512 GB/s / 8 / 1.35 GHz = 47.4 B a cycle.
The banks' endpoint tiles: NOC0 coordinates x 0 (banks 0-3) and x 9 (banks 4-7), 3 tiles a bank
(blackhole_140_arch.yaml, the map test_qwen38_dataflow_dram printed: /tmp/q38/dram_map.txt rows 'bank')."""
import collections
import math

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

NX, NY = 17, 12
BANK = 512e9 / 8 / 1.35e9
LINK = 64.0
G = 4096                     # granule: 4 KiB (2 KiB-or-larger contiguous requests: 27B README section 3 GDDR rule)
# the banks' 3 endpoint tiles each, NOC0 (x, y): from the device map (dram_map.txt rows 'bank ... noc0 x y')
BANK_EPS = [[(0, 11), (0, 1), (0, 0)], [(0, 2), (0, 10), (0, 3)], [(0, 9), (0, 4), (0, 8)], [(0, 5), (0, 7), (0, 6)],
            [(9, 11), (9, 1), (9, 0)], [(9, 3), (9, 10), (9, 2)], [(9, 8), (9, 4), (9, 9)], [(9, 6), (9, 7), (9, 5)]]


def path(src, dst, noc):
    (x, y), (dx, dy) = src, dst
    links = []
    if noc == 0:
        while x != dx:
            links.append((x, y, "E"))
            x = (x + 1) % NX
        while y != dy:
            links.append((x, y, "S"))
            y = (y + 1) % NY
    else:
        while y != dy:
            links.append((x, y, "N"))
            y = (y - 1) % NY
        while x != dx:
            links.append((x, y, "W"))
            x = (x - 1) % NX
    return links


def plan(core_bytes, nocs=(1,), U=0.5, time_limit=60):
    """core_bytes: {(x, y) NOC0 coords of a Tensix core: bytes}; returns dict(flows={core: [(bank, ep, noc, bytes)]},
    bank_bytes, umax, T (cycles), eff (= total / (8 x BANK x T)))"""
    cores = sorted(c for c, b in core_bytes.items() if b > 0)
    Q = {c: max(1, math.ceil(core_bytes[c] / G)) for c in cores}
    tot = sum(Q.values())
    NB = 8
    per_bank = tot // NB
    rem = tot - per_bank * NB
    bank_q = [per_bank + (1 if b < rem else 0) for b in range(NB)]
    flows = [(b, e, n) for b in range(NB) for e in range(3) for n in nocs]
    NF = len(flows)
    NC = len(cores)
    T = max(bank_q) * G / BANK
    cap = U * LINK * T / G

    def gi(c, f):
        return c * NF + f

    def zi(c, f):
        return NC * NF + c * NF + f

    nvar = 2 * NC * NF
    rows = {}
    ent = []
    hops = np.zeros(nvar)
    for ci, c in enumerate(cores):
        for f, (b, e, n) in enumerate(flows):
            p = path(BANK_EPS[b][e], c, n)
            hops[gi(ci, f)] = len(p) + 1
            for l in p:
                ent.append((rows.setdefault(("link", n) + l, len(rows)), gi(ci, f)))
            ent.append((rows.setdefault(("ep", b, e, n), len(rows)), gi(ci, f)))
            ent.append((rows.setdefault(("core", ci, n), len(rows)), gi(ci, f)))
    A = lil_matrix((len(rows), nvar))
    for r, v in ent:
        A[r, v] += 1.0
    cons = [LinearConstraint(A.tocsr(), -np.inf, cap)]
    E = lil_matrix((NC + NB, nvar))
    for ci in range(NC):
        for f, (b, e, n) in enumerate(flows):
            E[ci, gi(ci, f)] = 1
            E[NC + b, gi(ci, f)] = 1
    lo = np.array([Q[c] for c in cores] + bank_q, dtype=float)
    cons.append(LinearConstraint(E.tocsr(), lo, lo))
    Z = lil_matrix((NC * NF + NC, nvar))
    for ci, c in enumerate(cores):
        for f in range(NF):
            Z[ci * NF + f, gi(ci, f)] = 1
            Z[ci * NF + f, zi(ci, f)] = -Q[c]
            Z[NC * NF + ci, zi(ci, f)] = 1
    cons.append(LinearConstraint(Z.tocsr(), np.full(NC * NF + NC, -np.inf), np.concatenate([np.zeros(NC * NF), np.full(NC, 2)])))
    ub = np.concatenate([np.array([Q[c] for c in cores for _ in range(NF)], dtype=float), np.ones(NC * NF)])
    res = milp(c=hops, constraints=cons, integrality=np.ones(nvar), bounds=Bounds(np.zeros(nvar), ub),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": 0.02})
    if res.x is None:
        return None
    x = np.round(res.x).astype(int)
    out = {}
    load = collections.Counter()
    bank_b = [0] * NB
    for ci, c in enumerate(cores):
        fl = []
        for f, (b, e, n) in enumerate(flows):
            q = x[gi(ci, f)]
            if q:
                fl.append((b, e, n, q * G))
                bank_b[b] += q * G
                for l in path(BANK_EPS[b][e], c, n):
                    load[("link", n) + l] += q * G
        out[c] = fl
    umax = max(load.values()) / (LINK * T) if load else 0.0
    return dict(flows=out, bank_bytes=bank_b, umax=umax, T=T, eff=tot * G / (NB * BANK * T), status=res.status)
