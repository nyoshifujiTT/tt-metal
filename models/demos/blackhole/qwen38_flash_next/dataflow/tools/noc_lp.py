#!/usr/bin/env python
r"""I-1b' (worklog 14): the DRAM-read link-load model of one Blackhole chip and the assignment that saturates the 8
banks.  Inputs: the chip map (test_qwen38_dataflow_dram Map: worker NOC0 coords, the banks' endpoint tiles).
Physical rules (tt-isa-documentation NoC/RoutingPaths.md): NOC0 responses go right (x+1 mod 17) then down (y+1 mod 12),
NOC1 responses go up (y-1) then left (x-1); every router-router link carries at most 64 B/cycle (one flit) per NoC;
an endpoint injects at most 64 B/cycle per NoC; a bank delivers at most BANK B/cycle (512 GB/s / 8 / 1.35 GHz = 47.4)
in total over its endpoints; a core receives at most 64 B/cycle per NoC.
Variables: x[c, b, e, n] = bytes core c reads from bank b through endpoint tile e on NoC n.
Demand: every reader core reads S bytes in total (its slice); the slices can be placed in any bank (the weight layout is
ours) -- or, with pinned=1, core c's slice is in bank pin[c] (then only e and n are free).
Objective: minimise T (cycles) s.t. per-link loads <= 64 T, per-bank <= BANK T, per-endpoint-NoC <= 64 T,
per-core-NoC <= 64 T.  Efficiency = total / (8 BANK T)."""
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

NX, NY = 17, 12
BANK = 512e9 / 8 / 1.35e9
LINK = 64.0


def load_map(path):
    workers = {}
    banks = []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "worker":
            workers[(int(t[1]), int(t[2]))] = (int(t[7]), int(t[8]))
        elif t[0] == "bank":
            eps = []
            i = t.index("ep")
            while i < len(t):
                eps.append((int(t[i + 4]), int(t[i + 5])))
                i += 6
            banks.append(eps)
    return workers, banks


def path(src, dst, noc):
    """links (x, y, dir) used from src to dst; dir 'E'/'S' for NOC0, 'N'/'W' for NOC1; plus the ejection 'J' at dst"""
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


def solve(workers, banks, readers, S, pinned=None, nocs=(0, 1), eps_allowed=None):
    """readers: list of logical cores; returns (T, efficiency, flows)"""
    var = []
    for ci, c in enumerate(readers):
        for b in range(len(banks)):
            if pinned is not None and pinned[ci] != b:
                continue
            for ei, e in enumerate(banks[b]):
                if eps_allowed is not None and ei not in eps_allowed:
                    continue
                for n in nocs:
                    var.append((ci, b, ei, n))
    nv = len(var) + 1  # last = T
    rows = {}
    def row(key):
        if key not in rows:
            rows[key] = len(rows)
        return rows[key]
    entries = []
    for vi, (ci, b, ei, n) in enumerate(var):
        src = banks[b][ei]
        dst = workers[readers[ci]]
        for l in path(src, dst, n):
            entries.append((row(("link", n) + l), vi, 1.0))
        entries.append((row(("bank", b)), vi, 1.0))
        entries.append((row(("ep", b, ei, n)), vi, 1.0))
        entries.append((row(("core", ci, n)), vi, 1.0))
    nr = len(rows)
    A = lil_matrix((nr, nv))
    for r, v, a in entries:
        A[r, v] += a
    cap = np.zeros(nr)
    for k, r in rows.items():
        cap[r] = BANK if k[0] == "bank" else LINK
    for r in range(nr):
        A[r, nv - 1] = -cap[r]
    Aeq = lil_matrix((len(readers), nv))
    for vi, (ci, b, ei, n) in enumerate(var):
        Aeq[ci, vi] = 1.0
    beq = np.full(len(readers), float(S))
    c = np.zeros(nv)
    c[-1] = 1.0
    res = linprog(c, A_ub=A.tocsr(), b_ub=np.zeros(nr), A_eq=Aeq.tocsr(), b_eq=beq, bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        raise SystemExit(res.message)
    T = res.x[-1]
    total = S * len(readers)
    eff = total / (len(banks) * BANK * T)
    flows = [(var[i], res.x[i]) for i in range(len(var)) if res.x[i] > 1e-6]
    # binding constraints
    slack = {k: cap[r] * T - (A[r, :nv - 1] @ res.x[:nv - 1]).item() for k, r in rows.items()}
    tight = sorted((v, k) for k, v in slack.items())[:12]
    return T, eff, flows, tight


if __name__ == "__main__":
    workers, banks = load_map(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/dram_map.txt")
    allc = sorted(workers)
    S = 262144
    for name, readers, pinned, nocs, epsa in (
        ("130 cores, any bank, endpoint and NoC", allc, None, (0, 1), None),
        ("130 cores, any bank, NOC0 only, all endpoints", allc, None, (0,), None),
        ("130 cores, any bank, NOC0 only, the NOC0 preferred endpoint only", allc, None, (0,), (0,)),
        ("130 cores, bank = core index mod 8 (the run), NOC0 preferred endpoint", allc, [i % 8 for i in range(len(allc))], (0,), (0,)),
    ):
        T, eff, flows, tight = solve(workers, banks, readers, S, pinned, nocs, epsa)
        print(f"{name}: T {T:.0f} cycles = {T/1.35e3:.2f} us for {S*len(readers)/1e6:.1f} MB -> {100*eff:.1f}% of the 8 banks")
        print("   tightest:", [(round(v), k) for v, k in tight[:6]])
