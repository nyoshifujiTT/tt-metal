#!/usr/bin/env python
r"""I-1b': the DRAM-read assignment as an exact integer program (worklog 14): every core reads exactly S bytes in 4 KiB
granules, from at most 2 (bank, endpoint tile, NoC) flows, every bank delivers exactly 130 S / 8, and the largest
link / endpoint-NoC / core-NoC load is at most U x (64 B/cycle x T) for the bank time T; objective: the total
link load (bytes x hops).  Variables: g[c, f] >= 0 integer granules on flow f of core c, z[c, f] binary (flow used),
g <= (S/G) z, sum_f z <= 2.  Solved with scipy's HiGHS MILP.
argv: S, out file, NoCs (e.g. 1 or 0,1), U, time limit [s]."""
import collections, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import noc_lp as m

S = int(sys.argv[1]); out = sys.argv[2]
NOCS = tuple(int(x) for x in sys.argv[3].split(','))
U = float(sys.argv[4]); TL = float(sys.argv[5]) if len(sys.argv) > 5 else 300
G = 4096
w, banks = m.load_map('/tmp/q38/dram_map.txt')
tr = {}
for line in open('/tmp/q38/dram_map.txt'):
    t = line.split()
    if t and t[0] == 'bank':
        i = t.index('ep')
        while i < len(t):
            tr[(int(t[i + 4]), int(t[i + 5]))] = (int(t[i + 1]), int(t[i + 2]))
            i += 6
cores = sorted(w, key=lambda c: (c[1], c[0]))
NC, NB = len(cores), len(banks)
Q = S // G
flows = [(b, e, n) for b in range(NB) for e in range(len(banks[b])) for n in NOCS]
NF = len(flows)
nvar = 2 * NC * NF
def gi(c, f): return c * NF + f
def zi(c, f): return NC * NF + c * NF + f
T = NC * S / (NB * m.BANK)          # cycles
cap = U * m.LINK * T / G            # granules per link
rows = {}
def row(k):
    if k not in rows:
        rows[k] = len(rows)
    return rows[k]
ent = []
hops = np.zeros(nvar)
for c in range(NC):
    for f, (b, e, n) in enumerate(flows):
        p = m.path(banks[b][e], w[cores[c]], n)
        hops[gi(c, f)] = len(p)
        for l in p:
            ent.append((row(('link', n) + l), gi(c, f)))
        ent.append((row(('ep', b, e, n)), gi(c, f)))
        ent.append((row(('core', c, n)), gi(c, f)))
A = lil_matrix((len(rows), nvar))
for r, v in ent:
    A[r, v] += 1.0
cons = [LinearConstraint(A.tocsr(), -np.inf, cap)]
E = lil_matrix((NC + NB, nvar))
for c in range(NC):
    for f, (b, e, n) in enumerate(flows):
        E[c, gi(c, f)] = 1
        E[NC + b, gi(c, f)] = 1
cons.append(LinearConstraint(E.tocsr(), np.concatenate([np.full(NC, Q), np.full(NB, NC * Q / NB)]),
                             np.concatenate([np.full(NC, Q), np.full(NB, NC * Q / NB)])))
Z = lil_matrix((NC * NF + NC, nvar))
for c in range(NC):
    for f in range(NF):
        Z[c * NF + f, gi(c, f)] = 1
        Z[c * NF + f, zi(c, f)] = -Q
        Z[NC * NF + c, zi(c, f)] = 1
lo = np.full(NC * NF + NC, -np.inf)
hi = np.concatenate([np.zeros(NC * NF), np.full(NC, 2)])
cons.append(LinearConstraint(Z.tocsr(), lo, hi))
assert (NC * Q) % NB == 0, 'banks cannot be exactly equal'
integrality = np.ones(nvar)
ub = np.concatenate([np.full(NC * NF, Q), np.ones(NC * NF)])
res = milp(c=hops, constraints=cons, integrality=integrality, bounds=Bounds(np.zeros(nvar), ub),
           options={'time_limit': TL, 'disp': False, 'mip_rel_gap': 0.02})
print(res.status, res.message)
assert res.x is not None, 'no feasible assignment'
x = np.round(res.x).astype(int)
load = collections.Counter()
srcs = collections.defaultdict(set)
off = collections.Counter()
lines = []
for c in range(NC):
    parts = []
    for f, (b, e, n) in enumerate(flows):
        q = x[gi(c, f)]
        if q:
            parts.append((b, e, n, q * G))
    for b, e, n, by in parts:
        for l in m.path(banks[b][e], w[cores[c]], n):
            load[('link', n) + l] += by
            srcs[('link', n) + l].add(b)
    lines.append((cores[c], parts))
umax = max(load.values()) / (m.LINK * T)
print(f'S {S} NoCs {NOCS} U {U}: max link utilisation {umax:.3f}, total hop-bytes {sum(hops*np.concatenate([x[:NC*NF]*G, np.zeros(NC*NF)])):.3e}; two-flow cores {sum(1 for _, p in lines if len(p) == 2)}; sources per link {collections.Counter(len(v) for v in srcs.values())}')
with open(out, 'w') as fo:
    for cc, parts in lines:
        s_ = []
        for b, e, n, by in parts:
            tx, ty = tr[banks[b][e]]
            s_.append(f'{b} {off[b]} {by} {tx} {ty} {n}')
            off[b] += by
        fo.write(f'{cc[0]} {cc[1]} {len(parts)} ' + ' '.join(s_) + '\n')
print('->', out, 'bank bytes', dict(sorted(off.items())))
