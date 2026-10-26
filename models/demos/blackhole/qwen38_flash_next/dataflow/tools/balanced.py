#!/usr/bin/env python
r"""I-1b': a balanced DRAM-read assignment -- every one of the 130 cores holds the same share S of the weights (so the
matmul work per core is even) -- with the 8 banks at full rate and the smallest maximum link load.
Step 1 (LP): variables x[c, b, e, n] >= 0 (bytes core c reads from bank b's endpoint tile e on NoC n); sum over
(b, e, n) = S per core; every bank delivers 130 S / 8 (equal banks, the time is the bank bound); minimise the largest
link / endpoint-NoC / core-NoC load L (in units of the bank's rate: the bank's 47.4 B/cycle -> 1).
Step 2 (rounding): each core keeps its two largest flows (4 KiB granules), re-check the loads.
Routing (tt-isa-documentation NoC/RoutingPaths.md): NOC0 right then down, NOC1 up then left."""
import collections, sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import noc_lp as m

S = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
out = sys.argv[2] if len(sys.argv) > 2 else f'/tmp/q38/noc/balanced_{S}.txt'
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
NC = len(cores)
NB = len(banks)
var = [(ci, b, e, n) for ci in range(NC) for b in range(NB) for e in range(len(banks[b])) for n in (0, 1)]
nv = len(var) + 1
rows = {}
def row(k):
    if k not in rows:
        rows[k] = len(rows)
    return rows[k]
ent = []
for vi, (ci, b, e, n) in enumerate(var):
    src = banks[b][e]
    dst = w[cores[ci]]
    for l in m.path(src, dst, n):
        ent.append((row(('link', n) + l), vi))
    ent.append((row(('ep', b, e, n)), vi))
    ent.append((row(('core', ci, n)), vi))
A = lil_matrix((len(rows), nv))
for r, v in ent:
    A[r, v] += 1.0
# loads in bytes; the bank rate per byte of the round: a round moves NC*S bytes in T = NC*S/(NB*BANK) cycles, so a
# load of x bytes on a 64 B/cycle link is a utilisation x / (64 T); express L as a utilisation
T = NC * S / (NB * m.BANK)
for r in range(len(rows)):
    A[r, nv - 1] = -m.LINK * T
Aeq = lil_matrix((NC + NB, nv))
for vi, (ci, b, e, n) in enumerate(var):
    Aeq[ci, vi] = 1.0
    Aeq[NC + b, vi] = 1.0
beq = np.concatenate([np.full(NC, float(S)), np.full(NB, NC * S / NB)])
c = np.zeros(nv)
c[-1] = 1.0
res = linprog(c, A_ub=A.tocsr(), b_ub=np.zeros(len(rows)), A_eq=Aeq.tocsr(), b_eq=beq, bounds=[(0, None)] * nv, method='highs')
assert res.status == 0, res.message
L = res.x[-1]
print(f'S {S}: LP max link utilisation {L:.3f} at the bank bound (100%)')
per = collections.defaultdict(list)
for vi, (ci, b, e, n) in enumerate(var):
    if res.x[vi] > 1e-6:
        per[ci].append([res.x[vi], b, e, n])
print('flows per core', collections.Counter(len(v) for v in per.values()))
# round: the two largest flows, 4 KiB granules
G = 4096
segs = {}
for ci in range(NC):
    v = sorted(per[ci], reverse=True)[:2]
    if len(v) == 1 or v[1][0] < G / 2:
        segs[ci] = [(v[0][1], v[0][2], v[0][3], S)]
    else:
        a = int(round(v[0][0] * S / (v[0][0] + v[1][0]) / G)) * G
        a = min(max(a, G), S - G)
        segs[ci] = [(v[0][1], v[0][2], v[0][3], a), (v[1][1], v[1][2], v[1][3], S - a)]
# exact bank balance: move 4 KiB granules between a core's two segments until every bank holds exactly NC*S/NB
# (each move keeps the core's total S; a core with one segment cannot move -- it is fixed)
target = NC * S // NB
def bank_tot():
    t = collections.Counter()
    for ss in segs.values():
        for b, e, n, x in ss:
            t[b] += x
    return t
for it in range(100000):
    t = bank_tot()
    over = max(t, key=lambda b: t[b] - target)
    under = min(t, key=lambda b: t[b] - target)
    if t[over] - target < G and target - t[under] < G:
        break
    moved = False
    # a two-segment core with one segment in `over`: shift G to its other segment, preferring one in `under`
    cands = [(ci, ss) for ci, ss in segs.items() if len(ss) == 2 and any(s_[0] == over and s_[3] > G for s_ in ss)]
    cands.sort(key=lambda t_: 0 if any(s_[0] == under for s_ in t_[1]) else 1)
    for ci, ss in cands:
        i = 0 if ss[0][0] == over else 1
        j = 1 - i
        if t[ss[j][0]] - target >= 0 and ss[j][0] != under:
            continue
        a = list(ss[i]); b2 = list(ss[j])
        a[3] -= G; b2[3] += G
        segs[ci] = [tuple(a), tuple(b2)] if i == 0 else [tuple(b2), tuple(a)]
        moved = True
        break
    if not moved:
        break
print('bank imbalance after balancing (bytes over target):', {b: v - target for b, v in sorted(bank_tot().items())})
load = collections.Counter()
bank = collections.Counter()
for ci, ss in segs.items():
    for b, e, n, x in ss:
        for l in m.path(banks[b][e], w[cores[ci]], n):
            load[('link', n) + l] += x
        load[('ep', b, e, n)] += x
        load[('core', ci, n)] += x
        bank[b] += x
Tb = max(bank.values()) / m.BANK
Tl = max(load.values()) / m.LINK
print(f'rounded: bank time {Tb/1.35e3:.2f} us (ideal {T/1.35e3:.2f}), max link utilisation {max(load.values())/(m.LINK*Tb):.3f}, efficiency {100*T/max(Tb,Tl):.2f}%')
srcs = collections.defaultdict(set)
for ci, ss in segs.items():
    for b, e, n, x in ss:
        for l in m.path(banks[b][e], w[cores[ci]], n):
            srcs[('link', n) + l].add((b, e))
print('links carrying flows of k source tiles:', collections.Counter(len(v) for v in srcs.values()))
off = collections.Counter()
with open(out, 'w') as f:
    for ci, cc in enumerate(cores):
        parts = []
        for b, e, n, x in segs[ci]:
            tx, ty = tr[banks[b][e]]
            parts.append(f'{b} {off[b]} {x} {tx} {ty} {n}')
            off[b] += x
        f.write(f'{cc[0]} {cc[1]} {len(parts)} ' + ' '.join(parts) + '\n')
print('->', out, 'bank bytes', dict(sorted(off.items())))
