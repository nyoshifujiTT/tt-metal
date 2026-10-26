#!/usr/bin/env python
"""I-1b': the row-local assignment -- worker row y reads only the endpoint tiles in its own NOC0 row (left cores from
column 0's tile, right cores from column 9's), so no two banks' responses share a link (NOC0 responses go right then
down: they stay in row y).  Bank b's bytes = total / 8, split evenly over its readers.  Output per core:
x y 1 bank offset bytes tile_tx tile_ty (the tile's translated coordinates for the explicit-endpoint read)."""
import collections, sys
import noc_lp as m

bank_bytes = int(sys.argv[1]) if len(sys.argv) > 1 else 2 * 1024 * 1024
out = sys.argv[2] if len(sys.argv) > 2 else f'/tmp/q38/noc/rowlocal_{bank_bytes}.txt'
w, b = m.load_map('/tmp/q38/dram_map.txt')
tr = {}
for line in open('/tmp/q38/dram_map.txt'):
    t = line.split()
    if t and t[0] == 'bank':
        i = t.index('ep')
        while i < len(t):
            tr[(int(t[i + 4]), int(t[i + 5]))] = (int(t[i + 1]), int(t[i + 2]))
            i += 6
tile = {e: (bb, ei) for bb, eps in enumerate(b) for ei, e in enumerate(eps)}
readers = collections.defaultdict(list)
for c in sorted(w, key=lambda c: (c[1], c[0])):
    p = w[c]
    e = (0 if p[0] < 9 else 9, p[1])
    readers[tile[e][0]].append((c, e))
G = 4096
off = collections.Counter()
lines = []
for bb in range(8):
    n = len(readers[bb])
    s = (bank_bytes // n) // G * G
    for c, e in readers[bb]:
        tx, ty = tr[e]
        lines.append((c, f'{c[0]} {c[1]} 1 {bb} {off[bb]} {s} {tx} {ty}'))
        off[bb] += s
lines.sort(key=lambda t: (t[0][1], t[0][0]))
open(out, 'w').write('\n'.join(l for _, l in lines) + '\n')
tot = sum(off.values())
print(out, 'readers/bank', {k: len(v) for k, v in sorted(readers.items())}, 'bank bytes', dict(sorted(off.items())), 'total', tot)
