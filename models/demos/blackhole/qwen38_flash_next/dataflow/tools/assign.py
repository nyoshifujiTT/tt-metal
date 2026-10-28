#!/usr/bin/env python
"""I-1b': round the LP's DRAM-read assignment to an implementable one (each core: <= 2 segments (bank, bytes), 4 KiB
granules, NOC0, the bank's NOC0 preferred endpoint = what get_noc_addr_from_bank_id addresses), lay the segments out in
the banks (per bank, segments back to back), check the link loads of the rounded flows, write the assignment file."""
import collections, sys
import noc_lp as m

G = 4096
w, b = m.load_map('/tmp/q38/dram_map.txt')
allc = sorted(w, key=lambda c: (c[1], c[0]))
S = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
out = sys.argv[2] if len(sys.argv) > 2 else f'/tmp/q38/noc/assign_{S}.txt'
T, eff, flows, tight = m.solve(w, b, allc, S, None, (0,), (0,))
per = collections.defaultdict(list)
for (ci, bb, ei, n), x in flows:
    per[ci].append([bb, x])
segs = {}
for ci in range(len(allc)):
    v = sorted(per[ci], key=lambda t: -t[1])
    if len(v) == 1:
        segs[ci] = [(v[0][0], S)]
    else:
        a = int(round(v[0][1] / G)) * G
        a = min(max(a, G), S - G)
        segs[ci] = [(v[0][0], a), (v[1][0], S - a)] + ([] if len(v) == 2 else None)
# evaluate the rounded flows
def evaluate(segs):
    load = collections.Counter()
    for ci, ss in segs.items():
        dst = w[allc[ci]]
        for bb, x in ss:
            for l in m.path(b[bb][0], dst, 0):
                load[('link',) + l] += x
            load[('bank', bb)] += x
            load[('ep', bb)] += x
            load[('core', ci)] += x
    T = max(v / (m.BANK if k[0] == 'bank' else m.LINK) for k, v in load.items())
    return T, S * len(allc) / (8 * m.BANK * T), load
Tr, effr, load = evaluate(segs)
print(f'S {S}: LP {100*eff:.2f}%, rounded {100*effr:.2f}% ({Tr/1.35e3:.2f} us)')
bank_off = collections.Counter()
with open(out, 'w') as f:
    for ci, c in enumerate(allc):
        parts = []
        for bb, x in segs[ci]:
            parts.append(f'{bb} {bank_off[bb]} {x}')
            bank_off[bb] += x
        f.write(f'{c[0]} {c[1]} {len(parts)} ' + ' '.join(parts) + '\n')
print('bank bytes', dict(sorted(bank_off.items())), '->', out)
