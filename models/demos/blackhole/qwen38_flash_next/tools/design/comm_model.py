#!/usr/bin/env python
r"""Packet-level communication model of a TP group on the 8x4 Blackhole Galaxy torus (physical values only, README
11.2.3 / 27B README section 3): link 25 GB/s per direction, hop latency 280 ns, store-and-forward of a packet at the
sender 96 B/cycle (2 KiB: 16 ns), relay through an intermediate chip over the NoC 92 cycles + 32 flits (92 ns), 56 B of
line overhead per packet, payload per packet <= 4,096 B (2 KiB used: the tile size), a reduce level (NoC distribution of
the arrived tiles to cores + one DEST accumulate) 92 + 64 + 32 cycles = 139 ns.
Submesh r x c of the torus: horizontal ring when c == 4 (the torus column count), vertical ring when r == 8; otherwise
lines.  Routes: vertical first, then the shorter ring direction (ties alternate).  Every link-direction is a FIFO.
Collectives (per chip payload P):
  AG_tree    each chip's chunk multicast along its BFS tree (a relay forwards to each child after RELAY + SAF)
  AG_direct  each chip sends its chunk point-to-point to every other chip
  RS_tree    reduce-scatter: for each owner, the others' slices (P/n) flow up the owner's inverted BFS tree; a node
             forwards its partial sum after its children arrived (+ one reduce level per node)
  RS_direct  each chip sends P/n slices point-to-point to their owners, the owner reduces (one level)
  AR = min(AG_tree+local reduce, AG_direct+local, RS_tree+AG_tree, RS_direct+AG_direct)   (all-reduce of P)
A collective returns (completion time of the last byte at its destination, the per-chip arrival times of the chunks
(AG), the per-link bytes).  consume_overlap() gives the end of a matmul that consumes the all-gathered row as its K
chunks arrive (the cores accumulate in fp32 DEST across chunks; the K-split tree reduce follows the last chunk)."""
import heapq
import itertools
import math

BW = 25e9
HOP = 280e-9
CLK = 1.35e9
SAF_RATE = 96
RELAY_CYC = 92 + 32
HDR = 56
PKT = 2048
ADD_LEVEL = (92 + 64 + 32) / CLK
TORUS_R = 8
TORUS_C = 4


def saf(b):
    return b / SAF_RATE / CLK


RELAY = RELAY_CYC / CLK


class Mesh:
    def __init__(s, r, c):
        s.r = r
        s.c = c
        s.chips = [(i, j) for i in range(r) for j in range(c)]
        s.n = r * c
        s.ring_c = c == TORUS_C
        s.ring_r = r == TORUS_R

    def step(s, a, b):
        """link id for the hop a -> b (adjacent)"""
        return (a, b)

    def route(s, src, dst, alt=0):
        (r1, c1), (r2, c2) = src, dst
        path = []
        cur = src
        # vertical
        dr = r2 - r1
        if s.ring_r and abs(dr) > s.r // 2:
            dr = dr - s.r if dr > 0 else dr + s.r
        if s.ring_r and abs(dr) == s.r // 2 and alt % 2:
            dr = -dr
        while dr != 0:
            step = 1 if dr > 0 else -1
            nxt = ((cur[0] + step) % s.r if s.ring_r else cur[0] + step, cur[1])
            path.append((cur, nxt))
            cur = nxt
            dr -= step
        dc = c2 - c1
        if s.ring_c and abs(dc) > s.c // 2:
            dc = dc - s.c if dc > 0 else dc + s.c
        if s.ring_c and abs(dc) == s.c // 2 and alt % 2:
            dc = -dc
        while dc != 0:
            step = 1 if dc > 0 else -1
            nxt = (cur[0], (cur[1] + step) % s.c if s.ring_c else cur[1] + step)
            path.append((cur, nxt))
            cur = nxt
            dc -= step
        return path

    def hops(s, a, b):
        return len(s.route(a, b))

    def diameter(s):
        return max(s.hops(a, b) for a in s.chips for b in s.chips)

    def bfs_tree(s, root):
        """BFS tree from root following the dimension-ordered shortest routes: parent of x = the chip before x on
        route(root, x) (the union of shortest routes is a tree)"""
        T = {}
        for x in s.chips:
            if x == root:
                continue
            p = s.route(root, x)
            parent = p[-1][0]
            T.setdefault(parent, []).append(x)
        return T


# ---------------- packet simulation ----------------
def simulate_mc(flows, pkt=PKT):
    """flows: (t_ready, root, tree, bytes, tag).  Multicast of bytes from root along tree (dict parent -> children);
    returns (last arrival, arrivals {(tag, leaf): time of the last byte}, link bytes {link: bytes}, link busy {link: s})
    """
    link_free = {}
    link_bytes = {}
    link_busy = {}
    events = []
    seq = itertools.count()
    arrivals = {}
    done = 0.0
    for t0, root, tree, nbytes, tag in flows:
        n = max(1, math.ceil(nbytes / pkt))
        left = nbytes
        tr = t0
        for p in range(n):
            b = min(pkt, left)
            left -= b
            tr += saf(b)  # the sender stores packet after packet
            heapq.heappush(events, (tr, next(seq), root, tree, b, tag, True))
    while events:
        t, s, node, tree, b, tag, last = heapq.heappop(events)
        kids = tree.get(node, [])
        for kid in kids:
            link = (node, kid)
            start = max(t, link_free.get(link, 0.0))
            ser = (b + HDR) / BW
            link_free[link] = start + ser
            link_bytes[link] = link_bytes.get(link, 0) + b
            link_busy[link] = link_busy.get(link, 0.0) + ser
            arrive = start + ser + HOP
            arrivals[(tag, kid)] = max(arrivals.get((tag, kid), 0.0), arrive)
            done = max(done, arrive)
            if tree.get(kid):
                heapq.heappush(events, (arrive + RELAY + saf(b), next(seq), kid, tree, b, tag, last))
    return done, arrivals, link_bytes, link_busy


def path_tree(mesh, src, dst, alt=0):
    T = {}
    for a, b in mesh.route(src, dst, alt):
        T[a] = [b]
    return T


def ag_direct(mesh, P, t0=0.0, pkt=PKT):
    fl = []
    k = 0
    for s in mesh.chips:
        for d in mesh.chips:
            if s != d:
                fl.append((t0, s, path_tree(mesh, s, d, k), P, s))
                k += 1
    return simulate_mc(fl, pkt)


def ag_tree(mesh, P, t0=0.0, pkt=PKT):
    return simulate_mc([(t0, root, mesh.bfs_tree(root), P, root) for root in mesh.chips], pkt)


def rs_direct(mesh, P, t0=0.0, pkt=PKT):
    d, arr, lb, lu = ag_direct(mesh, P / mesh.n, t0, pkt)
    return d + ADD_LEVEL, arr, lb, lu


def rs_tree(mesh, P, t0=0.0, pkt=PKT):
    """for every owner: leaves send P/n up; a node sends after all its children arrived (+ ADD_LEVEL); FIFO links
    shared among the owners' trees; simulated node by node in dependency order (per owner the tree is a DAG)"""
    link_free = {}
    lb = {}
    lu = {}
    done = 0.0
    slice_ = P / mesh.n
    # event-driven: a node's send time = max(children's arrivals) + ADD_LEVEL (leaves: t0 + saf); process in order of
    # readiness over all owners jointly (a link FIFO orders packets by readiness)
    ready = []
    seq = itertools.count()
    trees = {o: mesh.bfs_tree(o) for o in mesh.chips}
    parent = {o: {} for o in mesh.chips}
    pending = {o: {} for o in mesh.chips}
    arr = {o: {} for o in mesh.chips}
    for o, T in trees.items():
        for p, kids in T.items():
            for k in kids:
                parent[o][k] = p
        for x in mesh.chips:
            if x == o:
                continue
            pending[o][x] = len(T.get(x, []))
            if pending[o][x] == 0:
                heapq.heappush(ready, (t0 + saf(min(pkt, slice_)), next(seq), o, x))
    finish = {}
    while ready:
        t, s, o, x = heapq.heappop(ready)
        p = parent[o][x]
        link = (x, p)
        n = max(1, math.ceil(slice_ / pkt))
        left = slice_
        tcur = t
        last = None
        for i in range(n):
            b = min(pkt, left)
            left -= b
            start = max(tcur + (0 if i == 0 else 0.0), link_free.get(link, 0.0))
            ser = (b + HDR) / BW
            link_free[link] = start + ser
            lb[link] = lb.get(link, 0) + b
            lu[link] = lu.get(link, 0.0) + ser
            last = start + ser + HOP
            tcur = start + ser
        arr[o].setdefault(p, 0.0)
        arr[o][p] = max(arr[o][p], last)
        if p == o:
            finish[o] = last + ADD_LEVEL
            done = max(done, finish[o])
            continue
        pending[o][p] -= 1
        if pending[o][p] == 0:
            heapq.heappush(ready, (arr[o][p] + ADD_LEVEL + saf(min(pkt, slice_)), next(seq), o, p))
    return done, finish, lb, lu


def all_reduce(mesh, P, t0=0.0, pkt=PKT):
    """returns (time, form, forms): all-reduce of P per chip (every chip needs the full sum)"""
    forms = {}
    d, arr, _, _ = ag_tree(mesh, P, t0, pkt)
    forms["AG_tree+local"] = d + ADD_LEVEL
    d, arr, _, _ = ag_direct(mesh, P, t0, pkt)
    forms["AG_direct+local"] = d + ADD_LEVEL
    d1 = rs_tree(mesh, P, t0, pkt)[0]
    forms["RS_tree+AG_tree"] = ag_tree(mesh, P / mesh.n, d1, pkt)[0]
    d1 = rs_direct(mesh, P, t0, pkt)[0]
    forms["RS_direct+AG_direct"] = ag_direct(mesh, P / mesh.n, d1, pkt)[0]
    best = min(forms, key=forms.get)
    return forms[best] - t0, best, {k: v - t0 for k, v in forms.items()}


def all_gather(mesh, P, t0=0.0, pkt=PKT):
    forms = {"AG_tree": ag_tree(mesh, P, t0, pkt)[0], "AG_direct": ag_direct(mesh, P, t0, pkt)[0]}
    best = min(forms, key=forms.get)
    return forms[best] - t0, best, {k: v - t0 for k, v in forms.items()}


def reduce_scatter(mesh, P, t0=0.0, pkt=PKT):
    forms = {"RS_tree": rs_tree(mesh, P, t0, pkt)[0], "RS_direct": rs_direct(mesh, P, t0, pkt)[0]}
    best = min(forms, key=forms.get)
    return forms[best] - t0, best, {k: v - t0 for k, v in forms.items()}


def ag_arrivals(mesh, P, dst, pkt=PKT):
    """arrival times at chip dst of every other chip's chunk (AG_tree), sorted; own chunk at 0"""
    d, arr, _, _ = ag_tree(mesh, P, 0.0, pkt)
    return sorted([0.0] + [t for (tag, leaf), t in arr.items() if leaf == dst])


def consume_overlap(arrivals, t_mm, t_tree):
    """end of a matmul of t_mm (K-split over cores, tree reduce t_tree after the last chunk) that consumes the n
    chunks as they arrive, in arrival order; the K-range of each core is processed chunk by chunk"""
    n = len(arrivals)
    busy = 0.0
    for a in arrivals:
        busy = max(busy, a) + t_mm / n
    return busy + t_tree


def point_to_point(mesh, src, dst, nbytes, t0=0.0, pkt=PKT):
    return simulate_mc([(t0, src, path_tree(mesh, src, dst), nbytes, src)], pkt)[0] - t0


if __name__ == "__main__":
    pass

    for r, c in ((1, 1), (2, 1), (1, 2), (1, 4), (2, 4), (4, 4), (8, 4), (8, 2)):
        m = Mesh(r, c)
        if m.n == 1:
            continue
        for P in (32, 1280, 2592, 3840, 10240, 15360, 20480, 40960):
            t, f, fs = all_reduce(m, P)
            ta, fa, _ = all_gather(m, P)
            tr, fr, _ = reduce_scatter(m, P)
            print(
                f"{r}x{c} n={m.n:2d} diam {m.diameter()} P={P:6d}: AR {t*1e6:.3f} ({f}) AG {ta*1e6:.3f} ({fa}) RS {tr*1e6:.3f} ({fr}); AG arrivals at (0,0): {[round(x*1e6,2) for x in ag_arrivals(m,P,(0,0))]}"
            )
