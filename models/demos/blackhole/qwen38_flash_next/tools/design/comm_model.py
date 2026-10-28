#!/usr/bin/env python
r"""Packet-level communication model of a TP group on the 8x4 Blackhole Galaxy torus (physical values only, README
11.2.3 / 27B README section 3): toward each neighbour, per direction, 2 channels (2 Ethernet cores of the link, each
25 GB/s one way), hop
latency 280 ns, store-and-forward of a packet at the sender 96 B/cycle (2 KiB: 16 ns), relay through an intermediate
chip over the NoC 92 cycles + 32 flits (92 ns), 56 B of line overhead per packet, payload per packet <= 4,096 B (2 KiB
used: the tile size), a reduce level (NoC distribution of the arrived partial rows to cores + the add: 92 cycles + the
rows' flits + their unpack, 98 cycles at rows 2 = 73 ns; revision 5: 92 + 64 + 32 = 139 ns for a full tile).
Submesh r x c of the torus: horizontal ring when c == 4 (the torus column count), vertical ring when r == 8; otherwise
lines.  Routes: vertical first, then the shorter ring direction (ties alternate).  Every link-direction is CH = 2
channels, each a FIFO; a packet takes the channel that frees first.  A chunk is cut into packets of <= pkt bytes, either
the fewest packets or that count rounded up to a multiple of CH (equal packets, so a single chunk uses both channels);
simulate_mc keeps the faster of the two cuts.
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
CH = 2
HOP = 306e-9  # D-4 (rev8): 280 ns (EthernetSsDesignSpec Table 16, 400G) + 25.6 ns RS-FEC codeword pair at 200G (27B README 3)
CLK = 1.35e9
SAF_RATE = 96
# D-4 (rev8): a relay inside a chip = the receiving Ethernet tile's RISC sees the packet's trailer (fence + L1 load, a
# full poll period: 22), issues a NoC write to the onward Ethernet tile (3 stores + cmd_buf ready 7 = 10) + the NIU's
# payload read (budget N_RD 16) + the flight over NOC_ETH_ETH NoC hops between the paired tiles (5 + 9 x 2 + 5) + the
# packet's flits; the onward tile's sender sees it (poll 22) and issues the Ethernet send (4 TXQ registers + cmd_ongoing:
# 31, 27B README 3); the send's store-and-forward read (saf) is added separately by the simulation
ETH_POLL = 22
NOC_ISSUE = 10
N_RD = 8     # = stages.N_RD (bh noc deepdive slide 21)
ETH_SEND = 31
# D-4 (ii)(b), corrected 10-27 15:1x: a relay's receiving and sending tiles are a pair on the same side of the chip
# (the used Ethernet tiles are x 1-4 and 13-16 on NoC row 1: same-side pairs are at most 3 NoC hops apart)
NOC_ETH_ETH = 3
# D-4 (ii)(a), corrected 10-27 15:1x: additive meetings (partial sums: AR, RS) are made and summed by the owners of the
# producing matmul (at most ~80 columns), placed among the 91+ cores within 6 NoC hops of an Ethernet tile (both
# directions, all 14 harvest cases); gathers (AG) are made by the producing pool's hub, placed 2 hops from an Ethernet
# tile, and delivered from the receiving Ethernet tile to all 130 cores by 2 multicast rectangles (worst 22 hops)
NOC_ETH_CORE = 6
NOC_HUB_ETH = 2
NOC_ETH_ALL = 22
RELAY_CYC = ETH_POLL + NOC_ISSUE + N_RD + (5 + 9 * NOC_ETH_ETH + 5) + ETH_POLL + ETH_SEND
RELAY_FLITS = True
HDR = 56
PKT = 2048
ROWS_AR = 2
# a reduce level (revision 6): the NoC distribution of the arrived partial rows to the cores + the add, from mm_cost.py's
# tree level at the meeting's row count (rows 2: 98 cycles; revision 5 counted a full 4 KiB tile, 188)
# D-4 (rev8): a reduce level = the arrived partial rows go from the receiving Ethernet tile to the owner cores (poll 22 +
# NoC issue 10 + N_RD 16 + flight over NOC_ETH_CORE NoC hops Ethernet -> Tensix (5 + 9 x 6 + 5) + the rows' flits), the
# owner sees them (poll 22), unpacks them into Dst (2 + bytes/128 + budget P_UNP 17) and adds them on the SFPU (per 4-row
# group x half x face: 2 loads, add, nop, store)
P_UNP = 17   # NO SOURCE for BH (= stages.P_UNP; D-2 removes it from the chain)
ADD_LEVEL = (ETH_POLL + NOC_ISSUE + N_RD + (5 + 9 * NOC_ETH_CORE + 5) + math.ceil(ROWS_AR * 32 * 4 / 64) + ETH_POLL
             + 2 + math.ceil(ROWS_AR * 32 * 4 / 128) + P_UNP + math.ceil(ROWS_AR / 4) * 2 * 2 * 5) / CLK
TORUS_R = 8
TORUS_C = 4


def saf(b):
    return b / SAF_RATE / CLK


def cut(nbytes, pkt, even):
    """the packet sizes of a chunk: the fewest packets of <= pkt, or that count rounded up to a multiple of CH"""
    n = max(1, math.ceil(nbytes / pkt))
    if even and n % CH:
        n += CH - n % CH
    return [nbytes / n] * n


class Link:
    """one link-direction: CH channels, each a FIFO; a packet takes the channel that frees first"""

    def __init__(s):
        s.free = [0.0] * CH

    def send(s, t, ser):
        i = min(range(CH), key=lambda k: s.free[k])
        start = max(t, s.free[i])
        s.free[i] = start + ser
        return start


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
# origin of a flow = the maker's hand-over to the sending Ethernet tile, by meeting kind
ORIGIN_ADD = (2 * (NOC_ISSUE + N_RD) + (5 + 9 * NOC_ETH_CORE + 5) + ETH_POLL + ETH_SEND) / CLK      # + the chunk's flits
ORIGIN_GATHER = (NOC_ISSUE + N_RD + (5 + 9 * NOC_HUB_ETH + 5) + ETH_POLL + ETH_SEND) / CLK          # + the chunk's flits
# delivery of a gathered chunk at the receiving chip: the Ethernet tile sees it (poll), multicasts it to all 130 cores
# (issue + N_RD + worst 22 hops + flits), the cores see it (poll)
DELIVER_GATHER = (ETH_POLL + NOC_ISSUE + N_RD + (5 + 9 * NOC_ETH_ALL + 5) + ETH_POLL) / CLK          # + the chunk's flits


def deliver(nbytes):
    return DELIVER_GATHER + math.ceil(nbytes / 64) / CLK


def owner_leg(nbytes):
    """arrival of partial rows at their owners: the receiving Ethernet tile sees them (poll), writes them to the owner
    (issue + N_RD + NOC_ETH_CORE hops + flits), the owner sees them (poll); the add itself is counted by the caller"""
    return (ETH_POLL + NOC_ISSUE + N_RD + (5 + 9 * NOC_ETH_CORE + 5) + ETH_POLL) / CLK + math.ceil(nbytes / 64) / CLK


def chosen_cut(nbytes, pkt=PKT):
    """the cut a meeting of chunks of nbytes uses: simulate_mc / rs_tree take the faster of the two cuts; for the headers
    (H-1 (c)) the cut is decided the same way on a 2 x 4 all-gather tree of that chunk (the packets' count and size)"""
    m = Mesh(2, 4)
    a = simulate_mc([(0.0, r, m.bfs_tree(r), nbytes, r) for r in m.chips], pkt, False)[0]
    b = simulate_mc([(0.0, r, m.bfs_tree(r), nbytes, r) for r in m.chips], pkt, True)[0]
    return cut(nbytes, pkt, b < a)


def simulate_mc(flows, pkt=PKT, even=None, origin=None):
    """flows: (t_ready, root, tree, bytes, tag).  Multicast of bytes from root along tree (dict parent -> children);
    returns (last arrival, arrivals {(tag, leaf): time of the last byte}, link bytes {link: bytes}, link busy {link: s}).
    even=None: the faster of the two cuts (cut()).
    """
    if origin is None:
        origin = ORIGIN_ADD
    if even is None:
        a = simulate_mc(flows, pkt, False, origin)
        b = simulate_mc(flows, pkt, True, origin)
        return a if a[0] <= b[0] else b
    links = {}
    link_bytes = {}
    link_busy = {}
    events = []
    seq = itertools.count()
    arrivals = {}
    done = 0.0
    for t0, root, tree, nbytes, tag in flows:
        # the chunk's maker hands it to the sending Ethernet tile (ORIGIN_ADD / ORIGIN_GATHER above)
        tr = t0 + origin + math.ceil(nbytes / 64) / CLK
        for b in cut(nbytes, pkt, even):
            tr += saf(b)  # the sender stores packet after packet
            heapq.heappush(events, (tr, next(seq), root, tree, b, tag, True))
    while events:
        t, s, node, tree, b, tag, last = heapq.heappop(events)
        kids = tree.get(node, [])
        for kid in kids:
            link = (node, kid)
            ser = (b + HDR) / BW
            start = links.setdefault(link, Link()).send(t, ser)
            link_bytes[link] = link_bytes.get(link, 0) + b
            link_busy[link] = link_busy.get(link, 0.0) + ser
            arrive = start + ser + HOP
            arrivals[(tag, kid)] = max(arrivals.get((tag, kid), 0.0), arrive)
            done = max(done, arrive)
            if tree.get(kid):
                heapq.heappush(events, (arrive + RELAY + math.ceil(b / 64) / CLK + saf(b), next(seq), kid, tree, b, tag, last))
    return done, arrivals, link_bytes, link_busy


def path_tree(mesh, src, dst, alt=0):
    T = {}
    for a, b in mesh.route(src, dst, alt):
        T[a] = [b]
    return T


def ag_direct(mesh, P, t0=0.0, pkt=PKT, origin=None):
    fl = []
    k = 0
    for s in mesh.chips:
        for d in mesh.chips:
            if s != d:
                fl.append((t0, s, path_tree(mesh, s, d, k), P, s))
                k += 1
    return simulate_mc(fl, pkt, None, origin)


def ag_tree(mesh, P, t0=0.0, pkt=PKT, origin=None):
    return simulate_mc([(t0, root, mesh.bfs_tree(root), P, root) for root in mesh.chips], pkt, None, origin)


def rs_direct(mesh, P, t0=0.0, pkt=PKT):
    d, arr, lb, lu = ag_direct(mesh, P / mesh.n, t0, pkt)
    return d + ADD_LEVEL, arr, lb, lu


def rs_tree(mesh, P, t0=0.0, pkt=PKT, even=None):
    """for every owner: leaves send P/n up; a node sends after all its children arrived (+ ADD_LEVEL); CH-channel
    links shared among the owners' trees; simulated node by node in dependency order (per owner the tree is a DAG).
    even=None: the faster of the two cuts."""
    if even is None:
        a = rs_tree(mesh, P, t0, pkt, False)
        b = rs_tree(mesh, P, t0, pkt, True)
        return a if a[0] <= b[0] else b
    links = {}
    lb = {}
    lu = {}
    done = 0.0
    slice_ = P / mesh.n
    sizes = cut(slice_, pkt, even)
    # event-driven: a node's send time = max(children's arrivals) + ADD_LEVEL (leaves: t0 + saf); process in order of
    # readiness over all owners jointly (a link channel orders packets by readiness)
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
                heapq.heappush(ready, (t0 + saf(sizes[0]), next(seq), o, x))
    finish = {}
    while ready:
        t, s, o, x = heapq.heappop(ready)
        p = parent[o][x]
        link = (x, p)
        tcur = t
        last = 0.0
        for b in sizes:
            ser = (b + HDR) / BW
            start = links.setdefault(link, Link()).send(tcur, ser)
            lb[link] = lb.get(link, 0) + b
            lu[link] = lu.get(link, 0.0) + ser
            last = max(last, start + ser + HOP)
            tcur = tcur + saf(b)
        arr[o].setdefault(p, 0.0)
        arr[o][p] = max(arr[o][p], last)
        if p == o:
            finish[o] = last + ADD_LEVEL
            done = max(done, finish[o])
            continue
        pending[o][p] -= 1
        if pending[o][p] == 0:
            heapq.heappush(ready, (arr[o][p] + ADD_LEVEL + saf(sizes[0]), next(seq), o, p))
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
    """a gather (AG) of P per chip: made by the producing pool's hub (ORIGIN_GATHER), delivered at every chip from its
    receiving Ethernet tiles to all 130 cores (deliver)"""
    forms = {
        "AG_tree": ag_tree(mesh, P, t0, pkt, ORIGIN_GATHER)[0] + deliver(P),
        "AG_direct": ag_direct(mesh, P, t0, pkt, ORIGIN_GATHER)[0] + deliver(P),
    }
    best = min(forms, key=forms.get)
    return forms[best] - t0, best, {k: v - t0 for k, v in forms.items()}


def reduce_scatter(mesh, P, t0=0.0, pkt=PKT):
    forms = {"RS_tree": rs_tree(mesh, P, t0, pkt)[0], "RS_direct": rs_direct(mesh, P, t0, pkt)[0]}
    best = min(forms, key=forms.get)
    return forms[best] - t0, best, {k: v - t0 for k, v in forms.items()}


def ag_arrivals(mesh, P, dst, pkt=PKT):
    """arrival times at the cores of chip dst of every other chip's chunk (a gather: AG_tree from the hubs, delivered
    by the receiving Ethernet tiles' multicast), sorted; own chunk at 0 (the hub's own multicast)"""
    d, arr, _, _ = ag_tree(mesh, P, 0.0, pkt, ORIGIN_GATHER)
    return sorted([0.0] + [t + deliver(P) for (tag, leaf), t in arr.items() if leaf == dst])


def hand_over(flows, nbytes, pkt=PKT):
    """rows from a group's cores to the next group's cores: made by the producing cores (ORIGIN_ADD), delivered at the
    receiving chip to all its cores (deliver)"""
    return simulate_mc(flows, pkt, None, ORIGIN_ADD)[0] + deliver(nbytes)


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
