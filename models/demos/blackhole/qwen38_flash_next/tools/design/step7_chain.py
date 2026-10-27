#!/usr/bin/env python
r"""Step 7 of the detailed design (worklog 12.7): the steady state of the chosen form, found by following the chain of
units through N tokens with the DRAM lanes, the L1 capacity and the collectives of steps 2-6 (physical values only).
Form: DP1 x PP4(R2) x 8chip[TP8 | EP1 x TPx8 tiles | DCP1], b=1, k=1 (rows 2), ctx 1; frames = step 3; unit times =
step 4 (the hyper-connection read's down/up/out_proj per the hc form of step 6); meetings and hand-offs = step 6;
routed sets = census_stage2.json.
Simulation (discrete events; b=1 so the token's chain is one sequential process):
  chain: units in HF order; a unit starts at the previous unit's end + the meeting exposure between them (+ the
         hand-off at a frame cut); it ends at max(start + t, last of its reads done + t/8) (the compute consumes the
         weights/tiles as they arrive and finishes one K-chunk after the last one).
  DRAM lane per chip group (512 GB/s; the 8 chips of a group read the same bytes at the same time): every read is
         cut into 512 KiB pieces (1 us: the issue granularity of the reader, so an urgent read waits at most one
         piece).  Two queues per lane in consumption order: the dense weights / S / rows (always triggered) and the
         routed tiles (triggered by their stage).  A piece is issued when the lane is free, its trigger has happened
         and the L1 has room for it without taking the room the earliest not-yet-issued read of the lane will need
         (27B principle 1: the next frame goes only into L1 the previous frame has freed; the check is
         live(items consumed at or after that read) + piece + one piece <= capacity).  Among the issuable heads the
         earlier deadline goes first.  An item occupies L1 from its issue until its unit ends; S until written back.
  routed stages per MoE layer l (busiest chip, tiles x 138,240 B / 20 per expert-tile; gate_up 2/3, down 1/3):
         P2 at pred2 time (router l on the residual at the start of layer l-1's attention block: that block's start +
         the speculative read's own hc read + its meetings + router + issue; across a frame cut: + the residual 1/8
         hop), P1 \ P2 at pred1 time (same at layer l's attention block), misses at the real router + issue latency
         (+ the hop if the experts sit on the next group).  Cases: hit (all tiles at the P2 time; L00's at the token's
         argmax of the previous pass + the token hop + the read; L01's at its pred1 time), census (the sets of
         census_stage2.json: wasted tiles are read too), miss (all after the router).
Outputs: the steady-state pass time (mean of the last half), its breakdown, the per-frame table of the last pass, the
lane utilisation, the L1 high-water mark; json for steps 8/9."""
import bisect
import json
import math
import sys

MiB = 1024**2
KiB = 1024
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
D = ARGS.get("dir", "/tmp/q38/design")
PFX = "" if D == "/tmp/q38/design" else "design_"
A = json.load(open(f"{D}/{PFX}step2_account.json"))
U = A["units"]
ROWS = A["rows"]
F = json.load(open(f"{D}/{PFX}step3_frames.json"))
FR = F["frames"]
CAP = F["cap"]
PP = F["PP"]
T4 = json.load(open(f"{D}/{PFX}step4_item_time.json"))
C6 = json.load(open(f"{D}/{PFX}step6_comm.json"))
CEN = json.load(open(ARGS.get("census", f"{D}/census_stage2.json")))["rows2"]
CASE = ARGS.get("case", "hit")
NPASS = int(ARGS.get("npass", "24"))
FORM = ARGS.get("form", "A")
S_RES = ARGS.get("s_resident", "0") == "1"
PIECE = int(ARGS.get("piece", str(512 * KiB)))
BW = 512e9
EXPERT_TILE = 138240.0
GU = 2 / 3
DN = 1 / 3
FM = C6["forms"][FORM]
PARTS = C6["parts"]
AGARG = C6["AG_argmax"]["us"]
HAND = {h["cut"]: h["us"] + (C6["hand_B_extra"] if FORM != "A" else 0.0) for h in C6["hand"]}
HOP_RES = PARTS["res_hop"]
ISSUE = 0.16
# the speculative router and its hc read (the down + up matmuls of the hyper-connection read), from step 4 (revision 6)
SPEC = next(t["t_us"] for t in T4 if t["unit"].startswith("L00.moe.speculative router"))
HC_READ = next(t["t_us"] for t in T4 if t["unit"].startswith("L00.moe.gr.down")) + next(
    t["t_us"] for t in T4 if t["unit"].startswith("L00.moe.gr.up")
)
SPEC_MEET = PARTS["spec_meet_A"] if FORM == "A" else PARTS["spec_meet_B"]
SPEC_LOCAL = HC_READ + SPEC_MEET + SPEC
SPEC_REMOTE = HOP_RES + SPEC_LOCAL
TOKHOP = C6["hand"][0]["us"]
ARGMAX_LOCAL = next(t["t_us"] for t in T4 if t["unit"].startswith("argmax"))
AGX_FULL = C6["AG_x"]["us"]
EXTRA_MiB = FM["extra_MiB"]
IDS_HOP = 1.04
spec_binds = [0, 0]  # (predictions whose weights were not yet in L1, total)
N = len(U)
frame_of = [None] * N
cut_before = [None] * N
for f in FR:
    for i in range(f["a"], f["z"]):
        frame_of[i] = f["frame"]
    cut_before[f["a"]] = f["frame"]


def group_of(i):
    return frame_of[i] % PP


def layer_of(i):
    return U[i]["layer"]


layers = []
for i in range(N):
    l = layer_of(i)
    if (l.startswith("L") or l == "mtp") and l not in layers:
        layers.append(l)
LIDX = {l: k for k, l in enumerate(layers)}
gate_unit = {}
router_unit = {}
gu_unit = {}
dn_unit = {}
attn_start = {}
moe_start = {}
argmax_unit = next(i for i, x in enumerate(U) if x["unit"].startswith("argmax"))
for i, x in enumerate(U):
    n = x["unit"]
    l = layer_of(i)
    if l not in LIDX:
        continue
    if (".gdn.gr.norm" in n or ".qsa.gr.norm" in n) and l not in attn_start:
        attn_start[l] = i
    if ".gdn.gr.gate" in n or ".qsa.gr.gate" in n:
        gate_unit[l] = i
    if ".moe.router" in n and "speculative" not in n:
        router_unit[l] = i
    if ".moe.gr.norm" in n and l not in moe_start:
        moe_start[l] = i
    if "routed expert e: gate_up" in n:
        gu_unit[l] = i
    if "routed expert e: down" in n:
        dn_unit[l] = i


def cen(key, k):
    return CEN[key][min(k, 47)]


def block_of(i):
    n = U[i]["unit"]
    return "attn" if (".gdn." in n or ".qsa." in n) else ("moe" if ".moe." in n else None)


def comm_after(i):
    n = U[i]["unit"]
    b = block_of(i)
    if ".gr.down" in n:
        return FM[b]["meet_low"] if b else PARTS["ar_down"]
    if ".gr.gate x normalized" in n:
        nxt = U[i + 1]["unit"]
        return (
            0.0
            if FORM == "B+D+U"
            else (PARTS["ag_gdn"] if ".gdn." in nxt else PARTS["ag_qsa"] if ".qsa." in nxt else PARTS["ag_moe"])
        )
    if ".out_proj" in n or ".o_proj" in n:
        return FM["attn"]["meet_out"]
    if "routed expert e: down" in n:
        return FM["moe"]["meet_out"]
    if ".ple.norm_query" in n or ".ple.norm_conv" in n:
        return PARTS["ar_down_full"]
    if n.startswith("final hyper_connection_mixer") or n.startswith("mtp.hyper_connection_mixer"):
        return PARTS["ar_down_full"] + AGX_FULL
    if n.startswith("argmax"):
        return AGARG
    if n.startswith("mtp.lm_head"):
        return ARGMAX_LOCAL + AGARG + TOKHOP
    return 0.0


def dur(i):
    if T4[i]["spec"]:
        return 0.0
    n = U[i]["unit"]
    b = block_of(i)
    if b and ".gr.down" in n:
        return FM[b]["down"]
    if b and ".gr.up" in n:
        return FM[b]["up"]
    if ".out_proj" in n or ".o_proj" in n:
        return FM["attn"]["out"]
    return T4[i]["t_us"]


STAGES = {}
for l in layers:
    k = LIDX[l]
    if l not in gu_unit:
        continue
    if CASE == "miss":
        STAGES[l] = {"P2": 0.0, "P1": 0.0, "miss": cen("busy_A", k)}
    elif k == 0:
        STAGES[l] = {
            "P2": cen("busy_A", k),
            "P1": 0.0,
            "miss": 0.0,
        }  # exact router on the known embedding, at the pass start
    elif l == "mtp":
        STAGES[l] = (
            {"P2": 0.0, "P1": cen("busy_A", k), "miss": 0.0}
            if CASE == "hit"
            else {"P2": 0.0, "P1": cen("busy_P1mP2", k) + cen("busy_P2", k), "miss": cen("busy_miss", k)}
        )
    elif CASE == "hit":
        STAGES[l] = {"P2": cen("busy_A", k), "P1": 0.0, "miss": 0.0}
    else:
        if k == 1:
            STAGES[l] = {"P2": 0.0, "P1": cen("busy_P1mP2", k), "miss": cen("busy_miss", k)}
        else:
            STAGES[l] = {"P2": cen("busy_P2", k), "P1": cen("busy_P1mP2", k), "miss": cen("busy_miss", k)}
EXTRA_PER_LAYER = EXTRA_MiB * MiB


def extra_of(i):
    n = U[i]["unit"]
    b = block_of(i)
    return EXTRA_PER_LAYER if (b == "attn" and ".gr.down" in n and FORM != "A") else 0.0


# ---------------- lanes ----------------
class Lane:
    def __init__(s, g):
        s.g = g
        s.free = 0.0
        s.q = []
        s.rq = []
        s.busy = 0.0
        s.l1 = {}  # l1: key -> [free_time, bytes, issue_time, dl]

    def occupied(s, t):
        return sum(v[1] for v in s.l1.values() if v[0] > t and v[2] <= t)


lanes = [Lane(g) for g in range(PP)]
if S_RES:
    for g in range(PP):
        lanes[g].l1[("S", "resident")] = [math.inf, sum(x["S"] for i, x in enumerate(U) if group_of(i) == g), 0.0, -1.0]
reads = {}
seq = [0]


def add_read(key, g, nbytes, trig, deadline, unit, dep=None, hold=True):
    """a read of nbytes (L1 room = nbytes if hold) cut into pieces; returns the piece keys"""
    if nbytes <= 0:
        return []
    n = max(1, math.ceil(nbytes / PIECE))
    left = nbytes
    keys = []
    for c in range(n):
        b = min(PIECE, left)
        left -= b
        k = key + (c,)
        reads[k] = dict(
            key=k,
            bytes=b,
            l1=(b if hold else 0.0),
            trig=trig,
            dep=dep,
            done=None,
            issue=None,
            unit=unit,
            lane=g,
            dl=deadline,
        )
        lane = lanes[g]
        q = lane.rq if key[2] in ("P2", "P1", "miss") else lane.q
        bisect.insort(q, (deadline, seq[0], k))
        seq[0] += 1
        keys.append(k)
    return keys


def trig_of(r):
    if r["dep"] is not None:
        d = reads.get(r["dep"])
        if d is None or d["done"] is None:
            return None
        return max(r["trig"] or 0.0, d["done"] + r.get("dep_add", 0.0))
    return r["trig"]


def heads(lane):
    out = []
    if lane.q:
        out.append(lane.q[0])
    for e in lane.rq:
        if trig_of(reads[e[2]]) is not None:
            out.append(e)
            break
    return out


ROUTED_LOOKAHEAD = int(
    ARGS.get("lookahead", "3")
)  # the routed buffer kept free ahead of the chain: this many MoE layers' routed reads


def reserve_of(lane, dl):
    """the L1 the lane must keep free for the unissued routed reads earlier in consumption order than deadline dl:
    the first ROUTED_LOOKAHEAD layers' unissued routed reads (the routed tiles of a layer are read ~1.5 layers ahead
    and freed at the layer's end, so at most this many layers' tiles are live together); returns (min_dl, bytes)"""
    tot = 0.0
    first = None
    layers_seen = []
    for d, s, key in lane.rq:
        if d >= dl:
            break
        r = reads[key]
        if r["issue"] is not None:
            continue
        lay = layer_of(key[1])
        if lay not in layers_seen:
            if len(layers_seen) >= ROUTED_LOOKAHEAD:
                break
            layers_seen.append(lay)
        if first is None:
            first = d
        tot += r["l1"]
    return first, tot


def room_time(lane, r, t0, protect):
    """earliest t >= t0 at which piece r fits, leaving room for the protected reads: live(items with dl >= their first
    deadline) + r + reserve <= CAP"""
    items = [(v[0], v[1], v[3]) for v in lane.l1.values() if v[0] > t0]
    cands = [t0] + sorted({ft for ft, b, dl in items if ft != math.inf and ft > t0})
    pdl, res = protect
    for t in cands:
        live = [(b, dl) for ft, b, dl in items if ft > t]
        if sum(b for b, dl in live) + r["l1"] > CAP:
            continue
        if pdl is not None:
            later = sum(b for b, dl in live if dl >= pdl)
            if later + r["l1"] + res > CAP:
                continue
        return t
    return None


def try_issue(now):
    tmin = math.inf
    for lane in lanes:
        while True:
            hs = heads(lane)
            if not hs:
                break
            best = None
            for dl, s, key in hs:
                r = reads[key]
                tr = trig_of(r)
                if tr is None:
                    continue
                t0 = max(lane.free, tr)
                t1 = room_time(lane, r, t0, reserve_of(lane, dl))
                if t1 is None:
                    continue
                if best is None or (t1, dl) < (best[0], best[1]):
                    best = (t1, dl, s, key)
            if best is None:
                break
            t1, dl, s, key = best
            if t1 > now:
                tmin = min(tmin, t1)
                break
            r = reads[key]
            q = lane.rq if key[2] in ("P2", "P1", "miss") else lane.q
            q.remove((dl, s, key))
            r["issue"] = t1
            r["done"] = t1 + r["bytes"] / BW * 1e6
            lane.free = r["done"]
            lane.busy += r["bytes"] / BW * 1e6
            if key[2] == "Sw":
                for c in range(64):
                    kk = (key[0], key[1], "S", c)
                    if kk in lane.l1:
                        lane.l1[kk][0] = r["done"]
            elif r["l1"] > 0:
                lane.l1[key] = [math.inf, r["l1"], t1, dl]
    return tmin


# ---------------- run ----------------
prev_start = {}
unit_reads = {}


def deadline_of(p, i):
    return p * 1e6 + prev_start.get(i, i * 0.5)


def schedule_pass_reads(p):
    for i, x in enumerate(U):
        g = group_of(i)
        keys = []
        keys += add_read(
            (p, i, "w"),
            g,
            x["w"] + x["dram_read"] + extra_of(i) + x["in_"] + x["out"] + x["work"],
            0.0,
            deadline_of(p, i),
            i,
        )  # the unit's column (rows/work are L1 room without DRAM bytes: counted as bytes here for simplicity of order; they are < 1% of the weights)
        if x["S"] > 0 and not S_RES:
            keys += add_read(
                (p, i, "S"), g, x["S"], 0.0, deadline_of(p, i), i, dep=((p - 1, i, "Sw", 0) if p > 0 else None)
            )
        l = layer_of(i)
        if l in STAGES and i in (gu_unit.get(l), dn_unit.get(l)):
            frac = GU if i == gu_unit[l] else DN
            for st in ("P2", "P1", "miss"):
                tiles = STAGES[l][st]
                if tiles > 0:
                    keys += add_read((p, i, st), g, tiles * EXPERT_TILE * frac, None, deadline_of(p, i), i)
        unit_reads[(p, i)] = keys


def set_trig(p, l, st, t, dep=None):
    for i in (gu_unit.get(l), dn_unit.get(l)):
        for k, r in reads.items():
            if k[0] == p and k[1] == i and k[2] == st and r["trig"] is None:
                r["trig"] = t
                r["dep"] = dep
                r["dep_add"] = (
                    SPEC_LOCAL + ISSUE + (IDS_HOP if group_of(gu_unit[l]) != group_of(router_unit[l]) else 0.0)
                )


def spec_time(p, l, t_start, g_start):
    """the time the speculative router of layer l (fed by the residual as it is at t_start on group g_start) has its
    result at the experts' group: the prediction runs on the group holding l's router (residual hop if different),
    after its weights (l's MoE hc read + router) are in that group's L1; the ids hop to the experts' group if different
    """
    g_r = group_of(router_unit[l])
    g_x = group_of(gu_unit[l])
    t = t_start + (HOP_RES if g_r != g_start else 0.0)
    wkeys = [k for k in reads if k[0] == p and k[2] == "w" and moe_start[l] <= k[1] <= router_unit[l]]
    w_done = max([reads[k]["done"] for k in wkeys if reads[k]["done"] is not None], default=0.0)
    pending = [k for k in wkeys if reads[k]["done"] is None]
    spec_binds[1] += 1
    if w_done > t or pending:
        spec_binds[0] += 1
    t = max(t, w_done) + SPEC_LOCAL + (IDS_HOP if g_x != g_r else 0.0) + ISSUE
    return t, (max(pending) if pending else None)


pass_times = []
frame_rows = []
l1_hwm = [0.0] * PP
waits_by_kind = []
spec_load = {}
busy_by_pass = []
now = 0.0
T_end = 0.0
t_argmax_prev = None
schedule_pass_reads(0)
for p in range(NPASS):
    if p + 1 < NPASS:
        schedule_pass_reads(p + 1)
    T_start = T_end if p > 0 else 0.0
    t_prev_end = T_start
    comm_prev = 0.0
    rows_this = []
    wk = {"dense": 0.0, "S": 0.0, "routed": 0.0}
    big_waits = []
    busy0 = [lane.busy for lane in lanes]
    if "L00" in STAGES:
        set_trig(
            p, "L00", "P2", T_start + PARTS["down_r"] + PARTS["up_r"] + SPEC + ISSUE
        )  # the embedding row is on every chip: whole-row hc read (D and U replicated, step 6) + router, no meeting
    for f in FR:
        f_start = None
        comp = 0.0
        expo = 0.0
        wait = 0.0
        f_end = None
        hand = 0.0
        for i in range(f["a"], f["z"]):
            x = U[i]
            l = layer_of(i)
            g = group_of(i)
            t_nom = t_prev_end + comm_prev + (HAND[cut_before[i]] if (cut_before[i] is not None and i > 0) else 0.0)
            if cut_before[i] is not None and i > 0:
                hand += HAND[cut_before[i]]
            keys = unit_reads.get((p, i), [])
            while any(reads[k]["done"] is None for k in keys):
                tmin = try_issue(now)
                if all(reads[k]["done"] is not None for k in keys):
                    break
                if tmin == math.inf:
                    raise SystemExit(
                        f"deadlock at pass {p} unit {i} {x['unit'][:60]}: L1 {[round(lane.occupied(now)/MiB,1) for lane in lanes]} MiB, trig {[trig_of(reads[k]) for k in keys][:4]}"
                    )
                now = max(now, tmin)
            d = dur(i)
            start = t_nom
            t_data = max([reads[k]["done"] for k in keys], default=0.0)
            end = max(start + d, (t_data + d / 8) if keys else 0.0)
            rw = max(0.0, end - (start + d))
            wait += rw
            comp += d
            expo += comm_prev
            if rw > 0:
                lastk = max(keys, key=lambda k: reads[k]["done"])
                kind = "S" if lastk[2] == "S" else ("dense" if lastk[2] == "w" else "routed")
                wk[kind] += rw
                big_waits.append(
                    (
                        round(rw, 1),
                        i,
                        x["unit"][:55],
                        kind,
                        round(reads[lastk]["issue"] - T_start, 1),
                        round(reads[lastk]["done"] - T_start, 1),
                        round(start - T_start, 1),
                        trig_of(reads[lastk]) and round(trig_of(reads[lastk]) - T_start, 1),
                    )
                )
            if f_start is None:
                f_start = start
            lane = lanes[g]
            for k in keys:
                if k[2] != "S" and k in lane.l1:
                    lane.l1[k][0] = end
            if x["S"] > 0 and not S_RES:
                add_read((p, i, "Sw"), g, x["S"], end, end, i, hold=False)
            if i == attn_start.get(l):
                k = LIDX[l]
                nxt = layers[k + 1] if k + 1 < len(layers) else None
                if l in gu_unit and k > 0:
                    t1, dep1 = spec_time(p, l, start, g)
                    set_trig(p, l, "P1", t1, dep1)
                    if (k == 1 and CASE == "hit") or l == "mtp":
                        set_trig(p, l, "P2", t1, dep1)
                if nxt in gu_unit and nxt != "mtp":
                    t2, dep2 = spec_time(p, nxt, start, g)
                    set_trig(p, nxt, "P2", t2, dep2)
                spec_load[l] = 2 * (HC_READ + SPEC)
            if i == router_unit.get(l) and l in gu_unit:
                same = group_of(gu_unit[l]) == g
                set_trig(p, l, "miss", end + ISSUE + (0.0 if same else HOP_RES))
            if i == argmax_unit:
                t_argmax_prev = end + comm_after(i)
            now = max(now, end)
            t_prev_end = end
            comm_prev = comm_after(i)
            f_end = end
            prev_start[i] = start - T_start
            try_issue(now)
        rows_this.append(
            dict(
                frame=f["frame"],
                group=f["group"],
                start=f_start - T_start,
                end=f_end - T_start,
                comp=comp,
                expo=expo,
                wait=wait,
                hand=hand,
            )
        )
    T_end = t_prev_end + comm_prev
    pass_times.append(T_end - T_start)
    frame_rows.append(rows_this)
    waits_by_kind.append(wk)
    busy_by_pass.append([lane.busy - b0 for lane, b0 in zip(lanes, busy0)])
    for g in range(PP):
        l1_hwm[g] = max(l1_hwm[g], max(lanes[g].occupied(T_start + k * (T_end - T_start) / 64) for k in range(65)))
    now = max(now, T_end)
    for lane in lanes:
        for k in [k for k, v in lane.l1.items() if v[0] != math.inf and v[0] < now - 2 * pass_times[-1]]:
            del lane.l1[k]
    for key in [k for k in reads if k[0] < p - 1]:
        del reads[key]
half = NPASS // 2
steady = pass_times[half:]
Tss = sum(steady) / len(steady)
R_EFF = 1.376
print(
    f"# step 7: steady state, case {CASE}, form {FORM}{', S resident in L1' if S_RES else ''}, {NPASS} passes, pieces of {PIECE//KiB} KiB\n"
)
print(
    f"pass times [us]: first {pass_times[0]:.1f}, last {len(steady)} mean **{Tss:.1f}** (min {min(steady):.1f} max {max(steady):.1f}) -> **{1e6/Tss*R_EFF:,.0f} tsu** (k=1, R_eff {R_EFF})"
)
last = frame_rows[-1]
print(
    f"\n| frame | group | compute [us] | start -> end [us] (width) | meetings exposed [us] | read waits [us] | hand-off [us] |"
)
print("|---|---|---|---|---|---|---|")
for r in last:
    print(
        f"| {r['frame']} | {r['group']} | {r['comp']:.1f} | {r['start']:.1f} -> {r['end']:.1f} ({r['end']-r['start']:.1f}) | {r['expo']:.1f} | {r['wait']:.1f} | {r['hand']:.2f} |"
    )
tc = sum(r["comp"] for r in last)
te = sum(r["expo"] for r in last)
tw = sum(r["wait"] for r in last)
th = sum(r["hand"] for r in last)
spec_total = sum(spec_load.values())
print(
    f"speculative predictions whose weights were not yet in the predicting group's L1: {spec_binds[0]} of {spec_binds[1]} (the trigger waited for the weight read)"
)
print(
    f"\nspeculative routers (off the chain): {spec_total:.0f} us of compute per pass (2 x (hc read {HC_READ} + router {SPEC}) per attention block) against the chain's idle time (meeting exposure + read waits) {te+tw:.0f} us: {'fits' if spec_total<=te+tw else 'does NOT fit -> add the excess to the chain'}"
)
wk = waits_by_kind[-1]
print(
    f"\nlast pass: compute {tc:.1f} + meetings {te:.1f} + read waits {tw:.1f} (dense {wk['dense']:.1f}, S {wk['S']:.1f}, routed {wk['routed']:.1f}) + hand-offs {th:.1f} + wrap {pass_times[-1]-tc-te-tw-th:.1f} = {pass_times[-1]:.1f} us"
)
big_waits.sort(reverse=True)
print("\nbiggest read waits of the last pass (wait us, unit, kind, read issue -> done, unit start, trigger):")
for w in big_waits[:8]:
    print("  ", list(w))
T_window = sum(steady)
busy = [sum(b[g] for b in busy_by_pass[half:]) for g in range(PP)]
print(
    f"DRAM lanes (last {len(steady)} passes): "
    + " / ".join(f"group {g} {busy[g]/T_window*100:.0f}%" for g in range(PP))
    + f"; L1 high-water mark per group {[round(h/MiB,1) for h in l1_hwm]} MiB of {CAP/MiB:.1f}"
)
tiles = sum(sum(STAGES[l].values()) for l in STAGES)
rb = tiles * EXPERT_TILE
print(
    f"routed tiles read per pass (busiest chip): {tiles:.0f} tiles = {rb/MiB:.1f} MiB = {rb/BW*1e6:.1f} us of lane time; the form's extra dense {EXTRA_MiB*49/4:.1f} MiB per chip per pass"
)
json.dump(
    dict(
        case=CASE,
        form=FORM,
        s_resident=S_RES,
        piece=PIECE,
        T_steady=Tss,
        pass_times=pass_times,
        frames=last,
        busy=[b / T_window for b in busy],
        l1_hwm=l1_hwm,
        waits=wk,
        comp=tc,
        expo=te,
        wait=tw,
        hand=th,
        spec_total=spec_total,
        tiles=tiles,
    ),
    open(f"{D}/{PFX}step7_chain_{CASE}_{FORM}{'_S' if S_RES else ''}.json", "w"),
)
