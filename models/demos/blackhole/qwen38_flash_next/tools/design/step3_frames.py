#!/usr/bin/env python
r"""Step 3 of the detailed design (worklog 12.3): the rounds and the frames of the chosen form, from capacity only.
The group of 8 chips streams its frames in turn; a frame's column (the chip's weights + rows + work + S + routed tiles
of the frame's units) must fit the chip's L1 that is left after the fixed reservations.  The unit sequence of one pass
(step 2, per chip) is cut into PP x R frames; frame f sits on group f mod PP; the cut can fall inside a unit only at the
unit's grain (a value head, a q head, a tile column, an expert tile) -- here the cut is first placed by bytes, then
moved to the nearest unit boundary (the 27B README section 8 rule: the overflow goes to the next frame).
Per chip L1: 130 cores x 1,536 KiB = 195.0 MiB; fixed: firmware 40 KiB + kernel binaries/config 100 KiB per core
= 17.8 MiB; L1-resident state (conv taps, PLE taps) 0.29 MiB; stream buffers (step 2 of the limit: matmul tiles 80 KiB,
GDR unit 48 KiB, KV chunk 64 KiB per core) 24.4 MiB -> frame capacity.
Outputs: the frames table (markdown) and step3_frames.json for steps 4-7."""
import json
import math
import sys

A = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step2_account.json"))
U = A["units"]
PP = 4
CORES = 130
KiB = 1024
MiB = 1024**2
L1_CORE = 1536 * KiB
FIXED_CORE = 140 * KiB
L1_CHIP = CORES * (L1_CORE - FIXED_CORE)
STATE_L1 = sum(x["state_l1"] for x in U)
# D-3 correction 12 (10-27 17:0x): the per-core landing areas (flags, the hub multicast group, the Stream-K partials,
# the meeting and hand-over landings: stages.l1_landing, the same sizes chip_l1.h lays out) come out of the capacity;
# the largest meeting payload and hand-over are step 6's (run_design.sh runs step 6 before step 3)
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import stages as SG  # noqa: E402

S6 = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
MEET_MAX = max(v["bytes"] for v in S6.values() if isinstance(v, dict) and "form" in v) if S6 else 0
HAND_MAX = max(h["bytes"] for h in S6["hand"]) if S6 else 0
# correction 15: the partial landing holds the largest fan-in of the design's Stream-K trees (step 4's pools)
T4 = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else []
FORM = sys.argv[5] if len(sys.argv) > 5 else "A"     # corrections 36-41: the frames of each hyper-connection form


def pools_of(i):
    t = T4[i]
    return t.get("pools_by_form", {}).get(FORM, t.get("pools", []))
FANIN = max([SG.pool_fanin(tuple(tuple(tuple(y) if isinstance(y, list) else y for y in m) for m in p)) for i in range(len(T4)) for p in pools_of(i)] or [0])
# correction 33-34: no stream buffers in section 12 (weights are the frame's columns, S units live in the frame, ctx 1
# has no KV chunk); the x receive area holds the whole next input of the largest-K pool
import pool_plan as PPL  # noqa: E402

def _multicast_input(i):
    """correction 45-46: a pool whose input is the residual (the hc read's down) reads it from L1 (the replicated
    residual in the B forms; the chip's residual columns in form A) and needs no x receive area"""
    return ".gr.down" not in U[i]["unit"] and not U[i]["unit"].startswith(("final hyper", "mtp.hyper"))


KMAX = max([sum(kt for kt, f in PPL.norm_mm(m)[2]) * 32 for i in range(len(T4)) if _multicast_input(i) for p in pools_of(i) for m in p] or [0])
STREAM = 0
LANDING_CORE = SG.landing_core(A["rows"], KMAX, FANIN, MEET_MAX, HAND_MAX)
LANDING = CORES * LANDING_CORE
REGION = L1_CORE - FIXED_CORE - LANDING_CORE        # the per-core frame region (chip_l1.h kFrameBytes)
CAP = L1_CHIP - LANDING - STATE_L1


def _stored(i, x):
    """correction 32: a unit's weight as stored (the pools' columns packed to 8-column groups and 16-column half tiles:
    pool_plan.stored_bytes); a unit costed inside another unit's pool stores nothing here; shared weights (the speculative
    router, mtp.lm_head) are stored once"""
    if "speculative router" in x["unit"] or x["unit"].startswith("mtp.lm_head"):
        return 0.0
    ps = pools_of(i) if T4 else []
    if ps:
        return sum(PPL.stored_bytes(tuple(tuple(tuple(y) if isinstance(y, list) else y for y in m) for m in p)) for p in ps)
    if T4 and "(in the" in T4[i].get("pool", ""):
        return 0.0
    return x["w"]


STORED = [_stored(i, x) for i, x in enumerate(U)]


_IDX = {id(x): i for i, x in enumerate(U)}


def col(x):
    """correction 42: a unit's lasting L1 in its frame = its stored weights + its GDN state S; its rows and work live only
    while it runs (inputs land in the landing areas, outputs go to the next unit's landing): the frame adds the largest
    work of its units once (frame_extra) and the residual (always resident)"""
    i = _IDX[id(x)]
    return STORED[i] + U[i]["S"]


RESIDUAL = A["rows"] * 4 * 2560 * 2          # the replicated residual [rows, 4H] bf16 (forms B; form A holds 1/8)


def frame_extra(us):
    return max((x["work"] for x in us), default=0.0) + RESIDUAL


def dram(x):
    i = _IDX[id(x)]
    return STORED[i] + 2 * U[i]["S"] + U[i]["dram_read"]  # the unit's DRAM traffic per pass on this chip (stored weights: correction 32)


total_col = sum(col(x) for x in U)
CAP_COL = CAP - frame_extra(U)              # the columns of a frame fit what the frame's lasting rows leave (an upper bound on the work)
total_dram = sum(dram(x) for x in U)
print(
    f"chip L1 {L1_CHIP/MiB:.1f} MiB after the fixed {CORES*FIXED_CORE/MiB:.1f}; stream buffers {STREAM/MiB:.1f}; landing areas {LANDING/MiB:.2f} (x receive 2 x rows x {KMAX} x 2 B, meeting max {MEET_MAX:.0f} B, hand-over max {HAND_MAX:.0f} B, Stream-K fan-in {FANIN}; frame region {REGION/KiB:.1f} KiB per core); resident state {STATE_L1/MiB:.2f}; frame capacity {CAP/MiB:.1f} MiB"
)
print(
    f"one pass column on this chip {total_col/MiB:,.1f} MiB (stored weights {sum(STORED)/MiB:,.1f}, rows+work {sum(x['in_']+x['out']+x['work'] for x in U)/MiB:,.1f}, S {sum(x['S'] for x in U)/MiB:.1f}); DRAM per pass {total_dram/MiB:,.1f} MiB"
)
print(
    f"frames needed by capacity: {total_col/CAP:.2f} -> {math.ceil(total_col/CAP)} frames = {math.ceil(total_col/CAP)/PP:.2f} rounds of {PP}"
)


def cut(nf):
    """cut the unit sequence into nf frames of at most CAP bytes each, cutting only at unit boundaries (a unit is one grain
    here; the big units -- lm_head 80.5 MiB on this chip -- fit a frame); returns the frames as (a, z) unit index ranges
    """
    frames = []
    a = 0
    acc = 0.0
    for i, x in enumerate(U):
        c = col(x)
        if c > CAP:
            raise SystemExit(f"unit does not fit a frame: {x['unit']} {c/MiB:.1f} MiB")
        if acc + c > CAP_COL:
            frames.append((a, i))
            a = i
            acc = 0.0
        acc += c
    frames.append((a, len(U)))
    return frames


def balance(nf):
    """nf frames with equal columns (the limit), cutting at unit boundaries: binary search the cap"""
    lo, hi = max(col(x) for x in U), total_col

    def fits(limit):
        cnt = 1
        acc = 0.0
        for x in U:
            c = col(x)
            if acc + c > limit:
                cnt += 1
                acc = 0.0
            acc += c
        return cnt <= nf

    for _ in range(60):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    limit = hi * (1 + 1e-9)
    frames = []
    a = 0
    acc = 0.0
    for i, x in enumerate(U):
        c = col(x)
        if acc + c > limit:
            frames.append((a, i))
            a = i
            acc = 0.0
        acc += c
    frames.append((a, len(U)))
    return frames, limit


best = None
import os
for R in ((int(os.environ['ROUNDS']),) if os.environ.get('ROUNDS') else (1, 2, 3, 4)):
    nf = PP * R
    fr, limit = balance(nf)
    if len(fr) < nf and limit <= CAP_COL:
        # the minimal-max cut used fewer frames than nf: split the largest frames at their middle unit until nf
        while len(fr) < nf:
            cand = [j for j in range(len(fr)) if fr[j][1] - fr[j][0] >= 2]
            if not cand:
                break
            k = max(cand, key=lambda j: sum(col(x) for x in U[fr[j][0]:fr[j][1]]))
            a0, b0 = fr[k]
            acc, m = 0.0, a0
            half = sum(col(x) for x in U[a0:b0]) / 2
            while m < b0 - 1 and acc + col(U[m]) <= half:
                acc += col(U[m])
                m += 1
            m = max(m, a0 + 1)
            fr[k:k + 1] = [(a0, m), (m, b0)]
    if len(fr) != nf or limit > CAP_COL:
        print(
            f"R={R}: {nf} frames -> max column {limit/MiB:.1f} MiB {'> ' if limit>CAP_COL else '<= '}capacity {CAP_COL/MiB:.1f} ({len(fr)} frames placed)"
        )
        continue
    print(f"R={R}: {nf} frames, max column {limit/MiB:.1f} MiB <= {CAP_COL/MiB:.1f}")
    if best is None:
        best = (R, fr, limit)
R, fr, limit = best
print(
    f"\nchosen: R={R} ({PP*R} frames); the smallest round count whose frames fit the L1 (the 27B rule: rounds from capacity, time later)\n"
)
print(
    "| frame | group | units (first .. last) | column [MiB] | weights | rows+work | S | routed tiles | DRAM/pass [MiB] | L1 left [MiB] |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
out = []
for f, (a, z) in enumerate(fr):
    us = U[a:z]
    c = sum(col(x) for x in us) + frame_extra(us)
    w = sum(STORED[_IDX[id(x)]] for x in us)
    rw = sum(x["in_"] + x["out"] + x["work"] for x in us)
    S = sum(x["S"] for x in us)
    rt = sum(STORED[_IDX[id(x)]] for x in us if "routed expert" in x["unit"])
    d = sum(dram(x) for x in us)
    print(
        f"| {f} | {f%PP} | `{us[0]['unit'][:38]}` .. `{us[-1]['unit'][:38]}` | {c/MiB:.1f} | {w/MiB:.1f} | {rw/MiB:.1f} | {S/MiB:.1f} | {rt/MiB:.1f} | {d/MiB:.1f} | {(CAP-c)/MiB:.1f} |"
    )
    out.append(
        dict(frame=f, group=f % PP, a=a, z=z, column=c, weights=w, rows_work=rw, S=S, routed=rt, dram=d, left=CAP - c, extra=frame_extra(us))
    )
grp = [sum(o["dram"] for o in out if o["group"] == g) for g in range(PP)]
print(
    f"\nper group (a chip of the group) DRAM per pass: {' / '.join(f'{g/MiB:,.1f}' for g in grp)} MiB -> {' / '.join(f'{g/512e9*1e3:.3f}' for g in grp)} ms at 512 GB/s; spread {(max(grp)-min(grp))/max(grp)*100:.1f}%"
)
print(
    f"the pass's DRAM floor on the heaviest chip: {max(grp)/512e9*1e3:.3f} ms; the 32 chips together move {sum(grp)*8/MiB:,.0f} MiB per token"
)
json.dump(
    dict(R=R, PP=PP, cap=CAP, l1_chip=L1_CHIP, stream=STREAM, form=FORM, stored=STORED, landing=LANDING, landing_core=LANDING_CORE, region=REGION, kmax=KMAX, meet_max=MEET_MAX, hand_max=HAND_MAX, fanin=FANIN, state_l1=STATE_L1, frames=out),
    open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step3_frames.json", "w"),
)
