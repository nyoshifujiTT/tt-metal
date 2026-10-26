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
STREAM_CORE = (80 + 48 + 64) * KiB
L1_CHIP = CORES * (L1_CORE - FIXED_CORE)
STREAM = CORES * STREAM_CORE
STATE_L1 = sum(x["state_l1"] for x in U)
CAP = L1_CHIP - STREAM - STATE_L1


def col(x):
    return x["w"] + x["in_"] + x["out"] + x["work"] + x["S"]  # the unit's column on this chip


def dram(x):
    return x["w"] + 2 * x["S"] + x["dram_read"]  # the unit's DRAM traffic per pass on this chip


total_col = sum(col(x) for x in U)
total_dram = sum(dram(x) for x in U)
print(
    f"chip L1 {L1_CHIP/MiB:.1f} MiB after the fixed {CORES*FIXED_CORE/MiB:.1f}; stream buffers {STREAM/MiB:.1f}; resident state {STATE_L1/MiB:.2f}; frame capacity {CAP/MiB:.1f} MiB"
)
print(
    f"one pass column on this chip {total_col/MiB:,.1f} MiB (weights {sum(x['w'] for x in U)/MiB:,.1f}, rows+work {sum(x['in_']+x['out']+x['work'] for x in U)/MiB:,.1f}, S {sum(x['S'] for x in U)/MiB:.1f}); DRAM per pass {total_dram/MiB:,.1f} MiB"
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
        if acc + c > CAP:
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
for R in (1, 2, 3, 4):
    nf = PP * R
    fr, limit = balance(nf)
    if len(fr) != nf or limit > CAP:
        print(
            f"R={R}: {nf} frames -> max column {limit/MiB:.1f} MiB {'> ' if limit>CAP else '<= '}capacity {CAP/MiB:.1f} ({len(fr)} frames placed)"
        )
        continue
    print(f"R={R}: {nf} frames, max column {limit/MiB:.1f} MiB <= {CAP/MiB:.1f}")
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
    c = sum(col(x) for x in us)
    w = sum(x["w"] for x in us)
    rw = sum(x["in_"] + x["out"] + x["work"] for x in us)
    S = sum(x["S"] for x in us)
    rt = sum(x["w"] for x in us if "routed expert" in x["unit"])
    d = sum(dram(x) for x in us)
    print(
        f"| {f} | {f%PP} | `{us[0]['unit'][:38]}` .. `{us[-1]['unit'][:38]}` | {c/MiB:.1f} | {w/MiB:.1f} | {rw/MiB:.1f} | {S/MiB:.1f} | {rt/MiB:.1f} | {d/MiB:.1f} | {(CAP-c)/MiB:.1f} |"
    )
    out.append(
        dict(frame=f, group=f % PP, a=a, z=z, column=c, weights=w, rows_work=rw, S=S, routed=rt, dram=d, left=CAP - c)
    )
grp = [sum(o["dram"] for o in out if o["group"] == g) for g in range(PP)]
print(
    f"\nper group (a chip of the group) DRAM per pass: {' / '.join(f'{g/MiB:,.1f}' for g in grp)} MiB -> {' / '.join(f'{g/512e9*1e3:.3f}' for g in grp)} ms at 512 GB/s; spread {(max(grp)-min(grp))/max(grp)*100:.1f}%"
)
print(
    f"the pass's DRAM floor on the heaviest chip: {max(grp)/512e9*1e3:.3f} ms; the 32 chips together move {sum(grp)*8/MiB:,.0f} MiB per token"
)
json.dump(
    dict(R=R, PP=PP, cap=CAP, l1_chip=L1_CHIP, stream=STREAM, state_l1=STATE_L1, frames=out),
    open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step3_frames.json", "w"),
)
