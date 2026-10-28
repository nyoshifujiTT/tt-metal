#!/usr/bin/env python
r"""Step 3p of the detailed design (corrections 30-35, worklog 10-27 19:xx): the physical placement of every matmul pool
of every frame on each of the 14 harvest cases, the pool times on it, and the per-core L1 check that decides the
frames.
- Per core L1 (chip_l1.h): 1,536 KiB - fixed 140 KiB (firmware 40 + kernel 100) - the landing areas (stages.landing_core:
  flags, the x receive area (2 x rows x the largest pool K x 2 B: the whole next input, no credit ring), the Stream-K
  partials (fan-in x rows x 32 x 4 B), the meeting landing (2 x the largest meeting), the hand-over landing) = the frame
  region.  Correction 33: no separate 'stream buffers' (the weights are the frame's columns, the S units live in the
  frame, ctx 1 has no KV chunks).
- A frame's per-core load = its pools' stored bytes on the core (placement.place_frame_opt: the rotations of the raster
  path that minimize the pools' total time subject to every core <= the region) + the frame's non-weight columns (rows,
  work, S, L1-resident state) spread over the cores + the frame's non-pool weights (element units' small weights).
- The rounds: the smallest R whose step-3 cut (balanced by bytes) fits on all 14 harvest cases (27B rule: rounds from
  capacity); for that R the frames are re-cut by moving a boundary unit to the next frame while a frame overflows.
- The design value of a pool = its worst time over the 14 harvest cases.
Inputs: step 2, step 3 (its frame cut for the chosen R), step 4 (pools), step 6 (meeting payloads), step 6b's bound.
Outputs: design_step3p_place.json: per frame the frame's cut, per harvest the per-core max, per unit the placed time
(worst harvest) and per (harvest, pool) the rotation; markdown."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import placement as PL  # noqa: E402
import pool_plan as PP  # noqa: E402
import stages as SG  # noqa: E402

D = sys.argv[1]
FORM = sys.argv[2] if len(sys.argv) > 2 else "A"
A = json.load(open(f"{D}/design_step2_account.json"))
F = json.load(open(f"{D}/design_step3_frames_{FORM}.json"))
T4 = json.load(open(f"{D}/design_step4_item_time.json"))
U = A["units"]
ROWS = A["rows"]
KiB = 1024
REGION = F["region"]


def pools_of(i):
    return T4[i].get("pools_by_form", {}).get(FORM, T4[i].get("pools", []))


def frame_pools(a, z):
    out = []
    for i in range(a, z):
        u = U[i]["unit"]
        if "speculative router" in u or u.startswith("mtp.lm_head"):
            continue
        for p in pools_of(i):
            out.append((i, PL.tup(p)))
        if ".gated delta rule" in u and U[i]["S"] > 0:
            # correction 54: the layer's S on the cores of its GDR units (6 value heads x 16 units of 8 columns)
            out.append((i, ("S", float(U[i]["S"]), GDR_UNITS)))
    return out


GDR_UNITS = 6 * 16


def frame_base(a, z):
    nonpool = 0.0
    for i in range(a, z):
        u = U[i]
        if "speculative router" in u["unit"] or u["unit"].startswith("mtp.lm_head"):
            continue
        if not pools_of(i) and "(in the" not in T4[i].get("pool", ""):
            nonpool += u["w"]
    # correction 42: the frame's lasting non-weight L1 = its S + its L1-resident state + the residual + its largest work
    fr = next(f for f in F["frames"] if f["a"] == a and f["z"] == z)
    return (sum(U[i]["state_l1"] for i in range(a, z)) + fr["extra"] + nonpool) / 130      # correction 54: S is placed


def run(args):
    harv, a, z = args
    pools = frame_pools(a, z)
    L, out, fits = PL.place_frame_fit(pools, harv, REGION, frame_base(a, z))
    return harv, a, z, max(L), fits, [(i, o, t, sch, fan) for i, mms, o, t, sch, fan in out]   # sch "S": a GDN state pseudo-pool


def main():
    frames = [(f["a"], f["z"]) for f in F["frames"]]
    jobs = [(h, a, z) for (a, z) in frames for h in PL.XS]
    with ProcessPoolExecutor(max_workers=min(56, os.cpu_count() or 8)) as ex:
        res = list(ex.map(run, jobs))
    per_frame = {}
    for harv, a, z, mx, fits, out in res:
        per_frame.setdefault((a, z), {})[harv] = dict(max=mx, fits=fits, pools=out)
    unit_t = {}
    for (a, z), hs in per_frame.items():
        for harv, d in hs.items():
            for i, o, t, sch, fan in d["pools"]:
                unit_t.setdefault(i, {}).setdefault(harv, 0.0)
                unit_t[i][harv] += t
    worst = {i: max(v.values()) for i, v in unit_t.items()}
    model = {}
    for i in unit_t:
        model[i] = sum(PP.plan(PL.tup(p))["t"] for p in pools_of(i))   # S pseudo-pools add 0
    allfit = all(d["fits"] for hs in per_frame.values() for d in hs.values())
    print(f"frame region {REGION/KiB:.1f} KiB per core; {len(frames)} frames x 14 harvest cases; all fit: {allfit}\n")
    print("| frame | units | per-core max over the harvests [KiB] | fits | pools' time placed (worst harvest) / model [cycles] |")
    print("|---|---|---|---|---|")
    out = []
    for k, (a, z) in enumerate(frames):
        hs = per_frame[(a, z)]
        mx = max(d["max"] for d in hs.values())
        tp = sum(worst.get(i, 0.0) for i in range(a, z))
        tm = sum(model.get(i, 0.0) for i in range(a, z))
        print(f"| {k} | {a}..{z-1} | {mx/KiB:,.1f} | {all(d['fits'] for d in hs.values())} | {tp:,.0f} / {tm:,.0f} (+{(tp/max(tm,1)-1)*100:.1f}%) |")
        out.append(dict(frame=k, a=a, z=z, max=mx, fits=all(d["fits"] for d in hs.values()),
                        rot={str(h): [(i, o) for i, o, t, sch, fan in d["pools"] if sch != "S"] for h, d in hs.items()},
                        srot={str(h): [(i, o) for i, o, t, sch, fan in d["pools"] if sch == "S"] for h, d in hs.items()}))
    json.dump(dict(form=FORM, region=REGION, allfit=allfit, frames=out, unit_t={str(i): v for i, v in worst.items()},
                   unit_model={str(i): v for i, v in model.items()}), open(f"{D}/design_step3p_place_{FORM}.json", "w"))


if __name__ == "__main__":
    main()
