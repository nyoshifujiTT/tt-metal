#!/usr/bin/env python
r"""Step 5 of the detailed design (worklog 12.5): the frames on the time axis.
The group of 8 chips holds 2 frames (R=2): frames g and g+4.  A pass: the token visits f0 (group 0) .. f7 (group 3).
While group g computes frame g, the other 3 groups are idle; while the token is in groups g+1..g+3 (3 frames) group g
is idle and can stream its NEXT frame's column into the L1 that frame g's column freed (the dense weights of a frame are
consumed in HF order; their L1 is free once the frame's chain is past them).  What cannot be prefetched while idle:
the routed experts of the next frame (they depend on this token's hidden state: read inside the window after the
speculative router one block earlier -- hit -- or after the router -- miss), the state S (read before the GDR, written
after it) and the n-gram/embedding rows.
Per frame f on group g, with the previous frame of the same group f' = f - 4 (or f + 4 of the previous pass):
  idle_f  = the time between the end of occ_{f'} and the start of occ_f = sum of the 3 other groups' occupancies in between
  prefetch_f = min(dense_f, idle_f x BW, L1 free during the idle time)  -- L1 free = CAP - (what frame f' leaves: nothing
              after it finished) - (frame f's rows/work/routed buffers) = CAP - (column_f - dense_f)
  stream_f (inside the window) = (dense_f - prefetch_f + routed_f + S2_f + reads_f) / BW
  occ_f   = max(stream_f, chain_f)          (compute consumes weights as they arrive; the chain is step 4's)
The pass is the fixed point of T = sum_f occ_f with idle_f = T - occ_f - occ_{f'} ... (3 frames between): iterate.
Also reported: the chip's DRAM per pass (everything it reads/writes), and the two bounds (nothing prefetched /
everything prefetched).  Collectives are added in step 6/7."""
import json
import sys

F = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/design/step3_frames.json"))
T = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/q38/design/step4_item_time.json"))
A = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "/tmp/q38/design/step2_account.json"))["units"]
BW = 512e9
MiB = 1024**2
PP = F["PP"]
R = F["R"]
CAP = F["cap"]
NF = PP * R
frames = F["frames"]
for f in frames:
    us = A[f["a"] : f["z"]]
    ts = T[f["a"] : f["z"]]
    f["chain_us"] = sum(t["t_us"] for t in ts if not t["spec"])
    f["dense"] = sum(x["w"] for x in us if "routed expert" not in x["unit"])
    f["rows"] = sum(x["in_"] + x["out"] + x["work"] for x in us)
    f["S2"] = 2 * sum(x["S"] for x in us)
    f["reads"] = sum(x["dram_read"] for x in us)
    f["window_fixed"] = (f["routed"] + f["S2"] + f["reads"]) / BW * 1e6  # must be inside the window whatever happens
    f["l1_free_idle"] = CAP - (
        f["column"] - f["dense"]
    )  # L1 for the prefetched dense while the frame's other buffers are reserved


def solve(prefetch_allowed=True, iters=50):
    """returns (T_pass, occupancies, per-frame notes); the notes are kept per solve (the first version wrote them into
    the frame dicts and the second solve overwrote the first's -- the 'prefetched 0.0 MiB' of the first print)"""
    occ = [max(f["window_fixed"] + f["dense"] / BW * 1e6, f["chain_us"]) for f in frames]
    notes = [dict(prefetch=0.0, stream_us=0.0, idle_us=0.0) for _ in frames]
    for _ in range(iters):
        new = []
        for i, f in enumerate(frames):
            # idle before frame i on its group: the frames of the other groups between f' and f: the 3 preceding frames
            idle = sum(occ[(i - k) % NF] for k in (1, 2, 3))
            pre = min(f["dense"], idle * 1e-6 * BW, max(0.0, f["l1_free_idle"])) if prefetch_allowed else 0.0
            stream = f["window_fixed"] + (f["dense"] - pre) / BW * 1e6
            new.append(max(stream, f["chain_us"]))
            notes[i] = dict(prefetch=pre, stream_us=stream, idle_us=idle)
        if max(abs(a - b) for a, b in zip(new, occ)) < 1e-3:
            occ = new
            break
        occ = new
    return sum(occ), occ, notes


Tp, occ, notes = solve(True)
Tn, occn, notesn = solve(False)
for f, n in zip(frames, notes):
    f.update(n)
dram = [
    sum((f["dense"] + f["routed"] + f["S2"] + f["reads"]) for f in frames if f["group"] == g) / BW * 1e6
    for g in range(PP)
]
print(
    f"frame capacity {CAP/MiB:.1f} MiB (L1 free for the prefetched dense while a frame's rows/routed are reserved: {min(f['l1_free_idle'] for f in frames)/MiB:.1f}..{max(f['l1_free_idle'] for f in frames)/MiB:.1f} MiB)"
)
print(
    "\n| frame | group | dense [MiB] | routed [MiB] | S x2 [MiB] | chain [us] | idle before [us] | prefetched [MiB] | stream in window [us] | occupancy [us] |"
)
print("|---|---|---|---|---|---|---|---|---|---|")
for i, f in enumerate(frames):
    print(
        f"| {f['frame']} | {f['group']} | {f['dense']/MiB:.1f} | {f['routed']/MiB:.1f} | {f['S2']/MiB:.1f} | {f['chain_us']:.1f} | {f['idle_us']:.1f} | {f['prefetch']/MiB:.1f} | {f['stream_us']:.1f} | {occ[i]:.1f} |"
    )
R_EFF = 1.376
print(
    f"\nT_pass (compute chain + the routed/S window, no collectives yet) = sum of the 8 occupancies = **{Tp:.1f} us** ({1e6/Tp*R_EFF:,.0f} tsu at k=1, R_eff {R_EFF}); without idle-time prefetch {Tn:.1f} us ({1e6/Tn*R_EFF:,.0f} tsu)"
)
print(
    f"chip DRAM per pass per group {' / '.join(f'{d:.1f}' for d in dram)} us (the DRAM lane is {max(dram)/Tp*100:.0f}% busy); the window is bound by {'the routed/S stream' if any(f['stream_us']>f['chain_us'] for f in frames) else 'the compute chain'} in {sum(1 for f in frames if f['stream_us']>f['chain_us'])} of 8 frames"
)
print(
    f"routed tiles inside the windows: {sum(f['routed'] for f in frames)/MiB:.1f} MiB = {sum(f['routed'] for f in frames)/BW*1e6:.1f} us of the {Tp:.1f} (hit case: streamed under the frame's compute from one block earlier); S round trip {sum(f['S2'] for f in frames)/BW*1e6:.1f} us"
)
# the per-layer view of the window: the routed tiles of a layer are read after the speculative router one block earlier,
# i.e. under the previous block's chain; they are hidden only if the layer's chain (+ its exposed collectives, step 6/7)
# is at least the routed read time
# the limit model's H-21 form (the buffer holds one layer's routed tiles ahead): a layer takes at least its routed
# bytes / BW, so T_layer = max(chain_l + comm_l, routed_l / BW); here comm_l = 0 (step 6 adds it).  This is stricter
# than max(stream, chain) over the whole frame (which lets a layer's routed read be spread over the frame).
layers = []
for u in A:
    if u["layer"] not in [x[0] for x in layers]:
        layers.append((u["layer"],))
per_layer = []
for (lay,) in layers:
    rt = sum(x["w"] for x in A if x["layer"] == lay and x["kind"] == "mm_expert") / BW * 1e6
    ch = sum(t["t_us"] for t in T if t["layer"] == lay and not t["spec"])
    per_layer.append((lay, rt, ch))
moe = [p for p in per_layer if p[1] > 0]
T_h21 = sum(max(p[1], p[2]) for p in per_layer) + sum(f["S2"] + f["reads"] for f in frames) / BW * 1e6
print(
    f"\nper MoE block ({len(moe)} incl. the MTP layer): routed read {min(p[1] for p in moe):.1f}..{max(p[1] for p in moe):.1f} us (busiest chip, census, all hit) vs the layer's compute chain {min(p[2] for p in moe):.1f}..{max(p[2] for p in moe):.1f} us (no collectives yet) -> the routed read is longer than the chain in {sum(1 for p in moe if p[1]>p[2])} of {len(moe)} layers; with the one-layer-ahead buffer (limit H-21) T_pass >= sum_l max(chain_l, routed_l/BW) + S/reads = **{T_h21:.1f} us** ({1e6/T_h21*R_EFF:,.0f} tsu); the collectives (step 6) lengthen the chain, the steady state (step 7) decides"
)
json.dump(
    dict(T_pass_us=Tp, T_noprefetch_us=Tn, occ=occ, dram=dram, frames=frames),
    open(sys.argv[4] if len(sys.argv) > 4 else "/tmp/q38/design/step5_frame_time.json", "w"),
)
