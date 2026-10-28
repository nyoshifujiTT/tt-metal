#!/usr/bin/env python
r"""Routing census for the two-stage speculative prefetch of the detailed design (worklog 12.5 addendum / 12.6):
  stage 2 (two blocks early, predictor pred2 = layer l's router on the residual after layer l-2): the set P2 is read
          during layer l-1 (window ~ one layer);
  stage 1 (one block early, pred1 = layer l's router on the residual after layer l-1): the set P1 \ P2 is read during
          layer l's attention block;
  after the router: the misses A \ (P1 u P2) are read after the issue latency;
  wasted: (P1 u P2) \ A are read and not used (DRAM lane only).
All sets are per layer, over the rows of one decode item (rows = k+1 = 2: the verify row and the draft row of one user,
B=1), tiles form EP1 x TPx8 (expert e's 20 column tiles round-robin from chip e mod 8): the busiest chip's tile count of
each set, and the per-chip mean.  Layers 0 and 1 have no pred2 (layer 1 no pred1): their P2 (P1) is empty, so the whole
set is read one block early (layer 1) or after the router (layer 0) -- the design reads layer 0's experts after its
router and layer 1's during its attention block.
Also: per-layer hit rates (for the README), and the same for rows 1 (k=0).
Input: /data/yoshifuji/qwen38/routing/*.npz (indices, pred1, pred2, kind); output: census_stage2.json."""
import glob
import json
import os
import sys

import numpy as np

E = 512
L = 48
K = 10
N_TILES = 20


def busiest_tiles(experts, ep, tpx):
    if not experts:
        return 0, np.zeros(ep * tpx, dtype=np.int64)
    e = np.asarray(sorted(experts), dtype=np.int64)
    g = e // (E // ep)
    j = np.arange(N_TILES)
    idx = (g[:, None] * tpx + (e[:, None] + j[None, :]) % tpx).ravel()
    cnt = np.bincount(idx, minlength=ep * tpx)
    return int(cnt.max()), cnt


def load(paths):
    items = {}
    for p in paths:
        z = np.load(p)
        nm = os.path.basename(p)[:-4]
        if "pred1" not in z.files or (z["kind"] == 1).sum() < 8:
            continue
        k = z["kind"] == 1
        items[nm] = dict(
            ix=z["indices"][k].astype(np.int32), p1=z["pred1"][k].astype(np.int32), p2=z["pred2"][k].astype(np.int32)
        )
    return items


def main(rdir="/data/yoshifuji/qwen38/routing", out="/data/yoshifuji/qwen38/routing/census_stage2.json"):
    paths = sorted(p for p in glob.glob(rdir + "/*.npz") if "/smoke-" not in p)
    items = load(paths)
    names = sorted(items)
    print("items", len(names))
    res = {}
    for rows in (1, 2):
        per_layer = {
            k: np.zeros(L)
            for k in (
                "A",
                "P2",
                "P1mP2",
                "miss",
                "wasted",
                "hit2",
                "hit12",
                "busy_A",
                "busy_P2",
                "busy_P1mP2",
                "busy_miss",
                "busy_wasted",
                "busy_total_read",
            )
        }
        n = np.zeros(L)
        per_chip = np.zeros((L, 8))
        for nm in names:
            ix, p1, p2 = items[nm]["ix"], items[nm]["p1"], items[nm]["p2"]
            T = len(ix)
            for t in range(rows, T - rows + 1, rows):  # non-overlapping row groups along the decode
                for l in range(L):
                    A = set()
                    P1 = set()
                    P2 = set()
                    for r in range(rows):
                        A |= (
                            set(ix[t + r - rows + 1 if False else t - rows + r, l].tolist())
                            if False
                            else set(ix[t - rows + r, l].tolist())
                        )
                        if l >= 1:
                            P1 |= set(p1[t - rows + r, l].tolist())
                        if l >= 2:
                            P2 |= set(p2[t - rows + r, l].tolist())
                    P1 = P1 - {-1}
                    P2 = P2 - {-1}
                    P1mP2 = P1 - P2
                    miss = A - P1 - P2
                    wasted = (P1 | P2) - A
                    per_layer["A"][l] += len(A)
                    per_layer["P2"][l] += len(P2)
                    per_layer["P1mP2"][l] += len(P1mP2)
                    per_layer["miss"][l] += len(miss)
                    per_layer["wasted"][l] += len(wasted)
                    per_layer["hit2"][l] += 1 - len(A - P2) / len(A)
                    per_layer["hit12"][l] += 1 - len(miss) / len(A)
                    for key, S in (
                        ("busy_A", A),
                        ("busy_P2", P2),
                        ("busy_P1mP2", P1mP2),
                        ("busy_miss", miss),
                        ("busy_wasted", wasted),
                        ("busy_total_read", A | P1 | P2),
                    ):
                        b, cnt = busiest_tiles(S, 1, 8)
                        per_layer[key][l] += b
                        if key == "busy_total_read":
                            per_chip[l] += cnt
                    n[l] += 1
        R = {k: (v / np.maximum(n, 1)).round(3).tolist() for k, v in per_layer.items()}
        R["samples"] = n.tolist()
        R["per_chip_total_read_mean"] = (per_chip / np.maximum(n[:, None], 1)).round(2).tolist()
        # pass totals (48 layers; the MTP layer is counted as a 49th with layer 47's statistics)
        tot = lambda key: float(sum(R[key]) + R[key][-1])
        R["pass"] = dict(
            busy_A=tot("busy_A"),
            busy_P2=tot("busy_P2"),
            busy_P1mP2=tot("busy_P1mP2"),
            busy_miss=tot("busy_miss"),
            busy_wasted=tot("busy_wasted"),
            busy_total_read=tot("busy_total_read"),
            hit2_mean=float(np.mean(R["hit2"][2:])),
            hit12_mean=float(np.mean(R["hit12"][1:])),
            distinct_mean=float(np.mean(R["A"])),
        )
        res[f"rows{rows}"] = R
        P = R["pass"]
        print(
            f"rows {rows}: per pass (49 layers, busiest chip, tiles EP1xTPx8): actual {P['busy_A']:.1f} tiles, P2 {P['busy_P2']:.1f}, P1-P2 {P['busy_P1mP2']:.1f}, miss {P['busy_miss']:.1f}, wasted {P['busy_wasted']:.1f}, total read {P['busy_total_read']:.1f}; hit2 {P['hit2_mean']:.3f} hit12 {P['hit12_mean']:.3f} distinct/layer {P['distinct_mean']:.1f}"
        )
        print("  per layer busiest actual:", [round(x, 1) for x in R["busy_A"]])
        print("  per layer hit12:", R["hit12"])
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
