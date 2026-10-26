#!/usr/bin/env python
r"""Two-stage speculative-prefetch census per batch (the limit model v4's routing numerator for the census case).
Same sampling as routing_stats.py (seed 20261023, the same batch keys, test/train split, sample counts) so the
'actual' busiest values coincide with stats.json; per sample and layer the four sets A (actual top-10 union of the
rows), P2 (pred2 union), P1 \ P2, miss = A \ (P1 u P2), wasted = (P1 u P2) \ A are formed (layers 0/1 as in
census_stage2.py: no pred2 for l < 2, no pred1 for l < 1 -> those sets are empty and the whole A is a miss (l=0) or
P1 (l=1)), and for every placement the busiest chip/group of each set is averaged:
  tiles form (ep, tpx) in TILE_PAIRS: busiest chip in tiles;  slice form chips in (1, 2, 4, 8, 16, 32): busiest
  group in experts.
Output: routing/stats_stage2.json = {batch key: {"tiles": {"1x8": {A, P2, P1mP2, miss, wasted}}, "groups": {...},
hit2, hit12, distinct}}."""
import glob
import json
import os
import random
import sys

import numpy as np

E = 512
L = 48
K = 10
N_TILES = 20
rng = random.Random(20261023)
TILE_PAIRS = [(ep, tpx) for tpx in (8, 16, 32) for ep in (1, 2, 4, 8, 16, 32) if ep * tpx <= 32]
CHIPS = (1, 2, 4, 8, 16, 32)


def owners(experts, n):
    return np.asarray(sorted(experts), dtype=np.int64) // (E // n) if experts else np.zeros(0, dtype=np.int64)


def busiest(experts, n):
    return int(np.bincount(owners(experts, n), minlength=n).max()) if experts else 0


def busiest_tiles(experts, ep, tpx):
    if not experts:
        return 0
    e = np.asarray(sorted(experts), dtype=np.int64)
    g = e // (E // ep)
    j = np.arange(N_TILES)
    idx = (g[:, None] * tpx + (e[:, None] + j[None, :]) % tpx).ravel()
    return int(np.bincount(idx, minlength=ep * tpx).max())


def load(paths):
    items = {}
    for p in paths:
        z = np.load(p)
        nm = os.path.basename(p)[:-4]
        items[nm] = dict(
            indices=z["indices"].astype(np.int32),
            kind=z["kind"],
            pred1=z["pred1"].astype(np.int32) if "pred1" in z.files else None,
            pred2=z["pred2"].astype(np.int32) if "pred2" in z.files else None,
        )
    return items


def main(rdir="/data/yoshifuji/qwen38/routing", out="/data/yoshifuji/qwen38/routing/stats_stage2.json"):
    paths = sorted(p for p in glob.glob(rdir + "/*.npz") if "/smoke-" not in p)
    items = load(paths)
    names = sorted(items)
    # the same preamble as routing_stats.main (its rng draws before the batch loop: none -- the h1 and hot-set parts do not draw)
    dec = {nm: items[nm]["indices"][items[nm]["kind"] == 1] for nm in names}
    names = [nm for nm in names if len(dec[nm]) >= 8]
    train = names[::2]
    test = names[1::2]
    p1 = {nm: items[nm]["pred1"][items[nm]["kind"] == 1] for nm in names}
    p2 = {nm: items[nm]["pred2"][items[nm]["kind"] == 1] for nm in names}
    results = {}
    for B, rows in (
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (16, 1),
        (32, 1),
        (64, 1),
        (128, 1),
        (256, 1),
        (1, 2),
        (2, 2),
        (4, 2),
        (8, 2),
        (16, 2),
        (32, 2),
        (64, 2),
        (128, 2),
    ):
        pool = test if B <= len(test) else names
        samples = 2000 if B == 1 else (400 if B <= 32 else 100)
        key = f"{B}" if rows == 1 else f"{B}x{rows}"
        acc = {k: [] for k in ("distinct", "hit2", "hit12", "wasted")}
        tl = {f"{ep}x{tpx}": {s: [] for s in ("A", "P2", "P1mP2", "miss", "wasted")} for ep, tpx in TILE_PAIRS}
        gr = {c: {s: [] for s in ("A", "P2", "P1mP2", "miss", "wasted")} for c in CHIPS}
        for _ in range(samples):
            chosen = rng.sample(pool, min(B, len(pool)))
            while len(chosen) < B:
                chosen.append(rng.choice(pool))
            tmin = min(len(dec[nm]) for nm in chosen)
            t = rng.randrange(rows, tmin - rows + 1)
            for l in range(L):
                A = set()
                P1 = set()
                P2 = set()
                for i, nm in enumerate(chosen):
                    tt = t if i < len(pool) else rng.randrange(rows, len(dec[nm]) - rows + 1)
                    for r in range(rows):
                        A |= set(dec[nm][tt + r, l].tolist())
                        if l >= 1:
                            P1 |= set(p1[nm][tt + r, l].tolist())
                        if l >= 2:
                            P2 |= set(p2[nm][tt + r, l].tolist())
                P1.discard(-1)
                P2.discard(-1)
                sets = dict(A=A, P2=P2, P1mP2=P1 - P2, miss=A - P1 - P2, wasted=(P1 | P2) - A)
                acc["distinct"].append(len(A))
                acc["hit2"].append(1 - len(A - P2) / len(A))
                acc["hit12"].append(1 - len(sets["miss"]) / len(A))
                acc["wasted"].append(len(sets["wasted"]))
                for ep, tpx in TILE_PAIRS:
                    d = tl[f"{ep}x{tpx}"]
                    for s, S in sets.items():
                        d[s].append(busiest_tiles(S, ep, tpx))
                for c in CHIPS:
                    for s, S in sets.items():
                        gr[c][s].append(busiest(S, c))
        results[key] = dict(
            rows_per_user=rows,
            samples=samples,
            distinct_mean=float(np.mean(acc["distinct"])),
            hit2=float(np.mean(acc["hit2"])),
            hit12=float(np.mean(acc["hit12"])),
            wasted_mean=float(np.mean(acc["wasted"])),
            tiles={k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in tl.items()},
            groups={c: {s: float(np.mean(v)) for s, v in d.items()} for c, d in gr.items()},
        )
        R = results[key]
        print(
            f"B={key:5s} distinct {R['distinct_mean']:6.1f} hit2 {R['hit2']:.3f} hit12 {R['hit12']:.3f} wasted {R['wasted_mean']:5.1f} | tiles 1x8 A {R['tiles']['1x8']['A']:.1f} P2 {R['tiles']['1x8']['P2']:.1f} P1mP2 {R['tiles']['1x8']['P1mP2']:.1f} miss {R['tiles']['1x8']['miss']:.1f} wasted {R['tiles']['1x8']['wasted']:.1f} | groups 4 A {R['groups'][4]['A']:.2f} miss {R['groups'][4]['miss']:.2f}",
            flush=True,
        )
    json.dump(results, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
