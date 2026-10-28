#!/usr/bin/env python
r"""Step 6b of the detailed design (correction 23, worklog 10-27 18:5x): the hand-offs between groups at the REAL frame cuts
of step 3.  At each cut the values alive across it (liveness.py: produced before the cut, consumed at or after it, from the
HF dataflow of the units) go from every chip of the group to its counterpart in the next group (the 2 x 4 blocks of the
8 x 4 torus, 2 hops, the column's two chips sharing the inter-group link; packet model, made by the owners, delivered to
all the receiving chip's cores).  The pass wrap (the last frame -> frame 0) carries the token ids.  The forms B* add the
receiving group's all-gather of the residual columns (step 6's hand_B_extra).
Inputs: step 2 (units), step 3 (frames), step 6 (the landing bound HAND_MAX the frame capacity assumed).  Asserts that
every hand-off fits that bound (else step 3 must be redone with the larger bound).
Output: design_step6b_hand.json (per cut: unit pair, alive values, bytes, us) and markdown."""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import comm_model as cm  # noqa: E402
import liveness as LV  # noqa: E402

A = json.load(open(sys.argv[1]))
F = json.load(open(sys.argv[2]))
S6 = json.load(open(sys.argv[3]))
out_path = sys.argv[4]
U = A["units"]
ROWS = A["rows"]
g = LV.graph(U)
big = cm.Mesh(4, 4)


def hand_us(b):
    fl = [(0.0, (0, c), cm.path_tree(big, (0, c), (2, c)), b, (0, c)) for c in range(4)] + [
        (0.0, (1, c), cm.path_tree(big, (1, c), (3, c)), b, (1, c)) for c in range(4)
    ]
    return cm.hand_over(fl, b) * 1e6


rows = []
wrap_b = ROWS * 4
rows.append(dict(cut=0, frame=0, before="(pass wrap)", after=U[F["frames"][0]["a"]]["unit"].split(" ")[0], alive={"token ids": wrap_b}, bytes=wrap_b, us=hand_us(wrap_b)))
for fr in F["frames"][1:]:
    a = fr["a"]
    alive = LV.live_at(U, a, g=g)
    b = sum(alive.values())
    rows.append(dict(cut=fr["frame"], frame=fr["frame"], before=U[a - 1]["unit"].split(" ")[0], after=U[a]["unit"].split(" ")[0], alive=alive, bytes=b, us=hand_us(b)))
bound = F.get("hand_max", max(h["bytes"] for h in S6["hand"]))
worst = max(r["bytes"] for r in rows)
assert worst <= bound, f"a hand-off of {worst} B exceeds the landing bound {bound} B the frame capacity assumed: redo step 3"
print("## hand-offs at the frame cuts of step 3 (liveness from the HF dataflow)\n")
print("| cut | before \\| after | alive values (bytes on this chip) | bytes/chip | time [us] |")
print("|---|---|---|---|---|")
for r in rows:
    print(f"| {r['cut']} | {r['before']} \\| {r['after']} | {', '.join(f'{k} {v:,.0f}' for k, v in r['alive'].items())} | {r['bytes']:,.0f} | {r['us']:.3f} |")
print(f"\ntotal {sum(r['us'] for r in rows):.2f} us per pass; the largest {worst:,.0f} B within the landing bound {bound:,.0f} B")
json.dump(dict(hand=rows, bound=bound), open(out_path, "w"), indent=1)
