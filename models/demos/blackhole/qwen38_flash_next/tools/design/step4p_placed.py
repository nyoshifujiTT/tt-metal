#!/usr/bin/env python
r"""Step 4p (corrections 30-41, worklog 10-27 19:xx): per hyper-connection form, every unit's time with its matmul pools
timed on their placement (step 3p: the worst of the 14 harvest cases), and its stored weight bytes (padding and the
form's folds included: step 3's 'stored').  A unit whose pools are off the chain in the form (the MoE read's down in the
B forms: g_b at the attention block's start) keeps its stored bytes and gets no pool time on the chain.
Output: design_step4p_<form>.json: per unit t_us (chain), pool_us (placed), model_us (stage model), stored, and per
form the pass sums."""
import json
import sys

D = sys.argv[1]
FORM = sys.argv[2]
CLK = 1.35e9
T4 = json.load(open(f"{D}/design_step4_item_time.json"))
F3 = json.load(open(f"{D}/design_step3_frames_{FORM}.json"))
P3 = json.load(open(f"{D}/design_step3p_place_{FORM}.json"))
UT = {int(k): v for k, v in P3["unit_t"].items()}
UM = {int(k): v for k, v in P3["unit_model"].items()}
out = []
for i, t in enumerate(T4):
    n = t["unit"]
    off_chain = FORM != "A" and ".moe.gr.down" in n
    mm_model = t["mm_us"]
    placed = UT.get(i)
    if placed is not None and not off_chain:
        mm = placed / CLK * 1e6
    elif off_chain:
        mm = 0.0
    else:
        mm = mm_model
    tu = t["t_us"] - t["mm_us"] + mm if not t["spec"] else t["t_us"]
    out.append(dict(unit=n, layer=t["layer"], spec=t["spec"], t_us=tu, mm_us=mm, mm_model_us=mm_model,
                    pool_model_cycles=UM.get(i, 0.0), pool_placed_cycles=placed or 0.0, stored=F3["stored"][i],
                    off_chain=off_chain))
tot = sum(r["t_us"] for r in out if not r["spec"])
tot_m = sum(r["mm_model_us"] for r in out if not r["spec"])
tot_p = sum(r["mm_us"] for r in out if not r["spec"])
print(f"form {FORM}: chain compute {tot:.1f} us per pass on one chip's units; matmul part placed {tot_p:.1f} us vs the stage model {tot_m:.1f} us; stored weights {sum(r['stored'] for r in out)/2**20:,.1f} MiB")
json.dump(dict(form=FORM, units=out), open(f"{D}/design_step4p_{FORM}.json", "w"))
