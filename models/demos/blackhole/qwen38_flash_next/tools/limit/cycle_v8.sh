#!/bin/bash
# one full regeneration cycle: census stage 2 -> the v8 enumeration (section 11) -> the design (section 12: per form,
# the placement on the 14 harvest cases) -> README 11 and 12 -> the design headers of the chosen form + the host test ->
# sha256 check against routing/V8.sha256
set -e
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
D=tt-qwen-old/models/demos/blackhole/qwen38_flash_next
$PY scripts/design/census_stage2_batches.py routing routing/stats_stage2.json > logs/routing/census_stage2_batches.log 2>&1
bash scripts/run_free5d_v8.sh
bash scripts/design/run_design.sh
bash scripts/install_free5d_v8.sh
bash scripts/design/install_design.sh
F=$($PY - <<'PYX'
import json
fs=("A","B","Bn","B+D")
d={f: json.load(open(f"routing/design_step7_chain_hit_{f}.json"))["T_steady"] for f in fs}
st={f: sum(json.load(open(f"routing/design_step3_frames_{f}.json"))["stored"]) for f in fs}
print(min(fs, key=lambda f: (round(d[f],1), st[f])))
PYX
)
bash $D/tools/design/gen_headers.sh /data/yoshifuji/qwen38 /data/yoshifuji/qwen38/$D $F > logs/routing/gen_headers.log 2>&1
sha256sum -c routing/V8.sha256 > logs/routing/V8_check.log 2>&1 || true
