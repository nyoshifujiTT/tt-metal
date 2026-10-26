#!/bin/bash
# Regenerate the whole detailed design (worklog section 12) from the HF definition, the census and the physical
# values: steps 0-7 in order, then the README section 12 (render_design.py) and install_design.sh.  Deterministic.
set -e
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
S=scripts/design; O=routing
$PY $S/step0_units.py $O/design_step0_units.json > $O/design_step0_units.md
$PY $S/step2_account.py $O/design_step0_units.json $O/stats.json $O/design_step2_account.json > $O/design_step2_account.md
$PY $S/step3_frames.py $O/design_step2_account.json $O/design_step3_frames.json > $O/design_step3_frames.md
$PY $S/step4_item_time.py $O/design_step2_account.json $O/design_step0_units.json $O/design_step4_item_time.json > $O/design_step4_item_time.md
$PY $S/step5_frame_time.py $O/design_step3_frames.json $O/design_step4_item_time.json $O/design_step2_account.json $O/design_step5_frame_time.json > $O/design_step5_frame_time.md
$PY $S/census_stage2.py $O $O/census_stage2.json > $O/census_stage2.md
$PY $S/step6_comm.py $O/design_step6_comm.json > $O/design_step6_comm.md
for c in hit census miss; do $PY $S/step7_chain.py dir=$O case=$c form=A npass=12 > $O/design_step7_chain_${c}_A.md; done
for f in B Bn B+D; do $PY $S/step7_chain.py dir=$O case=hit form=$f npass=12 > $O/design_step7_chain_hit_$f.md; done
$PY $S/step7_chain.py dir=$O case=hit form=A s_resident=1 npass=12 > $O/design_step7_chain_hit_A_S.md
