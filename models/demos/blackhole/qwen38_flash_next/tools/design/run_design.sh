#!/bin/bash
# Regenerate the whole detailed design (worklog section 12, revision 8: the stage-level design with the physical
# placement, corrections 1-44) from the HF definition, the census and the specification values, then README section 12
# (render_design.py) via install_design.sh.  Deterministic.  Order (worklog 10-27 20:0x):
#   step0 -> step2 -> step4 (the pools per hyper-connection form) -> step6 (the meeting payloads; its form-A file is the
#   landing bound) -> per form F: step3 (frames from per-core stored bytes) -> step3p (the placement on the 14 harvest
#   cases, per-core L1 check) -> step4p (unit times on the placement) -> step6 F (exposures on the placed times) ->
#   step6b F (hand-offs at the frame cuts) -> step5 -> step7 F x {hit, census, miss}
set -e
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
S=scripts/design; O=routing
$PY $S/step0_units.py $O/design_step0_units.json > $O/design_step0_units.md
$PY $S/step2_account.py $O/design_step0_units.json $O/stats.json $O/design_step2_account.json > $O/design_step2_account.md
$PY $S/step4_item_time.py $O/design_step2_account.json $O/design_step0_units.json $O/design_step4_item_time.json > $O/design_step4_item_time.md
$PY $S/step6_comm.py $O/design_step6_comm.json $O/design_step4_item_time.json > $O/design_step6_comm.md
$PY $S/census_stage2.py $O $O/census_stage2.json > $O/census_stage2.md
for f in A B Bn B+D; do
  $PY $S/step3_frames.py $O/design_step2_account.json $O/design_step3_frames_$f.json $O/design_step6_comm.json $O/design_step4_item_time.json $f > $O/design_step3_frames_$f.md
  $PY $S/step3p_place.py $O $f > $O/design_step3p_place_$f.md
  $PY $S/step4p_placed.py $O $f > $O/design_step4p_$f.md
  $PY $S/step6_comm.py $O/design_step6_comm_$f.json $O/design_step4_item_time.json $O/design_step4p_$f.json > $O/design_step6_comm_$f.md
  $PY $S/step6b_hand.py $O/design_step2_account.json $O/design_step3_frames_$f.json $O/design_step6_comm.json $O/design_step6b_hand_$f.json > $O/design_step6b_hand_$f.md
  $PY $S/step5_frame_time.py $O/design_step3_frames_$f.json $O/design_step4_item_time.json $O/design_step2_account.json $O/design_step5_frame_time_$f.json $O/design_step4p_$f.json > $O/design_step5_frame_time_$f.md
done
for f in A B Bn B+D; do for c in hit census miss; do $PY $S/step7_chain.py dir=$O case=$c form=$f npass=12 > "$O/design_step7_chain_${c}_$f.md" & done; done
wait
