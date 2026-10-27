#!/bin/bash
# the files whose bytes the regeneration must reproduce (V8.sha256)
cd /data/yoshifuji/qwen38
D=tt-qwen-old/models/demos/blackhole/qwen38_flash_next
ls routing/stats_stage2.json routing/free5d_v8_results*.json routing/design_step*.json routing/census_stage2.json \
   $D/README.md $D/dataflow/design/*.h \
   scripts/free5d_v8.py scripts/render_free5d_v8.py scripts/merge_free5d_v8.py scripts/run_free5d_v8.sh scripts/cycle_v8.sh \
   scripts/design/*.py scripts/design/run_design.sh \
   $D/tools/design/gen_*.py $D/tools/design/gen_headers.sh $D/dataflow/design_tests/test_qwen38_design_host.cpp | sort
