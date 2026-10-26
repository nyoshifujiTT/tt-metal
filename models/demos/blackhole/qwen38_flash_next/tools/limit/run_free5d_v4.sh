#!/bin/bash
# the full revision-4 enumeration, the 7 contexts in parallel (one process each; ~2 h on one core)
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
$PY scripts/free5d_v4.py routing/stats.json 192 routing/free5d_v4_results.json all > logs/routing/free5d_v4_full.log 2>&1 &
for c in 1 128 8192 32768 262144 1000000; do $PY scripts/free5d_v4.py routing/stats.json $c routing/free5d_v4_results_ctx$c.json k1 > logs/routing/free5d_v4_ctx$c.log 2>&1 & done
wait
