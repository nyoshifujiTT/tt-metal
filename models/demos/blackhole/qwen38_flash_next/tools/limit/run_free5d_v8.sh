#!/bin/bash
# the full revision-8 enumeration: one process per (context, case, k) in parallel, then merged per context (the same
# json as a single process per context: merge_free5d_v8.py keeps the CASES_K order)
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
T=routing/parts; mkdir -p $T logs/routing/parts
for ck in miss:0 miss:1 census:0 census:1 hit:0 hit:1 all:0 all:1; do
  $PY scripts/free5d_v8.py routing/stats.json 192 $T/192_${ck/:/_}.json $ck > logs/routing/parts/192_${ck/:/_}.log 2>&1 &
done
for c in 1 128 8192 32768 262144 1000000; do
  for ck in miss:1 census:1 hit:1 all:1; do
    $PY scripts/free5d_v8.py routing/stats.json $c $T/${c}_${ck/:/_}.json $ck > logs/routing/parts/${c}_${ck/:/_}.log 2>&1 &
  done
done
wait
$PY scripts/merge_free5d_v8.py routing/free5d_v8_results.json $T/192_*.json
for c in 1 128 8192 32768 262144 1000000; do $PY scripts/merge_free5d_v8.py routing/free5d_v8_results_ctx$c.json $T/${c}_*.json; done
cat logs/routing/parts/192_*.log > logs/routing/free5d_v8_full.log
for c in 1 128 8192 32768 262144 1000000; do cat logs/routing/parts/${c}_*.log > logs/routing/free5d_v8_ctx$c.log; done
