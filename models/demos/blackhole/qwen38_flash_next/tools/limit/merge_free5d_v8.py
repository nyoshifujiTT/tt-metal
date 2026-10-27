#!/usr/bin/env python
"""merge the per-case result files of free5d_v8.py (run_free5d_v8.sh runs one process per (context, case, k)) into the
per-context result file, in the CASES_K order (byte-identical to a single-process run's json)"""
import json
import sys

sys.argv[0]
out, parts = sys.argv[1], sys.argv[2:]
order = ["miss|k=0", "miss|k=1", "census|k=0", "census|k=1", "hit|k=0", "hit|k=1", "all|k=0", "all|k=1"]
d = {}
for p in parts:
    d.update(json.load(open(p)))
json.dump({k: d[k] for k in order if k in d}, open(out, "w"))
