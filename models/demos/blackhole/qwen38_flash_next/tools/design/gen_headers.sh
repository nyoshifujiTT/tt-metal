#!/bin/bash
# H-1 / H-2: regenerate the seven design headers (README section 12, revision 8) of the chosen form from the generators and
# the design jsons, then compile each and run the host-only joint checks.  Deterministic: byte-identical for the same inputs.
# usage: gen_headers.sh <qwen38 root (scripts/, routing/)> <repo demo dir> <form>
set -e
Q=$1; D=$2; F=$3
PY=$Q/tt-qwen-old/python_env/bin/python
T=$D/tools/design; O=$D/dataflow/design
mkdir -p $O
$PY $T/gen_core_map.py $O/core_map.h > /dev/null
$PY $T/gen_stage_costs.py $Q/scripts/design $O/stage_costs.h
$PY $T/gen_design_plan.py $Q/routing $O/design_plan.h $F > /dev/null
$PY $T/gen_chip_l1.py $Q/routing $O/chip_l1.h $F > /dev/null
$PY $T/gen_coll_routes.py $Q/scripts $Q/routing/design_step6_comm_$F.json $O/coll_routes.h
$PY $T/gen_unit_assign.py $Q/scripts/design $Q/routing $F $O/unit_assign.h > /dev/null
$PY $T/gen_dram_plan.py $Q/scripts/design $Q/routing $F $O/dram_plan.h > /dev/null
for h in core_map stage_costs design_plan chip_l1 coll_routes unit_assign dram_plan; do
  if g++ -std=c++20 -fconstexpr-ops-limit=4000000000 -fsyntax-only -x c++ $O/$h.h 2>&1 | grep -v "pragma once" | grep -E "error"; then exit 1; fi
done
g++ -std=c++20 -fconstexpr-ops-limit=4000000000 -I $O $D/dataflow/design_tests/test_qwen38_design_host.cpp -o /tmp/test_qwen38_design_host
/tmp/test_qwen38_design_host
(cd $O && sha256sum core_map.h stage_costs.h design_plan.h chip_l1.h coll_routes.h unit_assign.h dram_plan.h)
