#!/bin/bash
# Replace README section 11 (the five-axis decode limit, revision 4) with the renderer's output.
set -e
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
D=tt-qwen-old/models/demos/blackhole/qwen38_flash_next
$PY scripts/render_free5d_v4.py > /tmp/readme_sec11.md
$PY - <<'PYX'
D="tt-qwen-old/models/demos/blackhole/qwen38_flash_next/README.md"
s=open(D).read(); sec=open("/tmp/readme_sec11.md").read()
a=s.find("## 11. "); b=s.find("## 12. "); assert a>=0 and b>=0
s=s[:a] + sec + ("\n" if not sec.endswith("\n") else "") + s[b:]
open(D,"w").write(s)
PYX
