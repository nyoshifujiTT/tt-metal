#!/bin/bash
# Replace README section 12 (the detailed design at ctx 1, B=1, k=1) with render_design.py's output.
set -e
cd /data/yoshifuji/qwen38
PY=./tt-qwen-old/python_env/bin/python
$PY scripts/design/render_design.py routing > /tmp/readme_sec12.md
$PY - <<'PYX'
D="tt-qwen-old/models/demos/blackhole/qwen38_flash_next/README.md"
s=open(D).read(); sec=open("/tmp/readme_sec12.md").read()
a=s.find("## 12. "); b=s.find("## More\n"); assert b>=0
s=(s[:a] if a>=0 else s[:b]) + sec + ("\n" if not sec.endswith("\n") else "") + s[b:]
open(D,"w").write(s)
PYX
