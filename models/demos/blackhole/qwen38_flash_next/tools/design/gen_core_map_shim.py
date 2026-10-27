"""the hub of each harvest case, as tools/design/gen_core_map.place chooses it (one source for the core map)"""
import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "../../tt-qwen-old/models/demos/blackhole/qwen38_flash_next/tools/design"), "/tmp/q38/h1"):
    if os.path.exists(os.path.join(_p, "gen_core_map.py")):
        sys.path.insert(0, _p)
        break


@functools.lru_cache(maxsize=None)
def hub_of(harv):
    import gen_core_map as G
    return G.place(harv)["hub"]
