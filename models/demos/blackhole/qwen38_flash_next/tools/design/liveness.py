"""Correction 23 (10-27 18:5x): the values alive across a cut between units i-1 and i, from the HF dataflow of the
units (step 0 / step 2 order).  A value is produced by one unit and consumed by later units; it crosses the cut when
its producer is before the cut and a consumer is at or after it.  The residual stream is always alive.  Sizes are this
chip's bytes after the value's meeting (TP8, rows 2): the block output y after its all-reduce is bf16 [rows, H]; the
low-rank after AR_down fp32 [rows, 320]; the router's result 10 x (score fp32 + id int16) per row; other values as
step 2's 'out' of the producing unit.  The residual crosses as its 1/TP columns (form B: the receiver all-gathers it)."""
import re

ROWS = 2


def _key(name):
    """(layer, block, the rest of the FULL unit name after the block prefix)"""
    m = re.match(r"^(L\d\d|mtp)\.(gdn|qsa|moe|ple)\.(.*)$", name, re.S)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"^(mtp)\.(.*)$", name)
    if m:
        return "mtp", "tail", m.group(2)
    return "-", "tail", name


def graph(units, H=2560, TP=8):
    """returns (produces, consumes, size): per unit index the value it produces (or None), the values it consumes, and
    each value's crossing bytes"""
    prod = {}
    cons = {}
    size = {}
    last = {}
    for i, u in enumerate(units):
        lay, blk, rest = _key(u["unit"])
        full = u["unit"]
        P = f"{lay}.{blk}"
        c = []
        v = f"{P}:{rest.split(' ')[0]}"
        if "speculative router" in full:
            prod[i] = None
            cons[i] = []
            continue
        if rest.startswith("gr.norm"):
            c = ["residual"]
        elif rest.startswith("gr.down"):
            c = [f"{P}:gr.norm"]
            size[v] = ROWS * 320 * 4
        elif rest.startswith("gr.up"):
            c = [f"{P}:gr.down"]
        elif rest.startswith("gr.gate"):
            c = [f"{P}:gr.up", f"{P}:gr.norm"]
            v = f"{P}:x"
        elif rest.startswith("gr.inject"):
            c = [f"{P}:gr.norm"]
        elif rest.startswith("gr.write"):
            c = [f"{P}:gr.inject", f"{P}:y", "residual"]
            v = None
        elif blk == "gdn":
            if rest.startswith("in_proj"):
                c = [f"{P}:x"]
            elif rest.startswith("conv1d"):
                c = [f"{P}:in_proj_qkv"]
            elif rest.startswith("l2norm"):
                c = [f"{P}:conv1d"]
            elif rest.startswith("gated"):
                c = [f"{P}:l2norm", f"{P}:conv1d", f"{P}:in_proj_a", f"{P}:in_proj_b"]
            elif rest.startswith("norm"):
                c = [f"{P}:gated", f"{P}:in_proj_z"]
            elif rest.startswith("out_proj"):
                c = [f"{P}:norm"]
                v = f"{P}:y"
                size[v] = ROWS * H * 2
        elif blk == "qsa":
            if rest.startswith(("indexer", "q+gate", "k")):
                c = [f"{P}:x"]
            elif rest.startswith("index"):
                c = [f"{P}:indexer"]
            elif rest.startswith("partial"):
                c = [f"{P}:q+gate", f"{P}:k"]
            elif rest.startswith("SDPA"):
                c = [f"{P}:partial", f"{P}:k", f"{P}:index", f"{P}:q+gate"]
            elif rest.startswith("o_proj"):
                c = [f"{P}:SDPA"]
                v = f"{P}:y"
                size[v] = ROWS * H * 2
        elif blk == "moe":
            if rest.startswith("router"):
                c = [f"{P}:x"]
                size[v] = ROWS * 10 * 6
            elif rest.startswith("shared gate"):
                c = [f"{P}:x"]
                v = f"{P}:shared_act"
            elif rest.startswith("shared down"):
                c = [f"{P}:shared_act", f"{P}:x"]
                v = f"{P}:shared_out"
                size[v] = 0   # pooled with the routed down: the down's accumulator (y)
            elif rest.startswith("routed expert e: gate_up"):
                c = [f"{P}:x", f"{P}:router"]
                v = f"{P}:expert_act"
            elif rest.startswith("routed expert e: down"):
                c = [f"{P}:expert_act", f"{P}:router", f"{P}:shared_act"]
                v = f"{P}:y"
                size[v] = ROWS * H * 2
            elif rest.startswith("routed + shared"):
                c = [f"{P}:y"]
                v = None
        elif blk == "ple":
            if rest.startswith("n-gram"):
                c = []
            elif rest.startswith("key_proj"):
                c = [f"{P}:n-gram"]
            elif rest.startswith("norm_query"):
                c = ["residual", f"{P}:key_proj"]
            elif rest.startswith("value_proj"):
                c = [f"{P}:n-gram", f"{P}:norm_query(residual)"]
            elif rest.startswith("norm_conv"):
                c = [f"{P}:value_proj", "residual"]
                v = None
        else:
            prev = [x for x in prod.values() if x and x.split(":")[0] in ("-.tail", "mtp.tail")]
            if full.startswith("embed_tokens"):
                c = []
                v = None
            elif full.startswith("final"):
                c = ["residual"]
            elif full.startswith("lm_head"):
                c = [prev[-1]]
            elif full.startswith("argmax"):
                c = [prev[-1]]
                size[v] = ROWS * 8
            elif full.startswith("mtp.hyper"):
                c = ["residual"]
            elif full.startswith("mtp.embed"):
                c = [[x for x in prev if x.split(":")[1].startswith("argmax")][-1], prev[-1]]
            elif full.startswith("mtp.lm_head"):
                c = ["residual"]
            else:
                raise SystemExit("liveness: unmatched unit " + full)
        prod[i] = v
        cons[i] = c
        if v and v not in size:
            size[v] = u["out"]
        for x in c:
            last[x] = i
    return prod, cons, size, last


def live_at(units, cut, H=2560, TP=8, g=None):
    """the values alive across the cut before unit `cut` and their bytes on this chip (residual: its 1/TP columns)"""
    prod, cons, size, last = g or graph(units, H, TP)
    alive = {"residual": ROWS * 4 * H * 2 / TP}
    for i in range(cut):
        v = prod.get(i)
        if v and last.get(v, -1) >= cut:
            alive[v] = size[v]
    return alive


def max_live(units, H=2560, TP=8):
    g = graph(units, H, TP)
    best = (0, None, None)
    for cut in range(1, len(units)):
        a = live_at(units, cut, H, TP, g)
        b = sum(a.values())
        if b > best[0]:
            best = (b, cut, a)
    return best
