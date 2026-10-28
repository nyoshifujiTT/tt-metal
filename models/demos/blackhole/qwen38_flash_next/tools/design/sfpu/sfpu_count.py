#!/usr/bin/env python
r"""D-1 (worklog 10-28): cycle count of an SFPU function, from the disassembly of the function compiled by the repo's
kernel toolchain (sfpi 7.74.0 = tt_metal/sfpi-version, -O3 -mcpu=tt-bh-tensix: the compute-kernel flags of
tt_metal/jit_build/build.cpp:427 and bh_hal.cpp:218).  Desk count, nothing is run on a device.

Rules (tt-isa-documentation be83310, BlackholeA0):
  - the Tensix thread dequeues at most one instruction a cycle (BabyRISCV/PushTensixInstruction.md:19); every
    sfp* / tt* instruction and every `sw` into the instruction buffer (a runtime-immediate push) is one cycle;
  - REPLAY with Load and Exec records and executes its N instructions; REPLAY with Exec replays N instructions
    (TensixCoprocessor/REPLAY.md);
  - SFPMAD / SFPMUL / SFPADD / SFPMULI / SFPADDI / SFPLUT / SFPLUTFP32 / SFPMUL24 take two cycles; hardware stalls the
    next SFPU instruction one cycle when it reads a register the previous one writes (SFPMAD.md:70);
  - SFPSWAP: on the next cycle only SFPNOP is accepted, otherwise one cycle of stall (SFPSWAP.md);
  - loops: the `li rX,N` ... `bnez rX` around a body run the body N times.
Register reads are taken conservatively: an operand register of the next instruction counts as read unless it is the
destination of SFPLOAD or of a full-width SFPLOADI (mod 0/1/2/4: whole register written)."""
import re
import sys

TWO = {"sfpmad", "sfpmul", "sfpadd", "sfpmuli", "sfpaddi", "sfplut", "sfplutfp32", "sfpmul24"}


def parse(dis):
    funcs = {}
    cur = None
    for line in open(dis):
        m = re.match(r"^[0-9a-f]+ <(\S+)>:", line)
        if m:
            name = m.group(1)
            if name.startswith(".L"):
                funcs[cur].append(("label", name))
            else:
                cur = name
                funcs[cur] = []
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)$", line)
        if m and cur:
            funcs[cur].append(("ins", int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))
    return funcs


def regs(ops):
    return re.findall(r"\bL(\d+)\b", ops)


def tensix_stream(body):
    """expand loops (li N ... bnez back) and replays into the dequeued Tensix instruction stream"""
    ins = [x for x in body if x[0] == "ins"]
    addr = {x[1]: i for i, x in enumerate(ins)}
    out = []
    replay = []
    i = 0
    loop_n = {}
    while i < len(ins):
        _, a, op, ops = ins[i]
        if op == "li":
            m = re.match(r"(\w+),(\d+)", ops)
            if m:
                loop_n[m.group(1)] = int(m.group(2))
        if op == "bnez":
            reg, tgt = ops.split(",")[0], int(ops.split(",")[1].split()[0], 16)
            n = loop_n.get(reg, 1)
            j = addr[tgt]
            seg = [x for x in out_seg(ins[j:i])]
            out.extend(seg * (n - 1))
        elif op == "ttreplay":
            s, n, ld, ex = [int(v) for v in ops.split(",")]
            if ld:
                replay = [("rec", n)]
                if ex:
                    out.append(("REPLAY_REC", n))
            else:
                out.append(("REPLAY_EXEC", n))
        elif op.startswith("sfp") or op.startswith("tt") or (op == "sw" and "(a5)" in ops) or op == ".ttinsn":
            out.append((op, ops))
        i += 1
    return out


def out_seg(seg):
    for _, a, op, ops in seg:
        if op.startswith("sfp") or op.startswith("tt") or (op == "sw" and "(a5)" in ops):
            yield (op, ops)


def cycles(stream, recorded):
    """count with stalls; REPLAY_REC n: the next n stream items are recorded (and executed); REPLAY_EXEC n: replays"""
    flat = []
    rec = None
    i = 0
    while i < len(stream):
        op, ops = stream[i]
        if op == "REPLAY_REC":
            rec = stream[i + 1:i + 1 + ops]
            flat.extend(rec)
            i += 1 + ops
            continue
        if op == "REPLAY_EXEC":
            flat.extend(rec if rec is not None else recorded[:ops])
            i += 1
            continue
        flat.append((op, ops))
        i += 1
    cyc = 0
    stall = 0
    prev = None
    for op, ops in flat:
        cyc += 1
        if prev is not None:
            pop, pops = prev
            if pop in TWO and op.startswith("sfp"):
                d = regs(pops)[:1]
                r = regs(ops)
                if op == "sfpload" or (op == "sfploadi" and ops.split(",")[-1].strip() in ("0", "1", "2", "4")):
                    r = r[1:]
                if d and d[0] in r:
                    stall += 1
            if pop == "sfpswap" and op != "sfpnop" and op.startswith("sfp"):
                stall += 1
        prev = (op, ops)
    return cyc, stall, len(flat)


def recorded_from(body):
    """instructions recorded by a REPLAY with Load and not Exec at the head (recorded in init, e.g. reciprocal):
    unknown here -> the caller supplies them"""
    return []


if __name__ == "__main__":
    funcs = parse(sys.argv[1])
    for name, body in funcs.items():
        st = tensix_stream(body)
        c, s, n = cycles(st, [])
        print(f"{name}\tinstructions {n}\tstalls {s}\tcycles {c + s}")
