"""Correction 33-34 (10-27 19:1x): the raster path of a chip's 130 Tensix cores per harvest case (columns x ascending,
each column y 2 -> 11).  A later core lies right / below an earlier one, so a child -> parent send (later -> earlier) is a
NoC1 (left / up) route: hops <= logical distance + 3 per column boundary crossed + the x gap and the torus wrap."""
import functools

import stages as SG

XS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
YS = list(range(2, 12))


def du(a, b):
    return min(SG.d0(a, b), SG.d1(a, b))


@functools.lru_cache(maxsize=None)
def path(harv):
    return tuple((x, y) for x in XS if x != harv for y in YS)


@functools.lru_cache(maxsize=None)
def hop_table(harv):
    """H[i][j] = hops from path core i to path core j (better NoC)"""
    P = path(harv)
    return tuple(tuple(du(P[i], P[j]) for j in range(130)) for i in range(130))
