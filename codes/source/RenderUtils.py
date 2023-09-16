import numpy as np
from numba import jit

kEpsilon = 1e-8

###
@jit(nopython=True)
def min3(ax, ay, bx, by, cx, cy):
    xmin = min(ax, min(bx, cx))
    ymin = min(ay, min(by, cy))
    return xmin, ymin
    
###    
@jit(nopython=True)
def max3(ax, ay, bx, by, cx, cy):
    xmax = max(ax, max(bx, cx))
    ymax = max(ay, max(by, cy))
    return xmax, ymax

###
@jit(nopython=True)
def edge_function(ax, ay, bx, by, cx, cy):
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

###
@jit(nopython=True)
def interleave2(x, y):
        return part1by1(x) | (part1by1(y) << 1)

###
@jit(nopython=True)
def deinterleave2(n):
        return unpart1by1(n), unpart1by1(n >> 1)

###
@jit(nopython=True)
def part1by2(n):
        n&= 0x000003ff
        n = (n ^ (n << 16)) & 0xff0000ff
        n = (n ^ (n <<  8)) & 0x0300f00f
        n = (n ^ (n <<  4)) & 0x030c30c3
        n = (n ^ (n <<  2)) & 0x09249249
        return n

###
@jit(nopython=True)
def unpart1by2(n):
        n&= 0x09249249
        n = (n ^ (n >>  2)) & 0x030c30c3
        n = (n ^ (n >>  4)) & 0x0300f00f
        n = (n ^ (n >>  8)) & 0xff0000ff
        n = (n ^ (n >> 16)) & 0x000003ff
        return n

###
@jit(nopython=True)
def interleave3(x, y, z):
        return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)

###
@jit(nopython=True)
def deinterleave3(n):
        return unpart1by2(n), unpart1by2(n >> 1), unpart1by2(n >> 2)  