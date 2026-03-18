#!/usr/bin/env python3
"""Check if our LCG seeds produce diverse first centers."""
import ctypes

def lcg(s):
    return (s * 1664525 + 1013904223) & 0xFFFFFFFF

n = 10000
k = 5
for trial in range(10):
    seed = 0xCAFE + trial * 7919 + k + ((n * 2654435761) >> 16) & 0xFFFFFFFF
    seed = seed & 0xFFFFFFFF
    seed = lcg(seed)
    idx = seed % n
    print(f"trial {trial}: seed={seed:#010x}  first_center_idx={idx}  (of {n})")
