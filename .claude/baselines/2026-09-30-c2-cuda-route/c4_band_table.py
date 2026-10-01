"""Summarise a band.sh output: per case base/head medians, head/base, the registered band, and
whether head's matrix hash and two distances equal base's. Usage: band_table.py band.txt"""
import re
import sys

# registered 2026-10-01 02:45 BST: (low, high) for head / base
BAND = {2751: (0.0, 0.95), 2754: (0.0, 0.95), 2757: (0.0, 0.95), 2750: (0.97, 1.03), 2400: (0.97, 1.03)}
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
cases = {}
contended = set()
for block in re.split(r"^== ", text, flags=re.M)[1:]:
    which, mode, L, N = block.split("\n", 1)[0].split()
    med = re.search(r"^median,\S+?,\d+,\d+,([0-9.]+) s,.*?min ([0-9.]+),max ([0-9.]+),d\(0,1\)=(\S+),d\(N-1,N-2\)=(\S+)$", block, re.M)
    mat = re.search(r"^matrix,.*fnv1a64 ([0-9a-f]+)$", block, re.M)
    cases.setdefault((mode, int(L), int(N)), {})[which] = (float(med.group(1)), med.group(4), med.group(5), mat.group(1), float(med.group(2)), float(med.group(3)))
    if "CONTENDED" in block:
        contended.add((mode, int(L), int(N)))
print("| case (N) | base median | head median | head / base | registered | verdict | matrix hash, d(0,1), d(N-1,N-2) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for (mode, L, N), r in cases.items():
    b, h = r["base"], r["head"]
    ratio = h[0] / b[0]
    lo, hi = BAND[L]
    reg = f"<= {hi}" if lo == 0.0 else f"{lo}-{hi}"
    verdict = "pass" if lo <= ratio <= hi else "OUTSIDE"
    if h[3] != b[3] or b[1:3] != h[1:3]:
        verdict += " (DISTANCES DIFFER)"
    same = "equal" if b[1:4] == h[1:4] else f"DIFFER {b[1:4]} vs {h[1:4]}"
    note = " CONTENDED" if (mode, L, N) in contended else ""
    print(f"| FP32 L {L} ({N}) | {b[0]:.4f} s | {h[0]:.4f} s | {ratio:.3f} | {reg} | {verdict}{note} | {same} ({h[3]}) |")
