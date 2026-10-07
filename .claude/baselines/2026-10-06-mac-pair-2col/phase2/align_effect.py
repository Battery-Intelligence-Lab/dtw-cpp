import re, statistics, sys
from collections import defaultdict
TIME = re.compile(r'^r(\d+) (base|k1|head) (p\d) time (f64|f32) (L1|Sq) nx (\d+)\s+ny (\d+)\s+band (-?\d+)\s+ns/cell ([\d.]+)')
t = defaultdict(list)
for line in open(sys.argv[1]):
    m = TIME.match(line)
    if m:
        r, v, p, ty, d, nx, ny, band, ns = m.groups()
        t[(v, ty, d, int(nx), int(ny), int(band), p)].append(float(ns))
med = {k: statistics.median(v) for k, v in t.items()}
shapes = sorted({k[:6] for k in med})
print("build ty cost nx ny band | ns p0 p1(aligned) p2 | p0/p1 p2/p1  (>1: aligned faster)")
worst = []
for s in shapes:
    if s[0] != 'head': continue
    p0, p1, p2 = (med.get(s + (p,)) for p in ('p0', 'p1', 'p2'))
    if None in (p0, p1, p2): continue
    worst.append(min(p0 / p1, p2 / p1))
    print(f"{' '.join(map(str, s)):34s} | {p0:.4f} {p1:.4f} {p2:.4f} | {p0/p1:.3f} {p2/p1:.3f}")
print("head: min over shapes of best-unaligned/aligned:", f"{min(worst):.3f}", "max:", f"{max(worst):.3f}")
