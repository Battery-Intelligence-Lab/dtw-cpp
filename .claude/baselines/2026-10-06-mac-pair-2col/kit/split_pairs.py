"""For each probe's Standard per-pair loops (no-threshold copies): does an fcmp/fcsel pair straddle a 64-byte
boundary (fcmp at 60 mod 64, fcsel at 0)? Printed beside the measured speed-up of the shapes that loop serves.
usage: split_pairs.py <results file>"""
import re
import statistics
import subprocess
import sys
from collections import defaultdict

sys.argv, results = sys.argv[:1], sys.argv[1]
import placement  # noqa: E402

TIME = re.compile(r'^r(\d+) (base|k1|head) (p\d) time (f64|f32) (L1|Sq) nx (\d+)\s+ny (\d+)\s+band (-?\d+)\s+ns/cell ([\d.]+)')
t = defaultdict(dict)
for line in open(results):
    if m := TIME.match(line):
        r, v, p, ty, d, nx, ny, band, ns = m.groups()
        t[(v, p, ty, d, int(nx), int(ny), int(band))][int(r)] = float(ns)
med = statistics.median


def splits(binary):
    """kernel key -> list of (insns, start mod 64, split pairs) for the binary's no-threshold Standard loops"""
    dis = subprocess.run(['objdump', '-d', '--no-show-raw-insn', binary], capture_output=True, text=True).stdout
    addr_ins = {}
    for line in dis.splitlines():
        m = re.match(r'^\s*([0-9a-f]+):\s+(\S+)', line)
        if m:
            addr_ins[int(m.group(1), 16)] = m.group(2)
    out = defaultdict(list)
    for k, start, n, mod, cr, dig, norm in placement.loops_of(binary):
        if '<true' in k:
            continue
        pairs = sum(1 for a in range(start, start + 4 * n, 4)
                    if addr_ins.get(a) == 'fcmp' and addr_ins.get(a + 4) == 'fcsel' and (a + 4) % 64 == 0)
        kern = k.split()[0].replace('<false', '')
        out[(kern, k.split()[1], k.split()[2])].append((n, mod, pairs))
    return out


for v in ('base', 'k1', 'head'):
    for p in ('p0', 'p1', 'p2'):
        sp = splits(f'pprobe_{v}_{p}')
        for (kern, ty, d), loops in sorted(sp.items()):
            tt = 'f64' if ty == 'double' else 'f32'
            served = [s for s in t if s[0] == v and s[1] == p and s[2] == tt and s[3] == d
                      and ((kern == 'dtw_linear') == (s[6] < 0))]
            ratios = []
            for s in served:
                b = t[('base', p) + s[2:]]
                x = t[s]
                ratios.append(med(b[r] / x[r] for r in b))
            desc = ', '.join(f'{n} insns at {mod} mod 64, {pairs} split' for n, mod, pairs in loops)
            rr = f'{min(ratios):.2f}-{max(ratios):.2f}' if ratios and v != 'base' else '-'
            print(f'{v:4s} {p} {kern:10s} {tt} {d:2s}  {desc:60s}  speed-up of its shapes against base: {rr}')
