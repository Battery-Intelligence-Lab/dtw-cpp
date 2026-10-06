"""Summarises a run.sh results file: per shape, the median over repeats for base and head, the head/base
speed-up (median and range of the per-repeat ratios), the fill matrices' hashes, and the bands the
orchestrator registered before any measurement. usage: summary.py <results file>"""
import re
import statistics
import sys
from collections import defaultdict

TIME = re.compile(r'^r(\d+) (base|head) time (f64|f32) (L1|Sq) L (\d+)\s+band (-?\d+)\s+W (\d+)\s+ns/cell ([\d.]+) .*?'
                  r'cycles/cell@4\.59GHz ([\d.]+)\s+cycles/cell@measured ([\d.]+)\s+clock ([\d.]+) GHz')
FILL = re.compile(r'^r(\d+) (base|head) fill (f64|f32) (L1|Sq) threads\s+(\d+) N (\d+)\s+L (\d+)\s*( ragged\+-10)? band (-?\d+)\s+'
                  r'W (\d+)\s+ms ([\d.]+) .*?Gcell/s ([\d.]+)\s+matrix hash ([0-9a-f]+)')

DRY = 'mode dry' in open(sys.argv[1]).readline()
if DRY:
    print('\nDRY RUN (tiny sizes, one repeat): checks that the kit runs; none of the numbers below is a measurement')
times = defaultdict(lambda: defaultdict(dict))  # shape -> build -> repeat -> (ns, cyc, cyc_meas, ghz, W)
fills = defaultdict(lambda: defaultdict(dict))  # shape -> build -> repeat -> (gcell, hash, W)
for line in open(sys.argv[1]):
    if m := TIME.match(line):
        r, b, t, d, L, band, W, ns, cyc, cycm, ghz = m.groups()
        times[(t, d, int(L), int(band))][b][int(r)] = (float(ns), float(cyc), float(cycm), float(ghz), int(W))
    elif m := FILL.match(line):
        r, b, t, d, th, N, L, rag, band, W, ms, g, h = m.groups()
        fills[(t, d, int(th), int(N), int(L), bool(rag), int(band))][b][int(r)] = (float(g), h, int(W))

med = statistics.median
verdicts = []
print('\nsingle thread, the lane function alone (median over repeats; speed-up = base ns / head ns per repeat)')
print(f'{"shape":24s} {"W b/h":7s} {"base ns/cell":>12s} {"head ns/cell":>12s} {"base cyc":>9s} {"head cyc":>9s} '
      f'{"speed-up":>9s} {"range":>13s} {"clock GHz":>9s}')
for key in sorted(times, key=lambda k: (k[0], k[1], k[2], k[3])):
    t, d, L, band = key
    b, h = times[key]['base'], times[key]['head']
    reps = sorted(set(b) & set(h))
    if not reps:
        continue
    ratios = [b[r][0] / h[r][0] for r in reps]
    sp = med(ratios)
    name = f'{t} {d} L{L} {"unbanded" if band < 0 else f"band {band}"}'
    print(f'{name:24s} {b[reps[0]][4]:>2d}/{h[reps[0]][4]:<4d} {med(b[r][0] for r in reps):12.5f} {med(h[r][0] for r in reps):12.5f} '
          f'{med(b[r][1] for r in reps):9.4f} {med(h[r][1] for r in reps):9.4f} {sp:9.3f} '
          f'{min(ratios):6.3f}-{max(ratios):<6.3f} {med([b[r][3] for r in reps] + [h[r][3] for r in reps]):9.3f}')
    verdicts.append(('single', t, d, L, band, sp))

print('\nfill at the given threads (median Gcell/s over repeats; speed-up = head / base per repeat)')
hashes_ok = True
for key in sorted(fills):
    t, d, th, N, L, rag, band = key
    b, h = fills[key]['base'], fills[key]['head']
    reps = sorted(set(b) & set(h))
    if not reps:
        continue
    ratios = [h[r][0] / b[r][0] for r in reps]
    same = all(b[r][1] == h[r][1] for r in reps)
    hashes_ok = hashes_ok and same
    name = f'{t} {d} N{N} L{L}{"+-10 ragged" if rag else ""} {"unbanded" if band < 0 else f"band {band}"} T{th}'
    print(f'{name:40s} W {b[reps[0]][2]}/{h[reps[0]][2]}  base {med(b[r][0] for r in reps):7.2f}  head {med(h[r][0] for r in reps):7.2f} Gcell/s  '
          f'speed-up {med(ratios):.3f} ({min(ratios):.3f}-{max(ratios):.3f})  matrix {"identical" if same else "DIFFERS"}')
    verdicts.append(('fill', rag, med(ratios)))

print('\nbands registered before the run (the orchestrator, 2026-10-06):')
single = [v for v in verdicts if v[0] == 'single']
fill = [v for v in verdicts if v[0] == 'fill']
if single:
    every = min(v[5] for v in single)
    f64l1 = [v[5] for v in single if v[1] == 'f64' and v[2] == 'L1']
    print(f'  single thread, every shape >= 1.15x: min {every:.3f} -> {"PASS" if every >= 1.15 else "FAIL"}')
    if f64l1:
        print(f'  single thread, f64 L1 >= 1.3x at all four shapes: min {min(f64l1):.3f} -> {"PASS" if min(f64l1) >= 1.3 and len(f64l1) == 4 else "FAIL"}')
    print(f'  single thread, none below 0.97x: min {every:.3f} -> {"PASS" if every >= 0.97 else "FAIL"}')
eq = [v[2] for v in fill if not v[1]]
rg = [v[2] for v in fill if v[1]]
if eq:
    print(f'  18-thread equal-length fill >= 1.3x: {", ".join(f"{x:.3f}" for x in eq)} -> {"PASS" if min(eq) >= 1.3 else "FAIL"}')
if rg:
    print(f'  ragged fill 0.97-1.03x: {", ".join(f"{x:.3f}" for x in rg)} -> {"PASS" if all(0.97 <= x <= 1.03 for x in rg) else "FAIL"}')
print(f'  fill matrices base == head: {"yes" if hashes_ok else "NO"}')
if DRY:
    print('DRY RUN: the verdicts above are not measurements')
