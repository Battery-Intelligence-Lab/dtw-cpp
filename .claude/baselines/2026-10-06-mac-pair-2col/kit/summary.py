"""Summarises a run.sh results file: per shape and code placement, the median over repeats for base, k1 and
head, the speed-ups against base (median and range of the per-repeat ratios; wall clock, and normalised by the
add-chain clock measured around each run), the checksums and matrix hashes (every build must agree), the bands
the orchestrator registered before any measurement, and the decision for kernel 2. No verdict is given if a
probe failed or a shape is missing for some build, placement or repeat. usage: summary.py <results file>"""
import re
import statistics
import sys
from collections import defaultdict

TIME = re.compile(r'^r(\d+) (base|k1|head) (p\d) time (f64|f32) (L1|Sq) nx (\d+)\s+ny (\d+)\s+band (-?\d+)\s+'
                  r'ns/cell ([\d.]+) .*?cycles/cell@4\.59GHz ([\d.]+)\s+cycles/cell@measured ([\d.]+)\s+'
                  r'clock ([\d.]+) GHz.*checksum (\S+)')
FILL = re.compile(r'^r(\d+) (base|k1|head) (p\d) fill (f64|f32) (L1|Sq) threads\s+(\d+) N (\d+)\s+L (\d+)\s*'
                  r'( ragged\+-10)? band (-?\d+)\s+W (\d+)\s+ms ([\d.]+) .*?Gcell/s ([\d.]+)\s+matrix hash ([0-9a-f]+)')
VERSIONS = ('base', 'k1', 'head')
CLOCK_SPREAD = 0.03
med = statistics.median

lines = open(sys.argv[1]).read().splitlines()
DRY = 'mode dry' in lines[0]
print(lines[0])
if DRY:
    print('DRY RUN (tiny sizes, one repeat): checks that the kit runs; none of the numbers below is a measurement')

times = defaultdict(lambda: defaultdict(dict))  # (shape, placement) -> version -> repeat -> (ns, cyc, cycm, ghz, chk)
fills = defaultdict(lambda: defaultdict(dict))  # (shape, placement) -> version -> repeat -> (gcell, hash)
failed = [l for l in lines if ' FAILED ' in l]
for line in lines:
    if m := TIME.match(line):
        r, v, p, t, d, nx, ny, band, ns, cyc, cycm, ghz, chk = m.groups()
        times[((t, d, int(nx), int(ny), int(band)), p)][v][int(r)] = (float(ns), float(cyc), float(cycm), float(ghz), chk)
    elif m := FILL.match(line):
        r, v, p, t, d, th, n, l, rag, band, w, ms, g, h = m.groups()
        fills[((t, d, int(th), int(n), int(l), bool(rag), int(band)), p)][v][int(r)] = (float(g), h)

# completeness: every shape in every placement, for every build, in every repeat
problems = [f'probe failed: {l[:160]}' for l in failed]
repeats = {r for table in (times, fills) for rows in table.values() for v in rows.values() for r in v}
for table, what in ((times, 'single-thread'), (fills, 'fill')):
    placements_seen = {p for (_, p) in table}
    shapes = {s for (s, _) in table}
    for s in shapes:
        for p in placements_seen:
            rows = table.get((s, p), {})
            for v in VERSIONS:
                if set(rows.get(v, {})) != repeats:
                    problems.append(f'{what} shape {s} placement {p} build {v}: repeats {sorted(rows.get(v, {}))} of {sorted(repeats)}')
if problems:
    print(f'\nINCOMPLETE: {len(problems)} problem(s); the verdicts below are NOT VALID')
    for p in problems[:20]:
        print('  ' + p)


def shape_name(s):
    t, d, nx, ny, band = s
    return f'{t} {d} {nx}x{ny} {"unbanded" if band < 0 else f"band {band}"}'


wall, norm = {}, {}  # (version, shape, placement) -> median speed-up
spread_flags = 0
bitwise_ok = True
print('\nsingle thread, one pair at a time (median over repeats; x = base ns / build ns per repeat; clk-x = the same on\n'
      f'cycles at the add-chain clock; "!" = some repeat\'s base and build clocks differ by more than {CLOCK_SPREAD:.0%})')
print(f'{"shape":26s} {"pl":2s} {"base ns":>8s} {"base cyc":>8s} {"k1 cyc":>7s} {"head cyc":>8s} '
      f'{"k1 x":>6s} {"(range)":>13s} {"head x":>6s} {"(range)":>13s} {"k1 clk-x":>8s} {"head clk-x":>10s} {"GHz":>5s} check')
for key in sorted(times, key=lambda k: (k[0], k[1])):
    s, p = key
    rows = times[key]
    if not all(rows[v] for v in VERSIONS):
        continue
    cells, flag = {}, ''
    for v in ('k1', 'head'):
        reps = sorted(set(rows['base']) & set(rows[v]))
        rw = [rows['base'][r][0] / rows[v][r][0] for r in reps]
        rn = [rows['base'][r][2] / rows[v][r][2] for r in reps]
        spread = max(abs(rows[v][r][3] - rows['base'][r][3]) / rows['base'][r][3] for r in reps)
        if spread > CLOCK_SPREAD:
            flag, spread_flags = '!', spread_flags + 1
        wall[(v, s, p)], norm[(v, s, p)] = med(rw), med(rn)
        cells[v] = (med(rw), min(rw), max(rw), med(rn))
    chks = {rows[v][r][4] for v in VERSIONS for r in rows[v]}
    bitwise_ok = bitwise_ok and len(chks) == 1
    ghz = med([rows[v][r][3] for v in VERSIONS for r in rows[v]])
    b = rows['base'].values()
    print(f'{shape_name(s):26s} {p:2s} {med(x[0] for x in b):8.5f} {med(x[1] for x in b):8.3f} '
          f'{med(x[1] for x in rows["k1"].values()):7.3f} {med(x[1] for x in rows["head"].values()):8.3f} '
          f'{cells["k1"][0]:6.3f} ({cells["k1"][1]:.3f}-{cells["k1"][2]:.3f}) '
          f'{cells["head"][0]:6.3f} ({cells["head"][1]:.3f}-{cells["head"][2]:.3f}) '
          f'{cells["k1"][3]:8.3f} {cells["head"][3]:10.3f} {ghz:5.2f}{flag:1s} {"equal" if len(chks) == 1 else "DIFFER"}')

fill_speed = {}  # (version, kind, placement) -> median speed-up
print('\nfills (median Gcell/s over repeats; x = build / base per repeat)')
for key in sorted(fills, key=lambda k: (not k[0][5], k[0], k[1])):
    s, p = key
    t, d, th, n, l, rag, band = s
    rows = fills[key]
    if not all(rows[v] for v in VERSIONS):
        continue
    hashes = {rows[v][r][1] for v in VERSIONS for r in rows[v]}
    bitwise_ok = bitwise_ok and len(hashes) == 1
    sp = {}
    kind = ('ragged' if rag else 'equal') + (' unbanded' if band < 0 else ' banded')
    for v in ('k1', 'head'):
        reps = sorted(set(rows['base']) & set(rows[v]))
        rr = [rows[v][r][0] / rows['base'][r][0] for r in reps]
        sp[v] = (med(rr), min(rr), max(rr))
        fill_speed[(v, kind, p, s)] = sp[v][0]
    name = f'{t} {d} N{n} L{l}{"+-10 ragged" if rag else ""} {"unbanded" if band < 0 else f"band {band}"} T{th}'
    print(f'{name:40s} {p}  base {med(x[0] for x in rows["base"].values()):7.2f}  k1 x {sp["k1"][0]:.3f} '
          f'({sp["k1"][1]:.3f}-{sp["k1"][2]:.3f})  head x {sp["head"][0]:.3f} ({sp["head"][1]:.3f}-{sp["head"][2]:.3f})  '
          f'matrix {"identical" if len(hashes) == 1 else "DIFFERS"}')


def ok(flag):
    return 'PASS' if flag else 'FAIL'


def verdicts(speed, label):
    """The registered bands on one set of single-thread speed-ups (the fills are wall clock in both)."""
    print(f'\nbands registered before the run (the orchestrator, 2026-10-06), single thread by {label}:')
    placements = sorted({p for (_, _, p) in speed})
    k1_shapes = [('f64', 'L1', 100, 100, -1), ('f64', 'L1', 1000, 1000, -1)]
    for v in ('k1', 'head'):
        vals = [speed[(v, s, p)] for s in k1_shapes for p in placements if (v, s, p) in speed]
        if vals:
            print(f'  {v}: kernel 1 unbanded f64 L1 at L 100 and L 1000 >= 1.3x in every placement: min {min(vals):.3f} '
                  f'-> {ok(min(vals) >= 1.3 and len(vals) == 2 * len(placements))}')
    # kernel 2: the prescribed banded shapes (band L/10, and the ragged pair at band 20) >= 1.15x; nothing of head's
    # below 0.97x: every single-thread shape (band 2 included) and the ragged band-10 fill (registered 2026-10-06
    # by the implementer, before any measurement, as the reading of "none below 0.97x")
    k2 = [x for (v, s, p), x in speed.items() if v == 'head' and s[4] >= 0 and (s[4] == s[2] // 10 or s[2] != s[3])]
    head_all = [x for (v, s, p), x in speed.items() if v == 'head']
    head_rb = [x for (v, k, p, s), x in fill_speed.items() if v == 'head' and k == 'ragged banded']
    keep = bool(k2) and min(k2) >= 1.15 and min(head_all) >= 0.97 and (not head_rb or min(head_rb) >= 0.97)
    if k2:
        print(f'  kernel 2 (head against base) banded shapes L/10 and the ragged pair at band 20 >= 1.15x in every '
              f'placement: min {min(k2):.3f} -> {ok(min(k2) >= 1.15)}')
        print(f'  head none below 0.97x: single thread min {min(head_all):.3f} -> {ok(min(head_all) >= 0.97)}; ragged '
              f'band-10 fill min {min(head_rb) if head_rb else float("nan"):.3f} -> {ok(not head_rb or min(head_rb) >= 0.97)}')
        print(f'  => kernel 2 two columns per pass: {"KEEP (head is the candidate)" if keep else "DROP (revert 00fb9c36; k1 is the candidate)"}')
    final = 'head' if keep else 'k1'
    fin = [x for (v, s, p), x in speed.items() if v == final]
    if fin:
        worst = min((x, s, p) for (v, s, p), x in speed.items() if v == final)
        print(f'  {final}: no single-thread shape below 0.97x in any placement: min {worst[0]:.3f} ({shape_name(worst[1])} '
              f'{worst[2]}) -> {ok(worst[0] >= 0.97)}')
    rg = [x for (v, k, p, s), x in fill_speed.items() if v == final and k == 'ragged unbanded']
    eq = [x for (v, k, p, s), x in fill_speed.items() if v == final and k.startswith('equal')]
    if rg:
        print(f'  {final}: ragged unbanded fill (18 threads) >= 1.2x: {", ".join(f"{x:.3f}" for x in rg)} -> {ok(min(rg) >= 1.2)}')
    if eq:
        print(f'  {final}: equal-length (lanes) fill within 0.97-1.03x: {", ".join(f"{x:.3f}" for x in eq)} -> '
              f'{ok(all(0.97 <= x <= 1.03 for x in eq))}')
    if fin:
        unb = [x for (v, s, p), x in speed.items() if v == final and s[4] < 0]
        bnd = [x for (v, s, p), x in speed.items() if v == final and s[4] >= 0]
        print(f'  {final} against base, medians over all placements: unbanded single pair {min(unb):.2f}-{max(unb):.2f}x, '
              f'banded {min(bnd):.2f}-{max(bnd):.2f}x' + (f', ragged unbanded fill {min(rg):.2f}-{max(rg):.2f}x' if rg else ''))
    return keep


keep_wall = verdicts(wall, 'wall clock (the primary reading)')
keep_norm = verdicts(norm, 'cycles at the add-chain clock (the cross-check)')
print(f'\n  shape/placement pairs where a repeat\'s base and build clocks differ by more than {CLOCK_SPREAD:.0%}: {spread_flags}')
if keep_wall != keep_norm:
    print('  the kernel 2 decision differs between the two readings: rerun before deciding')
print(f'  checksums and fill matrices equal across base, k1 and head: {"yes" if bitwise_ok else "NO"}')
if problems:
    print('INCOMPLETE: the verdicts above are NOT VALID')
if DRY:
    print('DRY RUN: the verdicts above are not measurements')
