"""Diagnostic A/B timing (not a record): pinned single-thread fills on logical CPU 12, the builds alternating (each
round in a rotated order), `--runs` rounds. Each run is one probe process: `fill T D N lo hi band 1 1 12` (one untimed
fill, one timed), OMP_NUM_THREADS=1, PATH starting with LLVM's and Gurobi's bin. Prints every run and, per build, the
timed fill's ms (min, median, max) and the clock. Appends everything to diag/ab.txt.

usage: ab.py --shape "f64 L1 24 8000 8000 -1" [--runs 5] <name=exe> [<name=exe> ...]
"""
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

KIT = pathlib.Path(__file__).resolve().parents[1]
args = sys.argv[1:]
shape = args[args.index("--shape") + 1].split()
runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 5
builds = [(n, str((KIT / e).resolve())) for n, e in (a.split("=", 1) for a in args if "=" in a and not a.startswith("--"))]
# --envpad n1,n2,...: each build also runs with an environment variable RLPAD of n characters, which moves the
# process heap's first allocations (data placement), as separate entries name@pad<n>.
if "--envpad" in args:
    pads = [int(x) for x in args[args.index("--envpad") + 1].split(",")]
    builds = [(f"{n}@pad{k}", e, k) for n, e in builds for k in pads]
else:
    builds = [(n, e, 0) for n, e in builds]
env = {**os.environ, "OMP_NUM_THREADS": "1",
       "PATH": "C:/Program Files/LLVM/bin;C:/gurobi1301/win64/bin;" + os.environ.get("PATH", "")}
LINE = re.compile(r"ms ([\d.]+) \(min [\d.]+ max [\d.]+\)\s+Gcell/s ([\d.]+)\s+clock ([\d.]+) GHz.*matrix hash ([0-9a-f]+)")
res = {n: [] for n, _, _ in builds}
log = (KIT / "diag" / "ab.txt").open("a")
log.write(f"# ab {time.strftime('%Y-%m-%d %H:%M:%S')} shape {' '.join(shape)} runs {runs} builds "
          f"{' '.join(f'{n}={e}' for n, e, _ in builds)}\n")
for r in range(runs):
    k = r % len(builds)
    for name, exe, pad in builds[k:] + builds[:k]:
        e = {**env, "RLPAD": "x" * pad} if pad else env
        out = subprocess.run([exe, "fill", *shape, "1", "1", "12"], capture_output=True, text=True, env=e).stdout
        m = LINE.search(out)
        if not m:
            sys.exit(f"{name}: no fill line: {out}")
        ms, g, clk, h = float(m[1]), float(m[2]), float(m[3]), m[4]
        res[name].append((ms, clk, h))
        log.write(f"run {r} {name} ms {ms:.2f} Gcell/s {g:.3f} clock {clk:.3f} hash {h}\n")
        log.flush()
med = statistics.median
for name, v in res.items():
    ms = [x[0] for x in v]
    line = (f"{name:14s} ms min {min(ms):.1f} median {med(ms):.1f} max {max(ms):.1f}  clock {min(x[1] for x in v):.2f}-"
            f"{max(x[1] for x in v):.2f} GHz  hash {'/'.join(sorted({x[2] for x in v}))}")
    print(line)
    log.write(line + "\n")
base = builds[0][0]
for name, v in res.items():
    if name != base:
        ratio = med(x[0] for x in v) / med(x[0] for x in res[base])
        line = f"{name} / {base}: median ms ratio {ratio:.4f}"
        print(line)
        log.write(line + "\n")
log.close()
