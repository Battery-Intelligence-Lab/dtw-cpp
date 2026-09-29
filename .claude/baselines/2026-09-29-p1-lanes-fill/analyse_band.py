"""Summarise the P1 band runs in C:/D/git/wt/tmp/P1/bench (stdlib only)."""
import csv
import glob
import json
import os
import re
import statistics

import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/D/git/wt/tmp/P1/bench"


def med(v):
    return statistics.median(v)


def agg(path, stat="median"):
    """name -> real_time (ms) of the aggregate `stat` in a Google Benchmark JSON."""
    data = json.load(open(path))
    res = {}
    for b in data["benchmarks"]:
        if b.get("aggregate_name") == stat:
            t = b["real_time"]
            unit = b.get("time_unit", "ns")
            res[b["run_name"]] = t * {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}[unit]
    return res


def fills():
    rounds = sorted({int(m.group(1)) for f in os.listdir(OUT) if (m := re.match(r"fill_base_r(\d+)\.json", f))})
    names = sorted(agg(f"{OUT}/fill_base_r{rounds[0]}.json"))
    for name in names:
        base, tip, ratio = [], [], []
        for r in rounds:
            b = agg(f"{OUT}/fill_base_r{r}.json")[name]
            t = agg(f"{OUT}/fill_tip_r{r}.json")[name]
            base.append(b)
            tip.append(t)
            ratio.append(b / t)
        print(f"{name}: rounds={len(rounds)} base median {med(base):.3f} ms [{min(base):.3f}-{max(base):.3f}]  "
              f"tip median {med(tip):.3f} ms [{min(tip):.3f}-{max(tip):.3f}]  "
              f"ratio median {med(ratio):.2f} [{min(ratio):.2f}-{max(ratio):.2f}]")
        print("   per-round ratios: " + " ".join(f"{x:.2f}" for x in ratio))


def ucr():
    pat = re.compile(r"round (\d+) fill_s=([\d.]+) cpu_s=([\d.]+) hash=(\w+)")
    rows = {}
    hashes = set()
    for f in sorted(glob.glob(f"{OUT}/ucr_*_r*.txt")):
        kind, r = re.search(r"ucr_(\w+)_r(\d+)\.txt", f).groups()
        text = open(f).read()
        head = text.splitlines()[0] if text else ""
        vals = [m.groups() for m in pat.finditer(text)]
        for _, _, _, h in vals:
            hashes.add(h)
        warm = [v for v in vals if v[0] == "1"]
        if warm:
            rows.setdefault(int(r), {})[kind] = (float(warm[0][1]), float(warm[0][2]))
    print("ucr header:", head)
    wall, cpu = [], []
    for r in sorted(rows):
        p, l = rows[r]["perpair"], rows[r]["lanes"]
        wall.append(p[0] / l[0])
        cpu.append(p[1] / l[1])
        print(f"  r{r}: per-pair {p[0]:.3f} s (cpu {p[1]:.1f} s)  lanes {l[0]:.3f} s (cpu {l[1]:.1f} s)  "
              f"wall x{p[0] / l[0]:.2f}  cpu x{p[1] / l[1]:.2f}")
    pw = [rows[r]["perpair"][0] for r in rows]
    lw = [rows[r]["lanes"][0] for r in rows]
    print(f"ucr: rounds={len(rows)} per-pair median {med(pw):.3f} s [{min(pw):.3f}-{max(pw):.3f}]  "
          f"lanes median {med(lw):.3f} s [{min(lw):.3f}-{max(lw):.3f}]  wall ratio median {med(wall):.2f} "
          f"[{min(wall):.2f}-{max(wall):.2f}]  cpu ratio median {med(cpu):.2f} [{min(cpu):.2f}-{max(cpu):.2f}]")
    print("ucr hashes:", sorted(hashes))


def pinned():
    files = sorted(glob.glob(f"{OUT}/pinned_r*.json"))
    per = {}
    for f in files:
        for k, v in agg(f).items():
            per.setdefault(k, []).append(v)
    for k, v in sorted(per.items()):
        print(f"pinned {k}: medians " + " ".join(f"{x:.4f}" for x in v) + " ms")
    if files:
        lin = per["BM_dtwFull_L/1000"]
        ban = per["BM_dtwBanded/1000/100"]
        lf = per["BM_dtwLanes/1000/-1"]
        lb = per["BM_dtwLanes/1000/100"]
        rf = [8 * a / b for a, b in zip(lin, lf)]
        rb = [8 * a / b for a, b in zip(ban, lb)]
        print("pinned ratio full (8 x linear / lanes): " + " ".join(f"{x:.2f}" for x in rf))
        print("pinned ratio band 100 (8 x banded / lanes): " + " ".join(f"{x:.2f}" for x in rb))


def load():
    path = f"{OUT}/load.csv"
    if not os.path.exists(path):
        return
    vals = []
    with open(path, newline="") as fh:
        for row in list(csv.reader(fh))[1:]:
            try:
                vals.append(float(row[1]))
            except (ValueError, IndexError):
                pass
    if vals:
        q = statistics.quantiles(vals, n=4)
        print(f"load (total CPU %, 5 s samples): n={len(vals)} min {min(vals):.0f} q1 {q[0]:.0f} "
              f"median {med(vals):.0f} q3 {q[2]:.0f} max {max(vals):.0f}")


fills()
ucr()
pinned()
load()
for f in sorted(glob.glob(f"{OUT}/kernel_*.txt")):
    print(os.path.basename(f))
    print(open(f).read().strip())
