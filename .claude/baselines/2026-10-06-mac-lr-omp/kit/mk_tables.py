"""Markdown tables for the record from the kit's results.json files (stdlib only):  uv run --no-project python mk_tables.py
sweep1 (every N forks, loop 2 forked), sweep1b (loop 2 serial, every N forks, HiGHS-OFF), sweep2 (the final tree; its OFF cells), sweep2on (its ON cells,
one libhighs for base and head), confirm-ab / confirm-aa. Prints to stdout."""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def load(prefix):
    d = sorted(p for p in RUNS.glob(f"{prefix}-2026*") if p.is_dir())[-1]
    return {(c["flavor"], c["regime"], c["N"]): c for c in json.loads((d / "results.json").read_text())}, d.name


def row(cells, flavor, regime, sizes, key, fmt="{:.2f}"):
    out = []
    for n in sizes:
        c = cells.get((flavor, regime, n))
        v = None if c is None or key not in c else c[key]["ratio"]
        out.append("-" if v is None or v != v else fmt.format(v))
    return out


def table(title, cells, sizes, rows):
    print(f"\n{title}\n")
    print("| | " + " | ".join(f"N={n}" for n in sizes) + " |")
    print("| --- | " + " | ".join("---:" for _ in sizes) + " |")
    for label, flavor, regime, key in rows:
        print(f"| {label} | " + " | ".join(row(cells, flavor, regime, sizes, key)) + " |")


s1, n1 = load("sweep1")
s1b, n1b = load("sweep1b")
s2, n2 = load("sweep2")
s2on, n2on = load("sweep2on")
final = {k: v for k, v in s2.items() if k[0] == "off"}
final.update(s2on)
SIZES1 = [100, 200, 400, 800, 1600, 3200]
SIZES2 = [100, 200, 240, 280, 320, 400, 800, 1600, 3200]
print(f"<!-- sources: {n1}, {n1b}, {n2} (OFF cells), {n2on} (ON cells) -->")
table("Sweep 1 (both loops forked, every N forks, `HEAD_MIN_N=1`): root, base / head", s1, SIZES1,
      [("HiGHS-OFF noise", "off", "noise", "root"), ("HiGHS-OFF line", "off", "line", "root"),
       ("HiGHS-ON noise", "on", "noise", "root"), ("HiGHS-ON line", "on", "line", "root")])
table("Sweep 1b (loop 2 serial, every N forks, HiGHS-OFF): root, base / head", s1b, [200, 240, 280, 320, 360, 400],
      [("noise", "off", "noise", "root"), ("line", "off", "line", "root")])
table("Sweep 2 (the final tree): ratios base / head", final, SIZES2,
      [("OFF noise, root", "off", "noise", "root"), ("OFF noise, LR phase", "off", "noise", "lr"), ("OFF noise, `dtwc_cl` wall", "off", "noise", "e2e"),
       ("OFF line, root", "off", "line", "root"), ("OFF line, LR phase", "off", "line", "lr"), ("OFF line, `dtwc_cl` wall", "off", "line", "e2e"),
       ("ON noise, root", "on", "noise", "root"), ("ON noise, `dtwc_cl` wall", "on", "noise", "e2e"),
       ("ON line, root", "on", "line", "root"), ("ON line, `dtwc_cl` wall", "on", "line", "e2e")])
print("\nSweep 2, absolute (median over 3, ms for the root, s for `dtwc_cl`): base -> head\n")
print("| | N=800 | N=1600 | N=3200 |")
print("| --- | ---: | ---: | ---: |")
import statistics
for label, flavor, regime, key, unit in (("OFF noise root", "off", "noise", "root", 1), ("OFF line root", "off", "line", "root", 1),
                                         ("OFF noise `dtwc_cl`", "off", "noise", "e2e", 1), ("OFF line `dtwc_cl`", "off", "line", "e2e", 1),
                                         ("ON noise root", "on", "noise", "root", 1), ("ON line root", "on", "line", "root", 1)):
    cols = []
    for n in (800, 1600, 3200):
        c = final[(flavor, regime, n)][key]
        b, h = statistics.median(c["base"]), statistics.median(c["head"])
        cols.append(f"{b:,.0f} -> {h:,.0f}" if key == "root" else f"{b:.2f} -> {h:.2f}")
    print(f"| {label} | " + " | ".join(cols) + " |")
low = [(k, c["identical"]) for k, c in final.items() if not c["identical"]]
print(f"\ncells not bit-identical: {low}")
idle = sorted(round(c["idle_before"], 1) for c in final.values() if c.get("idle_before") is not None)
print(f"idle before the cells: min {idle[0]}, median {idle[len(idle) // 2]}, max {idle[-1]}")
worst = sorted(final.values(), key=lambda c: c["root"]["ratio"])[:4]
print("lowest root ratios:", [(c["flavor"], c["regime"], c["N"], round(c["root"]["ratio"], 3)) for c in worst])
on_hi = [c for c in final.values() if c["flavor"] == "on" and c["N"] >= 800]
print("ON root at N >= 800:", sorted(round(c["root"]["ratio"], 2) for c in on_hi))
off_hi = [c for c in final.values() if c["flavor"] == "off" and c["N"] >= 800]
print("OFF root at N >= 800:", sorted(round(c["root"]["ratio"], 2) for c in off_hi))
print("OFF e2e at N >= 800:", sorted(round(c["e2e"]["ratio"], 2) for c in off_hi))
print("ON e2e at N >= 800:", sorted(round(c["e2e"]["ratio"], 2) for c in on_hi))
