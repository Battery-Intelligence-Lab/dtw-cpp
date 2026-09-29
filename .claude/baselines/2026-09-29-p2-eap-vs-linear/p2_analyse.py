"""P2 analysis: per-dataset table, verdict count and crossover from the probe's per-pair CSV (stdlib only).

    uv run --no-project python p2_analyse.py p2_pairs.csv [p2_pairs_rock_raw.csv]

Ratios are t_EAP / t_linear (> 1: linear faster), per pair the median over rounds of the paired per-round ratio.
"cycles" = QueryThreadCycleTime (cycles charged to the probe's thread only); "wall" = QueryPerformanceCounter.
"""

import csv
import statistics
import sys

READ_CYCLES = 2246.0  # median cost of one QueryThreadCycleTime read, measured by the probe (p2_run.txt "clocks:")


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                for k in ("i", "j", "L", "visited", "eap_eq_lin", "eap_eq_full", "counted_eq_eap"):
                    r[k] = int(r[k])
                for k in ("computed_fraction", "eap_ns", "lin_ns", "ratio_eap_over_lin", "eap_cycles",
                          "lin_cycles", "ratio_cycles"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, round(p * (len(v) - 1))))]


def table(rows):
    order = []
    for r in rows:
        if r["dataset"] not in order:
            order.append(r["dataset"])
    print("| dataset | L | pairs | EAP µs/pair | linear µs/pair | ratio wall | ratio cycles | computed fraction "
          "median [p10–p90] | EAP wins | bitwise |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    verdict = {}
    for d in order:
        for kind in ("within", "across", "pooled"):
            s = [r for r in rows if r["dataset"] == d and (kind == "pooled" or r["kind"] == kind)]
            fr = [r["computed_fraction"] for r in s]
            wins = sum(r["ratio_cycles"] < 1 for r in s)
            ok = sum(r["eap_eq_lin"] and r["eap_eq_full"] and r["counted_eq_eap"] for r in s)
            rw = statistics.median(r["ratio_eap_over_lin"] for r in s)
            rc = statistics.median(r["ratio_cycles"] for r in s)
            print(f"| {d} | {s[0]['L']} | {kind} {len(s)} | {statistics.median(r['eap_ns'] for r in s) / 1e3:.1f} | "
                  f"{statistics.median(r['lin_ns'] for r in s) / 1e3:.1f} | {rw:.2f} | {rc:.2f} | "
                  f"{statistics.median(fr):.3f} [{q(fr, 0.1):.3f}–{q(fr, 0.9):.3f}] | {wins}/{len(s)} | "
                  f"{ok}/{len(s)} |")
            if kind == "pooled":
                verdict[d] = (rw, rc)
    return verdict


def crossover(rows):
    # Per-visited-cell cost of EAP against per-cell cost of linear, in thread cycles, the read cost removed.
    print("\nPer-cell costs (thread cycles, one clock read removed), median over pairs:")
    print("| dataset | L | EAP cycles / visited cell | linear cycles / cell | break-even computed fraction f* |")
    print("| --- | --- | --- | --- | --- |")
    seen = []
    for r in rows:
        if r["dataset"] not in seen:
            seen.append(r["dataset"])
    for d in seen:
        s = [r for r in rows if r["dataset"] == d]
        L = s[0]["L"]
        epv = statistics.median((r["eap_cycles"] - READ_CYCLES) / r["visited"] for r in s)
        lpc = statistics.median((r["lin_cycles"] - READ_CYCLES) / (L * L) for r in s)
        print(f"| {d} | {L} | {epv:.2f} | {lpc:.2f} | {lpc / epv:.3f} |")

    print("\nMedian cycle ratio by computed fraction (all pairs given):")
    edges = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.0001]
    pts = []
    for lo, hi in zip(edges, edges[1:]):
        s = [r["ratio_cycles"] for r in rows if lo <= r["computed_fraction"] < hi]
        if s:
            m = statistics.median(s)
            pts.append(((lo + min(hi, 1)) / 2, m))
            print(f"  [{lo:.2f}, {min(hi, 1):.2f}): {len(s):4d} pairs, median ratio {m:.3f}, "
                  f"EAP wins {sum(x < 1 for x in s)}")
    # Where the per-pair ratio crosses 1: least-squares line through the pairs near it, and the overlap zone.
    near = [(r["computed_fraction"], r["ratio_cycles"]) for r in rows if 0.20 <= r["computed_fraction"] <= 0.45]
    if len(near) > 2:
        mx = sum(f for f, _ in near) / len(near)
        my = sum(c for _, c in near) / len(near)
        b = sum((f - mx) * (c - my) for f, c in near) / sum((f - mx) ** 2 for f, _ in near)
        a = my - b * mx
        wins = [f for f, c in near if c < 1]
        losses = [f for f, c in near if c >= 1]
        print(f"\nFit over {len(near)} pairs with fraction 0.20-0.45: ratio = {a:.3f} + {b:.3f} f, "
              f"ratio 1 at f = {(1 - a) / b:.3f}; highest fraction EAP wins {max(wins, default=float('nan')):.3f}, "
              f"lowest fraction linear wins {min(losses, default=float('nan')):.3f}")
    total = sum(r["eap_cycles"] for r in rows) / sum(r["lin_cycles"] for r in rows)
    print(f"All {len(rows)} pairs: sum of median cycles EAP / linear = {total:.3f}; "
          f"EAP wins {sum(r['ratio_cycles'] < 1 for r in rows)}")
    lows = sorted((r for r in rows), key=lambda r: r["ratio_cycles"])[:5]
    print("\nFive best pairs for EAP (cycle ratio):")
    for r in lows:
        print(f"  {r['dataset']} {r['kind']} ({r['i']},{r['j']}) L={r['L']} computed {r['computed_fraction']:.3f} "
              f"ratio {r['ratio_cycles']:.3f}")


def main():
    rows = load(sys.argv[1:])
    verdict = table(rows)
    print("\nVerdict count (pooled median ratio > 1: linear faster):")
    counted = [d for d in verdict if d != "Rock"]
    lin = [d for d in counted if verdict[d][1] > 1]
    print(f"  cycles: linear faster on {len(lin)} of {len(counted)}: {lin}")
    lin_w = [d for d in counted if verdict[d][0] > 1]
    print(f"  wall:   linear faster on {len(lin_w)} of {len(counted)}: {lin_w}")
    crossover(rows)


if __name__ == "__main__":
    main()
