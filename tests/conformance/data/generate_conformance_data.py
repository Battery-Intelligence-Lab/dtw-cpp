#!/usr/bin/env python3
"""Deterministic generator for the cross-language conformance dataset.

This script emits ``conformance_series.csv`` — the SMALL recorded dataset that
the DTWC++ 2.0 cross-language conformance fixture (Phase 2 Task 2.4) loads in all
four language routes (C++ / Python / MATLAB / CLI). It takes NO random input:
every value is an explicit integer, so the CSV is stable committed text and
parses byte-identically under C++ ``operator>>``, Python ``float()`` and MATLAB
``str2double`` (integer-valued doubles are exact in IEEE-754).

Design rationale (why this data makes the parity gate deterministic)
--------------------------------------------------------------------
FastPAM's medoid ARRAY order and cluster-id numbering depend on the k-means++
initialisation, which draws from the global RNG — a state the Python/MATLAB/CLI
routes cannot be forced to share. The gate therefore canonicalises the result
(sort the medoid SET; label each point by the rank of its assigned medoid) in
every route, and the DATA is engineered so the canonical result is unique and
init-independent:

  * 3 clusters, baselines 0 / 100 / 200, so inter-cluster gaps (>=~86) dwarf the
    intra-cluster spread (<=~14). The 3-way grouping is unambiguous and any
    k-means++ init converges to the same partition (a cross-cluster medoid is
    always fixed by a huge-gain swap).
  * Within a cluster every member is the SAME phase-shifted integer pulse plus a
    vertical offset in {-4,...,+4} (9 members => ODD count => unique median).
    Pairwise intra-cluster DTW distance depends only on |offset_i - offset_j|
    (a constant vertical shift), so the offset-0 member is the UNIQUE medoid
    (strictly: total distance T(1)-T(0) = f(5)-f(4) > 0 for the DTW distance f).
  * Cluster pulse centres are 4 apart (> band 3), so the fixed Sakoe-Chiba band
    genuinely constrains cross-cluster warping — the banded-DTW kernel is
    exercised and the fixture's band value is load-bearing for the exact scores.

Regenerate with:  python generate_conformance_data.py
(then re-run the C++ route in regen mode to refresh conformance_reference.txt).
"""
from pathlib import Path

L = 16                      # series length (short)
BASELINES = [0, 100, 200]   # one per cluster; huge separation
PULSE_CENTRES = [3, 8, 12]  # phase-shifted per cluster; >band apart
OFFSETS = list(range(-4, 5))  # {-4..4}: 9 members, odd => unique median medoid


def pulse(centre: int, t: int) -> int:
    """Triangular integer pulse, height 6, half-width 3 (values 2,4,6,4,2)."""
    return max(0, 6 - 2 * abs(t - centre))


def main() -> None:
    rows = []
    for c, (base, centre) in enumerate(zip(BASELINES, PULSE_CENTRES)):
        for off in OFFSETS:
            rows.append([base + pulse(centre, t) + off for t in range(L)])

    out = Path(__file__).with_name("conformance_series.csv")
    with out.open("w", newline="\n") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {len(rows)} series x {L} samples -> {out}")


if __name__ == "__main__":
    main()
