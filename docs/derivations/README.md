# Mathematical derivations

This index is the reproducible science map for DTWC++. A derivation is listed
only after its source verification, unit analysis, assumptions, approximation
statement, code-conformance table, independent oracle, and decisive gate are
present. Remaining targets are tracked in `PLAN.md` Phase R2.

| ID | Topic | File | Verdict |
|---|---|---|---|
| D1 | DTW recurrence and Sakoe–Chiba adjustment window | [01-dtw-recurrence-sakoe-chiba.md](01-dtw-recurrence-sakoe-chiba.md) | CPU **CONFIRMED**; real-CUDA **CONFIRMED**; Metal source **CONFIRMED**, and real-device execution **CONFIRMED** on Apple M5 Pro (FX-13, 2026-09-23; F12 Metal half) |
| D2 | Envelopes and LB_Keogh admissibility | [02-envelopes-lb-keogh.md](02-envelopes-lb-keogh.md) | Scalar CPU L1/squared, feasible unequal prefix, and additive dependent/independent MV **CONFIRMED**; API/metric/domain/provenance **DISCREPANCY** F46–F49; GPU squared-L2, full-DTW, and `INT_MAX` bounds (F27/F28/F50) **CONFIRMED** on Metal since FX-13, CUDA `[BLOCKED-ENV]`; F29/F30 **OPEN** |

Verdict meanings:

- **CONFIRMED** — the derivation, live code, and named executable evidence
  agree inside the stated assumptions.
- **DISCREPANCY** — a live implementation or documented promise disagrees
  with the derivation; the table names its R3 finding.
- **OPEN** — available evidence is insufficient; the derivation names the
  probe that would decide it.
