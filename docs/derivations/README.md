# Mathematical derivations

This index is the reproducible science map for DTWC++. A derivation is listed
only after its source verification, unit analysis, assumptions, approximation
statement, code-conformance table, independent oracle, and decisive gate are
present. Remaining targets are tracked in `PLAN.md` Phase R2.

| ID | Topic | File | Verdict |
|---|---|---|---|
| D1 | DTW recurrence and Sakoe–Chiba adjustment window | [01-dtw-recurrence-sakoe-chiba.md](01-dtw-recurrence-sakoe-chiba.md) | CPU **CONFIRMED**; real-CUDA **CONFIRMED**; Metal source **CONFIRMED**, real-device execution **DISCREPANCY** F12 (`[BLOCKED-ENV]`) |
| D2 | Envelopes and LB_Keogh admissibility | [02-envelopes-lb-keogh.md](02-envelopes-lb-keogh.md) | Scalar CPU L1/squared, feasible unequal prefix, and additive dependent/independent MV **CONFIRMED**; API/metric/domain/provenance **DISCREPANCY** F46–F49; GPU **DISCREPANCY/OPEN** F27–F30/F50 |
| D3 | LB_Enhanced and local LB_Webb_NoLR plus tail cap | [03-lb-enhanced-webb.md](03-lb-enhanced-webb.md) | Finite equal-length scalar L1 and unrooted squared-L2 **CONFIRMED** in all three native matrices; Enhanced ordering, NoLR/Keogh dominance, F54 cascade, and F57 normal/WSL-UBSan arithmetic **CONFIRMED**; provenance/GPU/floating boundaries remain F46/F50/D17 |

Verdict meanings:

- **CONFIRMED** — the derivation, live code, and named executable evidence
  agree inside the stated assumptions.
- **DISCREPANCY** — a live implementation or documented promise disagrees
  with the derivation; the table names its R3 finding.
- **OPEN** — available evidence is insufficient; the derivation names the
  probe that would decide it.
