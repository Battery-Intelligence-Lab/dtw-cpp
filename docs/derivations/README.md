# Mathematical derivations

This index is the reproducible science map for DTWC++. A derivation is listed
only after its source verification, unit analysis, assumptions, approximation
statement, code-conformance table, independent oracle, and decisive gate are
present.

| ID | Topic | File | Verdict |
|---|---|---|---|
| D1 | DTW recurrence and Sakoe–Chiba adjustment window | [01-dtw-recurrence-sakoe-chiba.md](01-dtw-recurrence-sakoe-chiba.md) | CPU **CONFIRMED**; real-CUDA **CONFIRMED**; Metal source **CONFIRMED**, and real-device execution **CONFIRMED** on Apple M5 Pro (2026-09-23: the Metal half of the `[F12]` tests) |
| D2 | Envelopes and LB_Keogh admissibility | [02-envelopes-lb-keogh.md](02-envelopes-lb-keogh.md) | Scalar L1 envelopes and LB_Keogh, with the feasible unequal-length prefix, **CONFIRMED**; envelope provenance (an `Envelope` does not record its window) and TADPole's empty domain (zero bounds where DTW has no path) **DISCREPANCY** |

Verdict meanings:

- **CONFIRMED** — the derivation, live code, and named executable evidence
  agree inside the stated assumptions.
- **DISCREPANCY** — a live implementation or documented promise disagrees
  with the derivation; the table says what disagrees.
- **OPEN** — available evidence is insufficient; the derivation names the
  probe that would decide it.
