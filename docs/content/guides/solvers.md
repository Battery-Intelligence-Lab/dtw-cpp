---
title: "Exact solvers"
weight: 30
---

# Exact solvers

## HiGHS (default)

HiGHS is the default open-source MIP backend and is bundled in published Python
wheels. Native builds enable it with `-DDTWC_ENABLE_HIGHS=ON`. If the backend is
not compiled, requesting it raises `SolverError`; it never returns an empty or
partially solved clustering as though it succeeded.

## Gurobi (optional, external)

Gurobi remains available for licensed installations:

```sh
cmake -S . -B build -DDTWC_ENABLE_GUROBI=ON
```

The Python wheels do not bundle Gurobi. Use a source build linked against your
licensed SDK.

## LR-core exact route

`method="lrcore"` combines a Lagrangian root bound, reduced-cost candidate
fixing, and exact branch-and-bound over medoid-open decisions. The root bound is
matrix-free with respect to the compact MIP formulation, but it still streams a
dense distance matrix: memory is Θ(N²) doubles. It is strongest on separated,
clustered data where the LP root is tight. Uniform non-metric/adversarial data can
leave a wide gap and grow the tree; a node cap reports `certified_optimal=false`
rather than claiming an unproved optimum.

The recorded claims are intentionally not polished after the fact:

- Root exactness met its registered clustered-instance band (40/40 in the
  implementation gate), but a full quiet UCR certification sweep remains open.
- The universal reduced-cost fixing floor was **falsified**: mean elimination was
  80.3%, minimum 73.3%, and 77.8% of qualifying instances reached 80%.
- Fractional solutions are not generally half-integral (values of 1/4, 1/3, and
  3/4 occur), so odd-cycle cuts cannot replace branching.
- The N=10,000 throughput claim remains a quiet-machine benchmark requirement,
  not a published result.

See the [full derivation and registered-band ledger](../../math/lr-core/).

HiGHS PDLP is retained as an LP-bound cross-check, not as the production
p-median solver: the recorded comparison found matrix-free Kelley substantially
faster at tested sizes even when GPU PDLP crossed over its CPU variant.
