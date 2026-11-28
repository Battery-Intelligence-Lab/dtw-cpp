---
title: "Exact solvers"
weight: 30
---

# Exact solvers

## HiGHS (default)

HiGHS (1.16.0) is the default open-source MIP backend. `dtwc_cl`, C++ and the MATLAB MEX
link it (`-DDTWC_ENABLE_HIGHS=ON`, the default). The Python wheel does not:
`method="mip"` builds the same model in C++ and solves it with highspy, which
dtwcpp requires (`pip install dtwcpp` installs it). If neither can be loaded,
requesting HiGHS raises `SolverError` naming the fix; it never returns an empty or
partially solved clustering as though it succeeded.

Both HiGHS and Gurobi build Balinski's compact CSC model directly with $N$ binary diagonal medoid variables and $N(N-1)$ continuous assignment variables ($N^2$ columns, $\approx 3N^2$ nonzeros). Trivial shortcuts handle $k = 1$ (zero-swap FastPAM) and $k = N$ directly, and zero-cost warm starts decode immediately; otherwise objective coefficients are scaled by `min(max_distance / 2, pam_cost)`. HiGHS runs serially (`mip_rel_gap = mip_gap`, `mip_abs_gap = 0.0`, `parallel = "off"`, `threads = 1`, resetting the thread scheduler around each solve) so equal-cost medoid tie-breaking is deterministic across machines and root solves avoid multi-thread contention. Because CSC column/row indices use 32-bit `int`, `N > ~26,750` exceeds `INT_MAX` nonzeros and raises `SolverError` directing the caller to `Method::LRCore`.

## Gurobi (optional, external)

Gurobi remains available for licensed installations. It is off by default, because
a binary linked with it needs the Gurobi library to load:

```sh
cmake -S . -B build -DDTWC_ENABLE_GUROBI=ON
```

Point `GUROBI_HOME` at the installation; with the option on and no installation
found, the configure stops with an error that names the option.

The Python wheels do not bundle Gurobi. Use a source build linked against your
licensed SDK.

## LR-core exact route

`method="lrcore"` combines an Erlenkotter second-nearest Lagrangian root bound,
Volume-style exponential primal averages, cluster-grouped $O(\sum_c |C_c|^2)$
Lloyd + FastPAM 1-swap polish, Beasley reduced-cost candidate fixing, and
zero-allocation $y$-space branch-and-bound with precomputed suffix minima and
Erlenkotter client-floor lifts. `LR_core_clustering` reads the packed
lower-triangular `DistanceMatrix` (in-RAM `float`/`double` or memory-mapped
`.dtwm`) zero-copy via `PackedMatrixView<T>` and cache-line-blocked 8-row sweeps
without materialising a dense $N \times N$ double array or calling any external
MIP solver. It is strongest on separated,
clustered data where the LP root is tight. Uniform non-metric/adversarial data can
leave a wide gap and grow the tree; `mip::lagrangian_root_exact` then reports
`certified_optimal=false` rather than claiming an unproved optimum, and
`Method::LRCore` — an exact route — turns that into a `dtwc::SolverError`
instead of publishing the uncertified incumbent. Raise the cap with
`problem.mip_settings.lr_max_nodes` (default 2,000,000), or use
`Method::Kmedoids` when a heuristic answer is what you want.

The root bound is the subgradient root in every language and build; it needs no
solver.

What the route does and does not establish:

- On clustered instances the root gap was at most 0.1% on 40 of 40 test
  instances; certification across the full UCR archive has not been run.
- Reduced-cost fixing has no guaranteed floor: on 36 qualifying clustered test
  instances it eliminated 80.3% of the candidates on average (minimum 73.3%).
- Fractional solutions are not generally half-integral (values of 1/4, 1/3, and
  3/4 occur), so odd-cycle cuts cannot replace branching.
- No throughput at N=10,000 is claimed: it has not been benchmarked on a quiet
  machine.

See the [full derivation](../../math/lr-core/).
