---
title: "Exact solvers"
weight: 30
---

# Exact solvers

## HiGHS (default)

HiGHS is the default open-source MIP backend. `dtwc_cl`, C++ and the MATLAB MEX
link it (`-DDTWC_ENABLE_HIGHS=ON`, the default). The Python wheel does not:
`method="mip"` builds the same model in C++ and solves it with highspy, which
dtwcpp requires (`pip install dtwcpp` installs it). If neither can be loaded,
requesting HiGHS raises `SolverError` naming the fix; it never returns an empty or
partially solved clustering as though it succeeded.

## Gurobi (optional, external)

Gurobi remains available for licensed installations. It is off by default, because
a binary linked with it needs the Gurobi library to load:

```sh
cmake -S . -B build -DDTWC_ENABLE_GUROBI=ON
```

Point `GUROBI_HOME` at the installation; with the option on and no installation
found, the configure stops with an error that names the option. v1.0.0 linked
Gurobi whenever it found it.

The Python wheels do not bundle Gurobi. Use a source build linked against your
licensed SDK.

## LR-core exact route

`method="lrcore"` combines a Lagrangian root bound, reduced-cost candidate
fixing, and exact branch-and-bound over medoid-open decisions. The root bound is
matrix-free with respect to the compact MIP formulation, but it still streams a
dense distance matrix: memory is Θ(N²) doubles. It is strongest on separated,
clustered data where the LP root is tight. Uniform non-metric/adversarial data can
leave a wide gap and grow the tree; `mip::lagrangian_root_exact` then reports
`certified_optimal=false` rather than claiming an unproved optimum, and
`Method::LRCore` — an exact route — turns that into a `dtwc::SolverError`
instead of publishing the uncertified incumbent. Raise the cap with
`problem.mip_settings.lr_max_nodes` (default 2,000,000), or use
`Method::Kmedoids` when a heuristic answer is what you want.

The root bound is the subgradient root in every language and build; it needs no
solver. `--lr-root kelley` (Python `lr_root="kelley"`, MATLAB `'LrRoot', 'kelley'`)
runs Kelley's cutting planes instead, whose master LP solves on linked HiGHS in
`dtwc_cl`, C++ and the MATLAB MEX, and on highspy in Python, which dtwcpp requires.
Without either, asking for it raises `SolverError`
naming the fix before any distance is computed; it never falls back to the subgradient
root. Both roots reach the same optimum, at different costs: on some synthetic noisy
series the subgradient root certified at the root, with no branching, where Kelley's
needed up to 487,365 branch-and-bound nodes; on constant, level-shifted series (a
line metric) Kelley's certified in 8–13 iterations, where the subgradient root took
far longer.

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
