# Phase 8.2 F5 -- FastCLARA boundary planning

Date: 2026-07-12 (Europe/London)

## Confirmed failures

The resident and Parquet implementations did not share one execution plan.
The Parquet branch accepted `n_samples <= 0`, returned an empty/max-cost result,
and repeated a full-data sample for every nominal repetition. Both branches
used signed-`int` arithmetic for `40+2*k` and `10*k+100`. Comments claimed the
streaming path supported N beyond `INT_MAX`, although medoid indices and
`ClusteringResult` are int-indexed. The CLI added a fourth behavior for
N > 50,000 by replacing the documented formula with `sqrt(N)*k`.

The preregistered test called the new production planner before it existed.
Its expected red was the linker error:

```text
undefined symbol: dtwc::algorithms::detail::resolve_clara_plan(...)
ninja: build stopped: subcommand failed.
```

This establishes that the overflow oracle does not mirror local arithmetic in
the test.

## Resolution

`detail::resolve_clara_plan` is allocation-free and is called by both resident
and Parquet paths. It validates `n_samples`, `sample_size`, `max_iter`, N, and k
before sampling. N above `INT_MAX` is rejected before any size narrowing. The
auto policy uses int64 intermediates and then clamps to `[k,N]`; tests cover the
specific `10*k+100` overflow discriminator, maximal k, `INT_MAX`, and
`INT_MAX+1`.

A sample resolving to N has two explicit outcomes:

- resident data: exactly one invocation-local seeded FastPAM run, independent
  of `n_samples`;
- a Parquet file already exceeding `ram_limit_bytes`: typed rejection naming
  `sample_size < N` and raising the RAM limit, rather than loading all rows or
  repeating identical full samples.

The CLI now passes `sample_size=-1` unchanged, so the core policy is the single
source of truth.

## Validation

Focused Release gate, four OpenMP workers:

```text
unit_test_fast_clara
All tests passed (826 assertions in 19 test cases)
```

Clang 18.1.3 UBSan on Ubuntu 24.04 WSL used
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`; the same 826 assertions
passed with no report. MSVC 19.50 `/fsanitize=address` used
`ASAN_OPTIONS=halt_on_error=1:detect_leaks=0:strict_string_checks=1`; the same
826 assertions passed with no report.

The host has PyArrow 23 headers but no discoverable Arrow CMake package, and
the repository's Windows+Clang Arrow CPM route is explicitly unsupported.
The `DTWC_HAS_PARQUET` production branch was therefore compiled with
`clang++ -fsyntax-only` against PyArrow's Arrow/Parquet headers; it passed.
The exact streaming full-sample rejection is additionally pinned through the
same production validation function without an optional dependency.

Final canonical Windows Clang 21.1.8 Release gate (HiGHS, Gurobi, llfio ON):

```text
100% tests passed, 0 tests failed out of 113
Total Test time (real) = 125.94 sec
6 explicit capability skips
```

Independent adversarial review: PASS on all four queued boundary defects and
the strict streaming/full-data policy. It separately registered F6 (quadratic
parent cache allocation) and F7 (CLI preloads Parquet before applying its RAM
limit); those are intentionally isolated follow-up findings, not hidden in F5.
