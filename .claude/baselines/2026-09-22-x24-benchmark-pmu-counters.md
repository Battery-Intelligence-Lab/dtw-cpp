# X-24 — PMU counters: the option, and the silent fallback it exposed

**Date:** 2026-09-22 · **Machine:** Mac (Apple M5 Pro, 18 cores, 64 GiB), AppleClang 21.0.0,
`clang-macos` preset, Release · **Row:** X-24 (W0 option; R5 artefact still open, V-5)

macOS has no `perf_event_open`, so this machine can never produce a counter. Everything below is
therefore about the **refusal paths** — which is where the row's real content turned out to be.

## What was expected, and what was found

The row read as plumbing: forward `BENCHMARK_ENABLE_LIBPFM` and document that it needs Linux.
Reading google/benchmark v1.9.5 before writing it found the configure side already safe and the
**run** side not safe at all.

Configure side (safe): `CMakeLists.txt:338` runs `find_package(PFM REQUIRED)` under the option, so a
missing libpfm4 stops the configure instead of dropping the feature.

Run side (not safe): `src/perf_counters.cc:245-263` is the no-`HAVE_LIBPFM` stub. `Create()` logs
`"Performance counters not supported."` for a non-empty request and returns `NoCounters()`. The
benchmark then runs normally, writes a complete JSON, and exits 0.

## Reproduction `[confirmed]`

Build without libpfm4 (the only kind this machine can make):

```sh
cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_BUILD_BENCHMARK=ON
cmake --build build --target bench_dtw_baseline
./build/bin/bench_dtw_baseline --benchmark_filter='BM_dtwFull/100$' \
    --benchmark_min_time=0.01s --benchmark_perf_counters=CYCLES,INSTRUCTIONS \
    --benchmark_format=json
```

Result: `Performance counters not supported.` on stderr, a full JSON report on stdout with
**no `CYCLES` and no `INSTRUCTIONS` field**, and **exit 0**. Redirect stderr and the artefact is
indistinguishable from an ordinary wall-clock record.

## Why upstream's guard does not catch it `[inferred — read, not executed]`

`src/benchmark_runner.cc:323-325`:

```cpp
    BM_CHECK(FLAGS_benchmark_perf_counters.empty() ||
             (perf_counters_measurement_ptr->num_counters() == 0))
        << "Perf counters were requested but could not be set up.";
```

`BM_CHECK(b)` (`src/check.h:85`) invokes `CheckHandler` when `b` is **false**. The condition is false
exactly when counters were requested *and* `num_counters() != 0`, i.e. when they **were** set up — so
it aborts on success and stays silent on failure, and the message describes the case it does not
catch. It is also inside `if (b.aggregation_report_mode() != internal::ARM_Unspecified)`; no
benchmark in `benchmarks/` sets a report mode (`grep -rn "ReportAggregatesOnly\|DisplayAggregatesOnly\|Repetitions" benchmarks/*.cpp` → no matches), so it never ran here at all.

Tagged `[inferred]`: the inversion is read from source. Executing it needs working counters — the
probe is in V-5.

## The fix, and what proves it

`DTWC_BENCHMARK_PMU` (cache variable, so `machine_facts.py` harvests it into every machine record)
forwards to `BENCHMARK_ENABLE_LIBPFM`. Three configure-time refusals and one run-time refusal, all
`FATAL_ERROR` / non-zero rather than a downgrade.

| # | Condition | Proof on this machine |
| --- | --- | --- |
| 1 | `DTWC_BENCHMARK_PMU=ON` on non-Linux | `[confirmed]` configure exits 1, `Dependencies.cmake:145`, "needs Linux on bare metal; this is Darwin" |
| 2 | `DTWC_BENCHMARK_PMU=ON` with `DTWC_BUILD_BENCHMARK=OFF` | `[confirmed]` configure exits 1, `Dependencies.cmake:139`, "does nothing with DTWC_BUILD_BENCHMARK=OFF" |
| 3 | `benchmark::benchmark` already defined by an enclosing project | `[inferred]` — not reachable from this repo as top level |
| 4 | counters requested, JSON has none | `[confirmed]` `run_bench.sh` exits **65**, renames to `pmu.no-counters.json`, lists `missing: CYCLES INSTRUCTIONS` |
| 5 | no counters requested (the ordinary path) | `[confirmed]` exits 0, record written, host_name stripped — the guard does not fire on innocent runs |
| 6 | counters requested **and present** | `[confirmed]` exits 0, prints `PMU counters present`, no rename |
| 7 | counters requested, filter matched nothing | `[confirmed]` exits 65 naming `--benchmark_filter` rather than blaming libpfm4 |

Rows 4 and 5 together are the discrimination check: the gate distinguishes the case it exists for
from the case it must not disturb.

Row 6 needed a trick, because this machine cannot produce a counter. The check is a substring test
for `"<name>":` in the JSON, so passing `--benchmark_perf_counters=items_per_second` — a field
Google Benchmark always emits — drives the success branch with everything else unchanged. That
exercises the path the Linux host will take (empty `missing` array, no rename, exit 0) and, with
`/bin/bash 3.2.57`, confirms `${#missing[@]}` on an empty array is safe under `set -u`, which is the
one thing that would have broken the success path on a machine none of us can reach. It proves the
script's control flow, not libpfm4.

## Not done

The success path. `-DDTWC_BENCHMARK_PMU=ON` has never configured anywhere, because it stops on
`find_package(PFM REQUIRED)` on every machine available here. V-5 carries the command and the proof.
