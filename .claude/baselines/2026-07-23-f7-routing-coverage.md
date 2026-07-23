# F7 planner/guard coverage adjudication

Date: 2026-07-23

Base commit: `424738e` (`docs: record FastPAM index-width lesson`)

Build: `build/highs-1151` (canonical Arrow-OFF clang + Ninja + Release gate)

## Classification

**[confirmed]** The interrupted `unit_test_cli_args.cpp` addition is not F8.
It does not compare resident and streamed labels, medoids, or checkpoint bytes.
It adds executable coverage for F7's uncapped/exact-boundary Parquet planner
arithmetic plus the public `fast_clara` resident/forced-streaming guards that
remain reachable with Arrow disabled. F8 stays open.

`resolve_parquet_cli_plan` is a file-local production seam included into the
test translation unit with `DTWC_CL_NO_MAIN`; it is not a public API and this
test alone is not real-CLI reachability evidence. The real Arrow-OFF
`dtwc_cl.exe` is therefore a separate required smoke below.

## Bands registered before decisive runs

- **BAND-MUTANT-ZERO [HARD]:** removing the production
  `ram_limit == 0` disjunct makes the new focused case run and fail.
- **BAND-MUTANT-PREFLIGHT [HARD]:** weakening the production forced-streaming
  prerequisite from `missing limit OR missing path` to `missing limit AND
  missing path` makes the new FastCLARA guard case run and fail.
- **BAND-MUTANT-RESIDENT [HARD]:** inverting the production resident-series
  forced-streaming guard makes the new FastCLARA guard case run and fail on its
  expected message. Each mutant is restored immediately; no mutant is evidence
  for the final product.
- **BAND-FOCUSED [HARD]:** the repaired zero-limit planner case runs 5
  assertions and the FastCLARA guard case runs 4 assertions in Arrow-OFF, with
  zero failures and zero skips. The complete `unit_test_cli_args` and
  `unit_test_fast_clara` binaries run with zero failures and no skips.
- **BAND-REAL-CLI [HARD]:** fresh `dtwc_cl.exe` rejects CSV plus a non-zero
  `--ram-limit` with exit 1 and the exact “cannot be honoured for this input”
  remediation; the same CSV with no cap exits 0 and emits normal clustering
  artifacts.
- **BAND-CANONICAL [HARD]:** rebuild succeeds and CTest reports `114/114`,
  zero failed, with exactly the six registered capability skips.

## Runs

### Mutation probe: zero-limit disjunct

Temporary mutant:

```diff
- if (ram_limit == 0 || estimated_resident_bytes <= ram_limit)
+ if (estimated_resident_bytes <= ram_limit)
```

Focused command:

```text
.\build\highs-1151\bin\unit_test_cli_args.exe "Parquet CLI plan treats a zero RAM limit as uncapped"
```

Verbatim verdict:

```text
C:/D/git/dtw-cpp/tests/unit/unit_test_cli_args.cpp(669): FAILED:
  CHECK_FALSE( uncapped.stream_payload )
with expansion:
  !true

C:/D/git/dtw-cpp/tests/unit/unit_test_cli_args.cpp(670): FAILED:
  CHECK( uncapped.materialize_payload() )
with expansion:
  false

===============================================================================
test cases: 1 | 1 failed
assertions: 5 | 1 passed | 4 failed
```

The other two failures were the expected unexpected exceptions for uncapped
PAM/scalar and CLARA/directory plans. The production condition was restored
immediately.

**BAND-MUTANT-ZERO: PASS.**

### Mutation probe: forced-stream prerequisites

Temporary mutant:

```diff
- && (opts.ram_limit_bytes == 0 || opts.parquet_path.empty()))
+ && (opts.ram_limit_bytes == 0 && opts.parquet_path.empty()))
```

Verbatim verdict:

```text
C:/D/git/dtw-cpp/tests/unit/algorithms/unit_test_fast_clara.cpp(430): FAILED:
  REQUIRE_THROWS_WITH( algorithms::fast_clara(settings_only, opts), "fast_clara: force_parquet_streaming requires ram_limit_bytes and " "parquet_path." )
with expansion:
  "fast_clara: force_parquet_streaming requires a build with Parquet support."
  equals: "fast_clara: force_parquet_streaming requires ram_limit_bytes and
  parquet_path."

===============================================================================
test cases: 1 | 1 failed
assertions: 4 | 2 passed | 2 failed
```

The second failure was the corresponding empty-path section. The production
condition was restored immediately.

**BAND-MUTANT-PREFLIGHT: PASS.**

### Mutation probe: resident-series guard

Temporary mutant:

```diff
- if (opts.force_parquet_streaming && prob.size() != 0)
+ if (opts.force_parquet_streaming && prob.size() == 0)
```

Verbatim verdict:

```text
C:/D/git/dtw-cpp/tests/unit/algorithms/unit_test_fast_clara.cpp(449): FAILED:
  REQUIRE_THROWS_WITH( algorithms::fast_clara(resident, opts), "fast_clara: force_parquet_streaming requires a settings-only Problem " "without resident series." )
with expansion:
  "fast_clara: force_parquet_streaming requires a build with Parquet support."
  equals: "fast_clara: force_parquet_streaming requires a settings-only Problem
  without resident series."

===============================================================================
test cases: 1 | 1 failed
assertions: 4 | 2 passed | 2 failed
```

The second failure was the inverted settings-only capability section. The
production condition was restored immediately, and
`git diff --exit-code -- dtwc/dtwc_cl.cpp dtwc/algorithms/fast_clara.cpp`
returned zero before the final build.

**BAND-MUTANT-RESIDENT: PASS.**

### Repaired focused suites

Verbatim outputs:

```text
Filters: "Parquet CLI plan treats a zero RAM limit as uncapped"
Randomness seeded to: 3598608336
===============================================================================
All tests passed (5 assertions in 1 test case)

Filters: "FastCLARA forced streaming validates its route before reader I/O"
Randomness seeded to: 561316691
===============================================================================
All tests passed (4 assertions in 1 test case)

Randomness seeded to: 2852219247
===============================================================================
All tests passed (149 assertions in 25 test cases)

Randomness seeded to: 665885005
===============================================================================
All tests passed (842 assertions in 21 test cases)
```

**BAND-FOCUSED: PASS.**

### Real Arrow-OFF CLI

The fresh executable was rebuilt first. Capped CSV command verdict:

```text
Error: --ram-limit caps Parquet series materialisation and cannot be honoured for this input; drop --ram-limit, or convert the series to a list-per-row Parquet file to stream them under the cap.
CAP_EXIT=1
```

The same 27-series conformance CSV without `--ram-limit` reported:

```text
Reading data:
27 time-series data are read.

=== Results ===
  Method:     pam
  Clusters:   3
  Total cost: 828
  Converged:  yes
  Iterations: 3
  Output:     build\highs-1151\f7-cli-smoke-20260723/
UNCAPPED_EXIT=0
```

Artifacts emitted under the ignored canonical build directory:

```text
f7_uncapped_checkpoint.bin
f7_uncapped_distance_matrix.csv
f7_uncapped_labels.csv
f7_uncapped_medoids.csv
f7_uncapped_silhouettes.csv
```

**BAND-REAL-CLI: PASS.**

### Canonical gate

Commands:

```text
cmake --build build/highs-1151
ctest --test-dir build/highs-1151 -j4 -C Release --output-on-failure
```

Verbatim verdict:

```text
100% tests passed, 0 tests failed out of 114

Total Test time (real) =  30.19 sec

The following tests did not run:
	 48 - test_cuda_correctness (Skipped)
	 50 - test_cuda_lb_keogh (Skipped)
	 54 - test_io_readers (Skipped)
	 55 - test_metal_correctness (Skipped)
	 56 - test_metal_lb_keogh (Skipped)
	 57 - test_metal_mmap (Skipped)
```

**BAND-CANONICAL: PASS.**

## Verdict

**REPAIR / KEEP as F7 coverage.** The original mixed case overstated a static
planner seam as a public entry and mislabeled guard coverage as an “Arrow-OFF
half” of F8. The repaired tests separate the file-local CLI planner from the
public FastCLARA API, remove already-covered boundary/result-shape assertions,
and pin every forced-stream pre-reader guard reachable without Arrow.

F8 remains open in full: there is still no committed multi-row-group Parquet
fixture or resident-versus-stream comparison of f64/f32/Soft-DTW labels,
medoids, and checkpoint bytes.

Rollback: revert the test commit; production behavior and external state are
unchanged. The ignored smoke artifacts remain recoverable under
`build/highs-1151/f7-cli-smoke-20260723`.

The claim most likely to be wrong is that direct coverage of the static planner
will remain compositionally equivalent to its Arrow-ON CLI call site. Only the
F9 Arrow-ON executable gate plus F8's real Parquet parity fixture can confirm
that composition.
