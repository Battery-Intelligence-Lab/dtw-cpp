# F22 final serial full-gate adjudication

Date: 2026-07-30 (Europe/London)

## Scope and immutable prior verdict

Current clean HEAD before execution:

```text
e0df96ea09d5f7c7a6dad620c399ec12edbcbb32
```

This run completes only the already-registered F22 full-gate adjudication.
It does not rerun or reinterpret the exhausted C++ mutation campaign.
The F22 C++ mutation band remains **FALSIFIED at 33/46** after both permitted
attempts, exactly as recorded in
`.claude/baselines/2026-07-29-f22-deprecation-policy.md`.

## Registered bands

These bands are copied unchanged from the 2026-07-29 F22 record and
`AGENTS.md` before any decisive run:

- canonical `build/highs-1151`: 122/122, zero failed, exactly six capability
  skips (`test_cuda_correctness`, `test_cuda_lb_keogh`, `test_io_readers`,
  `test_metal_correctness`, `test_metal_lb_keogh`, `test_metal_mmap`);
- llfio-OFF `build/nollfio`: 122/122, zero failed, exactly nine capability
  skips (the six above plus `unit_test_mmap_data_store`,
  `unit_test_mmap_distance_matrix`, and `unit_test_benders`);
- Arrow-ON `build/arrow-pyarrow-23`: 124/124, zero failed, exactly eight
  capability skips, with `test_io_readers` executing 390 assertions in
  11 cases and no skip marker;
- Python focused F22: 18/18, the exact
  `F22_PYTHON_GATE ... verdict=PASS` marker, a forced-fresh built/imported
  extension identity, and no skip;
- Python full: 1,041 collected and either
  `1028 passed, 12 skipped, 1 failed` with the sole F39 inventory failure, or
  the arithmetically equivalent `1029 passed, 12 skipped` if F39 has closed;
- MATLAB focused F22: 1/1 on both R2024b and R2025b through the same
  forced-fresh OpenMP MEX, exact F22 marker, no skip;
- MATLAB full on each release: 85 total, 82 passed, two failed, three
  incomplete; the failed names are exactly the two retained F18 cases and the
  only additional incomplete name is exactly
  `test_test_api/test_parallelisation_serial_is_honest`;
- every native matrix and MATLAB release runs serially. No configured matrices
  run concurrently because their tests share source-root artifacts.

## Adversarial pre-run checks

The following checks were completed before any build or test:

1. `git status --short` printed no entries.
2. `git diff --name-status 43e1c44..HEAD` contains only campaign records,
   `AGENTS.md`, and `PLAN.md`; no F22 product/test source changed after the
   documented policy commit.
3. `ctest -N` reports exactly 122, 122, and 124 tests in the canonical,
   llfio-OFF, and Arrow-ON build directories respectively.
4. Cache inspection confirms Release/Ninja and the intended feature matrix:
   canonical = Arrow OFF / HiGHS ON / llfio ON; llfio-OFF = Arrow OFF /
   HiGHS OFF / llfio OFF; Arrow-ON = Arrow ON / HiGHS OFF / llfio OFF.
5. A dry build reports that all three directories must first re-run CMake, so
   no stale “ninja: no work to do” claim is assumed.
6. In all three generated CTest inventories, `test_problem_api_2_0` launches
   `scripts/test_f22_cpp_deprecations.py`. Its `PASS_REGULAR_EXPRESSION`
   requires the exact 33/33 diagnostic marker, the exact 33/33 behavior
   marker, and the Catch2 assertion/case floor; its failure regex rejects skip
   text.
7. Python collection reports 18 focused cases and 1,041 full cases. The
   current package/core provenance is:

   ```text
   PKG=C:\D\git\dtw-cpp\python\dtwcpp\__init__.py
   CORE=C:\D\git\dtw-cpp\.venv\Lib\site-packages\dtwcpp\_dtwcpp_core.cp313-win_amd64.pyd
   CORE_SHA256=9FE527ADC2C813FF59618D34000D6C9707536131DD7033DDD255F1C88AD54165
   F22_NATIVE=True
   HIGHS=False
   ```

   The decisive gate will nevertheless force a native rebuild, copy the built
   extension and `libomp.dll`, and re-check built/imported hashes.
8. The pre-existing MEX and Python extension timestamps follow their owning
   source timestamps, but neither timestamp is accepted as final provenance;
   both decisive binding gates force compilation.
9. A combined Python discovery/provenance wrapper printed the correct
   inventories but exceeded its 60-second outer timeout. No DTWC++ child
   process remained. This is classified as a harness failure and contributes
   no pass/fail evidence; subsequent checks use one bounded process per
   subject.
10. An independent read-only adversarial review found that the native
    `LastTestsFailed.log` files still name earlier focused failures
    (`test_problem_api_2_0` in canonical/llfio-OFF and
    `unit_test_distance_matrix_csv` in Arrow-ON). More importantly, the C++
    mutation campaign restored source bytes but did not leave a separately
    evidenced final clean native rebuild after the timed-out runtime mutant.
    Source identity therefore cannot certify object-file identity. All three
    native matrices and both binding artifacts must be rebuilt clean-first
    before receiving evidence credit.
11. The F22 record's R6 bullets are registered expectations, not completed
    final-run evidence; this log is the owner of the missing verbatim outputs.
12. `AGENTS.md` still states the pre-F22 Python and MATLAB floors. The
    preregistered F22 ledgers above govern this run; the working-rule floors
    will be updated only after the decisive inventories reproduce them.

Verdict before execution: **CONDITIONAL GO only after clean-first rebuilds;
no test credit yet.**

## Decisive outputs

Pending.
