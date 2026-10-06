# 2026-10-06 — macOS at `33302535` (W9b and the Mac pass merged): the Mac's second base check

Machine and toolchain as `2026-10-05-macos-design-2-0.md` (Apple M5 Pro, Apple clang 21, libomp, MATLAB R2026a; no
Arrow, CUDA or Gurobi). Purpose: the Windows handoff's step "Mac: pull, rebuild, pytest again (W9b)". Everything
below is `[confirmed]` unless marked.

## Build (`build/`, the existing clang-macos Release cache)

`cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp` (configure 0.1 s, cache already Gurobi OFF) and
`cmake --build --preset clang-macos`: 189 steps, zero warnings, exit 0.

## ctest (`ctest --test-dir build -C Release -j1 --output-on-failure`)

`100% tests passed out of 95`, `test_cuda_correctness` skipped (registered MAY_SKIP), 44.31 s. The codegen gate
(`test_codegen_no_calls`) passes inside that run.

## Conformance

`DTWC_CONFORMANCE_REGEN=1 ./build/bin/cpp_conformance` then `git diff tests/conformance/conformance_reference.txt`:
only `silhouette,0.96894972764334841 → 0.9689497276433483`, the one ulp of 2026-10-05 (D-19); reference restored.

## Gates

`check_docs.py --cli build/bin/dtwc_cl`: `DOCS flags checked=392 pages=59 live=60 VERDICT=PASS`;
`check_pins.py`: `PINS cmake=18 actions=37 failures=0`; `generate_docs.py --check`: current.

## Python (fresh venv, `uv venv --python 3.12`, `uv pip install ".[test,dev,io]" matplotlib`, wheel built in the venv)

`DTWC_CL_PATH=build/bin/dtwc_cl python -m pytest tests/python -q -p no:cacheprovider -rs`:
**970 passed, 12 skipped, 0 failed, 93.76 s** (982 ids, the count the Windows record reconciled: 962 + 20 there).
Skips: 9 `CUDA not available` (test_cuda.py), 1 `GPU present` (test_device.py:120), 1 `scipy IS installed`
(test_preprocess.py:111), 1 `could not import 'pandas'` (test_api.py:625, W9b's "pandas DataFrame" form of the
one-conversion case; `[io]` installs h5py and pyarrow, not pandas). One warning: sklearn's `check_array_api_input`
skipped (`SCIPY_ARRAY_API` unset), as before.

Note for the CI Python job (`.github/workflows/python-tests.yml:49` installs `".[test]"` only): the pyarrow, pandas
and scikit-learn cases skip there, so that gate does not prove them ran. Follow-up, not changed here (M1 edits that
file).

## AddressSanitizer + UBSan (`build-asan/`, incremental from `98e986fc`, same flags as 2026-10-05)

`cmake --build build-asan` (116 instrumented units, incremental; the only warnings are the 8 `-Wpass-failed` notes on
`z_normalize.hpp`'s `vectorize(enable)` pragma, as on 10-05) then
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --test-dir build-asan -C RelWithDebInfo -j1 -E test_codegen_no_calls --output-on-failure`:
**94/94 passed, `test_cuda_correctness` skipped, 62.20 s, no sanitizer report** (grep for `runtime error`,
`AddressSanitizer`, `UndefinedBehaviorSanitizer`, `SUMMARY`: 0 lines). W9b's byte-mode readers (Ctrl-Z, bare CR, the
UTF-8 messages) ran under both sanitizers for the first time.
