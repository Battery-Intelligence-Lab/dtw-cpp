# 2026-10-08 — Windows after the Mac's batch (8a63cda3): every tree green; one Windows-only pytest failure

Question: does the Mac's batch (`33302535..263e0e24`, about 140 commits: L2b, M1, W9c/e/f, W13c/e, PF, PQ, pair-2col,
lr-omp, arm-lanes, W14a/c, FU) build and pass on Windows, and does it change any CLI output on x86? Code `8a63cda3`
(origin `263e0e24` plus TB's record). Run by two verification agents (Sonnet 5.5, xhigh) on a shared machine; logs
outside the repo in `C:/D/git/wt/tmp/int/wv_*`. The summary lines, the codegen line and the three CLI diffs below were
re-read from those logs by the orchestrator [confirmed].

| tree | build | ctest (serial) | notes |
| --- | --- | --- | --- |
| `build/` (clang++ 21.1.8, GNU-style flags) | no new warning (the 5 CRT `getenv`/`fopen` deprecations of bdc47410) | 94 = 91 passed + 3 MAY_SKIP (CUDA, Metal × 2) | W13e removed the barycenter allocation test (95 → 94); `CODEGEN_NO_CALLS tool=CLANG_~1.EXE inner_loops=154 calls=0 verdict=PASS` (146 loops at bdc47410: pair-2col's two-column loops); `cpp_conformance` passes; `check_docs` 423 flags, 58 pages, 0 undocumented; `check_pins` 0 failures; `generate_docs --check` current |
| `build/arrow-pyarrow-23` (pyarrow 23.0.1 shim) | built | 97 = 94 + 3 skips | `test_io_readers` ran its 20 cases (1,544 assertions), the 12 Parquet ones included: the first Windows run of L2b's `dtwc_io` with Arrow ON and of PQ's reader |
| `build/cuda-verify-0928` (MSVC 14.50 + nvcc 13.0) | 0 warnings (FU's C4244 casts hold) | 93 = 91 + 2 (Metal × 2); `test_cuda_correctness` ran | `mip-solvers` compiles with `/openmp:experimental` (lr-omp's `#error` satisfied) |
| `build/mex` (MATLAB R2024b) | built, HiGHS linked statically | `matlab_suite: 138 run, 138 passed, 0 failed, 0 incomplete` | W9e closed on Windows |
| wheel (MSVC cl 19.50, fresh venv, `.[test,dev,io,mip]` matplotlib pandas, `DTWC_REQUIRE_HIGHSPY=1`) | the binding file at `/O2 /Ob2 /GL /arch:AVX2 /fp:precise /fp:contract`, no `/Os`: NOMINSIZE holds on MSVC | pytest 886 passed, 19 skipped, 1 failed (906 ids, as the Mac's 895 + 11) | the failure is below |

CLI outputs against bdc47410's (`w9b_mac_*`): clang 156/156 files byte-identical; CUDA tree CPU 65/65 and GPU 51/51
byte-identical; the 16 GPU CLARA files equal after normalising wall-clock times and paths (cost 153333, the
"CUDA DTW: 75 pairs (25 series x 3 medoids)" line still printed). So pair-2col's two-column kernel, lr-omp and the
rest change no CLI output bit on x86 under clang++ or MSVC: the first non-Apple evidence for pair-2col's bitwise claim.

The pytest failure: `test_examples.py::test_example_runs[08_parquet_io.py]` (added by W9f, `6246172f`; its first
Windows run). The example runs to its end; `tempfile.TemporaryDirectory()` then cannot delete `timeseries.arrow`
(`PermissionError: [WinError 32]`): `ipc.open_file(str)` memory-maps the file and `reader`/`batch` stay alive after the
`with` block, and Windows refuses to delete a mapped file (POSIX does not). A probe with pyarrow 25.0.1 unlinked it
after `del batch, reader` and `gc.collect()` [inferred cause; the fix is unit WU8's]. Skips (19): 9 CUDA (CPU wheel),
5 no GPU, 2 Metal, `test_hpc.py:1000` (Git Bash's ssh/scp precede the fakes on PATH; Windows only),
`test_preprocess.py:111` (scipy installed), `test_typed_errors.py:39` (POSIX permission bits).
