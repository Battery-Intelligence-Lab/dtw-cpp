# CUDA first runtime verification — 2026-07-07

Machine: local win32 dev box, NVIDIA RTX 4000 Ada Generation (compute 8.9, 20474 MB per `dtwc::test::gpu()` device_name), nvcc 13.0 + MSVC 14.50 host. Build dir: `build/cuda-verify` (`-DDTWC_ENABLE_CUDA=ON`; separate from the clang baseline dir — nvcc rejects clang 21 as host on Windows).

This is the FIRST run of the CUDA test suites — permanently skipped on every prior gate (no CUDA build until Phase 3 task 3.5). It runtime-verifies the Phase 0 CUDA audit fixes (task 0.1 wavefront max_L>2048 routing; task 0.7 int64 pair indexing) on real hardware.

## Orchestrator re-run (independent, 2026-07-07, this file's primary evidence)

`ctest -R cuda --output-on-failure` in `build/cuda-verify`, verbatim:

```text
    Start 26: test_cuda_correctness
1/2 Test #26: test_cuda_correctness ............   Passed    8.29 sec
    Start 27: test_cuda_lb_keogh
2/2 Test #27: test_cuda_lb_keogh ...............   Passed    0.25 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   8.55 sec
```

## Gate agent run (wave B gate, run `wf_abeeee64-25f`, journal.jsonl)

Direct binaries, verbatim per gate report: `"All tests passed (7312 assertions in 55 test cases)"` (test_cuda_correctness) and `"All tests passed (688 assertions in 8 test cases)"` (test_cuda_lb_keogh). Gate additionally built `test_test_api` with CUDA ON: `gpu available=1 backend=cuda device_name="NVIDIA RTX 4000 Ada Generation (compute 8.9, 20474 MB)" validated=1 pass=1` — real kernel vs FP64 CPU oracle.

## Scope honesty (adversarial-review finding M1, upheld)

The committed code delta of task 3.5 is one line (`#include <numeric>` in tests/unit/test_cuda_correctness.cpp — latent MSVC-STL compile break). The CUDA fat-binary arch list `60;70;75;80;86;89;90` PREDATES Phase 3 (git blame CMakeLists.txt:131 → 3550bb9 2026-04-09; :133-135 → 367a9b4 2026-04-04). Task 3.5's deliverable is the build recipe + this verification event, NOT new dispatch code. H100 (sm_90) perf claims remain ADVISORY until a real ARC run.
