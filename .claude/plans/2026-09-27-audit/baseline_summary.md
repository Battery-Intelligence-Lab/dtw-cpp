# Baseline — Windows 11, clang (LLVM) + Ninja, Release, build/ (Arrow ON, HiGHS ON, Gurobi found, CUDA OFF)
HEAD cd5d449. 2026-09-27.
- cmake --build build: exit 0. Project warnings: 4 × MSVC-CRT `getenv` deprecation (tests/conformance/cpp_conformance.cpp ×2, dtwc/dtwc_cl.cpp, dtwc/base/env.cpp); rest third-party (Gurobi, HiGHS, llfio).
- ctest -j1: 142 tests, 136 pass, 5 MAY_SKIP skips (test_cuda_correctness, test_cuda_lb_keogh — no CUDA build; test_metal_{correctness,lb_keogh,mmap} — Windows), 1 FAIL: test_config_spellings,
  case "Config{} holds the defaults dtwc_cl reports": 16 CHECKs like `"cpu" == "cpu"` fail (Device, Dtype, GPU Prec, CLARA sample_size/n_samples/seed, Linkage) — values print equal → hidden char / Windows-only harness issue [inferred]. 222 s total.
- Logs: scratchpad/logs/baseline_build.log, baseline_ctest.log

# Phase A head (9422a92), Windows, 2026-09-28 — reference for phase B
- `ctest --test-dir build -C Release -j1`: 141 tests, 136 pass, 5 MAY_SKIP (test_cuda_correctness, test_cuda_lb_keogh,
  test_metal_{correctness,lb_keogh,mmap}), 0 fail. (`build/` builds WITHOUT Arrow: Arrow not found with clang on Windows.)
- Gates: check_docs.py PASS (428 flags, 59 pages), check_pins.py PASS (cmake=12 actions=40), generate_docs --check current.
- pytest (fresh uv venv, Python 3.12, wheel built by scikit-build-core/MSVC from the checkout, `.[test,dev,io]` + matplotlib,
  DTWC_CL_PATH=build/bin/dtwc_cl.exe): 1174 passed, 19 skipped, 0 failed (174 s).
- MSVC + CUDA (`build/cuda-verify-0928`, at 0068386): 141 tests, 134 pass, 5 skip, 2 fail (squared-L2 last-bit
  cross-path `==`: test_problem_metric, test_run_resolution); 4/4 CUDA tests pass.
- MATLAB suite: not yet run on this machine.
