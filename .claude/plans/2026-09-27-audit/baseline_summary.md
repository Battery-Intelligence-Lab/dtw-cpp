# Baseline — Windows 11, clang (LLVM) + Ninja, Release, build/ (Arrow ON, HiGHS ON, Gurobi found, CUDA OFF)
HEAD cd5d449. 2026-09-27.
- cmake --build build: exit 0. Project warnings: 4 × MSVC-CRT `getenv` deprecation (tests/conformance/cpp_conformance.cpp ×2, dtwc/dtwc_cl.cpp, dtwc/base/env.cpp); rest third-party (Gurobi, HiGHS, llfio).
- ctest -j1: 142 tests, 136 pass, 5 MAY_SKIP skips (test_cuda_correctness, test_cuda_lb_keogh — no CUDA build; test_metal_{correctness,lb_keogh,mmap} — Windows), 1 FAIL: test_config_spellings,
  case "Config{} holds the defaults dtwc_cl reports": 16 CHECKs like `"cpu" == "cpu"` fail (Device, Dtype, GPU Prec, CLARA sample_size/n_samples/seed, Linkage) — values print equal → hidden char / Windows-only harness issue [inferred]. 222 s total.
- Logs: scratchpad/logs/baseline_build.log, baseline_ctest.log
