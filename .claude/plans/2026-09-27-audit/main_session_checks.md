# Main-session checks (2026-09-27, HEAD cd5d449) — lines opened by the orchestrator, not agents

Confirmed:
- Pruned fill re-runs an abandoned pair with a full recompute: `dtwc/core/pruned_distance_matrix.cpp:290-294` (`dist = dtw_with_abandon(-1.0)` after abandon). Auto selects Pruned: `dtwc/Problem.cpp:1189-1193`.
- MPI filler reached only by `benchmarks/bench_mpi_dtw.cpp` and the umbrella `dtwc/dtwc.hpp:60`; no Problem / CLI / binding caller.
- GPU one-vs-all / k-vs-all: declared only in `dtwc/metal/metal_dtw.hpp:132-160` (and defined in the .cu/.mm); no caller outside the backends.
- v1.0.0 CLI flags (`git show v1.0.0:dtwc/dtwc_cl.cpp:43-54`: `--Nc/--number_of_clusters`, `--probName`, `--in/--out`, `--skipRows`, `--skipCols/--skipColumns`, `--maxIter/--iter`, `--repeat/--Nrepeat/--Nrepetition/--Nrep`, `--mip_solver/--mipSolver`, `--bandwidth/--bandw/--bandlength`, `--distMat/--distance_matrix/--distances`) are absent at HEAD (git grep in dtwc/: 0 hits). Kept at HEAD: `--clusters`, `--name`, `-i/--input`, `-o/--output`, `--method`, `--solver`.
- `load_checkpoint` returns `false` on identity/N mismatch and `catch (...) { return false; }` swallows every exception → silent recompute: `dtwc/checkpoint.cpp:626-667`.
- `route_series_storage` builds the `.dtws` store from an already heap-resident `Data` (`MmapDataStore::create(cache, resident)`, `dtwc/DataLoader.hpp:200-222`), so the auto-spill cannot serve data larger than RAM.
- `dtwcpp` was never on PyPI (`https://pypi.org/pypi/dtwcpp/json` → 404). The PyPI package `dtwc` is unrelated (a wallpaper changer). v1.0.0 Python users could only build from a source checkout.

Baseline on this machine (Windows 11, LLVM clang + Ninja, Release, `build/`, Arrow ON, HiGHS ON, Gurobi found, CUDA OFF):
- build exit 0; project warnings: 4 × `getenv` deprecation (MSVC CRT).
- `ctest --test-dir build -C Release -j1`: 142 tests — 136 pass, 5 `MAY_SKIP` (2 CUDA: no CUDA build; 3 Metal), **1 FAIL: `test_config_spellings`** case "Config{} holds the defaults dtwc_cl reports": 16 CHECKs such as `"cpu" == "cpu"` fail (Device, Dtype, GPU Prec, CLARA sample_size / n_samples / seed, Linkage) — strings print equal, so a hidden character on Windows [inferred, not diagnosed].
- Machine: 24 cores; CUDA 13.0 + RTX 4000 Ada (sm_89) — nvcc needs an MSVC host compiler (separate build dir, e.g. `build/cuda-verify`); MATLAB R2024a / R2024b / R2025b on PATH; MS-MPI; Gurobi 13.0.1; uv; no Metal (macOS CI only).

Owner preferences on record (auto-memory): design approved before code; small steps; max 4 concurrent agents; prefer maintained libraries over custom platform code; lock-free, high-performance hot paths; "no custom binary formats (HDF5 + CSV)" for data; CasADi-style cross-language consistency; targets 100M series × 8K samples via CLARA, GPU, HPC.
