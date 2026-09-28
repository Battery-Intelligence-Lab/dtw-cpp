# Phase A measurements (2026-09-28, Windows, machine shared with COMSOL and builds — wall-clock advisory)

## Q2 — mio vs llfio (agent report, verbatim)

**mio is as fast as llfio wherever the file setup is the same. In this measurement it was 1.8× faster than the library's current llfio setup on random reads after reopen.** The speedup comes from llfio making a sparse file on Windows, not from the mapping itself. With sparse creation turned off, llfio matches mio on every access phase.

Probe: packed triangle including the diagonal (`tri_index`), N=20000, 1.60 GB. Single thread, clang 21 at -O3 -march=native. Five interleaved runs; each cell is median seconds [min, max]. Checksums were identical across all four backings in every run. The machine was busy during the run (2 ninja, dtwc_cl and python agents), so absolute times are rough and should be re-run on a quiet machine.

| phase | llfio (as library) | mio | heap | llfio, not sparse |
|---|---|---|---|---|
| a create+map | 0.027 [0.001,0.029] | 0.070 [0.009,0.121] | 0.553 (zero-fill) | 0.005 |
| b fill | 1.448 [0.97,1.66] | 1.232 [0.86,1.55] | 0.132 | 1.508 [0.92,1.77] |
| c sync | 0.450 [0.34,0.52] | 0.506 [0.37,0.60] | 0 | 0.454 |
| d1 close | 0.010 | 0.119 [0.075,0.213] | 0 | 0.011 |
| d2 reopen+map | 0.159 | 0.047 | 0 | 0.087 |
| e 1e8 random reads | **6.785 [6.09,8.00]** | **3.704 [2.96,4.94]** | 2.428 | **3.575 [2.91,4.52]** |
| f 2000 row gathers | 0.271 [0.18,0.35] | 0.247 [0.21,0.33] | 0.303 | 0.292 |
| g sync after 1 MiB dirty (extra) | 0.074 [0.041,0.089] | 0.127 [0.068,0.168] | 0 | 0.061 |

An earlier 3-backing run showed the same pattern: random reads 7.72 s for llfio and 4.79 s for mio.

**Verdict against the pre-registered criterion (±5 % or within the measured spread):**
- **Fill, sync and row gathers:** equivalent. The mio/llfio ratios (0.85, 1.13, 0.91) all fall inside the overlapping ranges.
- **Random reads:** the hypothesis fails for the library as it is configured today. mio/llfio is 0.55 and the ranges do not overlap.
- **Cause, tested directly:** the only change in the fourth backing is `flag::win_disable_sparse_file_creation`, and it brings llfio to 3.58 s against mio's 3.70 s. llfio makes new files sparse by default on Windows (`llfio-src/.../windows/file_handle.ipp:158,170`). The library uses `creation::only_if_not_exist` (`mmap_distance_matrix.hpp:626`), so its cache files are sparse.
- **Heap is still faster (2.43 s):** the likely reason is first-touch page faults after reopen, which both mapped backings pay. I did not test this.
- **So the pre-registered hypothesis holds once file attributes are equal.** The mapping itself costs nothing extra; the sparse file does.

**One-time costs (mio compared with llfio):**
- **Create:** 70 ms against 27 ms. mio's figure is an ofstream create plus `resize_file`. Both are small next to the fill.
- **Close:** mio 119 ms against 10 ms, because the mio destructor runs a full sync with errors ignored (`mio.hpp:975-978`, `1212-1219`).
- **Reopen:** mio 47 ms against 159 ms.
- **Sync of a small dirty block:** mio takes about 1.7× longer. Its Windows sync does `FlushViewOfFile(whole map)` + `FlushFileBuffers` twice (`mio.hpp:1127-1128`, `1138`). The library's `wait_data_only` does one `NtFlushBuffersFileEx(FILE_DATA_ONLY)` (`mapped_file_handle.hpp:218-240`, `byte_io_handle.ipp:263-280`). This matters for the design's "one sync() per row block".

**API checks against the mio source:**
- **Paths:** `std::filesystem::path` is not accepted, because `c_str` needs `.data()` (`mio.hpp:732-738`). Pass `p.native()`, which goes to `CreateFileW` (`826-839`), or a UTF-8 `std::string`, which goes through `CP_UTF8` to `CreateFileW` (`797-820`). Both work: tested with a non-ASCII name. `path.string()` would use the ANSI code page, so avoid it.
- **Mappings over 4 GB:** 64-bit size and offset are split correctly (`786-795`, `921-937`) and the size comes from `GetFileSizeEx` (`887`). A 5 GiB map worked on x64: writing the first and last double, syncing and reading back via the UTF-8 path gave the right values (`check_4g.exe`).
- **Errors:** every call reports a `std::error_code` from `system_category` (`848-858`); a missing file gave code 2. Two problems:
  - `CreateFileMapping` returns NULL on failure, but mio compares it with `INVALID_HANDLE_VALUE` (`928`, issue #102). The failure still surfaces, but from `MapViewOfFile`, with the wrong code.
  - `unmap` ignores errors (`1150`).
- **Sync:** it always covers the whole mapping; there is no ranged sync. It is `MS_SYNC` on POSIX (`1130`).
- **No create, truncate or exclusive create:** mio only opens existing files (`OPEN_EXISTING` at `821`, `O_RDWR` at `872-873`). The library's exclusive create (`only_if_not_exist`) needs its own step, such as `fopen "wx"`. Once the lease is gone, that is the only protection against two runs creating the same file.
- **No `FILE_SHARE_DELETE`** (`819`, `834`), which llfio sets (`file_handle.ipp:45`). On Windows the file cannot be deleted or renamed while it is mapped.
- **POSIX write maps use `PROT_WRITE` without `PROT_READ`** (`949`). This works on Linux and macOS in practice but POSIX does not guarantee it.
- **Nothing else is lost:** after the cut, the library needs create, size, map, sync, file length and reopen. mio covers all of these, with `resize_file` for sizing. The lock and ranged barrier are only needed for the lease and header publication, which the design drops.
- **Not measured:** without a sparse file, NTFS may zero-fill up to the highest written page at the first flush. That would be at most one extra write of the file, and a parallel fill that writes the tail first would trigger it. The 5 GiB tail write plus sync took 2.1 s in total, which is consistent with this.

**Open issues (52 open, including PRs; last code commit 2023-03-03, no tagged release, #115/#116):**
- No data-loss reports.
- Windows-relevant: #102 (the wrong error check above), #83 (file size looks stale while another handle is writing), #51 (2019, "invalid parameter" on a 4 GB file), and PR #105 (32-bit Windows offsets). I did not reproduce #51 on x64.
- #81/#75 (the `s_2_ws` multiple-definition link error) are fixed at the pinned commit (`797`, `inline`).
- #53 (POSIX `mmap` rather than `mmap64`) does not matter on 64-bit builds.

mio pin: commit `8b6b7d878c89e81614d05edca7936de41ccdd2da`, `single_include/mio/mio.hpp`, sha256 `634db76c...a7`. The 1.6 GB and 5 GiB files are deleted and `data/` is empty.

**Re-run command** (Git Bash, quiet machine):
`sh "C:/Users/engs2321/AppData/Local/Temp/claude/c--D-git-dtw-cpp/cfebc8b3-5c47-4d5a-b010-07838c4a9144/scratchpad/mio_ab/run.sh" 20000 5`
It builds `probe.exe` if missing, writes `results_<stamp>.csv`, prints the median table and deletes its data files.

Files are in `C:\Users\engs2321\AppData\Local\Temp\claude\c--D-git-dtw-cpp\cfebc8b3-5c47-4d5a-b010-07838c4a9144\scratchpad\mio_ab\`:
- `probe.cpp`
- `build.sh`
- `run.sh`
- `summarize.py`
- `check_4g.cpp`
- `MIO_PIN.txt`
- `RERUN.txt`
- `results_20260928_031218.csv` (3-backing run)
- `results_20260928_031733.csv` (4-backing run)
## Q3 — OneBatchPAM vs CLARA (agent report, verbatim)

**Verdict: TRADE-OFF, so they are not equivalent.** In 8 of 10 cells OneBatchPAM gets a total cost 1.4–4.7 % lower. CLARA uses fewer DTW evaluations whenever k ≤ 20. OneBatchPAM uses fewer at k=100. Neither method dominates the other, so under the owner's rule both can stay.

**How they differ (from the code)**
- **OneBatchPAM.** Every one of the N points can become a medoid (`one_batch_pam.cpp:334-335`). Swaps are judged on an estimated cost over one fixed batch of m = max(64, 20·⌈log2(N+1)⌉) points (`:28-34`, `:174-191`, `:340-363`). The final assignment of all N points is exact (`:383-397`).
  Work = (N−1)(m + k_out), where k_out ≤ k is the number of chosen medoids outside the batch (`:106-128`, `:193-206`). It does not depend on the number of sweeps.
- **CLARA.** Only points in the current subsample can become medoids (`fast_clara.cpp:510-549`). FastPAM minimises the exact cost on the s×s subsample. Each subsample's medoids are then scored by the exact cost over all N, and the best is kept (`:559-569`). Defaults: s = max(40+2k, min(N, 10k+100)) (`:83-88`), n_samples = 5 (`fast_clara.hpp:40`).
  Work = n_samples·[s(s−1)/2 + k(N−1)]. This is a lower bound, because the pruned fill of the sub-matrix can repeat abandoned attempts.

**PREREG** (full text in `obp_clara\PREREG.md`, written before any run)
- **Data:** synthetic families of smoothed random walks (z-normalised) with sinusoidal time warps up to 8 % of L, amplitude 0.8–1.2, offset sd 0.2, noise sd 0.3. L = 128, k_true = k, fixed seeds. One real set: UCR ECG5000 (local copy, TRAIN+TEST, N = 5000, L = 140).
- **Grid:** N ∈ {2k, 10k, 50k} × k ∈ {5, 20, 100}, skipping k ≥ N/20; ECG5000 at k = 5 and 20. band = 10 % of L (13 synthetic, 14 ECG). Seeds 1–3, run order rotated between seeds. PAM as the reference where N ≤ 10k.
- **Metrics:** exact cost from the CLI "Total cost:" line; work = DTW evaluations (formulas above; PAM = N(N−1)/2); median wall-clock; ARI against the true labels.
- **Rule:** as given in the task.
- **Prediction:** the work formulas alone predicted TRADE-OFF.

**Checks**
- Both methods report the exact cost, not a batch estimate. For all 18 runs at N = 2000, recomputing the cost from their medoids with the PAM distance matrix matched the reported value to ≤ 1e-6.
- OneBatchPAM work is rebuilt from the verbose "distance-matrix fraction", which is printed to 3 significant digits, so it carries ≤ 0.5 % error.

**Results** (median of 3 seeds; C = clara, O = onebatch, P = pam)

| cell | cost C/O | work C/O | time C/O | ARI C / O / P | C/P | O/P |
|---|---|---|---|---|---|---|
| N2k k5 | 1.019 | 0.24 | 0.48 | 0.996 / 1.000 / 1.000 | 1.028 | 1.009 |
| N2k k20 | 1.014 | 0.88 | 0.62 | 0.923 / 0.945 / 0.925 | 1.029 | 1.014 |
| N10k k5 | 1.026 | 0.11 | 0.15 | 1.000 / 1.000 / – | 1.028 | 1.002 |
| N10k k20 | 1.039 | 0.41 | 0.50 | 0.944 / 0.984 / – | 1.048 | 1.009 |
| N10k k100 | **0.987** | **2.11** | 0.55 | 0.727 / 0.703 / – | 1.040 | 1.053 |
| N50k k5 | 1.033 | 0.08 | 0.09 | 0.999 / 0.999 / – | – | – |
| N50k k20 | 1.041 | 0.31 | 0.18 | 0.956 / 0.973 / – | – | – |
| N50k k100 | 1.002 | **1.33** | 0.31 | 0.653 / 0.672 / – | – | – |
| ECG k5 | 1.047 | 0.14 | 0.18 | 0.505 / 0.407 / – | 1.052 | 1.005 |
| ECG k20 | 1.037 | 0.52 | 0.29 | 0.157 / 0.126 / – | 1.063 | 1.025 |

- **Seed spread** of cost within a cell is ≤ ±0.8 % for both methods. The 1.4–4.7 % cost gaps are larger than that; the k=100 cells (1.3 % and 0.2 %) are near it.
- **Largest wall-clock:** OneBatchPAM at N = 50k, k = 100 took 141 s; PAM at N = 10k took 81–117 s. No cell was dropped. Total compute was about 40 minutes.

**Reading the results**
- **k ≤ 20:** OneBatchPAM gives lower cost (close to PAM, 0.2–2.5 % above it) for 1.1–12× more DTW evaluations. CLARA is 2.8–6.3 % above PAM.
- **k = 100:** OneBatchPAM does less work (1.3–2.1× fewer evaluations), and the costs are equal or CLARA's is 1.3 % better.
- **Wall-clock:** CLARA is faster in every cell, even where OneBatchPAM does fewer DTW evaluations. This is an inference from the code, not measured: OneBatchPAM's swap loop (`:332-366`) and its final exact assignment (`:383-393`) are serial, while CLARA's fill and assignment use OpenMP. So OneBatchPAM's time is set by its implementation, not by how many DTW calls it makes.

**Caveats**
- The machine was shared with other builds and measurement agents, so wall-clock is advisory. Costs and work counts are deterministic.
- Both methods ran with their defaults. An equal-work comparison (CLARA with a larger s or more n_samples) was not run; that is the test that decides whether OneBatchPAM's cost advantage holds at the same work.
- The synthetic families are regular and balanced. ECG5000 is the only real set and its classes are imbalanced, so its low ARI does not tell the methods apart.
- Only 3 seeds per cell.

**Re-running the timing on a quiet machine**
```
cd "C:/Users/engs2321/AppData/Local/Temp/claude/c--D-git-dtw-cpp/cfebc8b3-5c47-4d5a-b010-07838c4a9144/scratchpad/obp_clara" && mv results.jsonl results_busy.jsonl && uv run --no-project python run_grid.py && uv run --no-project python summarize.py
```

All files are in `C:\Users\engs2321\AppData\Local\Temp\claude\c--D-git-dtw-cpp\cfebc8b3-5c47-4d5a-b010-07838c4a9144\scratchpad\obp_clara\`:
- `PREREG.md`
- `gen_data.py`
- `run_grid.py`
- `summarize.py`
- `crosscheck.py`
- `results.jsonl`
- `logs\`
- `out\`
- `bin\`
## CUDA build proof (agent report, verbatim)

**Can this machine build and run the CUDA backend? Yes.** All 4 CUDA tests pass, and `dtwc_cl --device gpu` runs on the RTX 4000 Ada. The given recipe needs one extra flag: `CMAKE_CUDA_FLAGS` alone leaves CUDA off.

**Directory:** I used `build/cuda-verify-0928`, not `build/cuda-verify`. The old `build/cuda-verify` is a stale MSVC+CUDA build from 2026-09-02, and I was not allowed to delete an untracked `build*/` dir.

**Why the given recipe turned CUDA off:** `check_language(CUDA)` (CMakeLists.txt, about line 160) runs a separate probe project that ignores `CMAKE_CUDA_FLAGS`. Without `-allow-unsupported-compiler`, nvcc rejects MSVC 19.50 in that probe. The log says "Looking for a CUDA compiler - NOTFOUND", "CUDA requested but nvcc not found — disabling", and the summary shows `CUDA: OFF`. A standalone nvcc probe gives the same first error without the flag: `crt/host_config.h(164): fatal error C1189: unsupported Microsoft Visual Studio version! Only 2019..2022 supported`. With the flag it compiles.
**Fix:** pass `CMAKE_CUDA_COMPILER` explicitly. `check_language` is then skipped and `enable_language(CUDA)` does see the flag. No source change is needed.

**Configure (worked; in `scratchpad\cuda_cfg.bat`, run from `C:\D\git\dtw-cpp`):**
```
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --fresh -S . -B build/cuda-verify-0928 "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" -G Ninja -DCMAKE_MAKE_PROGRAM=C:/D/git/dtw-cpp/.venv/Scripts/ninja.exe -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DDTWC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DDTWC_BUILD_TESTING=ON -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_ARROW=OFF -DDTWC_ENABLE_LLFIO=OFF
```
- Result: "CUDA compiler identification is NVIDIA 13.0.48 with host compiler MSVC 19.50.35723.0", `CUDA: ON (13.0.48)`, arch 89.
- Configure took 602 s on the loaded machine; the earlier CUDA-off attempt took 113 s.
- OpenMP came out as MSVC `-openmp:experimental`, which is version 2.0.

**Build:** `cmake --build build/cuda-verify-0928 -j 10 -- -k 0` under vcvars64. Took 7 min 12 s; 692/694 steps done, exit 2.
- **Only failure (MSVC only):** `benchmarks/bench_openmp_schedule.cpp:48-50` — `error C2065: 'omp_sched_dynamic': undeclared identifier` and `error C3861: 'omp_set_schedule': identifier not found`.
- Cause: MSVC's `omp.h` declares the OpenMP 3.0 API only under `/openmp:llvm` (`_OPENMP_LLVM_RUNTIME`, omp.h:10); `/openmp:experimental` gets the 2.0 header.
- Proposed patch at `benchmarks/CMakeLists.txt:54`: wrap `add_executable(bench_openmp_schedule …)` in `if(NOT MSVC OR OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0)`. Alternatively configure with `-DOpenMP_RUNTIME_MSVC=llvm`. Neither is tested.
- All library, test and `dtwc_cl` targets built.

**ctest** (`ctest --test-dir build/cuda-verify-0928 -C Release -j1 --output-on-failure`, 184 s): 141 tests, 134 passed, 5 skipped, 2 failed.
- 141 rather than 142 because `test_io_readers` (Arrow off) and `unit_test_mpi` are not registered.
- Skipped: `unit_test_mmap_data_store` and `unit_test_mmap_distance_matrix` (llfio off), plus `test_metal_correctness`, `test_metal_lb_keogh`, `test_metal_mmap`.
- CUDA tests all **Passed**, none skipped: `test_cuda_correctness` (10.4 s), `test_cuda_kernel_override`, `test_cuda_launch_guards`, `test_cuda_lb_keogh`. `test_cli_device_matrix` (gpu=cuda) also passed.
- `test_config_spellings` passed on this build.

**MSVC-only test failures (real second-compiler findings):** the distance-matrix fill differs from a direct `dtw<double>` call by the last bit, only for the squared-L2 metric (`SquaredL2`).
- `test_problem_metric`, 5 failed assertions:
  - `test_problem_metric.cpp:109` `CHECK( prob.dist_by_ind(i,j) == dtwc::distance::dtw<double>(series[i], series[j], band, MetricType::SquaredL2) )`, e.g. `19.42691249460886382 == 19.42691249460886738` (band -1, i=0, j=1). The same kind of mismatch appears for pairs (0,2), (1,5) and (4,5).
  - `test_problem_metric.cpp:322` resumed checkpoint: `4.83893133432279754 == 4.83893133432279665`.
- `test_run_resolution`, 4 failed assertions: `test_run_resolution.cpp:148` `CHECK( matrix[i*6+j] == dtw<double>(…, -1, SquaredL2) )`, e.g. `520.20000000000004547 == 520.20000000000015916` and `460.79999999999995453 == 460.79999999999989768`.
- Likely cause (inferred, not tested): `/fp:contract` at `cmake/StandardProjectSettings.cmake:81` lets MSVC fuse multiply-adds differently in the two code paths. The L1 checks in the same loop pass, which fits.
- Next test: rebuild these two tests without `/fp:contract` and expect both to pass.

**`dtwc_cl` GPU smoke:** `bin/dtwc_cl.exe -i tests/conformance/data/conformance_series.csv -k 3 --device gpu -v` exited 0.
- It printed "CUDA DTW: 351 pairs [FP32] in 0.169376ms on NVIDIA RTX 4000 Ada Generation (compute 8.9, 20474 MB)", then pam, total cost 828, converged in 3 iterations.
- The distance matrix and labels are byte-identical to `--device cpu`. `--gpu-precision fp32` and `fp64` also exit 0 with byte-identical matrices. The fixture's values are integers, so FP32 is exact here.

**F42:** not reproduced here. The default `--gpu-precision auto` resolved to FP32 and ran cleanly in `dtwc_cl` and in the CUDA tests, and the ctest log has no `0xc0000005`, access violation or exception. F42 was recorded inside a MATLAB MEX, which this run did not cover, so it stays open for that context.

Files are in `C:\Users\engs2321\AppData\Local\Temp\claude\c--D-git-dtw-cpp\cfebc8b3-5c47-4d5a-b010-07838c4a9144\scratchpad\`:
- cfg.log
- cfg2.log
- build.log
- ctest.log
- cuda_cfg.bat
- cuda_build.bat
- cuda_ctest.bat