# 2026-10-01 — W14b: an option set ON that cannot be honoured stops the configure

Question: does every dependency option set ON without its dependency stop the configure naming the option, does a
default configure of each tree still succeed, and does the default MEX stop importing Gurobi? Branch pb/W14b, base
f61af854. Windows 11, clang 21.1.8 (Ninja), MSVC 19.50 + nvcc 13.0 for the CUDA tree, CMake 4.2.3. All `[confirmed]`.

## Configure-only probes (head) [confirmed]

Common: `cmake -S . -B build-probe-<n> -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
-DCMAKE_BUILD_TYPE=Release` plus the arguments below; every probe exits 1.

| Option | Extra arguments | First FATAL line |
|---|---|---|
| CUDA, no usable compiler (nvcc exists here; the clang host cannot drive it) | `-DDTWC_ENABLE_CUDA=ON` | `DTWC_ENABLE_CUDA=ON but there is no usable CUDA compiler (nvcc is missing, or it cannot compile with this host compiler).` |
| CUDA, macOS (simulated on Windows) | `-DDTWC_ENABLE_CUDA=ON -DAPPLE=ON` | `DTWC_ENABLE_CUDA=ON: CUDA is not supported on macOS.` |
| Metal off Apple | `-DDTWC_ENABLE_METAL=ON` | `DTWC_ENABLE_METAL=ON: Metal exists only on Apple platforms, and this is Windows.` |
| Arrow not found | `-DDTWC_ENABLE_ARROW=ON -DDTWC_ENABLE_HIGHS=OFF` | `DTWC_ENABLE_ARROW=ON: Arrow was not found, and the CPM build is not supported with Windows+Clang ...` |
| Gurobi not found | `-DDTWC_ENABLE_GUROBI=ON -DGUROBI_HOME=C:/does-not-exist` | `DTWC_ENABLE_GUROBI=ON but Gurobi was not found.` |
| OpenMP (unchanged) | `-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON` | `OpenMP NOT found — DTWC++ requires OpenMP for parallel execution.` |
| HiGHS, empty CPM dir | `-DDTWC_ENABLE_HIGHS=ON`, `CPM_SOURCE_CACHE` with an empty `highs/<hash>` | `DTWC_ENABLE_HIGHS=ON but HiGHS made no highs::highs target` |
| HiGHS not downloadable | same, cache without highs, `HTTPS_PROXY=http://127.0.0.1:9` | CMake's own: `Each download failed!` ... `Build step for highs failed: 1`; the line before it is `HiGHS: fetching (DTWC_ENABLE_HIGHS=ON; ... -DDTWC_ENABLE_HIGHS=OFF)` |
| YAML, llfio not downloadable | as HiGHS, `-DDTWC_ENABLE_YAML=ON` / `-DDTWC_ENABLE_LLFIO=ON` | the same two CMake lines; a status line names the option |
| Arrow found, no `Arrow::arrow_shared` (a config defining only `Arrow::arrow_static`) | `-DDTWC_ENABLE_ARROW=ON -DArrow_DIR=<dir with that config>` | `DTWC_ENABLE_ARROW=ON: Arrow was found but it defines no Arrow::arrow_shared target to link.` |
| MATLAB not found | `-DDTWC_BUILD_MATLAB=ON -DCMAKE_DISABLE_FIND_PACKAGE_Matlab=ON`, with and without `-DDTWC_BUILD_TESTING=ON` | `DTWC_BUILD_MATLAB=ON but MATLAB was not found.` (`bindings/matlab/CMakeLists.txt:9`); with testing on, tests/CMakeLists.txt:376 stops first: `DTWC_BUILD_MATLAB=ON but MATLAB (with its executable) was not found, so matlab_suite cannot be registered.` |

`-DMatlab_ROOT_DIR=C:/does-not-exist` alone is not a probe on this machine: it is only a hint, MATLAB R2025b is on PATH
and the configure exits 0 having found it. Arrow through the shim prints `Arrow + Parquet linked` and compiles with both
`DTWC_HAS_ARROW` and `DTWC_HAS_PARQUET`; with `-DArrow_DIR` alone it prints `Arrow linked, IPC only: Parquet not found
— reading a Parquet file raises`, configures, and compiles with `DTWC_HAS_ARROW` only. The shim defines
`Arrow::arrow_shared` and `Parquet::parquet_shared` (Parquet linking Arrow). The MEX recipe (clang, R2024b) configures
with testing off and on (`ctest -N` lists `matlab_suite`).

Before the change the empty-`highs/<hash>` probe exited 0 with `HiGHS: OFF` and a "No MIP solver" warning. Arrow found
through the shim (`-DArrow_DIR`, `-DParquet_DIR` = `build/arrow-pyarrow-23/pyarrow-config`) configures and
`ctest -N` lists `test_io_readers` (101 tests).

A build directory configured before the change has `DTWC_ENABLE_METAL:BOOL=ON` cached (the old default), so its next
configure stops with the Metal message; `cmake -S . -B build` on this worktree's pre-change tree did. The integration
tree `C:/D/git/dtw-cpp/build` also caches `DTWC_ENABLE_ARROW:BOOL=ON` with `Arrow_DIR-NOTFOUND`: after the merge it needs
`-DDTWC_ENABLE_METAL=OFF -DDTWC_ENABLE_ARROW=OFF`, or a fresh configure.

## Default configures (base → head) [confirmed]

Logs, outside the repo: `C:/D/git/wt/W14b-base-configure{,-cuda,-mex}.log` (saved before editing) and
`W14b-head-configure{,-cuda,-mex}.log`. The diff of each is the Gurobi lines (`Found Gurobi`, `Gurobi found and enabled`,
`Gurobi: ON` → `OFF`) plus the three `fetching` status lines. Main tree: `cmake --preset clang-win -DDTWC_BUILD_BENCHMARK=ON`;
CUDA tree: MSVC cl + nvcc, Ninja, sm_89, `-allow-unsupported-compiler`, LLFIO off (recipe in phaseA_measurements.md);
MEX tree: clang, Ninja, `-DDTWC_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=.../R2024b`.

## Tests [confirmed]

| Leg | Base | Head |
|---|---|---|
| serial ctest, main tree | 98 tests, 0 failed; Skipped: test_cuda_correctness, test_metal_correctness, test_metal_mmap | the same 98 names, 0 failed, the same 3 Skipped |
| `test_mip_backend_guards` | with Gurobi (default at base) 29 assertions in 5 cases | default 20 in 5; `-DDTWC_ENABLE_GUROBI=ON` 29 in 5. The 9 are the Gurobi legs of "Method::MIP above N = 200 runs the selected solver" and "The compact MIP refuses a model whose size does not fit int", which return early without Gurobi |
| CUDA tree (build + ctest -R cuda) | not run | test_cuda_correctness and test_cuda_launch_guards Passed (the first 143.8 s, not skipped) |
| pytest `tests/python` (wheel, CMAKE_GENERATOR unset) | 1124 passed, 19 skipped | 1124 passed, 19 skipped; the 16 skip lines identical |
| `tests/conformance/test_conformance.py` | 1 passed, 1 skipped (CLI route skips: it read `DTWC_CL_BIN`) | 2 passed |
| check_docs --cli, check_pins, generate_docs --check | not run | PASS (386 flags, 59 pages), 0 failures, current |

## MEX [confirmed]

`llvm-objdump -p dtwc_mex.mexw64 | grep "DLL Name"`: base (Gurobi found, default) lists `gurobi130.dll`; head lists no
Gurobi DLL (7,974,912 → 7,750,656 bytes). MATLAB R2024b with Gurobi's directory removed from PATH:
`dtwc_mex('test_parallelisation')` on the base MEX fails with "Invalid MEX-file ...: The specified module could not be
found."; on the head MEX it loads (`available=1`).

## The CUDA CI assertion bites [confirmed]

`grep -Eq 'CUDA: +ON'` on the configure log: real CUDA tree `CUDA: ON (13.0.48)` passes; main tree `CUDA: OFF` fails; with
`dtwc/cuda/CMakeLists.txt`'s condition forced to `if(FALSE)` a CUDA=ON configure exits 0, prints `CUDA: OFF`, and the
grep fails (mutation reverted).

## Not established

The workflows ran nowhere (CI is not run here): the macOS libomp step, the CLI build step and the `DTWC_CL_PATH` path on
Windows/macOS are untested. CUDA on a real macOS is simulated with `-DAPPLE=ON`. Arrow found without
Parquet builds IPC only (Parquet reads raise at run time), by design. The static-only Arrow stop was shown with a
hand-made config, not a real static Arrow install.
